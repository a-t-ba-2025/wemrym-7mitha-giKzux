import os
import csv
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor, DetrConfig
from dotenv import load_dotenv

##################################################
########## Configuration #########################
##################################################

load_dotenv()  # load .env

ROOT_DIR = os.getenv("ROOT_DIR")
TRAIN_JSON = os.path.join(ROOT_DIR, "coco", "train.json")
VAL_JSON = os.path.join(ROOT_DIR, "coco", "val.json")
IMAGE_DIR = os.path.join(ROOT_DIR, "png")
MODEL_DIR = "models/detrModels"
os.makedirs(MODEL_DIR, exist_ok=True)  # Create model directory if not exist

BATCH_SIZE = 4
START_EPOCH = 1  # Start training at this epoch (inklusive)
END_EPOCH = 2  # End training at this epoch (inklusive)
RESUME_NAME = None  # None=Do not load a checkpoint or "detr_epoch_1.pth"
RESUME_PATH = os.path.join(MODEL_DIR, RESUME_NAME) if RESUME_NAME else None
LEARNING_RATE = 5e-5  # E1-E20: 5e-5, E21-E35: 1e-5
MODEL_NAME = "facebook/detr-resnet-50"  # Pretrained DETR model from HuggingFace

TRAIN_SAMPLES = 100  #  None = All - max = 69375
VAL_SAMPLES = 50  #  None = All - max = 6489

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
print(f"used device: {DEVICE}")


##################################################
########## Collate Function ######################
##################################################
#  prepares batch during training by removing failed samples
def collate_fn(batch):
    batch = [b for b in batch if b is not None]  # Remove invalid items
    if not batch:
        return None
    pixel_values = torch.stack([b["pixel_values"] for b in batch])  # Stack images
    labels = [b["labels"] for b in batch]  # Keep list of annotations
    return {"pixel_values": pixel_values, "labels": labels}


##################################################
########## Class for COCO Format #################
##################################################
class CocoDetrDataset(Dataset):
    def __init__(self, image_dir, annotation_file, processor):
        self.image_dir = image_dir
        self.processor = processor

        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            coco = json.load(f)

        self.images = {img['id']: img for img in coco['images']}
        self.annotations = coco['annotations']
        self.categories = coco['categories']

        # Map category IDs to label-indices
        self.cat_id_to_label = {cat['id']: idx for idx, cat in
                                enumerate(sorted(self.categories, key=lambda x: x['id']))}

        # Group annotations by image
        self.image_to_anns = {}
        for ann in self.annotations:
            self.image_to_anns.setdefault(ann['image_id'], []).append(ann)

        self.image_ids = list(self.image_to_anns.keys())
        self.valid_count = 0

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        try:
            image_id = self.image_ids[idx]
            image_info = self.images[image_id]
            image_path = os.path.join(self.image_dir, image_info['file_name'])
            image = Image.open(image_path).convert("RGB")
            width, height = image.size

            # Prepare annotations for image
            anns = self.image_to_anns[image_id]
            coco_annotations = [
                {
                    "bbox": ann["bbox"],
                    "category_id": self.cat_id_to_label[ann["category_id"]],
                    "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),  # Falls area fehlt, berechne aus bbox
                    "iscrowd": ann.get("iscrowd", 0)
                }
                for ann in anns
            ]

            # DETR processor: prepare image and annotations
            encoding = self.processor(
                images=image,
                annotations={"image_id": image_id, "annotations": coco_annotations},
                return_tensors="pt"
            )

            # Normalize labels
            if isinstance(encoding.get("labels"), list) and len(encoding["labels"]) == 1:
                encoding["labels"] = encoding["labels"][0]

            # Remove batch dimension
            encoding = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in encoding.items()}
            self.valid_count += 1
            return encoding

        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            return None  # Skip sample if something goes wrong


##################################################
########## Training ##############################
##################################################
def train_detr():
    # Load or save processor
    processor_path = os.path.join(MODEL_DIR, "processor")
    if os.path.exists(processor_path):
        processor = DetrImageProcessor.from_pretrained(processor_path)
        print(f"Loaded processor from {processor_path}")
    else:
        processor = DetrImageProcessor.from_pretrained(MODEL_NAME)
        processor.save_pretrained(processor_path)
        print(f"Saved processor to {processor_path}")

    #  model config and initializing the model
    config = DetrConfig.from_pretrained(MODEL_NAME)
    config.num_labels = 11  # Number of labela in dataset
    model = DetrForObjectDetection.from_pretrained(MODEL_NAME, config=config, ignore_mismatched_sizes=True).to(DEVICE)

    # Load datasets
    train_dataset = CocoDetrDataset(IMAGE_DIR, TRAIN_JSON, processor)
    val_dataset = CocoDetrDataset(IMAGE_DIR, VAL_JSON, processor)

    # Use subset if  TRAIN_SAMPLES or bzw. and VAL_SAMPLES not none
    if TRAIN_SAMPLES is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(TRAIN_SAMPLES))
    if VAL_SAMPLES is not None:
        val_dataset = torch.utils.data.Subset(val_dataset, range(VAL_SAMPLES))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Optimizer (AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    #save losses- path
    losses_path = os.path.join(MODEL_DIR, "losses_detr.csv")
    all_train_losses, all_val_losses = [], []

    #  resume from saved model if set
    if RESUME_PATH:
        model.load_state_dict(torch.load(RESUME_PATH))
        print(f"Resumed from: {RESUME_PATH}")

    # Start training
    for epoch in range(START_EPOCH, END_EPOCH + 1):
        model.train()
        train_losses = []
        print(f"\nStarting epoch {epoch}...")

        for batch in tqdm(train_loader, desc="Training"):
            if batch is None:
                continue
            pixel_values = batch['pixel_values'].to(DEVICE)
            labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        mean_train_loss = np.mean(train_losses)
        all_train_losses.append(mean_train_loss)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                if batch is None:
                    continue
                pixel_values = batch['pixel_values'].to(DEVICE)
                labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]

                outputs = model(pixel_values=pixel_values, labels=labels)
                val_losses.append(outputs.loss.item())

        mean_val_loss = np.mean(val_losses)
        all_val_losses.append(mean_val_loss)

        print(f"Epoch {epoch}: Train Loss = {mean_train_loss:.4f}, Val Loss = {mean_val_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"detr_epoch_{epoch}.pth"))

        # Save losses in CSV
        with open(losses_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["epoch", "train_loss", "val_loss"])
            writer.writerow([epoch, mean_train_loss, mean_val_loss])

    # Draw loss curve and save
    plot_losses_from_csv(os.path.join(MODEL_DIR, "losses_detr.csv"),
                         save_path=os.path.join(MODEL_DIR, "loss_curve.png"))


##################################################
########## Plot loss #############################
##################################################
def plot_losses_from_csv(csv_path, save_path=None):
    if not os.path.exists(csv_path):
        print(f"No CSV file found at {csv_path}")
        return

    epochs, train_losses, val_losses = [], [], []

    with open(csv_path, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    plt.figure()
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses, marker='o', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Saved loss plot to {save_path}")
    plt.show()


##################################################
########## save loss #############################
##################################################
def save_loss_to_csv(epoch, train_loss, val_loss, csv_path):
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_old = df_old[df_old["epoch"] != epoch]  # remove ggf. existing Row
    else:
        df_old = pd.DataFrame(columns=["epoch", "train_loss", "val_loss"])

    df_new = pd.DataFrame([{
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss
    }])

    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all = df_all.sort_values("epoch").reset_index(drop=True)
    df_all.to_csv(csv_path, index=False)
    print(f"Saved updated loss entry for epoch {epoch} to {csv_path}")


if __name__ == "__main__":
    train_detr()
