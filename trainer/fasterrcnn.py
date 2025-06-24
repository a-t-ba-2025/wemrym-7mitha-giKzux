import os
import csv
import time
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # for progress bar
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from dotenv import load_dotenv

##################################################
########## Configuration #########################
##################################################
load_dotenv()  # load .env file
ROOT_DIR = os.getenv("ROOT_DIR")
TRAIN_JSON = os.path.join(ROOT_DIR, "coco", "train.json")
VAL_JSON = os.path.join(ROOT_DIR, "coco", "val.json")
IMAGE_DIR = os.path.join(ROOT_DIR, "png")
FASTRCNN_MODEL_DIR = "models/fasterRCNNmodels"

# Create folder if don't exist
os.makedirs(FASTRCNN_MODEL_DIR, exist_ok=True)

BATCH_SIZE = 4
START_EPOCH = 1  # Start training at this epoch (inklusive)
END_EPOCH = 2  # End training at this epoch (inklusive)
RESUME_NAME = None  # None=Do not load a checkpoint or z. B. "model_epoch_1.pth"
RESUME_PATH = os.path.join(FASTRCNN_MODEL_DIR, RESUME_NAME) if RESUME_NAME else None
LEARNING_RATE = 0.0005  # E1-E5: 0.005, E6-E10: 0.0005

TRAIN_SAMPLES = 100  #  None = All - max = 69375
VAL_SAMPLES = 50  #  None = All - max = 6489

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
print(f"used device: {DEVICE}")


##################################################
########## Collate Function ######################
##################################################
def collate_fn(batch):
    return tuple(zip(*batch))


##################################################
########## Class for COCO Format #################
##################################################
class CocoDetectionTransforms(CocoDetection):
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        img = F.to_tensor(img)  # Convert image to tensor

        # for images with no annotations
        if len(target) == 0:
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64)
            }
        else:
            # Convert bounding boxes from [x, y, width, height] to [x1, y1, x2, y2]
            boxes = torch.tensor([obj["bbox"] for obj in target], dtype=torch.float32)
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            labels = torch.tensor([obj["category_id"] for obj in target], dtype=torch.int64)
            target = {'boxes': boxes, 'labels': labels}

        return img, target


##################################################
########## Load model #############################
##################################################
def get_fasterrcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


##################################################
########## Load data #############################
##################################################
def get_dataloaders():
    train_set = CocoDetectionTransforms(root=IMAGE_DIR, annFile=TRAIN_JSON)
    val_set = CocoDetectionTransforms(root=IMAGE_DIR, annFile=VAL_JSON)

    # Use only part of the dataset if set and not none
    if TRAIN_SAMPLES is not None:
        train_set = torch.utils.data.Subset(train_set, range(TRAIN_SAMPLES))
    if VAL_SAMPLES is not None:
        val_set = torch.utils.data.Subset(val_set, range(VAL_SAMPLES))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"Loaded {len(train_set)} training and {len(val_set)} validation images.")
    return train_loader, val_loader


##################################################
########## run one epoch #########################
##################################################
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = []
    for images, targets in tqdm(loader, desc="Training"):
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())  # Total loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    return np.mean(total_loss)


##################################################
########## validate one epoch ####################
##################################################
def validate_one_epoch(model, loader):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validating"):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            model.train()
            loss_dict = model(images, targets)
            model.eval()

            loss = sum(loss for loss in loss_dict.values())
            total_loss.append(loss.item())

    return np.mean(total_loss)


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
def save_loss_history(epoch, train_loss, val_loss):
    loss_history_path = os.path.join(FASTRCNN_MODEL_DIR, "loss_history.csv")

    # load file,
    if os.path.exists(loss_history_path):
        df_old = pd.read_csv(loss_history_path)
        df_old = df_old[df_old["epoch"] != epoch]
    else:
        df_old = pd.DataFrame(columns=["epoch", "train_loss", "val_loss"])

    df_new = pd.DataFrame([{
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss
    }])

    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all = df_all.sort_values("epoch").reset_index(drop=True)
    df_all.to_csv(loss_history_path, index=False)
    print(f"Saved model and loss history to {loss_history_path}")


##################################################
########## Training ##############################
##################################################
def train_fasterrcnn():
    model = get_fasterrcnn_model(num_classes=12).to(DEVICE)

    #  resume training from a checkpoint if set
    if RESUME_PATH and os.path.exists(RESUME_PATH):
        model.load_state_dict(torch.load(RESUME_PATH, map_location=DEVICE))
        print(f"Resumed model from {RESUME_PATH}")

    # optimizer
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    # Load data
    train_loader, val_loader = get_dataloaders()
    train_losses, val_losses = [], []

    # Train
    for epoch in range(START_EPOCH, END_EPOCH + 1):
        print(f"\nEpoch {epoch}/{END_EPOCH}")
        start_time = time.time()

        # Run training and validation
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = validate_one_epoch(model, val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Save model
        model_path = os.path.join(FASTRCNN_MODEL_DIR, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_path)

        # Save loss history
        save_loss_history(epoch, train_loss, val_loss)

        print(f"Saved model and loss history for epoch {epoch}")
        duration = int(time.time() - start_time)
        print(f"Epoch duration: {duration // 60}m {duration % 60}s")

    # Plot loss
    plot_losses_from_csv(
        csv_path=os.path.join(FASTRCNN_MODEL_DIR, "loss_history.csv"),
        save_path=os.path.join(FASTRCNN_MODEL_DIR, "loss_curve.png")
    )


if __name__ == "__main__":
    train_fasterrcnn()
