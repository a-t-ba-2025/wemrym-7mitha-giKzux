# Trainer_DocLayNet_FRCNN_DETR

This project contains training and evaluation scripts for **Faster R-CNN** and **DETR** models on the **DocLayNet** dataset.

## Requirements

### System Requirements

- Python **3.10**
- CUDA-compatible GPU recommended (tested with CUDA 11.8)
- Windows 10 or 11
- At least 16 GB RAM (more is recommended for large-scale training)

> **Note:** Make sure your **Torch version** matches your **CUDA version**. You can install the correct PyTorch packages from [https://pytorch.org](https://pytorch.org).

Example (for CUDA 11.8):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Setup Instructions

1. **Clone the repository**

```bash
git clone <repo-url>
cd Trainer_DocLayNet_FRCNN_DETR
```

2. **Create virtual environment (using Python 3.10)**

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Create the `.env` file**

Create a `.env` file in the root directory with the following content:

```env
ROOT_DIR=path_t0/Datasets/DocLayNet_core
```

> Replace the path with your actual dataset directory.

---

## Start Training

```bash
python main.py
```

Depending on the script logic, this will start training for either Faster R-CNN or DETR.

---

## Folder Structure

```
Trainer_DocLayNet_FRCNN_DETR/
│
├── models/
│   └── detrModels/
│       └── processor/   # Stores DETR processor (tokenizer/feature converter)
│   └── fasterRCNNmodels/
│
├── trainer/
│   ├── fasterrcnn.py    # Training logic for Faster R-CNN
│   └── detr.py          # Training logic for DETR
│
├── main.py              # Entry point for training
├── requirements.txt     # Dependency list
├── .env                 # Path configuration for dataset
```

---

## Compatibility Notes

- If switching CUDA versions, make sure to reinstall both `torch` and `torchvision`.
- If you encounter module crashes related to **NumPy 2.x**, downgrade to a compatible version:

```bash
pip install numpy<2 --force-reinstall
```

---

## Troubleshooting

- `torchvision.ops.nms` errors → Make sure CUDA is properly set up and your installed `torch`/**torchvision** are matching.
- `.env` not loaded? → Ensure `python-dotenv` is installed or load the variables manually.
- Training is slow? → Check if GPU is being used:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`, training runs on CPU.

---

## Status

- CUDA 11.8 support confirmed
- DETR & Faster R-CNN training works
- GPU is used successfully
- `.env` configuration active
- `requirements.txt` generated
