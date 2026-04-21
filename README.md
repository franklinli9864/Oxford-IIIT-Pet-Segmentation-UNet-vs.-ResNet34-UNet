# Oxford-IIIT Pet Segmentation: UNet vs. ResNet34-UNet

This repository contains the implementation of two deep learning architectures for **Binary Semantic Segmentation** on the **Oxford-IIIT Pet Dataset**. This project was developed as part of the Deep Learning course at **NTUST**.

## 🚀 Highlights
* **From Scratch Implementation**: All architectures, including a custom ResNet-34 backbone and Attention-enhanced UNet, were built independently using PyTorch without external model libraries.
* **No Pre-trained Weights**: Following strict academic requirements, all models were trained from random initialization (He Initialization).
* **High Performance**: Achieved a Dice Score of **0.913+** on Kaggle Leaderboards using an NVIDIA RTX 5090.

## 📂 Project Structure
Following the recommended organizational guidelines:
```text
.
├── dataset/               # Oxford-IIIT Pet images and trimaps
├── saved_models/          # Trained .pth checkpoints
├── src/                   # Source code
│   ├── models/            # Model definitions
│   │   ├── unet.py        # UNet with Attention Gates
│   │   └── resnet34_unet.py # ResNet-34 Encoder + UNet Decoder
│   ├── oxford_pet.py      # Custom Dataset and Preprocessing
│   ├── utils.py           # Dice Loss and Hybrid Loss functions
│   ├── train.py           # Training script with AMP and Scheduling
│   ├── evaluate.py        # Validation and threshold searching
│   └── inference.py       # Test set prediction and RLE encoding
├── requirements.txt       # Environment dependencies
└── README.md
```

## 🛠️ Methodology

### Data Preprocessing
* **Binary Masking**: Converted 3-class trimaps into binary masks, where foreground is 1, and background/boundary pixels are 0.
* **Augmentation**: Applied random horizontal flips, rotations ($\pm 15^\circ$), and Color Jitter to prevent overfitting during from-scratch training.

### Training Strategy
* **Loss Function**: A hybrid combination of **BCEWithLogitsLoss** and **Dice Loss** (0.5/0.5 ratio).
* **Optimization**: Adam optimizer with a learning rate of $10^{-4}$ and a batch size of 8.
* **Scheduling**: Used `ReduceLROnPlateau` to decay the learning rate when the validation Dice score plateaued.
* **Hardware**: Accelerated using **NVIDIA RTX 5090** with Automatic Mixed Precision (AMP).

## 📊 Results

| Model | Kaggle Dice Score |
| :--- | :--- |
| **UNet (Attention)** | **0.91313** |
| **ResNet34-UNet** | **0.91283** |

## 🔍 Key Observations
* **Attention Gates**: The integration of Attention Gates in the skip connections significantly improved boundary precision for smaller pet features.
* **Encoder Comparison**: While UNet converged faster, the ResNet34-UNet exhibited higher stability in images with complex, cluttered backgrounds.
* **TTA**: Utilizing Test-Time Augmentation (Horizontal Flipping) consistently improved the final submission score by ~0.5%.
