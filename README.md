# Dog Heart VHS Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)

A deep learning framework for predicting Vertebral Heart Score (VHS) from canine chest X-rays, with a focus on anatomical landmark detection using optimized attention mechanisms.

## Features

- Vision Transformer (ViT) backbone with optimized bidirectional attention
- Anatomical landmark detection with relative positional encoding
- Multi-path VHS prediction (direct and point-based)
- Efficient implementation using einops and optional Flash Attention
- Data augmentation preserving anatomical constraints
- Comprehensive training, evaluation, and visualization pipelines

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dog-heart-vhs.git
cd dog-heart-vhs

# Install dependencies
pip install -r requirements.txt

```

## Quick Start

### Google Colab Setup

```python
# Install the package directly from GitHub
!pip install git+https://github.com/yourusername/dog-heart-vhs.git

# Import the necessary modules
from VHS.data import create_dataloaders
from VHS.models import DogHeartViTWithOptimizedAttention
from VHS.models.loss import AnatomicalLoss
from VHS.scripts.train import train_model
```

### Training

```python
# Configure paths
base_path = '/content/drive/MyDrive'

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    base_path=base_path,
    batch_size=16, 
    augment=True
)

# Initialize model
model = DogHeartViTWithOptimizedAttention(
    pretrained=True,
    freeze_backbone=True,
    backbone='vit_base_patch16_224'
)

# Define loss function
criterion = AnatomicalLoss(
    points_weight=1.0, 
    vhs_weight=10.0,
    perimeter_weight=0.0  # Start with perimeter loss disabled
)

# Train model
trained_model, history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    epochs=30,
    initial_lr=1e-5,
    device='cuda'
)
```

### Prediction

```python
from dog_heart_vhs.scripts.predict import predict_test_set

# Predict on test set
predictions = predict_test_set(
    model=trained_model,
    test_loader=test_loader,
    output_path='vhs_predictions.csv',
    device='cuda'
)
```

## Data Format

The framework expects data organized as follows:

```
base_path/
├── Train/
│   ├── Images/           # Training images (.png)
│   └── Labels/           # Training labels (.mat files with VHS and six_points)
├── Valid/
│   ├── Images/           # Validation images (.png)
│   └── Labels/           # Validation labels (.mat files with VHS and six_points)
└── Test_Images/
    └── Images/           # Test images (.png)
```

## Model Architecture

Our model uses a Vision Transformer (ViT) backbone with a custom bidirectional attention mechanism optimized for landmark point detection:

1. **ViT Backbone**: Extracts high-level features from the X-ray image
2. **Bidirectional Attention**:
   - Points attend to image features (visual grounding)
   - Points attend to other points (anatomical consistency)
   - Image features attend to points (feature enhancement)
3. **Relative Position Encoding**: Preserves the circular/sequential nature of heart landmarks
4. **Dual VHS Prediction**:
   - Direct prediction from image features
   - Prediction from landmark points
   - Weighted combination for the final prediction

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dogheartvhs2025,
  author = {Your Name},
  title = {Dog Heart VHS Prediction},
  year = {2025},
  url = {https://github.com/yourusername/dog-heart-vhs}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
