# Fine-Grained Image Classification (FGIC) Competition

## Introduction

Fine-Grained Image Classification in Neural Networks for Computer Vision refers to the task of categorizing images into very specific and detailed classes, such as different species of birds or types of flowers. This type of classification requires the model to be highly
accurate and sensitive to subtle differences in visual features, as the distinctions between classes are often quite subtle.

## Proposed solution

The solution leveraged a timm's `vit_large_patch16_224` model achieving 81.2% Top-1 accuracy in classifying the 100 classes from the Mammalia dataset. The competition had a live format where each participant had 2 hours from the delivery of the dataset until the delivery of a final model. The implementation includes:

- Early Stopping.
- Data Augmentation.
- AdamW optimizer.
- A learning rate cosine scheduler.
- Mixed Precision Training.

## Project Structure

- `config.py` - Configuration file for all project parameters.
- `main.py` - Main script to load data, initialize the model, and start training.
- `models/` - Model architectures, including ViT.
- `training/` - Training scripts, including early stopping and training loop.
- `utils/` - Utility functions for data loading, transformations, seed setting, and submission.

## Requirements

All dependencies are listed in `requirements.txt`. Install them using:
```bash
pip install -r requirements.txt
