# Fine-Grained Image Classification (FGIC) Competition

This repository contains the code for a Fine-Grained Image Classification project using Vision Transformers (ViT) in PyTorch. The project is structured with modular, reusable components and supports training with early stopping and model checkpoints.

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