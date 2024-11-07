import timm
import torch.nn as nn
from config import IM_DIMENSION  # Import dimension for model consistency

def create_vit_model(num_classes):
    model = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=num_classes)
    return model