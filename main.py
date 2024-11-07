import torch
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from utils.data_loader import DatasetManager
from utils.seed_util import set_seed
from config import *
from models.vit_model import create_vit_model
from training.train import train_model
from torch.nn import CrossEntropyLoss

def main():
    # Define device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    set_seed()  # Set seed for reproducibility

    # Load data
    dataset_manager = DatasetManager(DATASET_DIR, TRAIN_DIR, TEST_DIR)
    dataloaders, dataset_sizes = dataset_manager.prepare_dataloaders(BATCH_SIZE, VAL_SPLIT)

    # Create model
    num_classes = len(dataloaders['train'].dataset.dataset.classes)
    model = create_vit_model(num_classes)
    model = model.to(device)

    # Set up training components
    criterion = CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    num_training_steps = NUM_EPOCHS * len(dataloaders['train'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=NUM_WARMUP_STEPS, num_training_steps=num_training_steps)

    # Train the model
    model = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=NUM_EPOCHS, checkpoint_path=CHECKPOINT_PATH)

if __name__ == '__main__':
    main()
