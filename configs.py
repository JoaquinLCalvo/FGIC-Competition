# Data parameters
IM_DIMENSION = 224
BATCH_SIZE = 12
VAL_SPLIT = 0.2

# Training parameters
SEED = 47
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
NUM_WARMUP_STEPS = 500
PATIENCE = 7
CHECKPOINT_PATH = "checkpoints/checkpoint.pth"

# Paths
DATASET_DIR = "/home/disi/COMPETITION_DATASET"
TRAIN_DIR = "train"
TEST_DIR = "test"

# Submission
GROUP_NAME = "Tanos Matadores" 

# Transformation parameters
NORMALIZATION_MEAN = [0.507, 0.487, 0.441]
NORMALIZATION_STD = [0.267, 0.256, 0.276]
