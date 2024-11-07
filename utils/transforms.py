from torchvision.transforms import v2
from config import IM_DIMENSION, NORMALIZATION_MEAN, NORMALIZATION_STD  # Import parameters

# Light data transformations (for heavier, uncomment those 6 lines)
data_transforms = {
    'train': v2.Compose([
        v2.Resize((244, 244)),
        v2.RandomRotation(15),
        v2.RandomCrop(IM_DIMENSION),
        #v2.RandomRotation(15,),
        #v2.ColorJitter(brightness=0.2,  #bad picture conditions (e.g. surveillance cameras)
        #               contrast=0.2,     # poor visibility (e.g. underwater images)
        #               saturation=0.2,
        #               hue=0.1),
        v2.RandomHorizontalFlip(),
        v2.ToTensor(),
        v2.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
        #v2.RandomErasing(p=0.1)
    ]),
    'valid': v2.Compose([
        v2.Resize((IM_DIMENSION, IM_DIMENSION)),
        v2.ToTensor(),
        v2.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
    ]),
    'test': v2.Compose([
        v2.Resize((IM_DIMENSION, IM_DIMENSION)),
        v2.ToTensor(),
        v2.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
    ]),
}