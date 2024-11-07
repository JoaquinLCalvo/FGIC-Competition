import os
import shutil
import zipfile
import tarfile
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from config import DATASET_DIR, TRAIN_DIR, TEST_DIR, BATCH_SIZE, VAL_SPLIT
from utils.transforms import data_transforms  # Ensures transforms are accessible

class DatasetManager:
    def __init__(self, dataset_dir=DATASET_DIR, train_dir=TRAIN_DIR, test_dir=TEST_DIR):
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(dataset_dir, train_dir)
        self.test_dir = os.path.join(dataset_dir, test_dir)

    def handle_dataset(self, source):
        if os.path.isfile(source):
            self._handle_local_file(source)
        else:
            raise ValueError("Only local files are supported in this setup")

    def _handle_local_file(self, source):
        if source.endswith('.zip'):
            self._extract_zip(source)
        elif source.endswith('.tar') or source.endswith('.tar.gz'):
            self._extract_tar(source)
        else:
            raise ValueError("Unsupported file format")

    def _extract_zip(self, filepath):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(self.dataset_dir)
        self._correct_directory_structure()

    def _extract_tar(self, filepath):
        with tarfile.open(filepath, 'r') as tar_ref:
            tar_ref.extractall(self.dataset_dir)
        self._correct_directory_structure()

    def _correct_directory_structure(self):
        extracted_folders = [name for name in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, name))]
        if len(extracted_folders) == 1:
            extracted_main_dir = os.path.join(self.dataset_dir, extracted_folders[0])
            for item in os.listdir(extracted_main_dir):
                shutil.move(os.path.join(extracted_main_dir, item), self.dataset_dir)
            os.rmdir(extracted_main_dir)

    def prepare_dataloaders(self, batch_size=BATCH_SIZE, val_split=VAL_SPLIT, random_state=42):
        from utils.transforms import data_transforms 

        image_datasets = {'train': datasets.ImageFolder(self.train_dir, data_transforms['train'])}

        train_dataset = image_datasets['train']
        train_size = int((1 - val_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        image_datasets['valid'] = val_dataset

        image_datasets['test'] = CustomImageDataset(self.test_dir, transform=data_transforms['test'])

        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
            'valid': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4),
            'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4)
        }

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
        return dataloaders, dataset_sizes

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        class_id = os.path.basename(img_path).split('_')[0]  # Adjust based on filename structure
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, class_id
