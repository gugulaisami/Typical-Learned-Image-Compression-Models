import os
import torch
import h5py
from PIL import Image
from pathlib import Path
import torch.utils.data
from torch.utils.data.dataset import Dataset

class VimeoDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, stage, transform=None):
        self.data_dir = os.path.join(data_path, stage)
        self.dataset_list = [f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir,f))]
        self.transform = transform

    def __len__(self):
        return len(self.dataset_list)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.dataset_list[idx], 'im1.png')
        img = Image.open(image_path).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

class KodakDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, transform):
        self.data_dir = data_path
        self.dataset_list = [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f))]
        self.transform = transform

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.dataset_list[idx])
        img = Image.open(image_path).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

class CLIC2020(torch.utils.data.Dataset):

    def __init__(self, data_path, transform):
        self.data_dir = data_path
        splitdir = Path(data_path)
        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

class H5Dataset_Flicker2W(Dataset):
    def __init__(self, path, transform=None):
        self.file_path = path
        self.dataset = h5py.File(self.file_path, 'r')
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file)
        self.transform = transform

    def __getitem__(self, index):
        img = self.dataset[str(index)][:]
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):

        return self.dataset_len