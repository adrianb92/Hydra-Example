import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torchvision.transforms import Resize, Compose, ToTensor, Normalize


import gdown
import zipfile
import os.path as osp

class AlienPredatorDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=10, data_dir: str = './alien_predator/', train_dataset_path='/content/alien_predator/data/train', test_dataset_path='/content/alien_predator/data/validation', **kwargs):
        super().__init__()
        print('batch_size:', batch_size)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.image_size = (256, 256)
        self.imagenet_transform = Compose([Resize(self.image_size), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.num_classes = 2
        self.zip_name = 'alien_predator.zip'

    def prepare_data(self):
        if not osp.isfile(self.zip_name): 
            gdown.download('https://drive.google.com/uc?id=19dSNIsEGoScG4AIxG0eOxjHxJu8AP5HZ', output=self.zip_name, quiet=False)
                            
        if not osp.isdir(self.data_dir):
            with zipfile.ZipFile(self.zip_name, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)    
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            dataset = ImageFolder(self.train_dataset_path, transform=self.imagenet_transform)
            train_dataset_size = int(len(dataset) * 0.9)
            self.train_dataset, self.val_dataset = random_split(dataset, [train_dataset_size, len(dataset) - train_dataset_size])
        if stage == 'test' or stage is None:
            self.test_dataset = ImageFolder(self.test_dataset_path, transform=self.imagenet_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
