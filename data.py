import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import random

class folder_dataset(Dataset):
    def __init__(self, path, threshold):
        super().__init__()
        self.path = path
        self.fnames = [os.path.join(root, file) for root, _, files in os.walk(path) for file in files]
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        self.normalize = transforms.Compose([transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        self.threshold = threshold
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        im = Image.open(self.fnames[idx])
        im = self.to_tensor(im)
        msk = (im > self.threshold).any(0, keepdim=True)

        return self.normalize(msk * im), self.normalize(im), msk 

        

class folder_dataset_with_fname(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.fnames = [os.path.join(root, file) for root, _, files in os.walk(path) for file in files]
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        random.shuffle(self.fnames)
   
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        im = Image.open(self.fnames[idx])
        im = self.transform(im)
        return im, self.fnames[idx]


