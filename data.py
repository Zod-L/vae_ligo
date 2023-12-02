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


class four_scale_dataset(Dataset):
    def __init__(self, path, threshold):
        super().__init__()
        self.path = path
        self.scale = ["0.5", "1.0", "2.0", "4.0"]
        self.fnames = [os.path.join(dir, f.replace("_4.0", "")) for dir in os.listdir(f"{path}/sub_4.0/") for f in os.listdir(os.path.join(f"{path}/sub_4.0/", dir))]
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        self.normalize = transforms.Compose([transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        self.threshold = threshold
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        msk_ims = []
        ims = []
        msks = []
        for scale in self.scale:
            im = Image.open(os.path.join(self.path, f"sub_{scale}", self.fnames[idx].replace(".png", f"_{scale}.png")))
            im = self.to_tensor(im)
            msk = (im > self.threshold).any(0, keepdim=True)
            msk_ims.append(self.normalize(msk * im))
            ims.append(self.normalize(im))
            msks.append(msk)

        return torch.concat(msk_ims, 0), torch.concat(ims, 0), torch.concat(msks, 0) 



class folder_dataset_with_fname(Dataset):
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

        return self.normalize(msk * im), self.normalize(im), msk, self.fnames[idx] 

class four_scale_datasett_with_fname(Dataset):
    def __init__(self, path, threshold):
        super().__init__()
        self.path = path
        self.scale = ["0.5", "1.0", "2.0", "4.0"]
        self.fnames = [f"{dir}/{f}" for dir in os.listdir(f"{path}/sub_4.0/") for f in os.listdir(dir)]
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        self.normalize = transforms.Compose([transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        self.threshold = threshold
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        msk_ims = []
        ims = []
        msks = []
        for scale in self.scale:
            im = Image.open(os.path.join(self.path, f"sub_{scale}", self.fnames[idx]))
            im = self.to_tensor(im)
            msk = (im > self.threshold).any(0, keepdim=True)
            msk_ims.append(self.normalize(msk * im))
            ims.append(self.normalize(im))
            msks.append(msk)

        return torch.concat(msk_ims, 0), torch.concat(ims, 0), torch.concat(msk, 0), self.fnames[idx]
