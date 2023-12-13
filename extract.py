from network import vae
import os
import torch
import numpy as np
from data import *
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data import DataLoader

encoder_blocks = [1, 2, 3, 1]
decoder_blocks = [1, 2, 3, 1]
encoder_base = 4
decoder_base = 5
latent_dim = 128
last_down = 2
threshold = 0

device = "cuda:1"
dataset = four_scale_dataset_with_fname("../gravityspy/processed/", threshold)
C = dataset[0][0].shape[0]
model = vae(encoder_blocks, decoder_blocks, encoder_base, decoder_base, latent_dim, last_down, C)
model.load_state_dict(torch.load("checkpoints/sub_2.0/1231-1231-4-5-128-threshold0-percept-4scale/100.pth"))
model = model.to(device)
batch = 16
dataloader = DataLoader(dataset=dataset, batch_size=batch, shuffle=True, num_workers=8)
for im, ori_im, msk, fnames in tqdm(dataloader):

    


    pred, mu, log_var = model(ori_im.to(device))

    





    for i, fname in enumerate(fnames):
        if not os.path.exists(os.path.join("test", fname.split("/")[-2])):
            os.makedirs(os.path.join("test", fname.split("/")[-2]))

        if not os.path.exists(os.path.join("test_im", fname.split("/")[-2])):
            os.makedirs(os.path.join("test_im", fname.split("/")[-2]))
        _im = torch.cat([im[i:i+1, 3*j : 3*(j+1), ...] for j in range(C // 3)], 0)
        _ori_im = torch.cat([ori_im[i:i+1, 3*j : 3*(j+1), ...] for j in range(C // 3)], 0)
        _pred = torch.cat([pred[i:i+1, 3*j : 3*(j+1), ...] for j in range(C // 3)], 0)
        res = torch.cat([_im.cpu(), _pred.cpu()], dim=0)
        np.save(os.path.join("test", fname.split("/")[-2],fname.split("/")[-1]), mu[i:i+1, ...].detach().cpu().numpy())
        save_image(res, os.path.join("test_im", fname.split("/")[-2],fname.split("/")[-1]), normalize=True, value_range=(-1, 1), nrow=4)
    
    