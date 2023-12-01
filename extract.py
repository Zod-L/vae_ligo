from network import vae
import os
import torch
import numpy as np
from data import folder_dataset_with_fname
from tqdm import tqdm
from torchvision.utils import save_image


encoder_blocks = [1, 2, 3, 1]
decoder_blocks = [1, 2, 3, 1]
encoder_base = 4
decoder_base = 5
latent_dim = 128
last_down = 2

device = "cuda:1"
dataset = folder_dataset_with_fname("../gravityspy/processed/sub_2.0")
model = vae(encoder_blocks, decoder_blocks, encoder_base, decoder_base, latent_dim, last_down)
model.load_state_dict(torch.load("checkpoints/sub_2.0/1231-1231-4-5-128-threshold0-percept/100.pth"))
model = model.to(device)
for im, fname in tqdm(dataset):
    im = im.to(device).unsqueeze(0)
    mu, log_var = model.encoder(im)
    #latent = model.sample(mu, log_var)
    latent = mu
    pred = model.decoder(latent)
    # save_image(pred, "./test.png", normalize=True, value_range=(-1, 1))
    # exit()
    
    if not os.path.exists(os.path.join("test", fname.split("/")[-2])):
        os.makedirs(os.path.join("test", fname.split("/")[-2]))

    if not os.path.exists(os.path.join("test_im", fname.split("/")[-2])):
        os.makedirs(os.path.join("test_im", fname.split("/")[-2]))
    
    np.save(os.path.join("test", fname.split("/")[-2],fname.split("/")[-1]), latent.detach().cpu().numpy())
    save_image(torch.cat((im, pred)), os.path.join("test_im", fname.split("/")[-2],fname.split("/")[-1]), normalize=True, value_range=(-1, 1))