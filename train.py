import torch
from network import vae
from data import *
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import random
from torchvision.utils import save_image
from tqdm import tqdm
import torch.multiprocessing as mp
import numpy as np
import torch.nn.init as init
import lpips
import torch.nn.functional as F

def kl_loss(mu, log_var):
    res = (-0.5 * log_var + (mu ** 2 + log_var.exp()) / 2)
    return res.mean()


def init_(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
        init.orthogonal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def main(rank, world_size, batch, start_epoch, epoch, data_path, out_path, save_epoch, pretrain_path, encoder_blocks, \
         decoder_blocks, encoder_base, decoder_base, latent_dim, l1_weight, kl_weight, precept_weight, threshold, last_down, use_vae):
    

    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12365'
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        dataset = four_scale_dataset(data_path, threshold)
        sampler = DistributedSampler(dataset=dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset=dataset, batch_size=batch, shuffle=False, sampler=sampler, num_workers=8)

    else:
        dataset = four_scale_dataset(data_path, threshold)
        dataloader = DataLoader(dataset=dataset, batch_size=batch, shuffle=True, num_workers=8)
    im_channel = dataset[0][0].shape[0]




    model = vae(encoder_blocks, decoder_blocks, encoder_base, decoder_base, latent_dim, last_down, im_channel, use_vae)
    #model.apply(init_)

    if rank == 0:
        print(f"Number of encoder parameters: {sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)}")
        print(f"Number of decoder parameters: {sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)}")


    im_loss_l2 = torch.nn.MSELoss(reduction="sum")
    im_loss_l1 = torch.nn.L1Loss(reduction="sum")
    loss_fn_alex = lpips.LPIPS(net='alex').to(rank)


    if rank == 0:
        print("Loading network.......")
    if pretrain_path is not None:
        model.load_state_dict(torch.load(pretrain_path))
    
    model = model.to(rank)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    model.train()
    if rank == 0:
        print("Loading finished.......")

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(start_epoch, epoch+1):
        if world_size > 1:
            sampler.set_epoch(epoch)
        if epoch % save_epoch == 0:
            model.eval()
            test_samples = random.sample(range(len(dataset)), 16)
            with torch.no_grad():
                im = torch.cat([dataset[i][1].unsqueeze(0) for i in test_samples], dim=0).to(rank)
                N, C, H, W = im.shape
                thresh_im = torch.cat([dataset[i][0].unsqueeze(0) for i in test_samples], dim=0).to(rank)
                pred, _, _ = model(thresh_im)

                im = torch.cat([im[:, 3*i : 3*(i+1), ...] for i in range(C // 3)], 0)
                thresh_im = torch.cat([thresh_im[:, 3*i : 3*(i+1), ...] for i in range(C // 3)], 0)
                pred = torch.cat([pred[:, 3*i : 3*(i+1), ...] for i in range(C // 3)], 0)
                res = torch.cat([im, pred], dim=0)
            model.train()

            if rank == 0:
                save_image(res, f"{out_path}/{epoch}.png", nrow=len(test_samples), normalize=True, value_range=(-1, 1))
                torch.save(model.module.state_dict() if world_size > 1 else model.state_dict(), f"{out_path}/{epoch}.pth")
        if world_size > 1:
            dist.barrier()

        pbar = tqdm(dataloader) if rank == 0 else dataloader
        total_im_loss_l2 = total_im_loss_l1 = total_kl_loss = total_percept_loss = total_loss = 0
        for im, _ , msk in pbar:
            im = im.to(rank)
            pred, mu, log_var = model(im)
            loss_im_l2 = im_loss_l2(im, pred) / msk.sum()
            loss_im_l1 = l1_weight * im_loss_l1(im, pred) / msk.sum()

            loss_percept = 0
            for i in range(C // 3):
                loss_percept += precept_weight * loss_fn_alex(F.interpolate(pred[:, 3*i : 3*(i+1), ...], 64), F.interpolate(im[:, 3*i : 3*(i+1), ...], 64)).mean()
            loss_kl = kl_weight * kl_loss(mu, log_var) if use_vae else 0
            loss = loss_im_l2 + loss_im_l1 + loss_kl + loss_percept

            total_im_loss_l2 += loss_im_l2.item()
            total_im_loss_l1 += loss_im_l1.item()
            total_kl_loss += loss_kl.item() if use_vae else 0 
            total_percept_loss += loss_percept.item()
            total_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            if rank == 0:
                pbar.set_description(f"epoch : {epoch} loss_im_l2 : {loss_im_l2 : .5f} loss_percept : {loss_percept : .5f} loss_kl : {loss_kl : .5f} loss : {loss : .5f}")

        if world_size > 1:
            dist.barrier()
        total_im_loss_l2 /= len(dataloader)
        total_im_loss_l1 /= len(dataloader)
        total_kl_loss /= len(dataloader)
        total_loss /= len(dataloader)
        print(f"epoch : {epoch} loss_im_l2 : {total_im_loss_l2 : .5f} total_percept_loss : {total_percept_loss}  loss_kl : {total_kl_loss : .5f} loss : {total_loss : .5f}")

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    world_size = 1
    batch = 8
    epoch = 200
    data_path = "../gravityspy/processed/"
    out_path = "./checkpoints/sub_2.0/1231-1231-4-5-128-threshold0-percept-4scale-ae"
    pretrain_path = None
    save_epoch = 20
    start_epoch = 0
    encoder_blocks = [1, 2, 3, 1]
    decoder_blocks = [1, 2, 3, 1]
    encoder_base = 4
    decoder_base = 5
    latent_dim = 128
    l1_weight = 0
    kl_weight = 2e-2
    precept_weight = 2e-1
    threshold = 0
    last_down = 2
    use_vae = True


    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if world_size > 1:
        mp.spawn(main, args=(world_size, batch, start_epoch, epoch, data_path, out_path, save_epoch, pretrain_path, encoder_blocks,\
                        decoder_blocks, encoder_base, decoder_base, latent_dim, l1_weight, kl_weight, precept_weight, threshold, last_down, use_vae), nprocs=world_size)
    else:
        main(0, world_size, batch, start_epoch, epoch, data_path, out_path, save_epoch, pretrain_path, encoder_blocks,\
                        decoder_blocks, encoder_base, decoder_base, latent_dim, l1_weight, kl_weight, precept_weight, threshold, last_down, use_vae)