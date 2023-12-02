import torch
from torchvision.models import resnet18



class bottle_neck_down(torch.nn.Module):
    def __init__(self, indim, middim, outdim, down=False):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(indim, middim, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(middim)

        stride = 2 if down else 1
        self.conv2 = torch.nn.Conv2d(middim, middim, 3, stride, 1)
        self.bn2 = torch.nn.BatchNorm2d(middim)

        self.conv3 = torch.nn.Conv2d(middim, outdim, 1, bias=False)

        self.conv_side = torch.nn.Conv2d(indim, outdim, 1, stride, bias=False) if down else torch.nn.Identity()
        self.bn3 = torch.nn.BatchNorm2d(outdim)
        self.relu = torch.nn.LeakyReLU()




    def forward(self, x):
        ori = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = x + self.conv_side(ori)
        x = self.relu(x)
        x = self.bn3(x)
        return x



class bottle_neck_up(torch.nn.Module):
    def __init__(self, indim, middim, outdim, up=False):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(indim, middim, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(middim)
        
        if up:
            self.conv2 = torch.nn.ConvTranspose2d(middim, middim, 3, 2, 1, 1)
        else:
            self.conv2 = torch.nn.Conv2d(middim, middim, 3, 1, 1)
        self.bn2 = torch.nn.BatchNorm2d(middim)

        self.conv3 = torch.nn.Conv2d(middim, outdim, 1, bias=False)

        self.conv_side = torch.nn.ConvTranspose2d(indim, outdim, 1, 2, 0, 1, bias=False) if up else torch.nn.Identity()
        self.bn3 = torch.nn.BatchNorm2d(outdim)
        self.relu = torch.nn.LeakyReLU()
        



    def forward(self, x):
        ori = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = x + self.conv_side(ori)
        x = self.relu(x)
        x = self.bn3(x)
        return x
    
    


class encoder(torch.nn.Module):
    def __init__(self, blocks, latent_dim, base_exp, last_down_exp, im_channel=3):
        super().__init__()
        self.num_block = len(blocks)
        self.base_exp = base_exp
        self.last_down_exp = last_down_exp
        self.im_channel = im_channel
        indims = [2 ** i for i in range(self.base_exp, self.base_exp+self.num_block)]
        middims = [2 ** i for i in range(self.base_exp, self.base_exp+self.num_block)]
        outdims = indims[1:] + [2 ** (self.base_exp+self.num_block)]
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(im_channel, indims[0], 3, padding=1), torch.nn.MaxPool2d(3, 2, 1))
        for i in range(self.num_block):
            setattr(self, f'layer{2+i}', torch.nn.Sequential(*([bottle_neck_down(indims[i], middims[i], outdims[i], True)] + [bottle_neck_down(outdims[i], middims[i], outdims[i]) for _ in range(blocks[i]-1)])))
        self.last_pool = torch.nn.MaxPool2d(2 ** self.last_down_exp, 2 ** self.last_down_exp, 0)

        #self.last_pool = torch.nn.Conv2d(2 ** self.last_down_exp, 2 ** self.last_down_exp, 0)

        self.mean_fc = torch.nn.Linear(int(outdims[-1] * pow(512 / pow(2, self.num_block+1+self.last_down_exp), 2)), latent_dim)
        self.var_fc = torch.nn.Linear(int(outdims[-1] * pow(512 / pow(2, self.num_block+1+self.last_down_exp), 2)), latent_dim)


    def forward(self, x):
        x = self.layer1(x)
        for i in range(self.num_block):
            x = getattr(self, f'layer{2+i}')(x)
        x = self.last_pool(x)
        mu = self.mean_fc(x.flatten(1))
        var = self.var_fc(x.flatten(1))

        return mu, var
    


class decoder(torch.nn.Module):
    def __init__(self, blocks, latent_dim, base_exp, first_up_exp, im_channel=3):
        super().__init__()
        self.num_block = len(blocks)
        self.base_exp = base_exp
        self.first_up_exp = first_up_exp
        self.im_channel = im_channel
        outdims = [2 ** i for i in range(self.base_exp, self.base_exp+self.num_block)]
        middims = [2 ** i for i in range(self.base_exp, self.base_exp+self.num_block)]
        indims = outdims[1:] + [2 ** (self.base_exp+self.num_block)]
        indims.reverse()
        middims.reverse()
        outdims.reverse()

        self.layer0 = torch.nn.Linear(latent_dim, int(indims[0] * pow(512 / pow(2, self.num_block+1+self.first_up_exp), 2)))
        self.layer1 = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2 ** (self.first_up_exp+1), mode="bilinear"), torch.nn.Conv2d(indims[0], indims[0], 3, padding=1))
        for i in range(self.num_block):
            setattr(self, f'layer{2+i}', torch.nn.Sequential(*([bottle_neck_up(indims[i], middims[i], outdims[i], True)] + [bottle_neck_up(outdims[i], middims[i], outdims[i]) for _ in range(blocks[i]-1)])))   
        self.layer_out = torch.nn.Conv2d(outdims[-1], im_channel, 3, padding=1)
    
    def forward(self, z):
        B, _ = z.shape
        x = self.layer0(z)
        x = x.view(B, -1, int(512 / pow(2, self.num_block+1+self.first_up_exp)), int(512 / pow(2, self.num_block+1+self.first_up_exp)))
        x = self.layer1(x)
        for i in range(self.num_block):
            x = getattr(self, f'layer{2+i}')(x)
        x = self.layer_out(x)
        return x
    
class vae(torch.nn.Module):
    def __init__(self, encoder_blocks, decoder_blocks, encoder_base_exp, decoder_base_exp, latent_dim, last_down, im_channel):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder(encoder_blocks, latent_dim, encoder_base_exp, last_down, im_channel)
        self.decoder = decoder(decoder_blocks, latent_dim, decoder_base_exp, last_down, im_channel)


    def sample(self, mu, log_var):
        var = torch.exp(0.5 * log_var)
        z = torch.randn_like(mu)
        z = var * z + mu
        return z


    def forward(self, im):
        mu, log_var = self.encoder(im)
        z = self.sample(mu, log_var)
        pred = self.decoder(z)
        return pred, mu, log_var

