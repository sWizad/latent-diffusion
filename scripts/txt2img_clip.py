import argparse, os, sys, glob
import torch
from torch import nn, optim
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision import transforms
#from kornia import create_meshgrid
import kornia.augmentation as K
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import math
from matplotlib import pyplot as plt
from fastprogress.fastprogress import master_bar, progress_bar
#from IPython.display import HTML
#from base64 import b64encode

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import clip
import pdb


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))
 
def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()
 
def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]

class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))
 
    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomSharpness(0.3,p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2,p=0.4),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7))
        self.noise_fac = 0.1 

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch

def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size
 
    input = input.view([n * c, 1, h, w])
 
    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])
 
    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])
 
    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)

class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward
 
    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)
 
 
replace_grad = ReplaceGrad.apply 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    if True:

        parser.add_argument(
            "--prompt",
            type=str,
            nargs="?",
            default="a painting of a virus monster playing guitar",
            help="the prompt to render"
        )

        parser.add_argument(
            "--outdir",
            type=str,
            nargs="?",
            help="dir to write results to",
            default="outputs/txt2img-samples"
        )
        parser.add_argument(
            "--ddim_steps",
            type=int,
            default=200,
            help="number of ddim sampling steps",
        )

        parser.add_argument(
            "--ddim_eta",
            type=float,
            default=0.0,
            help="ddim eta (eta=0.0 corresponds to deterministic sampling",
        )
        parser.add_argument(
            "--n_iter",
            type=int,
            default=1,
            help="sample this often",
        )

        parser.add_argument(
            "--H",
            type=int,
            default=256,
            help="image height, in pixel space",
        )

        parser.add_argument(
            "--W",
            type=int,
            default=256,
            help="image width, in pixel space",
        )

        parser.add_argument(
            "--n_samples",
            type=int,
            default=4,
            help="how many samples to produce for the given prompt",
        )

        parser.add_argument(
            "--scale",
            type=float,
            default=5.0,
            help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
        )
    opt = parser.parse_args()


    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")  # TODO: check path

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    #sampler = DDIMSampler(model)

    perceptor = clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device)
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])
    cut_size = perceptor.visual.input_resolution
    cutn=64
    cut_pow=1
    make_cutouts = MakeCutouts(cut_size, cutn, cut_pow=cut_pow)
    transtt = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
        ]),

    embed = perceptor.encode_text(clip.tokenize(opt.prompt).to(device)).float()
    prompt = Prompt(embed, 1, float('-inf')).to(device)
    # The ImStack. Layer sizes will be 16, 32, 64, 128, 256 and 512

    int_img = Image.open('outputs/other/samples/0018.png')
    with torch.no_grad():
        t_img = transtt[0](int_img).cuda()
        encoder_posterior = model.encode_first_stage(t_img[None])
        x_T = model.get_first_stage_encoding(encoder_posterior)
        x_T = x_T + torch.randn([1, 4, opt.H//8, opt.W//8], device=device)
        #x_T = torch.repeat_interleave(x_T, 4, dim=0)
    x_T = x_T.requires_grad_()

    # An optimizer (you can try others) with the image layers as parameters
    optimizer = optim.Adam([x_T], lr=0.01)

    # Somewhere to track our losses
    losses = []

    # A basic progress bar (using fastprogress)
    bar = progress_bar(range(800))
    for i in bar:
        optimizer.zero_grad()
        #pdb.set_trace()
        im  = model.differentiable_decode_first_stage(x_T) # Get the image from the ImStack
        im = torch.clamp((im+1)/2,0,1)
        #im2  = model.first_stage_model.decode(x_T)
        #pdb.set_trace()
        iii = perceptor.encode_image(normalize(make_cutouts(im))).float() # Encode image (using multiple cutouts)
        l = prompt(iii) # Calculate loss
        losses.append(float(l.detach().cpu())) # Store loss
        l.backward() # Backprop
        optimizer.step() # Update

    #plt.plot(losses)
    outpath = opt.outdir
    promptxt = opt.prompt

    x_sample = 255. * rearrange(im[0], 'c h w -> h w c').cpu().detach().numpy()
    Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(outpath, f'{promptxt.replace(" ", "-")}.png'))
