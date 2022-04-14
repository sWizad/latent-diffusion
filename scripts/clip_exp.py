import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision import transforms
#from kornia import create_meshgrid
from torch.autograd import Variable
import torch.optim as optim

from matplotlib import pyplot as plt
from fastprogress.fastprogress import master_bar, progress_bar

import clip
import pdb
from drawer.DwtDrawer import DwtDrawer
from drawer.LatentDrawer import LatentDrawer
from drawer.utils import *
from scripts.simple_clip_guide import prepare_clip

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

    outpath = opt.outdir
    os.makedirs(outpath,exist_ok=True)
    promptxt = opt.prompt
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #make_cutouts = MakeCutouts(224, 64, cut_pow=1)
    transtt = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        ]),
    int_img = Image.open('outputs/Rapunzel/0029.png')
    upshape = transforms.Resize((512,512), interpolation = transforms.InterpolationMode.NEAREST) 
    make_fixcutouts = MakeFixCutouts(224, 64, cut_pow=2)
    with torch.no_grad():
        t_img = transtt[0](int_img).cuda()
        #t_img = upshape(t_img)
        fixcut = make_fixcutouts(t_img[None])
    grid = make_grid(fixcut, nrow=8)
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'abc.png'))

    perceptors,cutoutSizeTable,cutoutsTable = prepare_clip('draft')
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])
    pMs = {}
    fMs = {}
    for clip_model in perceptors: 
        perceptor = perceptors[clip_model]
        embed = perceptor.encode_text(clip.tokenize(opt.prompt).to(device)).float()
        pMs[clip_model] = Prompt(embed, 1, float('-inf')).to(device)
        embed = perceptor.encode_image(normalize(fixcut)).float()
        fMs[clip_model] = Prompt(embed, 1, float('-inf')).to(device)

    if True:
        ims = DwtDrawer(int_img = transforms.Resize(256)(t_img), device=device)
        #ims = DwtDrawer( device=device)
    else:
        ims = LatentDrawer(device=device)

    # A basic progress bar (using fastprogress)
    bar = progress_bar(range(1000))
    for i in bar:
        ims.optimizer.zero_grad()
        im = ims() # Get the image from the ImStack
        loss = 0
        for clip_model in perceptors:
            perceptor = perceptors[clip_model]
            make_cutouts = cutoutsTable[cutoutSizeTable[clip_model]]
            iii = perceptor.encode_image(normalize(make_cutouts(im))).float() # Encode image (using multiple cutouts)
            loss = loss + pMs[clip_model](iii) # Calculate loss
            #iii = perceptor.encode_image(normalize(make_fixcutouts(upshape(im)))).float()
            #loss = loss + fMs[clip_model](iii) # Calculate loss
        loss.backward() # Backprop
        ims.optimizer.step() # Update

        if i % 20 == 0:   ims.save(os.path.join(outpath, f'{promptxt.replace(" ", "-")}.png'))
        #pdb.set_trace()