import argparse, os, sys, glob
import torch
from torchvision import transforms
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from scripts.simple_clip_guide import prepare_clip
from drawer.utils import *
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
    sampler = DDIMSampler(model)

    perceptors,cutoutSizeTable,cutoutsTable = prepare_clip('draft')
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])
    pMs = {}
    for clip_model in perceptors: 
        perceptor = perceptors[clip_model]
        embed = perceptor.encode_text(clip.tokenize(opt.prompt).to(device)).float()
        pMs[clip_model] = Prompt(embed, 1, float('-inf')).to(device)
    make_fixcutouts = MakeFixCutouts(224, 16, cut_pow=2)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt


    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_samples=list()
    best_samples = None
    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(opt.n_samples * [""])
            for n in trange(opt.n_iter*0, desc="Sampling"):
                c = model.get_learned_conditioning(opt.n_samples * [prompt])
                shape = [4, opt.H//8, opt.W//8]
                samples_ddim, intermed = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=opt.n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                for i, x_sample in enumerate(x_samples_ddim):
                    x_sample_pl = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_sample_pl.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png"))
                    base_count += 1
                    loss = 0
                    for clip_model in perceptors:
                        perceptor = perceptors[clip_model]
                        make_cutouts = cutoutsTable[cutoutSizeTable[clip_model]]
                        iii = perceptor.encode_image(normalize(make_fixcutouts(x_sample[None]))).float() 
                        loss = loss + pMs[clip_model](iii)
                    if best_samples is None:
                        best_loss = loss.item()
                        best_samples = samples_ddim[i:i+1]
                        best_img = x_sample
                        #pdb.set_trace()
                    elif loss.item()<best_loss:
                        best_loss = loss.item()
                        best_samples = samples_ddim[i:i+1]
                        best_img = x_sample
                all_samples.append(x_samples_ddim)
    if len(all_samples) > 0:
        # additionally, save as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=opt.n_samples)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}.png'))

    if False: #what in side intermed
        for samples_ddim in intermed['pred_x0']:
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
            for x_sample in x_samples_ddim:
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png"))
                base_count += 1
    if True:
        sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
        shape = [4, opt.H//8, opt.W//8]
        C, H, W = [4, opt.H//8, opt.W//8]
        batch_size=opt.n_samples
        size = (batch_size, C, H, W)
        b = batch_size
        timesteps = sampler.ddim_timesteps
        #intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = np.flip(timesteps[:15])
        total_steps = time_range.shape[0]
        if True:
            #print(f"Running DDIM Sampling with {total_steps} timesteps")
            transtt = transforms.Compose([
                    transforms.Resize(256),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5]),
                    ]),
            int_img = Image.open('outputs/test/0002_sk.png')
            with torch.no_grad():
                t_img = transtt[0](int_img).cuda()
                encoder_posterior = model.encode_first_stage(t_img[None])
                best_samples = model.get_first_stage_encoding(encoder_posterior)
            #pdb.set_trace()
        

        #img = best_samples #
         
        a_w = sampler.ddim_alphas_prev[total_steps]
        all_samples=list()
        for _ in range(8):
            c = model.get_learned_conditioning(opt.n_samples * [prompt])
            iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
            
            with torch.no_grad():
                img = torch.repeat_interleave(best_samples,batch_size,dim=0)
                noise =  torch.randn([1]+shape, device=device)
                img = np.sqrt(a_w) * img + np.sqrt(1-a_w) * noise
                for i, step in enumerate(iterator):
                    index = total_steps - i - 1
                    ts = torch.full((b,), step, device=device, dtype=torch.long)
                    outs = sampler.p_sample_ddim(img, c, ts, index=index, 
                                        unconditional_guidance_scale=1, unconditional_conditioning=uc)
                    img, pred_x0 = outs
                x_samples_ddim = model.decode_first_stage(img)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                #for i, x_sample in enumerate(x_samples_ddim):
                if False:
                    loss = 0
                    for clip_model in perceptors:
                        perceptor = perceptors[clip_model]
                        make_cutouts = cutoutsTable[cutoutSizeTable[clip_model]]
                        iii = perceptor.encode_image(normalize(make_fixcutouts(x_sample[None]))).float() 
                        loss = loss + pMs[clip_model](iii)
                    if loss.item()<best_loss:
                        best_loss = loss.item()
                        print(best_loss)
                        best_samples = samples_ddim[i:i+1]
                        best_img = x_sample
                #elif False:
                best_img = x_samples_ddim[0]
                x_sample_pl = 255. * rearrange(best_img.cpu().numpy(), 'c h w -> h w c')
                Image.fromarray(x_sample_pl.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png"))
                base_count += 1
                #all_samples.append(best_img[None])
                all_samples.append(x_samples_ddim)
            #x_sample = torch.clamp((x_sample+1.0)/2.0, min=0.0, max=1.0)
            #x_sample = 255. * rearrange(x_sample[0].cpu().numpy(), 'c h w -> h w c')
            #Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png"))
            #base_count += 1

        # additionally, save as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=opt.n_iter)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}_pack.png'))
    print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")
