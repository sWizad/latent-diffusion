import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
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

    #model.cuda()
    model.eval()
    return model

class LatentDrawer(nn.Module):
  """ 
  """
  def __init__(self, width=256, height=256, do_mono=False, shape=None, decay=None, lr=0.1, device='cuda', plan='fast'):
    """
    """
    super().__init__()
    self.h = height
    self.w = width
    self.device = device
    self.use_c2f = True
    self.num_rows, self.num_cols, self.num_step = 30, 30, 3
    fac = 2**self.num_step

    if plan in ['easy']:   self.plan = [25,50,100]
    elif plan in ['fast']: self.plan = [100,200,400]
    else:   self.plan = [400,800,1600]

    latent = torch.randn([1, 4, self.h//8, self.w//8], device=device)
    self.latent = latent.requires_grad_()
    self.optimizer = torch.optim.Adam([self.latent], lr =lr, betas = (0.9,0.999))
    self.upshape = transforms.Resize((height,width), interpolation = transforms.InterpolationMode.NEAREST) 
    if self.use_c2f:
        n = 3
        s_rows, s_cols = self.num_rows*fac//2**(n-1), self.num_cols*fac//2**(n-1)
        self.down = [transforms.Resize((s_rows*2**i,s_cols*2**i), interpolation = transforms.InterpolationMode.BILINEAR) for i in range(n)]
    #self.down = [transforms.Resize((self.h//2**(i+1),self.w//2**(i+1))) for i in range(3) ]
    #self.down.reverse()
    #self.upshape = transforms.Resize((self.h,self.w))
    self.load_model()

  def load_model(self):
    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")  # TODO: check path

    self.model = model.to(self.device)
    self.cur_iteration = 0
    self.img = self.render()
    #self.down = [d(self.img) for d in self.down_fn]

  def to_valid_rgb(self,img):
    img = img.permute(0,2,3,1)
    img = torch.matmul(img,self.to_valid)
    img = img.permute(0,3,1,2)
    return img

  def render(self):
    img = self.model.differentiable_decode_first_stage(self.latent)
    img = torch.clamp((img+1)/2,0,1)
    return img

  def synth(self, cur_iteration):
    img = self.render()
    if self.use_c2f:
        if cur_iteration<self.plan[0]:
            img = self.down[0](img)
        elif cur_iteration<self.plan[1]:
            a1 = self.down[0](img)
            a1 = self.down[1](a1)
            a2 = self.down[1](img)
            alpha_weight = (cur_iteration-self.plan[0])/(self.plan[1]-self.plan[0])
            img = a1*(1-alpha_weight) + a2*alpha_weight
        elif cur_iteration<self.plan[2]:
            a1 = self.down[1](img)
            a1 = self.down[2](a1)
            a2 = self.down[2](img)
            alpha_weight = (cur_iteration-self.plan[1])/(self.plan[2]-self.plan[1])
            img = a1*(1-alpha_weight) + a2*alpha_weight
        else:
            img = self.down[2](img)
    img = self.upshape(img)
    self.img = img
    return img

  def forward(self):
    """Sums """
    self.cur_iteration += 1
    #self.img = self.render()
    return self.synth(self.cur_iteration )
  
  def to_pil(self):
    x_sample = 255. * rearrange(self.img[0], 'c h w -> h w c').cpu().detach().numpy()
    return Image.fromarray(x_sample.astype(np.uint8))

  def save(self, fn):
    self.to_pil().save(fn)
