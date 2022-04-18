from numpy.core.numeric import False_
from torch._C import LongStorageBase
import argparse

def setup_parser():
    # Create the parser
    parser = argparse.ArgumentParser(description='Image generation using CLIP')
    parser.add_argument( "--prompt",
        type=str,
        nargs="?",
        default="",
        help="the prompt to render"
    )
    parser.add_argument( "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/test"
    )
    parser.add_argument( "--steps",
        type=int,
        default=300,
        help="number of ddim sampling steps",
    )
    parser.add_argument(  "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(  "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument( "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )
    parser.add_argument(  "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )
    parser.add_argument( "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for the given prompt",
    )
    parser.add_argument( "--seed",
        type=int,
        default=None,
        help="random seed",
    )
    parser.add_argument( "--init_image",
        type=str,
        default='outputs/test/0002.png',
        help="init_image",
    )
    parser.add_argument( "--save_interval",
        type=int,
        default=1000,
        help="save_interval",
    )
    parser.add_argument( "--color_space",
        type=str,
        default="YCoCg",
        help="color space pick from YCoCg or RGB",
    )
    
    opt = parser.parse_args()
    return opt

# Input prompts. Each prompt has "text" and a "weight"
# Weights can be negatives, useful for discouraging specific artifacts
texts = [
    {
        "text": "Little Mermaid",
        "weight": 1.0,
    },{
        "text": "Beautiful and detailed fantasy painting.",
        "weight": 0.2,
    # },{
    # #     "text": "Full body.",
    # #     "weight": 0.1,
    },{ # Improves contrast, object coherence, and adds a nice depth of field effect
        "text": "Rendered in unreal engine, trending on artstation.",
        "weight": 0.2,
    # },{
    # #     "text": "speedpainting",
    # #     "weight": 0.1,
    # # },{ # Seems to improve contrast and overall image structure
    #     "text": "matte painting, featured on artstation.",
    #     "weight": 0.1,
    # },{
    #     "text": "Vivid Colors",
    #     "weight": 0.15,
    # },{ # Doesn't seem to do much, but also doesn't seem to hurt. 
    #     "text": "confusing, incoherent",
    #     "weight": -0.25,
    },{ # Helps reduce pixelation, but also smoothes images overall. Enable if you're using scaling = 'nearest'
         "text":"pixelated",
         "weight":-0.25
    },{ # Not really strong enough to remove all signatures... but I'm ok with small ones
        "text":"text",
        "weight":-0.5
    }
]

#Image prompts
images = [  ]

# Resample image prompt vectors every iterations
# Slows things down a lot for very little benefit, don't bother
resample_image_prompts = False

# Optimizer settings for different training steps
stages = (
            { #First stage does rough detail.
        "cuts": 2,
        "cycles": 2000,
        "lr_luma": 1.5e-1, #1e-2 for RAdam #3e-2 for adamw
        "decay_luma": 1e-5,
        "lr_chroma": 7.5e-2, #5e-3 for RAdam #1.5e-2 for adamw
        "decay_chroma": 1e-5,
        "noise": 0.2,
        "denoise": 0.5,
        "checkin_interval": 100,
        # "lr_persistence": 0.95, # ratio of small-scale to large-scale detail
        "pyramid_lr_min" : 0.2, # Percentage of small scale detail
        # "lr_scales": [0.25,0.15,0.15,0.15,0.15,0.05,0.05,0.01,0.01], # manually set lr at each level
    }, { # 2nd stage does fine detail and
        "cuts": 2,
        "cycles": 1000,
        "lr_luma": 0.75e-1,
        "decay_luma": 1e-5,
        "lr_chroma": 3.75e-2,
        "decay_chroma": 1e-5,
        "noise": 0.2,
        "denoise": 0.5,
        "checkin_interval": 100,
        # "lr_persistence": 1.0,
        "pyramid_lr_min" : 1
        # "lr_scales": [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
    },
)


debug_clip_cuts = False

import sys, os, random, shutil, math
import torch, torchvision

import numpy as np
from PIL import Image
from fastprogress.fastprogress import progress_bar
import clip
from drawer.utils import *
from drawer.PyramidDrawer import PyramidDrawer 
from drawer.WaveletDrawer import WaveletDrawer

from models.Sketch.models import create_model
from models.Sketch.baseOption import *

import lpips
import pdb



bilinear = torchvision.transforms.functional.InterpolationMode.BILINEAR
bicubic = torchvision.transforms.functional.InterpolationMode.BICUBIC

torch.autograd.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type(torch.cuda.FloatTensor)




def normalize_image(image):
  R = (image[:,0:1] - 0.48145466) /  0.26862954
  G = (image[:,1:2] - 0.4578275) / 0.26130258 
  B = (image[:,2:3] - 0.40821073) / 0.27577711
  return torch.cat((R, G, B), dim=1)

# TODO: Use torchvision normalize / unnormalize
def unnormalize_image(image):
  
  R = image[:,0:1] * 0.26862954 + 0.48145466
  G = image[:,1:2] * 0.26130258 + 0.4578275
  B = image[:,2:3] * 0.27577711 + 0.40821073
  
  return torch.cat((R, G, B), dim=1)

@torch.no_grad()
def loadImage(filename):
  data = open(filename, "rb").read()
  image = torch.ops.image.decode_png(torch.as_tensor(bytearray(data)).cpu().to(torch.uint8), 3).cuda().to(torch.float32) / 255.0
  # image = normalize_image(image)
  return image.unsqueeze(0).cuda()

def getClipTokens(image, cuts, noise, perceptor):
    im = normalize_image(image)
    cut_data = torch.zeros(cuts, 3, perceptor["size"], perceptor["size"])
    for c in range(cuts):
      angle = random.uniform(-20.0, 20.0)
      img = torchvision.transforms.functional.rotate(im, angle=angle, expand=True, interpolation=bilinear)

      padv = im.size()[2] // 8
      img = torch.nn.functional.pad(img, pad=(padv, padv, padv, padv))

      size = img.size()[2:4]
      mindim = min(*size)

      if mindim <= perceptor["size"]-32:
        width = mindim - 1
      else:
        width = random.randint(perceptor["size"]-32, mindim-1 )

      oy = random.randrange(0, size[0]-width)
      ox = random.randrange(0, size[1]-width)
      img = img[:,:,oy:oy+width,ox:ox+width]

      img = torch.nn.functional.interpolate(img, size=(perceptor["size"], perceptor["size"]), mode='bilinear', align_corners=False)
      cut_data[c] = img

    cut_data += noise * torch.randn_like(cut_data, requires_grad=False)


    clip_tokens = perceptor['model'].encode_image(cut_data)
    return clip_tokens

def loadPerceptor(name):
  model, preprocess = clip.load(name, device="cuda")

  tokens = []
  imgs = []
  for text in texts:
    tok = model.encode_text(clip.tokenize(text["text"]).cuda())
    tokens.append( tok )

  perceptor = {"model":model, "size": preprocess.transforms[0].size, "tokens": tokens, }
  for img in images:
    image = loadImage(img["fpath"])
    if resample_image_prompts:
      imgs.append(image)
    else:
      tokens = getClipTokens(image, img["cuts"], img["noise"], False, perceptor )
      imgs.append(tokens)
  perceptor["images"] = imgs
  return perceptor

def lossClip(image, cuts, noise, perceptors):
  losses = []

  max_loss = 0.0
  for text in texts:
    max_loss += abs(text["weight"]) * len(perceptors)
  for img in images:
    max_loss += abs(img["weight"]) * len(perceptors)

  for perceptor in perceptors:
    clip_tokens = getClipTokens(image, cuts, noise, perceptor)
    for t, tokens in enumerate( perceptor["tokens"] ):
      similarity = torch.cosine_similarity(tokens, clip_tokens)
      weight = texts[t]["weight"]
      if weight > 0.0:
        loss = (1.0 - similarity) * weight
      else:
        loss = similarity * (-weight)
      losses.append(loss / max_loss)

    for img in images:
      for i, prompt_image in enumerate(perceptor["images"]):
        if resample_image_prompts:
          img_tokens = getClipTokens(prompt_image, images[i]["cuts"], images[i]["noise"], False, perceptor)
        else:
          img_tokens = prompt_image
        weight = images[i]["weight"] / float(images[i]["cuts"])
        for token in img_tokens:
          similarity = torch.cosine_similarity(token.unsqueeze(0), clip_tokens)
          if weight > 0.0:
            loss = (1.0 - similarity) * weight
          else:
            loss = similarity * (-weight)
          losses.append(loss / max_loss)
  return losses

def lossTV(image, strength):
  Y = (image[:,:,1:,:] - image[:,:,:-1,:]).abs().mean()
  X = (image[:,:,:,1:] - image[:,:,:,:-1]).abs().mean()
  loss = (X + Y) * 0.5 * strength
  return loss

def get_sk_loss(int_image):
    opt = SketOptions()
    sket_model = create_model(opt)
    loss_fn = lpips.LPIPS(net='vgg').cuda()
    with torch.no_grad():
      sk_img = sket_model.netG(int_image)
    def loss_fn(image):
        im_sket = sket_model.netG(image)
        return ((sk_img - im_sket)**2).mean() * 3
        #vgg = loss_fn.scaling_layer(sk_img)
        #return loss_fn(sk_img, im_sket).mean() * 3
    return loss_fn


def main():  ## This is an experiment on Sketch image guide

    args = setup_parser()
    os.makedirs(args.outdir+"/images",exist_ok=True)
    if args.prompt != "":
        texts[0]['text'] = args.prompt
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.use_deterministic_algorithms(mode=False)
        np.random.seed(args.seed)
    perceptors = (
        loadPerceptor("ViT-B/32"),
        loadPerceptor("ViT-B/16"),
        # loadPerceptor("RN50x16"),
        )
    drawer = PyramidDrawer(stages,args.init_image,color_space=args.color_space)
    #drawer = WaveletDrawer(stages,args.init_image,color_space=args.color_space)

    image = drawer.loadImage(args.init_image)
    image = torch.nn.functional.interpolate(image, size=drawer.dims[-1], mode='bilinear', align_corners=False)
    sk_loss = get_sk_loss(image)
    for n, stage in enumerate(drawer.stages): 
        stage["n"] = n
        if n > 0: drawer.update_optim(stage)
        bar = progress_bar(range(stage["cycles"]))
        for c in bar:
            do_checkin = (c+1) % stage["checkin_interval"] == 0 or c == 0
            with torch.enable_grad():
                losses = []
                image = drawer.paramsToImage(drawer.params_pyramid)
                losses += lossClip( image, stage["cuts"], stage["noise"], perceptors )
                losses += [lossTV( image, stage["denoise"] )]
                losses += [sk_loss(image)]

                loss_total = sum(losses).sum()
                drawer.optimizer.zero_grad(set_to_none=True)
                loss_total.backward(retain_graph=False)
                #pdb.set_trace()
                drawer.optimizer.step()

            if (c+1) % args.save_interval == 0 or c == 0:
                drawer.saveImage( args.outdir+f"/images/frame_{stage['n']:02}_{c:05}.png")
            if do_checkin:
                TV = losses[-1].sum().item()
                bar.comment = f'CLIP {loss_total.item():04f}'
                drawer.saveImage(os.path.join(args.outdir, f'{args.prompt.replace(" ", "-")}_{args.seed}.png' ))
                

if __name__ == "__main__":
    main()