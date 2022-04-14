from numpy.core.numeric import False_
from torch._C import LongStorageBase
import os
#! nvidia-smi -L

#! rm -rf images
# ! rm *.png
#! mkdir images

outdir = "outputs/Artemis_ldm"
# Input prompts. Each prompt has "text" and a "weight"
# Weights can be negatives, useful for discouraging specific artifacts
texts = [
    {
        "text": "Goddess Artemis",
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
    # # },{ # Helps reduce pixelation, but also smoothes images overall. Enable if you're using scaling = 'nearest'
    # #     "text":"pixelated",
    # #     "weight":-0.25
    },{ # Not really strong enough to remove all signatures... but I'm ok with small ones
        "text":"text",
        "weight":-0.5
    }
]

#Image prompts
images = [
          # {
          #     "fpath": "hod.png",
          #     "weight": 0.2,
          #     "cuts": 16,
          #     "noise": 0.0
          # }
          ]

# random seed
# Set to None for random seed
seed = None

color_space =  "YCoCg" # "YCoCg", "RGB"

# Number of times to run
images_n = 1

save_interval = 1000

# Initial image
initial_image = "outputs/Artemis_ldm/samples/0021.png"#None # "waste.png"

# Params for gaussian init noise
chroma_noise_scale = 0.5 # Saturation (0 - 2 is safe but you can go as high as you want)
luma_noise_mean = -0.0 # Brightness (-3 to 3 seems safe but around 0 seems to work better)
luma_noise_scale = 0.5 # Contrast (0-2 is safe but you can go as high as you want)
init_noise_clamp = 8.0 # Turn this down if you're getting persistent super bright or dark spots.

# High-frequency to low-frequency initial noise ratio. 
chroma_noise_persistence = 1.0
luma_noise_persistence = 1.0

# This doesn't seem to matter too much except for 'Nearest', which results in very crisp but pixelated images
# Lanczos is most advanced but also uses the largest kernel so has the biggest problem with image borders
# Bilinear is fastest (outside of nearest) but can introduce star-like artifacts
pyramid_scaling_mode = "lanczos" # "lanczos" # "lanczos" #'bicubic' "nearest" "bilinear"

# AdamW is real basic and gets the job done
# RAdam seems to work *extremely well* but seems to introduce some color instability?, use 0.5x lr
# Yogi is just really blurry for some reason, use 5x + lr
# Ranger works great. use 3-4x LR
optimizer_type = "Ranger" # "AdamW", "AccSGD","Ranger","RangerQH","RangerVA","AdaBound","AdaMod","Adafactor","AdamP","AggMo","DiffGrad","Lamb","NovoGrad","PID","QHAdam","QHM","RAdam","SGDP","SGDW","Shampoo","SWATS","Yogi"

# Resample image prompt vectors every iterations
# Slows things down a lot for very little benefit, don't bother
resample_image_prompts = False

# Size of the smallest pyramid layer
aspect_ratio = (1,1)#(4, 3)#(3, 4)

#Add an extra pyramid layer with dims (1, 1) to control global avg color
add_global_color = True

# Max dim of the final output image.
max_dim = 512

# Number of layers at different resolutions combined into the final image
# "optimal" number is log2(max_dim / max(aspect_ratio))
# Going below that can make things kinda pixelated but still works fine
# Seems like you can really go as high as you want tho. Not sure if it helps but you *can*
pyramid_steps = 11

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

#Calculate layer dims
pyramid_lacunarity = (max_dim / max(aspect_ratio))**(1.0/(pyramid_steps-1))
scales = [pyramid_lacunarity**step for step in range(pyramid_steps)]
dims = []
if add_global_color:
  dims.append([1,1])
for step in range(pyramid_steps):
  scale = pyramid_lacunarity**step
  dim = [int(round(aspect_ratio[0] * scale)), int(round(aspect_ratio[1] * scale))]
  # Ensure that no two levels have the same dims
  if len(dims) > 0:
    if dim[0] <= dims[-1][0]:
      dim[0] = dims[-1][0]+1
    if dim[1] <= dims[-1][1]:
      dim[1] = dims[-1][1]+1
  dims.append(dim)
print(dims)
display_size = [i * 160 for i in aspect_ratio]
pyramid_steps = len(dims)
for stage in stages:
  if "lr_scales" not in stage:
    if "lr_persistence" in stage:
      persistence = stage["lr_persistence"]
    elif "pyramid_lr_min" in stage:
      persistence = stage["pyramid_lr_min"]**(1.0/float(pyramid_steps-1))
    else:
      persistence = 1.0  
    lrs = [persistence**i for i in range(pyramid_steps)]
    sum_lrs = sum(lrs)
    stage["lr_scales"] = [rate / sum_lrs for rate in lrs]
    print(persistence, stage["lr_scales"])

debug_clip_cuts = False

import sys, os, random, shutil, math
import torch, torchvision
from IPython import display
import numpy as np
from PIL import Image
import pdb
os.makedirs(outdir+"/images",exist_ok=True)


bilinear = torchvision.transforms.functional.InterpolationMode.BILINEAR
bicubic = torchvision.transforms.functional.InterpolationMode.BICUBIC

torch.autograd.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type(torch.cuda.FloatTensor)

if seed is not None:
  random.seed(seed)
  torch.manual_seed(seed)
  # torch.use_deterministic_algorithms(mode=False)
  np.random.seed(seed)

#if not os.path.isdir("CLIP"):
  #! pip -q install ftfy
  #! git clone https://github.com/openai/CLIP.git --depth 1
  #! pip -q install torch_optimizer
import clip
#import torch_optimizer as optim

def normalize_image(image):
  R = (image[:,0:1] - 0.48145466) /  0.26862954
  G = (image[:,1:2] - 0.4578275) / 0.26130258 
  B = (image[:,2:3] - 0.40821073) / 0.27577711
  return torch.cat((R, G, B), dim=1)

@torch.no_grad()
def loadImage(filename):
  data = open(filename, "rb").read()
  image = torch.ops.image.decode_png(torch.as_tensor(bytearray(data)).cpu().to(torch.uint8), 3).cuda().to(torch.float32) / 255.0
  # image = normalize_image(image)
  return image.unsqueeze(0).cuda()

def getClipTokens(image, cuts, noise, do_checkin, perceptor):
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

    if debug_clip_cuts and do_checkin:
      displayImage(cut_data)

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

perceptors = (
  loadPerceptor("ViT-B/32"),
  loadPerceptor("ViT-B/16"),
  # loadPerceptor("RN50x16"),
)

@torch.no_grad()
def saveImage(image, filename):
  # R = image[:,0:1] * 0.26862954 + 0.48145466
  # G = image[:,1:2] * 0.26130258 + 0.4578275
  # B = image[:,2:3] * 0.27577711 + 0.40821073
  # image = torch.cat((R, G, B), dim=1)
  size = image.size()

  image = (image[0].clamp(0, 1) * 255).to(torch.uint8)
  png_data = torch.ops.image.encode_png(image.cpu(), 6)
  open(filename, "wb").write(bytes(png_data))

# TODO: Use torchvision normalize / unnormalize
def unnormalize_image(image):
  
  R = image[:,0:1] * 0.26862954 + 0.48145466
  G = image[:,1:2] * 0.26130258 + 0.4578275
  B = image[:,2:3] * 0.27577711 + 0.40821073
  
  return torch.cat((R, G, B), dim=1)

def paramsToImage(params_pyramid):
  pix = []
  for c in range(3):
    pixels = torch.zeros_like(params_pyramid[-1][c])
    for i in range(len(params_pyramid)):
      if pyramid_scaling_mode == "lanczos":
        pixels += resample(params_pyramid[i][c], params_pyramid[-1][c].shape[2:])
      else:
        if pyramid_scaling_mode == "nearest" or (params_pyramid[i][c].shape[2] == 1 and params_pyramid[i][c].shape[3] == 1):
          pixels += torch.nn.functional.interpolate(params_pyramid[i][c], size=params_pyramid[-1][c].shape[2:], mode="nearest")
        else:
          pixels += torch.nn.functional.interpolate(params_pyramid[i][c], size=params_pyramid[-1][c].shape[2:], mode=pyramid_scaling_mode, align_corners=True)
    pixels = torch.sigmoid(pixels)
    pix.append(pixels)
  if color_space == "YCoCg": 
    luma = pix[0]
    Co = pix[1] * 2 - 1
    Cg = pix[2] * 2 - 1
    tmp = luma - Cg/2
    G = Cg + tmp
    B = tmp - Co/2
    R = B + Co
  elif color_space == "RGB":
    R = pix[0]
    G = pix[1]
    B = pix[2]
  im_torch = torch.cat((R, G, B), dim=1)
  return im_torch

def imageToParams(image):
  image = image#.clamp(0,1)
  R, G, B = image[:,0:1], image[:,1:2], image[:,2:3]

  eps = 1e-5

  if color_space == "YCoCg":
    luma = R * 0.25 + G * 0.5 + B * 0.25
    Co = R  - B
    tmp = B + Co / 2
    Cg = G - tmp
    luma = tmp + Cg / 2

    nsize = luma.size()[2:4]
    Co = torch.logit((Co / 2.0 + 0.5), eps=eps)
    Cg = torch.logit((Cg / 2.0 + 0.5), eps=eps)
    luma = torch.logit(luma, eps=eps)
    return luma, Co, Cg
  R = torch.logit(R, eps=eps)
  G = torch.logit(G, eps=eps)
  B = torch.logit(B, eps=eps)
  return R, G, B

@torch.no_grad()
def displayImage(image):
  size = image.size()

  width = size[0] * size[3] + (size[0]-1) * 4
  image_row = torch.zeros( size=(3, size[2], width), dtype=torch.uint8 )

  nw = 0
  for n in range(size[0]):
    image_row[:,:,nw:nw+size[3]] = (image[n,:].clamp(0, 1) * 255).to(torch.uint8)
    nw += size[3] + 4

  jpeg_data = torch.ops.image.encode_png(image_row.cpu(), 6)
  image = display.Image(bytes(jpeg_data))
  display.display( image )

def lossClip(image, cuts, noise, do_checkin):
  losses = []

  max_loss = 0.0
  for text in texts:
    max_loss += abs(text["weight"]) * len(perceptors)
  for img in images:
    max_loss += abs(img["weight"]) * len(perceptors)

  for perceptor in perceptors:
    clip_tokens = getClipTokens(image, cuts, noise, do_checkin, perceptor)
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

def cycle(c, stage, optimizer, params_pyramid):
  do_checkin = (c+1) % stage["checkin_interval"] == 0 or c == 0
  with torch.enable_grad():
    losses = []
    image = paramsToImage(params_pyramid)
    losses += lossClip( image, stage["cuts"], stage["noise"], do_checkin )
    losses += [lossTV( image, stage["denoise"] )]

    loss_total = sum(losses).sum()
    optimizer.zero_grad(set_to_none=True)
    loss_total.backward(retain_graph=False)
    # if c <= warmup_its:
    #   optimizer.param_groups[0]["lr"] = stage["lr_luma"] * c / warmup_its
    #   optimizer.param_groups[1]["lr"] = stage["lr_chroma"] * c / warmup_its
    optimizer.step()

  if (c+1) % save_interval == 0 or c == 0:
    nimg = paramsToImage(params_pyramid)
    saveImage(nimg, outdir+f"/images/frame_{stage['n']:02}_{c:05}.png")
  if do_checkin:
    TV = losses[-1].sum().item()
    print( "Cycle:", str(stage["n"]) + ":" + str(c), "CLIP Loss:", loss_total.item() - TV, "TV loss:", TV)
    nimg = paramsToImage(params_pyramid)
    #displayImage(torch.nn.functional.interpolate(nimg, size=display_size, mode='nearest'))
    saveImage(nimg, os.path.join(outdir,texts[0]["text"] + f"_{stage['n']}" + ".png" ))

    # for i in range(len(dims)):
    #   print(i, "luma", params_luma[i].min().item(), params_luma[i].max().item())
    #   print(i, "chroma", params_chroma[i].min().item(), params_chroma[i].max().item())
      
    # for i in range(len(dims)):
    #   if pyramid_scaling_mode == "lanczos":
    #     nimg = paramsToImage([resample(params_luma[i], display_size)], [resample(params_chroma[i], display_size)])
    #   else:
    #     nimg = paramsToImage([params_luma[:i+1]], [params_chroma[:i+1]])
    #     nimg = torch.nn.functional.interpolate(nimg, size=display_size, mode=pyramid_scaling_mode)
    #   displayImage(nimg)
    

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

def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.reshape([n * c, 1, h, w])

    # if dh < h:
    kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
    pad_h = (kernel_h.shape[0] - 1) // 2
    input = torch.nn.functional.pad(input, (0, 0, pad_h, pad_h), 'reflect')
    input = torch.nn.functional.conv2d(input, kernel_h[None, None, :, None])

    # if dw < w:
    kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
    pad_w = (kernel_w.shape[0] - 1) // 2
    input = torch.nn.functional.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
    input = torch.nn.functional.conv2d(input, kernel_w[None, None, None, :])

    input = input.reshape([n, c, h, w])
    return torch.nn.functional.interpolate(input, size, mode='bicubic', align_corners=align_corners)

def init_optim(params_pyramid, stage):
  lr_scales = stage["lr_scales"]
  params = []
  for i in range(len(lr_scales)):
    if color_space == "YCoCg":
      params.append({"params": params_pyramid[i][0], "lr":stage["lr_luma"] * lr_scales[i], "weight_decay":stage["decay_luma"]})
      params.append({"params": params_pyramid[i][1], "lr":stage["lr_chroma"] * lr_scales[i], "weight_decay":stage["decay_chroma"]})
      params.append({"params": params_pyramid[i][2], "lr":stage["lr_chroma"] * lr_scales[i], "weight_decay":stage["decay_chroma"]})
    elif color_space == "RGB":
      params.append({"params": params_pyramid[i][0], "lr":stage["lr_luma"] * lr_scales[i], "weight_decay":stage["decay_luma"]})
      params.append({"params": params_pyramid[i][1], "lr":stage["lr_luma"] * lr_scales[i], "weight_decay":stage["decay_luma"]})
      params.append({"params": params_pyramid[i][2], "lr":stage["lr_luma"] * lr_scales[i], "weight_decay":stage["decay_luma"]})
  optimizer = getattr(optim, optimizer_type, None)(params)
  return optimizer

def main():
  params_pyramid = []
  if initial_image is not None:
    for dim in dims:
      pix = []
      for channel in range(3):
        pix_c = torch.zeros((1,1,dim[0], dim[1]))
        param_pix = torch.nn.parameter.Parameter( pix_c.cuda(), requires_grad=True)
        pix.append(param_pix)
      params_pyramid.append(pix)
    image = loadImage(initial_image)
    image = torch.nn.functional.interpolate(image, size=dims[-4], mode='bicubic', align_corners=False)
    image = torch.nn.functional.interpolate(image, size=dims[-1], mode='bicubic', align_corners=False)

    img_channels = imageToParams(image)
    pix = []
    for channel in range(3):
      c = img_channels[channel]
      if add_global_color:
        avg = img_channels[channel].mean()
        c -= avg
        params_pyramid[0][channel] += avg
      param_pix = torch.nn.parameter.Parameter( c.cuda(), requires_grad=True)
      pix.append(param_pix)
    params_pyramid[-1] = pix
    #pdb.set_trace()
  else:
    for i, dim in enumerate(dims):
      if color_space == "YCoCg":
        luma = (torch.randn(size = (1,1,dim[0], dim[1])) * luma_noise_scale * luma_noise_persistence**i / len(dims)).clamp(-init_noise_clamp / len(dims), init_noise_clamp / len(dims)) 
        Co = (torch.randn(size = (1,1,dim[0], dim[1])) * chroma_noise_scale * chroma_noise_persistence**i / len(dims)).clamp(-init_noise_clamp / len(dims), init_noise_clamp / len(dims)) 
        Cg = (torch.randn(size = (1,1,dim[0], dim[1])) * chroma_noise_scale * chroma_noise_persistence**i / len(dims)).clamp(-init_noise_clamp / len(dims), init_noise_clamp / len(dims)) 
        param_luma = torch.nn.parameter.Parameter( luma.cuda(), requires_grad=True)
        param_co = torch.nn.parameter.Parameter( Co.cuda(), requires_grad=True)
        param_cg = torch.nn.parameter.Parameter( Cg.cuda(), requires_grad=True)
        pix = [param_luma, param_co, param_cg]
      elif color_space == "RGB":
        pix = []
        for channel in range(3):
          pix_c = (torch.randn(size = (1,1,dim[0], dim[1])) * chroma_noise_scale * chroma_noise_persistence**i / len(dims)).clamp(-init_noise_clamp / len(dims), init_noise_clamp / len(dims)) 
          param_pix = torch.nn.parameter.Parameter( pix_c.cuda(), requires_grad=True)
          pix.append(param_pix)
      params_pyramid.append(pix)
  # if color_space == "YCoCg":
  #   params_pyramid[0][:, 0, :, :] += luma_noise_mean
  # elif color_space == "RGB":
  #   params_pyramid[0] += luma_noise_mean


  optimizer = init_optim(params_pyramid, stages[0])

  for n, stage in enumerate(stages):
    stage["n"] = n
    if n > 0:
      # if stage['dim'][0] != param_luma.shape[2]:
      #   if upscaling_mode == "lanczos":
      #     luma = resample(param_luma, ( stage['dim'][0], stage['dim'][1] ))
      #     chroma = resample(param_chroma, ( stage['dim'][0], stage['dim'][1] )) 
      #     param_luma = torch.nn.parameter.Parameter( luma.cuda(), requires_grad=True )
      #     param_chroma = torch.nn.parameter.Parameter( chroma.cuda(), requires_grad=True )
      #   else:
      #     param_luma = torch.nn.parameter.Parameter(torch.nn.functional.interpolate(param_luma.data, size=( stage['dim'][0], stage['dim'][1] ), mode=upscaling_mode, align_corners=False), requires_grad=True ).cuda()
      #     param_chroma = torch.nn.parameter.Parameter(torch.nn.functional.interpolate(param_chroma.data, size=( stage['dim'][0]//chroma_fraction, stage['dim'][1]//chroma_fraction ), mode=upscaling_mode, align_corners=False), requires_grad=True ).cuda()
      # if "init_noise" in stage:
      #   param_luma += torch.randn_like(param_luma) * stage["init_noise"]
      #   param_chroma += torch.randn_like(param_chroma) * stage["init_noise"]
      # optimizer = init_optim(params_luma, params_chroma, stage)
      for i in range(len(dims)):
        if color_space == "YCoCg":
          optimizer.param_groups[i*3]["lr"] = stage["lr_luma"] * stage["lr_scales"][i]
          optimizer.param_groups[i*3+1]["lr"] = stage["lr_chroma"] * stage["lr_scales"][i]
          optimizer.param_groups[i*3+2]["lr"] = stage["lr_chroma"] * stage["lr_scales"][i]
        elif color_space == "RGB":
          optimizer.param_groups[i*3]["lr"] = stage["lr_luma"] * stage["lr_scales"][i]
          optimizer.param_groups[i*3+1]["lr"] = stage["lr_luma"] * stage["lr_scales"][i]
          optimizer.param_groups[i*3+2]["lr"] = stage["lr_luma"] * stage["lr_scales"][i]
      #   optimizer.param_groups[0]["lr"] = stage["lr_luma"] * c / warmup_its
      #   optimizer.param_groups[1]["lr"] = stage["lr_chroma"] * c / warmup_its
        # params.append({"params":params_luma[i], "lr":stage["lr_luma"] * lr_scales[i], "weight_decay":stage["decay_luma"]})
        # params.append({"params":params_chroma[i], "lr":stage["lr_chroma"] * lr_scales[i], "weight_decay":stage["decay_chroma"]})

    for c in range(stage["cycles"]):
      cycle( c, stage, optimizer, params_pyramid)
    # for i in range(len(dims)):
    #   if pyramid_scaling_mode == "lanczos":
    #     nimg = paramsToImage(params_luma[:i+1], params_chroma[:i+1])
    #     nimg = torch.nn.functional.interpolate(nimg, size=display_size, mode="bicubic")
    #   else:
    #     nimg = paramsToImage(params_luma[:i+1], params_chroma[:i+1])
    #     nimg = torch.nn.functional.interpolate(nimg, size=display_size, mode=pyramid_scaling_mode)
    #   displayImage(nimg)

for _ in range(images_n):
  main()