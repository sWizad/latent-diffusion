import os
import torch
import numpy as np
from models.Sketch.models import create_model
from models.Sketch.baseOption import *
from PIL import Image
from torchvision import transforms
import pdb

opt = SketOptions()
model = create_model(opt)
int_img = Image.open('outputs/test/0002.png')
transtt = transforms.Compose([
                    transforms.Resize(256),
                    transforms.ToTensor(),
                    ])
t_img = transtt(int_img)[None].cuda()

skt = model.netG(t_img)
skt_img = 255. * skt[0,0].cpu().detach().numpy()
Image.fromarray(skt_img[...,0].astype(np.uint8)).save('outputs/test/0002_skt.png')
print("done")