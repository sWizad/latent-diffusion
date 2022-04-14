import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image
import pywt
from pytorch_wavelets import DWTForward, DWTInverse
import pdb

def dwt_scale(Ys, sharp):
    scale = []
    [h0,w0] = Ys[1].shape[3:5]
    for i in range(len(Ys)-1):
        [h,w] = Ys[i+1].shape[3:5]
        scale.append( ((h0*w0)/(h*w)) ** (1.-sharp) )
        # print(i+1, Ys[i+1].shape)
    return scale

class DwtDrawer(nn.Module):
  """ 
  """
  #num_rows, num_cols, num_step = 12,20, 3
  num_rows, num_cols, num_step = 32,32, 3
  decay = 1.5
  do_mono = False
  pixels = []
  def __init__(self, width=256, height=256, do_mono=False, 
                shape=None, decay=None, lr=0.01, device='cuda', plan = 'normal',
                int_img = None):
    """
    """
    super().__init__()
    self.canvas_width = width
    self.canvas_height = height
    self.do_mono = do_mono
    self.use_c2f = True

    if plan in ['easy']: 
        self.plan = [25,50,100]
    elif plan in ['fast']:
        self.plan = [100,200,400]
    else: 
        #self.plan = [400,800,1600]
        self.plan = [600,1000,1400]

    if shape is not None:
        self.num_rows, self.num_cols, self.num_step = shape
    fac = 2**self.num_step
    self.shape = [1, 3, self.num_rows*fac, self.num_cols*fac]
    #self.shape = [1, 3, height, width]
    #wp_fake = pywt.WaveletPacket2D(data=np.zeros(self.shape[2:]), wavelet='db1', mode='symmetric')
    #pdb.set_trace()
    #self.xfm = DWTForward(J=wp_fake.maxlevel, wave='coif2', mode='symmetric').cuda()
    self.xfm = DWTForward(J=8, wave='coif2', mode='symmetric').cuda()
    self.ifm = DWTInverse(wave='coif2', mode='symmetric').cuda() # symmetric zero periodization

    if decay is not None:
        self.decay = decay
    

    self.lr = lr
    self.upshape = transforms.Resize((height,width), interpolation = transforms.InterpolationMode.NEAREST) 
    if self.use_c2f:
        n = 3
        s_rows, s_cols = self.num_rows*fac//2**(n-1), self.num_cols*fac//2**(n-1)
        self.down = [transforms.Resize((s_rows*2**i,s_cols*2**i), interpolation = transforms.InterpolationMode.BILINEAR) for i in range(n)]


    color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                          [0.27, 0.00, -0.05],
                                          [0.27, -0.09, 0.03]]).astype("float32")
    color_correlation_svd_sqrt /= np.asarray([1.5, 1., 1.]) # saturate, empirical
    max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
    color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
    self.to_valid = torch.tensor(color_correlation_normalized.T).cuda()
    self.to_valid_inv = torch.tensor(np.linalg.inv(color_correlation_normalized.T)).cuda()
    if int_img is None:
        int_img = torch.zeros(self.shape).cuda()
    else:
        int_img = int_img.permute(1,2,0)
        #int_img = torch.matmul(int_img,self.to_valid_inv)
        int_img = int_img.permute(2,0,1)*2-1
        int_img = int_img[None]

    self.load_model(int_img)

  def load_model(self,int_img):
    self.cur_iteration = 0

    canvas_width, canvas_height = self.canvas_width, self.canvas_height
    num_rows, num_cols = self.num_rows, self.num_cols

    # Initialize Random Pixels
    Yl_in, Yh_in = self.xfm(int_img)
    #Yl_in, Yh_in = self.xfm(torch.zeros(self.shape).cuda())
    #Ys = [torch.randn(*Y.shape).cuda() for Y in [Yl_in, *Yh_in]]
    Ys = [0.1*torch.randn(*Yl_in.shape).cuda()] + Yh_in #[(0.01*torch.randn(*Y.shape).cuda()) for Y in Yh_in]
    #Ys = [Yl_in] + Yh_in
    self.Ys = [y.requires_grad_(True) for y in Ys]
    self.scale = dwt_scale(self.Ys, 0.3)

    lr = self.lr
    para = []
    bound = 1
    for i, y in enumerate(self.Ys):
        #print(torch.min(y),torch.max(y))
        if i>= bound**2:
            lr = lr/self.decay
            bound += 1
        para.append({'params':[y], 'lr':lr})
    img = self.synth(0)

    opt = [torch.optim.Adam(para, betas = (0.9,0.999))]

    self.img = img
    self.optimizer = opt[0] #+ [weight_optim]

  def to_valid_rgb(self,img):
    img = img.permute(0,2,3,1)
    img = torch.matmul(img,self.to_valid)
    img = img.permute(0,3,1,2)
    return img

  def render(self):
    img = self.ifm((self.Ys[0],self.Ys[1:]))
    #img = self.ifm((self.Ys[0], [self.Ys[i+1] * float(self.scale[i]) for i in range(len(self.Ys)-1)]))
    #print(img.std())
    #img = img/img.std()
    #img = self.to_valid_rgb(img)
    #img = torch.sigmoid(img)
    img = torch.clamp((img+1)/2 , 0,1)
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
      return self.synth(self.cur_iteration)
  
  def to_pil(self):
      x_sample = 255. * rearrange(self.img[0], 'c h w -> h w c').cpu().detach().numpy()
      return Image.fromarray(x_sample.astype(np.uint8))

  def save(self, fn):
      self.to_pil().save(fn)
