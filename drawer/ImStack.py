import torch
from torch import nn, optim

class ImStack(nn.Module):
  """ This class represents an image as a series of stacked arrays, where each is 1/2
  the resolution of the next. This is useful eg when trying to create an image to minimise
  some loss - parameters in the early (small) layers can have an affect on the overall 
  structure and shapes while those in later layers act as residuals and fill in fine detail.
  """

  def __init__(self, n_layers=4, base_size=32, scale=2,
               init_image=None, out_size=256, decay=0.7, device='cuda'):
    """Constructs the Image Stack

    Args:
        n_layers: How many layers in the stack
        base_size: The size of the smallest layer
        scale: how much larger each subsequent layer is
        init_image: Pass in a PIL image if you don't want to start from noise
        out_size: The output size. Works best if output size ~= base_size * (scale ** (n_layers-1))
        decay: When initializing with noise, decay controls scaling of later layers (avoiding too miuch high-frequency noise)

    """
    super().__init__()
    self.n_layers = n_layers
    self.base_size = base_size
    self.sig = nn.Sigmoid()
    self.layers = []

    for i in range(n_layers):
        side = base_size * (scale**i)
        tim = torch.randn((3, side, side)).to(device)*(decay**i)
        self.layers.append(tim)

    self.scalers = [nn.Upsample(scale_factor=out_size/(l.shape[1]), mode='bilinear', align_corners=False) for l in self.layers]
    
    self.preview_scalers = [nn.Upsample(scale_factor=224/(l.shape[1]), mode='bilinear', align_corners=False) for l in self.layers]
    
    if init_image != None: # Given a PIL image, decompose it into a stack
      downscalers = [nn.Upsample(scale_factor=(l.shape[1]/out_size), mode='bilinear', align_corners=False) for l in self.layers]
      final_side = base_size * (scale ** n_layers)
      im = torch.tensor(np.array(init_image.resize((out_size, out_size)))/255).clip(1e-03, 1-1e-3) # Between 0 and 1 (non-inclusive)
      im = im.permute(2, 0, 1).unsqueeze(0).to(device) # torch.log(im/(1-im))
      for i in range(n_layers):self.layers[i] *= 0 # Sero out the layers
      for i in range(n_layers):
        side = base_size * (scale**i)
        out = self.forward()
        residual = (torch.logit(im) - torch.logit(out))
        Image.fromarray((torch.logit(residual).detach().cpu().squeeze().permute([1, 2, 0]) * 255).numpy().astype(np.uint8)).save(f'residual{i}.png')
        self.layers[i] = downscalers[i](residual).squeeze()
    
    for l in self.layers: l.requires_grad = True
    self.optimizer = optim.Adam(self.layers, lr=0.1)

  def forward(self):
    """Sums the stacked layers (upsampling them all to out_size) and then runs the result through a sigmoid funtion."""
    im = self.scalers[0](self.layers[0].unsqueeze(0))
    for i in range(1, self.n_layers):
      im += self.scalers[i](self.layers[i].unsqueeze(0))
    return self.sig(im)

  def preview(self, n_preview=2):
    """Useful if you want to optimise the first few layers first"""
    im = self.preview_scalers[0](self.layers[0].unsqueeze(0))
    for i in range(1, n_preview):
      im += self.preview_scalers[i](self.layers[i].unsqueeze(0))
    return self.sig(im)
  
  def to_pil(self):
    """Return it as a PIL Image (useful for saving, transforming, viewing etc)"""
    return Image.fromarray((self.forward().detach().cpu().squeeze().permute([1, 2, 0]) * 255).numpy().astype(np.uint8))

  def preview_pil(self):
    return Image.fromarray((self.preview().detach().cpu().squeeze().permute([1, 2, 0]) * 255).numpy().astype(np.uint8))

  def save(self, fn):
    self.to_pil().save(fn)

  def plot_layers(self):
    """View the layers in the stack - nice to build intuition about what's happening."""
    fig, axs = plt.subplots(1, self.n_layers, figsize=(15, 5))
    for i in range(self.n_layers):
      im = (self.sig(self.layers[i].unsqueeze(0)).detach().cpu().squeeze().permute([1, 2, 0]) * 255).numpy().astype(np.uint8)
      axs[i].imshow(im)