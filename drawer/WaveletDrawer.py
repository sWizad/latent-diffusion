import torch
from torch import nn
import pdb
import torch_optimizer as optim
from IPython import display
from .utils import *
from pytorch_wavelets import DWTForward, DWTInverse

# This doesn't seem to matter too much except for 'Nearest', which results in very crisp but pixelated images
# Lanczos is most advanced but also uses the largest kernel so has the biggest problem with image borders
# Bilinear is fastest (outside of nearest) but can introduce star-like artifacts
pyramid_scaling_mode = "lanczos" # "lanczos" # "lanczos" #'bicubic' "nearest" "bilinear"

# Params for gaussian init noise
chroma_noise_scale = 2 # Saturation (0 - 2 is safe but you can go as high as you want)
luma_noise_mean = -0.0 # Brightness (-3 to 3 seems safe but around 0 seems to work better)
luma_noise_scale = 2 # Contrast (0-2 is safe but you can go as high as you want)
init_noise_clamp = 8.0 # Turn this down if you're getting persistent super bright or dark spots.

# High-frequency to low-frequency initial noise ratio. 
chroma_noise_persistence = 1.0
luma_noise_persistence = 1.0

# Size of the smallest pyramid layer
aspect_ratio = (1,1) #(4, 3)#(3, 4)

#Add an extra pyramid layer with dims (1, 1) to control global avg color
add_global_color = True

# Max dim of the final output image.
max_dim = 800

# Number of layers at different resolutions combined into the final image
# "optimal" number is log2(max_dim / max(aspect_ratio))
# Going below that can make things kinda pixelated but still works fine
# Seems like you can really go as high as you want tho. Not sure if it helps but you *can*
pyramid_steps = 8

# AdamW is real basic and gets the job done
# RAdam seems to work *extremely well* but seems to introduce some color instability?, use 0.5x lr
# Yogi is just really blurry for some reason, use 5x + lr
# Ranger works great. use 3-4x LR
optimizer_type = "Ranger" # "AdamW", "AccSGD","Ranger","RangerQH","RangerVA","AdaBound","AdaMod","Adafactor","AdamP","AggMo","DiffGrad","Lamb","NovoGrad","PID","QHAdam","QHM","RAdam","SGDP","SGDW","Shampoo","SWATS","Yogi"

class WaveletDrawer(nn.Module):
    def __init__(self,stages,initial_image=None, color_space =  "YCoCg"):
        """
        """
        super().__init__()
        self.dims = self.cal_layer_dim()
        self.stages = self.get_stages(stages)
        self.color_space = color_space
        self.add_global_color = True
        
        self.xfm = DWTForward(J=pyramid_steps, wave='db1', mode='symmetric').cuda() #coif2
        self.ifm = DWTInverse(wave='db1', mode='symmetric').cuda() # symmetric zero periodization

        params_pyramid = []
        if initial_image is not None:
            image = self.loadImage(initial_image)
            image = torch.nn.functional.interpolate(image, size=self.dims[-1], mode='bilinear', align_corners=False)

            img_channels = self.imageToParams(image)
            for channel in range(3):
                c = img_channels[channel]
                Yl_in, Yh_in = self.xfm(c)
                #Ys = [Yl_in] + Yh_in
                #Ys = [Yl_in] + [(0.01*torch.randn(*Y.shape).cuda()) for Y in Yh_in]
                Ys = [0.5*Yl_in] + [0.5*Y for Y in Yh_in]
                Ys = [torch.nn.parameter.Parameter( y, requires_grad=True) for y in Ys]
                params_pyramid.append(Ys)
        else: #TODO init without int img
            image = torch.zeros([1,3]+self.dims[-1]).cuda()
            img_channels = self.imageToParams(image)
            for channel in range(3):
                c = img_channels[channel]
                Yl_in, Yh_in = self.xfm(c)
                Ys = [0.5*Yl_in] + [0.5*Y for Y in Yh_in]
                Ys = [torch.nn.parameter.Parameter( y, requires_grad=True) for y in Ys]
                params_pyramid.append(Ys)

            dim_len = len(params_pyramid[0])
            #for i, dim in enumerate(self.dims):
            for i in range(1,dim_len):
                dim = params_pyramid[0][i].shape
                if color_space == "YCoCg":
                    luma = (torch.randn(size = dim) * luma_noise_scale * luma_noise_persistence**(dim_len-i) / dim_len).clamp(-init_noise_clamp / dim_len, init_noise_clamp / dim_len) 
                    Co = (torch.randn(size = dim) * chroma_noise_scale * chroma_noise_persistence**(dim_len-i) / dim_len).clamp(-init_noise_clamp / dim_len, init_noise_clamp / dim_len) 
                    Cg = (torch.randn(size = dim) * chroma_noise_scale * chroma_noise_persistence**(dim_len-i) / dim_len).clamp(-init_noise_clamp / dim_len, init_noise_clamp / dim_len) 
                    params_pyramid[0][i] = torch.nn.parameter.Parameter( luma.cuda(), requires_grad=True)
                    params_pyramid[1][i] = torch.nn.parameter.Parameter( Co.cuda(), requires_grad=True)
                    params_pyramid[2][i] = torch.nn.parameter.Parameter( Cg.cuda(), requires_grad=True)
                    #pix = [param_luma, param_co, param_cg]
                    
                elif color_space == "RGB":
                    pix = []
                    for channel in range(3):
                        pix_c = (torch.randn(size = dim) * chroma_noise_scale * chroma_noise_persistence**i / dim_len).clamp(-init_noise_clamp / dim_len, init_noise_clamp / dim_len) 
                        param_pix = torch.nn.parameter.Parameter( pix_c.cuda(), requires_grad=True)
                        params_pyramid[channel][i]
                #params_pyramid.append(pix)

        self.params_pyramid = params_pyramid
        # if color_space == "YCoCg":
        #   params_pyramid[0][:, 0, :, :] += luma_noise_mean
        # elif color_space == "RGB":
        #   params_pyramid[0] += luma_noise_mean
        
        self.optimizer = self.init_optim(params_pyramid, stages[0])

    def cal_layer_dim(self):
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
        print(dim)
        return dims

    def get_stages(self,stages):
        self.display_size = [i * 160 for i in aspect_ratio]
        pyramid_steps = len(self.dims)
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
                #print(persistence, stage["lr_scales"])   
        return stages

    @torch.no_grad()
    def loadImage(self,filename):
        data = open(filename, "rb").read()
        image = torch.ops.image.decode_png(torch.as_tensor(bytearray(data)).cpu().to(torch.uint8), 3).cuda().to(torch.float32) / 255.0
        # image = normalize_image(image)
        return image.unsqueeze(0).cuda()

    def paramsToImage(self,params_pyramid):
        #pdb.set_trace()
        pix = []
        for c in range(3):
            img_c = self.ifm((params_pyramid[c][0],params_pyramid[c][1:]))
            img_c = torch.sigmoid(img_c)
            pix.append(img_c)
            '''
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
            '''

        if self.color_space == "YCoCg": 
            luma = pix[0]
            Co = pix[1] * 2 - 1
            Cg = pix[2] * 2 - 1
            tmp = luma - Cg/2
            G = Cg + tmp
            B = tmp - Co/2
            R = B + Co
        elif self.color_space == "RGB":
            R = pix[0]
            G = pix[1]
            B = pix[2]
        im_torch = torch.cat((R, G, B), dim=1)
        self.img = im_torch
        return im_torch

    def imageToParams(self,image):
        #image = image.clamp(0,1)
        R, G, B = image[:,0:1], image[:,1:2], image[:,2:3]
        eps = 1e-5
        if self.color_space == "YCoCg":
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
        else:
            R = torch.logit(R, eps=eps)
            G = torch.logit(G, eps=eps)
            B = torch.logit(B, eps=eps)
            return R, G, B

    def init_optim(self,params_pyramid, stage):
        stage["lr_scales"].reverse()
        lr_scales = [stage["lr_scales"][-1]]+stage["lr_scales"][:-1]
        params = []
        
        for i in range(len(lr_scales)):
            if self.color_space == "YCoCg":
                params.append({"params": [params_pyramid[0][i]], "lr":stage["lr_luma"]*10 * lr_scales[i], "weight_decay":stage["decay_luma"]})
                params.append({"params": [params_pyramid[1][i]], "lr":stage["lr_chroma"]*10 * lr_scales[i], "weight_decay":stage["decay_chroma"]})
                params.append({"params": [params_pyramid[2][i]], "lr":stage["lr_chroma"]*10 * lr_scales[i], "weight_decay":stage["decay_chroma"]})
            elif self.color_space == "RGB":
                params.append({"params": params_pyramid[0][i], "lr":stage["lr_luma"] * lr_scales[i], "weight_decay":stage["decay_luma"]})
                params.append({"params": params_pyramid[1][i], "lr":stage["lr_luma"] * lr_scales[i], "weight_decay":stage["decay_luma"]})
                params.append({"params": params_pyramid[2][i], "lr":stage["lr_luma"] * lr_scales[i], "weight_decay":stage["decay_luma"]})
        
        #lr=0.01
        #decay = 1.5
        #for i, y in enumerate(self.Ys):
        #    if i>= bound**2:
        #        lr = lr/self.decay
        ##        bound += 1
        #    params.append({'params':[y], 'lr':lr})
        optimizer = getattr(optim, optimizer_type, None)(params)
        #optimizer = torch.optim.Adam(params)
        return optimizer

    def update_optim(self,stage):
        for i in range(len(self.dims)):
            if self.color_space == "YCoCg":
                self.optimizer.param_groups[i*3]["lr"] = stage["lr_luma"] * stage["lr_scales"][i] * 10
                self.optimizer.param_groups[i*3+1]["lr"] = stage["lr_chroma"] * stage["lr_scales"][i] *10
                self.optimizer.param_groups[i*3+2]["lr"] = stage["lr_chroma"] * stage["lr_scales"][i] *10
            elif self.color_space == "RGB":
                self.optimizer.param_groups[i*3]["lr"] = stage["lr_luma"] * stage["lr_scales"][i]
                self.optimizer.param_groups[i*3+1]["lr"] = stage["lr_luma"] * stage["lr_scales"][i]
                self.optimizer.param_groups[i*3+2]["lr"] = stage["lr_luma"] * stage["lr_scales"][i]
        #   optimizer.param_groups[0]["lr"] = stage["lr_luma"] * c / warmup_its
        #   optimizer.param_groups[1]["lr"] = stage["lr_chroma"] * c / warmup_its
        # params.append({"params":params_luma[i], "lr":stage["lr_luma"] * lr_scales[i], "weight_decay":stage["decay_luma"]})
        # params.append({"params":params_chroma[i], "lr":stage["lr_chroma"] * lr_scales[i], "weight_decay":stage["decay_chroma"]})

    @torch.no_grad()
    def saveImage(self, filename):
        self.img
        size = self.img.size()
        image = (self.img[0].clamp(0, 1) * 255).to(torch.uint8)
        png_data = torch.ops.image.encode_png(image.cpu(), 6)
        open(filename, "wb").write(bytes(png_data))

    @torch.no_grad()
    def displayImage(self,image): #currently not in used
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