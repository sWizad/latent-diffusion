class SketOptions():
    def __init__(self):
        self.model = 'pix2pix'
        self.isTrain = False
        self.use_cuda = True
        self.checkpoints_dir = 'symlink/sketch_ck/'
        self.name = 'pretrained'
        self.results_dir ='symlink/Results'
        self.input_nc = 3
        self.output_nc = 1 
        self.ngf=64
        self.ndf=64
        self.which_model_netG = 'resnet_9blocks'
        self.which_direction='AtoB'
        self.norm = 'batch'
        self.resize_or_crop = 'resize_and_crop'
        self.no_flip = True
        self.init_type='normal'
        self.no_dropout = True
        self.crop = False
        self.rotate = False
        self.color_jitter = False
        self.nGT = 5
        self.jitter_amount = 0.02
        self.inverse_gamma=False
        self.which_epoch='latest'
        self.how_many=50
        self.pretrain_path =''