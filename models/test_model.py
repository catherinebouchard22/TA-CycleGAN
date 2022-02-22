import torch
from .base_model import BaseModel
from . import networks
from torchvision import models, transforms
import itertools
import numpy
import tifffile

class TestModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_MSE * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='resnet_9blocks', netS='resnet_6blocks', dataset_mode='aligned', input_nc=1, output_nc=1)
        parser.add_argument('--num_gens', type=int, default=1, help='number of generations (test only)')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_PCT', 'D_real', 'D_fake', 'seg_fB']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'seg_fB0', 'seg_fB1', 'fake_B']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G', 'S']
        # define networks (generator, discriminator and reference VGG)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids) # not opt.no_dropout

        self.netS = networks.define_S(opt.output_nc, 2, opt.ngf, opt.netS, opt.norm, False, opt.init_type, opt.init_gain, self.gpu_ids)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device) # -1-1 (absolute)
        self.real_B = input['B' if AtoB else 'A'].to(self.device) # -1-1 (absolute)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        if self.opt.num_gens != 1:
            self.fake_B = torch.zeros((self.opt.num_gens,1,self.real_A.shape[2],self.real_A.shape[3]))
            self.seg_fB0 = torch.zeros((self.opt.num_gens,1,self.real_A.shape[2],self.real_A.shape[3]))
            self.seg_fB1 = torch.zeros((self.opt.num_gens,1,self.real_A.shape[2],self.real_A.shape[3]))
            for i in range(self.opt.num_gens):
                fake_B = self.netG(self.real_A)
                seg_fB = self.netS(fake_B)
                self.fake_B[i,:,:,:] = fake_B
                self.seg_fB0[i,:,:,:] = (seg_fB[:,0,:,:]>-0.5).float()
                self.seg_fB1[i,:,:,:] = (seg_fB[:,1,:,:]>-0.2).float()


    def optimize_parameters(self):
        pass
