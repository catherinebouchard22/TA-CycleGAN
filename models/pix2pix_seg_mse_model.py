import torch
from .base_model import BaseModel
from . import networks
from torchvision import models
import itertools

class Pix2PixSegMSEModel(BaseModel):
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
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_MSE', type=float, default=10, help='weight for MSE loss')
            parser.add_argument('--lambda_GAN', type=float, default=1, help='weight for GAN loss')
            parser.add_argument('--lambda_seg', type=float, default=10, help='weight for seg loss')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_GEN', 'D_real', 'D_fake', 'S_real', 'S_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'real_B', 'fake_B', 'seg_rB', 'seg_fB']
        if 'mask' in self.opt.dataset_mode:
            self.visual_names.append('seg_GT')

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        if 'mask' in self.opt.dataset_mode: #####
            self.model_names.append('S')
        # define networks (generator, discriminator and reference VGG)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids) # not opt.no_dropout
        if 'S' in self.model_names:
            self.netS = networks.define_S(opt.input_nc, opt.output_nc, opt.ngf, opt.netS, opt.norm, False, opt.init_type, opt.init_gain, self.gpu_ids)
        if 'two' in self.opt.dataset_mode:
            self.netS = networks.define_S(opt.input_nc, 2, opt.ngf, opt.netS, opt.norm, False, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionMSE = torch.nn.MSELoss()
            self.criterionSEG = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if 'mask' in self.opt.dataset_mode:
                self.optimizer_S = torch.optim.Adam(self.netS.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_S)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths']

        if 'mask' in self.opt.dataset_mode:
            self.seg_GT = input['C'].to(self.device)
            if 'two' in self.opt.dataset_mode:
                self.seg_GT2 = input['D'].to(self.device)
                self.seg_GT = torch.cat([self.seg_GT, self.seg_GT2], dim=1)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)
        if 'mask' in self.opt.dataset_mode or not self.opt.isTrain:
            self.seg_rB = self.netS(self.real_B)  # S_B(B)
            self.seg_fB = self.netS(self.fake_B)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False) * self.opt.lambda_GAN
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True) * self.opt.lambda_GAN
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and MSE loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_GAN
        # Second, G(A) = B
        self.loss_G_GEN = self.criterionMSE(self.fake_B, self.real_B) * self.opt.lambda_MSE

        # Added segmentation loss
        if self.isTrain and 'mask' in self.opt.dataset_mode:
            self.loss_S_fake = self.criterionSEG(self.seg_fB, self.seg_GT) * self.opt.lambda_seg # loss between fake B segmentation and GT

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_GEN
        if 'mask' in self.opt.dataset_mode:
            self.loss_G += self.loss_S_fake
        self.loss_G.backward()

    def backward_S(self):
        if self.isTrain and  'mask' in self.opt.dataset_mode:
            self.loss_S_real = self.criterionSEG(self.seg_GT, self.seg_rB) * self.opt.lambda_seg # loss between real B segmentation and GT
            self.loss_S_real.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G and S
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate gradients for G
        self.optimizer_G.step()             # udpate G's weights
        # update S
        if 'mask' in self.opt.dataset_mode and not self.opt.fine_tune:
            self.optimizer_S.zero_grad()        # set S's gradients to zero
            self.backward_S()                   # calculate gradients for S
            self.optimizer_S.step()             # udpate S's weights, don't if fine-tuning
