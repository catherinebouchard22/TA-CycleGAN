import torch
from .base_model import BaseModel
from . import networks
from torchvision import models, transforms
import itertools

class Pix2PixSegStochModel(BaseModel):
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
        parser.set_defaults(norm='batch', netG='resnet_9blocks', netS='resnet_6blocks', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_MSE', type=float, default=1, help='weight for MSE loss')
            parser.add_argument('--lambda_GAN', type=float, default=1, help='weight for GAN loss')
            parser.add_argument('--lambda_seg', type=float, default=10, help='weight for seg loss')
            parser.add_argument('--lambda_D', type=float, default=1, help='weight for discriminator loss') # added on 09/01/20, initially same as lambda_GAN
        parser.add_argument('--num_gen', type=int, default=5, help='number of fake Bs generated')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_GEN', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if self.opt.isTrain:
            self.visual_names = ['real_A', 'fake_B', 'real_B']
            if 'mask' in self.opt.dataset_mode:
                if not self.opt.fine_tune:
                    self.loss_names.append('S_real') # remove if fine-tuning
                    self.loss_names.append('S_fake')
                    self.visual_names.append('seg_GT')
                self.visual_names.append('seg_fB')
                self.visual_names.append('seg_rB')
                if 'two' in self.opt.dataset_mode:
                    self.visual_names.append('seg2_fB')
                    self.visual_names.append('seg2_rB')
                    self.visual_names.append('seg2_GT')
        else:
            self.visual_names = ['real_A', 'real_B']
            self.visual_names.append('fake_B')
            if 'mask' in self.opt.dataset_mode:
                self.visual_names.append('seg_GT')
                if 'two' in self.opt.dataset_mode:
                    self.visual_names.append('seg2_GT')
            #self.visual_names.append('seg2_fB')
            #self.visual_names.append('seg2_rB')
            self.visual_names.append('seg_fB')
            self.visual_names.append('seg_rB')
        # sapin
        # self.visual_names = ['fake_B0']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        if 'mask' in self.opt.dataset_mode:
            self.model_names.append('S')
            if 'two' in self.opt.dataset_mode:
                self.model_names.append('S2')
        # define networks (generator, discriminator and reference VGG)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids) # not opt.no_dropout
        if 'mask' in self.opt.dataset_mode or not self.opt.isTrain:
            self.netS = networks.define_S(opt.input_nc, opt.output_nc, opt.ngf, opt.netS, opt.norm, False, opt.init_type, opt.init_gain, self.gpu_ids)
            if 'two' in self.opt.dataset_mode or not self.opt.isTrain:
                self.netS2 = networks.define_S(opt.input_nc, opt.output_nc, opt.ngf, opt.netS, opt.norm, False, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionMSE = torch.nn.L1Loss()
            self.criterionSEG = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            if 'mask' in self.opt.dataset_mode:
                if 'two' in self.opt.dataset_mode:
                    self.optimizer_S = torch.optim.Adam([{'params':self.netS.parameters()},{'params':self.netS2.parameters()}], lr=opt.lr, betas=(opt.beta1, 0.999))
                else:
                    self.optimizer_S = torch.optim.Adam(self.netS.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_S)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if 'mask' in self.opt.dataset_mode:
            self.seg_GT = input['C'].to(self.device)
            if 'two' in self.opt.dataset_mode:
                self.seg2_GT = input['D'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if 'mask' in self.opt.dataset_mode or not self.opt.isTrain:
            self.seg_rB = self.netS(self.real_B)  # S_B(B)
            if 'two' in self.opt.dataset_mode or not self.opt.isTrain:
                self.seg2_rB = self.netS2(self.real_B)
        fake_B_list = []
        S_fake_list, seg2_fB_list = [], []
        loss_B_list = []
        for i in range(self.opt.num_gen):
            fake_B = self.netG(self.real_A)
            fake_B_list.append(fake_B)
            if 'mask' in self.opt.dataset_mode or not self.opt.isTrain:
                S_fake = self.netS(fake_B)  # S_B(G_A(A))
                S_fake_list.append(S_fake)
                if 'two' in self.opt.dataset_mode or not self.opt.isTrain:
                    seg2_fB = self.netS2(fake_B)
                    seg2_fB_list.append(seg2_fB)

            if self.isTrain:
                # all losses are computed for all fakes generated
                loss_G_GEN = self.criterionMSE(fake_B, self.real_B) * self.opt.lambda_MSE

                fake_AB = torch.cat((self.real_A, fake_B), 1)
                pred_fake = self.netD(fake_AB)
                loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_GAN

                if 'mask' in self.opt.dataset_mode:
                    loss_G_seg = self.criterionSEG(S_fake, self.seg_GT) * self.opt.lambda_seg
                    if 'two' in self.opt.dataset_mode:
                        loss_G_seg = 0.5*loss_G_seg + 0.5*self.criterionSEG(S_fake, self.seg2_GT)
                else:
                    loss_G_seg = 0

                loss_B_list.append(loss_G_GEN + loss_G_GAN + loss_G_seg)

        # Select the fake generated and its segmentation that minimize the loss
        if self.opt.isTrain:
            self.fake_B = fake_B_list[loss_B_list.index(min(loss_B_list))]
            if 'mask' in self.opt.dataset_mode:
                self.seg_fB = S_fake_list[loss_B_list.index(min(loss_B_list))]
                if 'two' in self.opt.dataset_mode:
                    self.seg2_fB = seg2_fB_list[loss_B_list.index(min(loss_B_list))]
        else:
            self.fake_B = torch.mean(torch.stack(fake_B_list),dim=1)
            if 'mask' in self.opt.dataset_mode:
                self.seg_fB = torch.mean(torch.stack(S_fake_list),dim=1)
                if 'two' in self.opt.dataset_mode:
                    self.seg2_fB = torch.mean(torch.stack(seg2_fB_list),dim=1)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
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
        if self.isTrain and 'mask' in self.opt.dataset_mode:
            self.loss_S_real = self.criterionSEG(self.seg_GT, self.seg_rB) * self.opt.lambda_seg # loss between real B segmentation and GT
            if 'two' in self.opt.dataset_mode:
                self.loss_S_real = 0.5*self.loss_S_real + self.criterionSEG(self.seg2_GT, self.seg2_rB)*0.5*self.opt.lambda_seg
            self.loss_S = self.loss_S_real
            self.loss_S.backward()

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
