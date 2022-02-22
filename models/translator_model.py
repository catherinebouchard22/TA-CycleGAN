import torch
from .base_model import BaseModel
from . import networks
from torchvision import models
import itertools

class TranslatorModel(BaseModel):
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
        parser.set_defaults(norm='batch', netG='resnet_9blocks', netS='resnet_6blocks', dataset_mode='fixed_segmentation', preprocess='crop_rotation', crop_size=224, batch_size=1, display_id=-1)
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_rec', type=float, default=1, help='weight for reconstruction loss')
            parser.add_argument('--lambda_cla', type=float, default=1, help='weight for classification loss')
            parser.add_argument('--lambda_seg', type=float, default=1, help='weight for segmentation loss')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['REC', 'CLA', 'SEG']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['fix', 'fake_live', 'recon_fix', 'seg_fix', 'seg_recon']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['T_fl', 'T_lf', 'S', 'C']
        else:  # during test time, only load G
            self.model_names = ['T_fl', 'T_lf', 'S']
        # define networks (generator, discriminator and reference VGG)
        self.netT_fl = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids) # not opt.no_dropout

        self.netT_lf = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids) # not opt.no_dropout

        self.netS = networks.define_S(opt.input_nc, 2, opt.ngf, opt.netS, opt.norm, False, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netC = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, 'instance', opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionREC = torch.nn.MSELoss()
            self.criterionCLA = torch.nn.BCELoss()
            self.criterionSEG = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_T = torch.optim.Adam(itertools.chain(self.netT_fl.parameters(), self.netT_lf.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_T)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.fix = input['STED'].to(self.device)
        self.image_paths = input['image_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_live = self.netT_fl(self.fix)
        self.recon_fix = self.netT_lf(self.fake_live)
        self.predL = self.netC(self.fake_live)
        self.predF = self.netC(self.recon_fix)
        self.seg_fix = self.netS(self.fix)
        self.seg_recon = self.netS(self.recon_fix)

    def backward_T(self):

        # Classification loss
        label_F = torch.zeros(1).expand_as(self.predF).cuda()
        label_L = torch.ones(1).expand_as(self.predL).cuda()
        self.loss_CLA = (self.criterionCLA(self.predF, label_F) + self.criterionCLA(self.predL, label_L)) * 0.5 * self.opt.lambda_cla

        # Reconstruction loss
        self.loss_REC = self.criterionREC(self.fix, self.recon_fix) * self.opt.lambda_rec

        # Segmentation loss
        self.loss_SEG = self.criterionSEG(self.seg_fix, self.seg_recon) * self.opt.lambda_seg
        self.loss_T = self.loss_CLA + self.loss_REC + self.loss_SEG
        self.loss_T.backward()
        

    def optimize_parameters(self):
        self.forward()                   # compute fake images
        # update Translators T_lf and T_fl
        self.optimizer_T.zero_grad()     # set T's gradients to zero
        self.backward_T()                # calculate gradients for T
        self.optimizer_T.step()          # update T's weights