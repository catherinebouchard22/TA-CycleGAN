import torch
from .base_model import BaseModel
from . import networks
from torchvision import models
import itertools
from torch.autograd import Variable

class ClassifierModel(BaseModel):
    """ This class implements a simple classification model.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['C']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['F', 'L']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['C']
        self.netC = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        print(self.netC)
        if self.isTrain:
            self.criterionC = torch.nn.BCELoss()
            self.optimizer_C = torch.optim.Adam(self.netC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_C)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.F = input['F'].to(self.device)
        self.L = input['L'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.predF = self.netC(self.F)  # F = 0
        self.predL = self.netC(self.L)  # L = 1
        if self.isTrain:
            self.loss_C = self.criterionC(self.predF, torch.zeros(1).expand_as(self.predF).cuda()) + self.criterionC(self.predL, torch.ones(1).expand_as(self.predL).cuda())  

    def backward_C(self):
        self.loss_C.backward()

    def optimize_parameters(self):
        self.forward()                      # compute fake images: G(A)
        self.optimizer_C.zero_grad()        # set S's gradients to zero
        self.backward_C()                   # calculate gradients for S
        self.optimizer_C.step()             # udpate S's weights, don't if fine-tuning
