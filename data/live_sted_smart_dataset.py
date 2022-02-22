import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy
import torch
import random
import tifffile
import skimage
from skimage import measure
from torchvision import transforms

class LiveSTEDSmartDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = tifffile.imread(AB_path)
        if AB.min() == 32768:
            AB = AB - 32768
        # split AB image into A and B
        h, w = AB.shape
        w2 = int(w / 2)
        A = AB[:, :w2]
        B = AB[:, w2:]
        A = Image.fromarray(A)
        B = Image.fromarray(B)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=True)
        B_transform = get_transform(self.opt, transform_params, grayscale=True)

        A = A_transform(A)
        B = B_transform(B)

        # Add channel consisting of real STED crops
        # First, simplify the confocal to a map of super-pixels
        # In this smart version, the STED is never taken in the center, which is defined as the ROI
        # The ROI side size if 60% of the whole image (for 500x500 image, ROI is 300x300)
        # STED pixel size is then 20% of the whole image (for 500x500, SP is 100x100)
        # The whole image is then always splitted in 25 regions (5x5)
        sup_px_map = torch.rand((1,5,5))
        sup_px_map[:,1:4,1:4] = 0
        # we randomly pick three regions in the periphery
        sorted_list = torch.sort(sup_px_map[sup_px_map!=0].view(-1))[0]#.values
        # for the first 100 epochs, all surrounding pixels (16 regions) are 'acquired'
        # every 50 epochs, one less region is acquired
        sup_px_map = sup_px_map <= sorted_list[(self.opt.npixels-1)]
        sup_px_map = sup_px_map.type(torch.FloatTensor)
        sup_px_map[:,1:4,1:4] = 0
        sup_px_map = sup_px_map.type(torch.IntTensor)

        ROI_map = torch.zeros((1,5,5)).type(torch.IntTensor)
        ROI_map[:,1:4,1:4] = 1
        tf = transforms.ToPILImage()
        ROI_map = tf(ROI_map).resize((A.shape[2], A.shape[1]), Image.NEAREST)
        sup_px_map = tf(sup_px_map).resize((A.shape[2], A.shape[1]), Image.NEAREST) # 0-1
        tf = transforms.ToTensor()
        sup_px_map = (tf(sup_px_map)[0].unsqueeze(0)).type(torch.FloatTensor)
        ROI_map = tf(ROI_map)[0].unsqueeze(0)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path, 'decision_map': sup_px_map, 'ROI_map': ROI_map}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
