import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy
from scipy.signal import convolve2d as conv2
import torch
import random
import tifffile

class FixedLiveDataset(BaseDataset):
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
        self.dir_F = os.path.join(opt.dataroot, opt.phase+'_fixed')  # get the image directory
        self.dir_L = os.path.join(opt.dataroot, opt.phase+'_STEDB')   # get the image directory
        self.F_paths = sorted(make_dataset(self.dir_F, opt.max_dataset_size))  # get image paths
        self.L_paths = sorted(make_dataset(self.dir_L, opt.max_dataset_size))  # get image paths
        self.F_size = len(self.F_paths)  # get the size of dataset F
        self.L_size = len(self.L_paths)  # get the size of dataset L
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image

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
        # read an image given a random integer index
        F_path = self.F_paths[index % self.F_size]
        if self.opt.serial_batches:   # make sure index is within range
            index_L = index % self.L_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_L = random.randint(0, self.L_size - 1)
        F = Image.fromarray(tifffile.imread(F_path)[0])

        L_path = self.L_paths[index_L]
        L = Image.open(L_path)

        # apply the same transform to F and S
        transform_params = get_params(self.opt, F.size)
        F_transform = get_transform(self.opt, transform_params, grayscale=True)
        # apply random transform to L
        L_transform_params = get_params(self.opt, L.size)
        L_transform = get_transform(self.opt, L_transform_params, grayscale=True)

        F = F_transform(F)
        L = L_transform(L)

        return {'A': F, 'B': L, 'A_paths': F_path, 'B_paths': L_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.F_size, self.L_size)
