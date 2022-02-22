import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy
from scipy.signal import convolve2d as conv2
import torch
import tifffile

class FixedSegmentationDataset(BaseDataset):
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
        self.dir = os.path.join(opt.dataroot, opt.phase+'_fixed')  # get the image directory
        self.image_paths = sorted(make_dataset(self.dir, opt.max_dataset_size))  # get image paths
        self.image_size = len(self.image_paths)  # get the size of dataset F
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
        image_path = self.image_paths[index]
        image = tifffile.imread(image_path)
        STED = image[0]
        STED = (STED - STED.min()) / (STED.max() - STED.min()) * 2.0 - 1.0
        STED = Image.fromarray(STED)
        rings = Image.fromarray((image[1] / 255.0) * 2.0 - 1.0)
        fibers = Image.fromarray((image[2] / 255.0) * 2.0 - 1.0)

        # apply transformation
        transform_params = get_params(self.opt, STED.size)
        image_transform = get_transform(self.opt, transform_params)

        STED = image_transform(STED)
        rings = image_transform(rings)
        fibers = image_transform(fibers)

        return {'STED': STED, 'rings': rings, 'fibers': fibers, 'image_paths': image_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.image_size
