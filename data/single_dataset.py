import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy
import torch
import tifffile

class SingleDataset(BaseDataset):
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
        self.image_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
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
        image_path = self.image_paths[index]
        image = tifffile.imread(image_path)#[0,0,:,:]
        if image.min() == 32768:
            image -= 32768

        # Make sure size is compatible with UNet-128 (must be a multiple of 128)
        if self.opt.netS == 'unet_128':
            image = image[:image.shape[0] - image.shape[0] % 128, :image.shape[1] - image.shape[1] % 128]

        # apply transformation
        STED = Image.fromarray(image)
        transform_params = get_params(self.opt, STED.size)
        image_transform = get_transform(self.opt, transform_params, grayscale=True)

        STED = image_transform(STED)
        rings = STED
        fibers = STED

        return {'A': STED, 'B': STED, 'C': STED, 'rings':STED, 'fibers':STED, 'A_paths': image_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)
