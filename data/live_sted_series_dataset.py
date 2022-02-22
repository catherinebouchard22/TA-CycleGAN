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

class LiveSTEDSeriesDataset(BaseDataset):
    """A dataset class for time series
    In this case, the data is in the form AB/A/AB/A/AB/A
    We only train with the last AB
    The relevance of the first AB is set at 0.6 and the second at 0.2 (linear decrease)
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        self.AB_paths = [s for s in AB_paths if 'STED_2' in s]
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
        AB2 = tifffile.imread(AB_path)
        AB1 = tifffile.imread(AB_path.replace('STED_2','STED_1').replace('2.msr','1.msr'))
        AB0 = tifffile.imread(AB_path.replace('STED_2','STED_0').replace('2.msr','0.msr'))
        # split AB image into A and B
        h, w = AB0.shape
        w2 = int(w / 2)
        CONF2 = Image.fromarray(AB2[:, :w2])
        STED2 = Image.fromarray(AB2[:, w2:])
        CONF1 = Image.fromarray(AB1[:, :w2])
        STED1 = Image.fromarray(AB1[:, w2:])
        CONF0 = Image.fromarray(AB0[:, :w2])
        STED0 = Image.fromarray(AB0[:, w2:])

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, CONF2.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=True)

        CONF2 = A_transform(CONF2)
        STED2 = A_transform(STED2)
        CONF1 = A_transform(CONF1)
        STED1 = A_transform(STED1)
        CONF0 = A_transform(CONF0)
        STED0 = A_transform(STED0)

        # Add channel consisting of real STED crops
        # First, simplify the confocal to a map of super-pixels
        px = random.choice([50,100]) # pixel size
        sup_px_map = skimage.measure.block_reduce(STED2.numpy()[0], (px,px), numpy.mean)
        # To choose a fixed number or super pixels
        sorted_list = numpy.sort(sup_px_map.ravel())
        STED_map = sup_px_map.copy() * 0 # this map combines all STED_crops
        decision_map = sup_px_map.copy() * 0 # this map wil give ponderation based on time elapsed since STED acquisition
        decision_map[sup_px_map == sorted_list[-1]] = 1
        decision_map[sup_px_map == sorted_list[-2]] = 0.6
        decision_map[sup_px_map == sorted_list[-3]] = 0.2

        # Resize all maps to orignal image size
        decision_map = Image.fromarray(decision_map).resize((STED2.shape[2], STED2.shape[1]), Image.NEAREST)
        STED_map = Image.fromarray(STED_map).resize((STED2.shape[2], STED2.shape[1]), Image.NEAREST)

        tf = transforms.ToTensor()
        STED_map = tf(STED_map)
        decision_map = tf(decision_map)

        # place STED_crops in STED_map
        crop0 = (decision_map == 0.2).type(torch.FloatTensor)
        crop1 = (decision_map == 0.6).type(torch.FloatTensor)
        crop2 = (decision_map == 1).type(torch.FloatTensor)
        STED_map = crop0 * STED0 + crop1 * STED1 + crop2 * STED2

        return {'A': CONF2, 'B': STED2, 'A_paths': AB_path, 'decision_map': decision_map, 'STED_map': STED_map}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
