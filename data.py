"""Module for processing animation data."""
import sys, os, glob

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from visualize import visualize_mpl

from visualization import pose2im_all

from PIL import Image

PARENTS = [-1, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13]

DEFAULT_MIXAMO_TRAIN = './data/mixamo/36_800_24/train'
DEFAULT_MIXAMO_VALID = './data/mixamo/36_800_24/valid'
DEFAULT_SOLODANCE_TRAIN = './data/solo_dance/train'

DEFAULT_TYPE = 'mixamo'

DATASET_ARGUMENTS = {
    'path': DEFAULT_MIXAMO_TRAIN,
    'type': DEFAULT_TYPE
}


class AnimDataset(Dataset):

    def __init__(self, *args, **kwargs):
        """Animation dataset constructor.

        """

        # Setup arguments
        for key, value in DATASET_ARGUMENTS.items():
            setattr(self, key, kwargs.get(key, value))
        for value, key in zip(args, DATASET_ARGUMENTS):
            setattr(self, key, value)

        if not os.path.exists(self.path):
            raise Exception('Invalid path! {}'.format(self.path))

        # Set up some parameters
        self._anims = None

        # Parse dataset
        if self.type.lower() == 'mixamo':
            self._load_mixamo()
        elif self.type.lower() == 'solo_dance':
            self._load_solo_dance()
        else:
            raise NotImplementedError


    def _load_mixamo(self):
        """Parses exported Mixamo dataset."""

        characters = sorted(os.listdir(self.path))

        for character in tqdm(characters):

            anims = sorted(os.listdir(os.path.join(self.path, character)))

            for anim_name in anims:

                anim_path = os.path.join(
                    self.path, character, anim_name, anim_name + '.npy'
                )

                anim = np.load(anim_path)
                
                visualize_mpl(anim)
                #skeleton = pose2im_all(anim[:,:,0])
                #im = Image.fromarray(skeleton)
                #im.show()


    def _load_dance(self):
        pass


    def _load_video(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """Get data."""
        pass



if __name__ == '__main__':
    ds = AnimDataset()
