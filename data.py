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


DEFAULT_MIXAMO_TRAIN = './data/mixamo/36_800_24/train'
DEFAULT_MIXAMO_VALID = './data/mixamo/36_800_24/valid'
DEFAULT_SOLODANCE_TRAIN = './data/solo_dance/train'

DEFAULT_TYPE = 'npy'

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
        if self.type.lower() == 'npy':
            self._load_npy()
        else:
            raise NotImplementedError


    def _load_npy(self):
        """Parses exported Mixamo dataset."""

        characters = sorted(os.listdir(self.path))

        for character in tqdm(characters):

            anims = sorted(os.listdir(os.path.join(self.path, character)))

            for anim_name in anims:
                anim_path = os.path.join(
                    self.path, character, anim_name, 
                )

                takes = sorted(os.listdir(anim_path))

                for take in takes:
                    take_path = os.path.join(
                        self.path, character, anim_name, take
                    )

                    anim = np.load(take_path)

                    if anim.shape[1] != 3:
                        anim = np.concatenate([anim, np.zeros_like(anim[:,0:1, :])], axis=1)
                    
                    visualize_mpl(anim)

    def _load_video(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """Get data."""
        pass



if __name__ == '__main__':
    ds = AnimDataset()
