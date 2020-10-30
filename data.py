"""Module for processing animation data."""
import sys, os, glob

from tqdm import tqdm
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from visualize import visualize_mpl

from transforms import *

from visualization import pose2im_all

from PIL import Image


DEFAULT_MIXAMO_TRAIN = './data/mixamo/36_800_24/train'
DEFAULT_MIXAMO_VALID = './data/mixamo/36_800_24/valid'
DEFAULT_SOLODANCE_TRAIN = './data/solo_dance/train'

DEFAULT_WINDOW_SIZE = 64
DEFAULT_INTERVAL = 32

DEFAULT_TYPE = 'npy'

DATASETBASE_ARGUMENTS = {
    'path': DEFAULT_MIXAMO_TRAIN,
    'type': DEFAULT_TYPE,
    'window_size': DEFAULT_WINDOW_SIZE,
    'window_interval': DEFAULT_INTERVAL,
    'visualize_loading': False,
}


class _AnimDatasetBase(Dataset):

    def __init__(self, transforms=None, *args, **kwargs):
        """Animation dataset constructor.

        """

        # Setup arguments
        for key, value in DATASETBASE_ARGUMENTS.items():
            setattr(self, key, kwargs.get(key, value))
        for value, key in zip(args, DATASETBASE_ARGUMENTS):
            setattr(self, key, value)

        if not os.path.exists(self.path):
            raise Exception('Invalid path! {}'.format(self.path))

        # Set up some parameters
        self.num_joints = 0
        self.num_frames = 0
        self.num_anims = 0
        self.transforms = transforms

        self._frames = None
        self._anim_ranges = []
        self._anim_names = []
            
    def process(self):
        """Processes the dataset."""
        if self.type.lower() == 'npy':
            self._load_npy()
        else:
            raise NotImplementedError
    
    def _load_npy(self):
        """Parses exported Mixamo dataset."""

        characters = sorted(os.listdir(self.path))

        stored_anims = []

        logging.info('Parsing animation files...')
        for character in tqdm(characters):

            char_path = os.path.join(self.path, character)

            if not os.path.isdir(char_path):
                continue

            anims = sorted(os.listdir(char_path))

            for anim_name in anims:

                anim_path = os.path.join(
                    self.path, character, anim_name, 
                )

                if not os.path.isdir(anim_path):
                    continue

                takes = sorted(os.listdir(anim_path))

                for take in takes:
                    take_path = os.path.join(
                        self.path, character, anim_name, take
                    )

                    anim = np.load(take_path)

                    if anim.shape[1] != 3:
                        anim = np.concatenate([anim, np.zeros_like(anim[:,0:1, :])], axis=1)
                    
                    # Put in-order [nframes, njoints, 3]
                    anim = anim.swapaxes(0, -1)
                    anim = anim.swapaxes(1, 2)

                    if self.num_joints == 0:
                        self.num_joints = anim.shape[-2]

                    self.num_frames += anim.shape[0]
                    self.num_anims += 1

                    stored_anims.append(anim)
                    self._anim_ranges.append(
                        (self.num_frames-1+anim.shape[0])
                    )
                    self._anim_names.append(
                        '{}_{}'.format(character, anim_name)
                    )


        # Makes sure we have loaded animations           
        if len(stored_anims) == 0:
            raise Exception('Failed to load any frames!')

        # Resizes everything
        self._frames = torch.empty([self.num_frames, self.num_joints, 3])
        
        curr_index = 0
        logging.info('Storing animation files in memory...')
        for anim in stored_anims:
            # Stores animation
            self._frames[curr_index: curr_index+anim.shape[0]] = torch.from_numpy(anim)
            curr_index += anim.shape[0]
            

        if self._frames is None or self._frames.shape[0] == 0:
            raise Exception('Failed to load any frames!')

        logging.info('Finished loading {} frames'.format(self._frames.shape[0]))

    def _load_video(self):
        raise NotImplementedError

    def __len__(self):
        if self._frames is None:
            return 0
        else:
            return len(self._frames)

    def __getitem__(self, index):
        """Get data."""

        if self._frames is None:
            raise Exception('Animations not loaded!')

        data = self._frames[index]

        if self.transforms is not None:
            data = self.transforms(data)

        return data


class AnimDataset(_AnimDatasetBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save(self, path=None):
        """Serializes the loaded dataset."""

        if path is None:
            path = os.path.join(self.path, 'ds.pt')

        if self.num_joints == 0 or self.num_frames == 0:
            logging.warning('Attempting to save empty database!')

        logging.info('Saving model to {}'.format(path))

        torch.save({
            '_frames': self._frames,
            '_anim_ranges': self._anim_ranges,
            '_anim_names': self._anim_names,
        }, path)

    def load(self, path=None):
        if path is None:
            path = os.path.join(self.path, 'ds.pt')
        logging.info('Loading model from {}'.format(path))
        params = torch.load(path)
        for k,v in params.items():
            setattr(self, k, v)
        self.num_joints = self._frames.shape[1]
        self.num_frames = self._frames.shape[0]



class AnimDatasetWindowed(_AnimDatasetBase):

    def __init__(self, window_size=DEFAULT_WINDOW_SIZE, *args, **kwargs):

        if self.window_size == 1:
            logging.warning('No point of creating windowed ds if use window_size of 1')
        super().__init__()

    def __getitem__(self, index):
        if self._anims is None:
            raise Exception('Animations not loaded!')
        return self._frames[index]


if __name__ == '__main__':

    transformations = torch.nn.Sequential(
        IK(),
        LimbScale(std=0.05),
        FK(),
    )

    ds = AnimDataset(transforms=transformations)
    #ds = AnimDataset()

    try:
        ds.load()
    except Excpetion:
        ds.process()
        ds.save()

    dl = DataLoader(ds, batch_size=256)

    for i, pose in enumerate(dl):
        visualize_mpl(pose)
