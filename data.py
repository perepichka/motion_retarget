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

DEFAULT_WINDOW_SIZE = 64
DEFAULT_INTERVAL = 32

DEFAULT_TYPE = 'npy'

DATASETBASE_ARGUMENTS = {
    'path': DEFAULT_MIXAMO_TRAIN,
    'type': DEFAULT_TYPE,
    'window_size': DEFAULT_WINDOW_SIZE,
    'window_interval': DEFAULT_INTERVAL,
    'visualize_loading': False,
    'save': True,
}


class _AnimDatasetBase(Dataset):

    def __init__(self, *args, **kwargs):
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

                    # Debug visualization
                    if self.visualize_loading:
                        visualize_mpl(anim)
                    
                    # Put in-order [nframes, njoints, 3]
                    anim = anim.swapaxes(0, -1)
                    anim = anim.swapaxes(1, 2)

                    if self.num_joints == 0:
                        self.num_joints = anim.shape[-2]

                    # Stores animation
                    if self._frames is None:
                        self._frames = torch.empty([0, self.num_joints, 3])

                    self._frames = torch.cat(
                        [self._frames, torch.from_numpy(anim)], dim=0)
                    self._anim_ranges.append(
                        (len(self._frames)-1,
                        len(self._frames)-1+anim.shape[0])
                    )
                    self._anim_names.append(
                        '{}_{}'.format(character, anim_name)
                    )


        if self._frames is None or self._frames.shape[0] == 0:
            raise Exception('Failed to load any frames!')

        logging.info('Finished loading {} frames').format(self._frames.shape[0])


    def _load_video(self):
        raise NotImplementedError

    def __len__(self, index):
        if self._frames is None:
            return 0
        else:
            return len(self._frames)

    def __getitem__(self, index):
        """Get data."""

        if self._anims is None:
            raise Exception('Animations not loaded!')

        return self._frames[index]


class AnimDataset(_AnimDatasetBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        if self._anims is None:
            raise Exception('Animations not loaded!')
        return self._frames[index]

    def save(self, path=None):
        """Serializes the loaded dataset."""
        if path is None:
            path = self.path
        logging.info('Saving model to {}'.format(path))
        torch.save({
            'num_joints': self.num_joints,
            '_frames': self.num_joints,
            '_anim_ranges': self.num_joints,
            '_anim_names': self.num_joints,
        }, path)

    def load(self, path=None):
        if path is None:
            path = os.path.join(self.path, 'ds.pt')
        logging.info('Loading model from {}'.format(path))
        params = torch.load(path)
        for k,v in params.items():
            setattr(self, key, value)



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
    ds = AnimDataset()
    ds.process()
    ds.save()
