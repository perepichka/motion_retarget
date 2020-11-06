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

from PIL import Image


DEFAULT_MIXAMO_TRAIN = './data/mixamo/36_800_24/train'
DEFAULT_MIXAMO_VALID = './data/mixamo/36_800_24/valid'
DEFAULT_SOLODANCE_TRAIN = './data/solo_dance/train'
DEFAULT_STATS_PATH = '../'

DEFAULT_DS_FILE = 'ds.pt'
DEFAULT_JOINTS_FILE = 'joints.txt'
DEFAULT_PARENTS_FILE = 'parents.txt'

DEFAULT_WINDOW_SIZE = 64
DEFAULT_INTERVAL = 32


DATASETBASE_ARGUMENTS = {
    'path': DEFAULT_MIXAMO_TRAIN,
    'stats_path': DEFAULT_STATS_PATH,
    'joints_file': DEFAULT_JOINTS_FILE,
    'parents_file': DEFAULT_PARENTS_FILE,
    'dataset_file': DEFAULT_DS_FILE,
    'window_size': DEFAULT_WINDOW_SIZE,
    'window_interval': DEFAULT_INTERVAL,
    'visualize_loading': False,
    '_reload': False,
}


class AnimDataset(Dataset):

    def __init__(self, transforms=None, pre_transforms=None, *args, **kwargs):
        """Animation Dataset constructor.
        
        :param transforms: Transformations to transform/augment animation data. 
        :param pre_transforms: Transformations to be pre-computed on entire dataset.
        :param path: Path to the folder containing the dataset.
        :param window_size: Size of the dataset Window.
        :param window_interval: Stride between window samples.
        :param visualize_loading: Whether to visualize animation data while 
            loading.

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

        self.joint_names = []
        self.parent_indices = []

        self._frames = None

        self._anim_ranges = []
        self._anim_info = []

        # Set up transforms
        self.transforms = transforms
        self.pre_transforms = pre_transforms

        # Loads data if needed
        if self.path is not None:
            self.load()

        # Sends loaded info to transforms
        self._update_transforms()

        # Preprocesses data if neeeded
        if self.pre_transforms is not None:
            self.precompute_transforms()
        
    
    def change_root(self, new_root):
        """Changes the root joint of the data"""

        logging.info('Changing root joint to {}'.format(new_root))
        if type(new_root) == str:
            if new_root == self.root_joint:
                logging.info('New root is the current root, skipping...')
                return
            elif new_root not in self.joint_names:
                raise Exception('Specified root {} not found in hierarchy'.format(new_root))
            
        elif type(new_root) == int:
            if new_root == self.joint_names.index(self.root_joint):
                logging.info('New root is the current root, skipping...')
                return
            if new_root < len(self.joint_names) and new_root > 0:
                new_root_index = new_root

        # Reprocess parent joints

        # All 


    def precompute_transforms(self):
        """Precomputes animation transforms."""

        if self._frames is None or len(self._frames) == 0:
            raise Exception('No data to precompute transforms on')

        
        self._frames_transformed = self.transforms((self._frames, self._anim_ranges))


    def save(self, path=None):
        """Serializes the loaded dataset."""

        if path is None:
            path = self.path

        path = os.path.join(path, self.dataset_file)

        if self.num_joints == 0 or self.num_frames == 0:
            logging.warning('Attempting to save empty database!')

        logging.info('Saving model to {}'.format(path))

        torch.save({
            '_frames': self._frames,
            '_anim_ranges': self._anim_ranges,
            '_anim_info': self._anim_info,
            'joint_names': self.joint_names,
            'parent_indices': self.parent_indices,
            'root_joint': self.root_joint,
        }, path)


    def load(self, path=None):
        """Load data set from serialized file(s)."""


        if self.path is None and path is None:
            raise Exception('No path specified!')

        if path is None:
            path = self.path

        if not self._reload:
            path = os.path.join(path, self.dataset_file)
            self._load_pt(path)
        else:
            self.path = path
            self._load_npy()

        logging.info('Finished loading {} frames'.format(self._frames.shape[0]))


    def _load_pt(self, path):
        """Loads saved dataset, found in path."""

        logging.info('Loading model from {}'.format(path))
        params = torch.load(path)
        for k,v in params.items():
            setattr(self, k, v)
        self.num_joints = self._frames.shape[1]
        self.num_frames = self._frames.shape[0]
        self.root_joint = self.joint_names[0]


    def _load_npy(self):
        """Parses exported Mixamo dataset."""

        characters = sorted(os.listdir(self.path))

        stored_anims = []

        logging.info('Parsing structure info..')
        try:
            with open(os.path.join(self.path, self.joints_file), 'r') as f:
                self.joint_names = [name.strip('\n') for name in f.readline().split(',')]
            logging.debug('Parsed joint names: {}'.format(self.joint_names))
            with open(os.path.join(self.path, self.parents_file), 'r') as f:
                self.parent_indices = [int(ind) for ind in f.readline().split(',')]
            logging.debug('Parsed parent indices: {}'.format(self.parent_indices))
            self.root_joint = self.joint_names[0]
        except Exception as e:
            logging.error('Unable to load structure file(s)')
            raise e

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
                        (self.num_frames-1, self.num_frames-1+anim.shape[0])
                    )
                    self._anim_info.append(
                            {'character': character, 'anim_name': anim_name, 'take': take}
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


    def _load_video(self):
        raise NotImplementedError

    def _update_transforms(self):
        """Passes updated info to data transforms"""

        for k,v in self.transforms._modules.items():
            v._anim_ranges = self._anim_ranges
            v._anim_info = self._anim_info
            v.joint_names = self.joint_names
            v.parent_indices = self.parent_indices
            v.root_joint = self.root_joint
            v.num_joints = self.num_joints
            v.num_frames = self.num_frames
        for k,v in self.pre_transforms._modules.items():
            v._anim_ranges = self._anim_ranges
            v._anim_info = self._anim_info
            v.joint_names = self.joint_names
            v.parent_indices = self.parent_indices
            v.root_joint = self.root_joint
            v.num_joints = self.num_joints
            v.num_frames = self.num_frames


    def compute_stats(self):
        """Computes statistics info."""

        print(self.anim_mean.shape)
        print(self.anim_std.shape)


    def __len__(self):
        if self._frames is None:
            return 0
        else:
            return len(self._frames)

    def __getitem__(self, index):
        """Get data."""

        if self._frames is None:
            raise Exception('Animations not loaded!')

        data = self._frames[index:index+1]

        if self.transforms is not None:
            data = self.transforms(data)

        return data


if __name__ == '__main__':

    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('data.log'),
            logging.StreamHandler()
        ]
    )

    pre_transforms = torch.nn.Sequential(
        ReplaceJoint('Mid_hip', ['R_hip', 'L_hip']),
        ReplaceJoint('Neck', ['R_shoulder', 'L_shoulder']),
    )

    transformations = torch.nn.Sequential(
        IK(),
        LimbScale(std=0.05),
        FK(),
        To2D(),
    )

    ds = AnimDataset(transforms=transformations, pre_transforms=pre_transforms)

    ds.save()
    
    #ds.precompute_transforms()

    dl = DataLoader(ds, batch_size=256)
    
    for i, pose in enumerate(dl):
        pass
        #visualize_mpl(pose)
