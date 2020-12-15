"""Module for processing animation data."""

import sys, os, glob
thismodule = sys.modules[__name__]

from tqdm import tqdm
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from visualize import visualize_mpl
from utils import get_config

from transforms import *

from PIL import Image


DEFAULT_MIXAMO_TRAIN = './data/mixamo/36_800_24/train'
DEFAULT_MIXAMO_TEST = './data/mixamo/36_800_24/test'
DEFAULT_SOLODANCE_TRAIN = './data/solo_dance/train'
DEFAULT_STATS_PATH = './data/mixamo/36_800_24/'

DEFAULT_DS_FILE = 'ds.pt'
DEFAULT_JOINTS_FILE = 'joints.txt'
DEFAULT_PARENTS_FILE = 'parents.txt'

DEFAULT_WINDOW_SIZE = 64
DEFAULT_INTERVAL = 32

#DEFAULT_PRETRANSFORMS = torch.nn.Sequential(
#    ReplaceJoint('Mid_hip', ['R_hip', 'L_hip']),
#    ReplaceJoint('Neck', ['R_shoulder', 'L_shoulder']),
#    FlipAxis('x'),
#    FlipAxis('y'),
#    ScaleAnim(128),
#    SlidingWindow(config.seq_len, config.stride)
#)

DEFAULT_TRANSFORMS = torch.nn.Sequential(
    RandomScaleAnim(),
    RandomRotateAnim()
)

#DEFAULT_PRETRANSFORMS_LIMBSCALE = torch.nn.Sequential(
#    SlidingWindow(config.seq_len, config.stride),
#)



DATASETBASE_ARGUMENTS = {
    'path': DEFAULT_SOLODANCE_TRAIN,
    'stats_path': DEFAULT_STATS_PATH,
    'joints_file': DEFAULT_JOINTS_FILE,
    'parents_file': DEFAULT_PARENTS_FILE,
    'dataset_file': DEFAULT_DS_FILE,
    'window_size': DEFAULT_WINDOW_SIZE,
    'window_interval': DEFAULT_INTERVAL,
    'visualize_loading': False,
    '_reload': False,
}

VIEWS = torch.tensor(
    [i * np.pi / 6 for i in range(-3, 4)], dtype=torch.float32
)

def get_dataloader_test(path, config):
    """Get dataloader for test.py"""

    if 'test' not in path:
        raise Exception('This is for the test set!')

    batch_size = 1

    dataset_cls_name = config.data.train_cls if 'train' in path else config.data.eval_cls

    dataset_cls = getattr(thismodule, dataset_cls_name)


    pre_transforms = torch.nn.Sequential(
        #SlidingWindow(config.seq_len, config.stride),
    )
    pre_anim = torch.nn.Sequential(
        InputData('basis', torch.eye(3), parallel_out=True),
        InputRandomData('view_angles', ((0,0,0), (0,0,2*np.pi)), parallel_out=True),
        RotateBasis(parallel_out=True),
        RotateAnimWithBasis(pass_along=False),
    )


    num_workers=1
    dataset = MixamoDatasset(path=path, config=config, pre_transforms=pre_transforms, pre_anim_transforms=pre_anim)
    dataset.batch_size = 1

    dataloader = DataLoader(dataset, batch_size=None, num_workers=8)

    return dataloader, dataset



def get_dataloader(path, config):
    """Gets a dataloader for a given path."""

    config.data.batch_size = config.batch_size
    config.data.seq_len = config.seq_len
    dataset_cls_name = config.data.train_cls if 'train' in path else config.data.eval_cls

    dataset_cls = getattr(thismodule, dataset_cls_name)

    if 'train' in path:
        pre_transforms = torch.nn.Sequential(
            SlidingWindow(config.seq_len, config.stride),
        )
        pre_anim = None
    else:
        pre_transforms = torch.nn.Sequential(
            SlidingWindow(config.seq_len, config.stride),
        )
        pre_anim = torch.nn.Sequential(
            InputData('basis', torch.eye(3), parallel_out=True),
            InputRandomData('view_angles', ((0,0,0), (0,0,2*np.pi)), parallel_out=True),
            RotateBasis(parallel_out=True),
            RotateAnimWithBasis(pass_along=False),
        )


    num_workers=8
    dataset = dataset_cls(path=path, config=config, pre_transforms=pre_transforms, pre_anim_transforms=pre_anim)

    if 'train' in path:
        dataset.batch_size = config.batch_size
        dl_batch_size = None
        dataloader = DataLoader(dataset, batch_size=dl_batch_size,
                                num_workers=(config.data.num_workers))
    else:
        dataset.batch_size = 1
        dl_batch_size = config.data.batch_size
        dataloader = DataLoader(dataset, shuffle=False,
                                batch_size=dl_batch_size,
                                num_workers=(config.data.num_workers),
                                drop_last=True)

    return dataloader, dataset


class AnimDataset(Dataset):

    def __init__(self, transforms=None, pre_transforms=None, pre_anim_transforms=None, batch_size=1, *args, **kwargs):
        """Generic animation dataset constructor.
        
        :param transforms: Transformations to transform/augment animation data. 
        :param pre_transforms: Transformations to be pre-computed on entire dataset.
        :param per_anim_transforms: Transformations to be pre-computed on individual animations. 
        :param path: Path to the folder containing the dataset.
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
        self.batch_size = batch_size
        self.num_joints = 0
        self.num_frames = 0
        self.num_anims = 0

        self.joint_names = []
        self.parent_indices = []
        
        # Frame storing the animation
        self._frames = None
        
        # Information on the animation
        self._anim_ranges = []
        self._anim_info = []
        self._anim_structure = {}
        self.characters = []
        self.anim_names = []

        # Set up transforms
        self.transforms = transforms
        self.pre_transforms = pre_transforms
        self.pre_anim_transforms = pre_anim_transforms

        # Stats
        self.__mean = None
        self.__std = None

        # Loads data if needed
        if self.path is not None:
            self.load()

        if self._reload:
            self.save()

        # Sends loaded info to transforms
        self.update_transforms()

        # Preprocesses data if neeeded
        if self.pre_transforms is not None:
            self.precompute_transforms()
        print('tst')
        if self.pre_anim_transforms is not None:
            self.compute_pre_anim_transforms()
        
    
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
        # @TODO finish this
        # All 


    def precompute_transforms(self):
        """Precomputes animation transforms."""

        if self._frames is None or len(self._frames) == 0:
            raise Exception('No data to precompute transforms on')
        print('pre-transform')
        self._frames = self.pre_transforms(self._frames)
        print('pre-transform done')

    def compute_pre_anim_transforms(self):
        """Computes pre-anim transforms."""

        print('happens')
        if self._frames is None or len(self._frames) == 0:
            raise Exception('No data to precompute transforms on')
        for rng in self._anim_ranges:
            self._frames[rng[0]:rng[1]] = self.pre_anim_transforms(
                self._frames[rng[0]:rng[1]]
            )


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
            '_anim_structure': self._anim_structure,
            'characters': self.characters,
            'anim_names': self.anim_names,
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


        num_dims = 0

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

                    # Put in-order [nframes, njoints, 3]
                    anim = anim.swapaxes(0, -1)
                    anim = anim.swapaxes(1, 2)

                    if self.num_joints == 0:
                        self.num_joints = anim.shape[-2]

                    if num_dims == 0:
                        num_dims = anim.shape[-1]

                    stored_anims.append(anim)


                    if len(takes) == 1:
                        anim_name_str = anim_name
                    else:
                        anim_name_str = '{}_{}'.format(anim_name, take)


                    # Set up animation information
                    self._anim_ranges.append(
                        (self.num_frames, self.num_frames+anim.shape[0])
                    )
                    self._anim_info.append(
                            {'character': character, 'anim_name': anim_name_str}
                    )

                    if character not in self._anim_structure:
                        self._anim_structure[character] = {}
                    self._anim_structure[character][anim_name_str] = (self.num_frames, self.num_frames+anim.shape[0])

                    
                    # Set up character information
                    if character not in self.characters:
                        self.characters.append(character)

                    # Set up anim name information
                    if anim_name_str not in self.anim_names:
                        self.anim_names.append(anim_name_str)

                    self.num_frames += anim.shape[0]
                    self.num_anims += 1

        # Makes sure we have loaded animations           
        if len(stored_anims) == 0:
            raise Exception('Failed to load any frames!')

        assert num_dims != 0, 'Something is wrong the the loaded data, num_dims is 0!!!'

        # Resizes everything
        self._frames = torch.empty([self.num_frames, self.num_joints, num_dims])
        
        curr_index = 0
        logging.info('Storing animation files in memory...')
        for anim in stored_anims:
            assert not np.all(anim == 0), "Invalid frame in animation"
            # Stores animation
            self._frames[curr_index: curr_index+anim.shape[0]] = torch.from_numpy(anim)
            curr_index += anim.shape[0]
            
        self._anim_ranges = np.array(self._anim_ranges)

        if self._frames is None or self._frames.shape[0] == 0:
            raise Exception('Failed to load any frames!')

    def _load_video(self):
        raise NotImplementedError


    def update_transforms(self):
        """Passes updated info to data transforms"""
        
        if self.transforms is not None:
            for k,v in self.transforms._modules.items():
                v.ds = self
        if self.pre_transforms is not None:
            for k,v in self.pre_transforms._modules.items():
                v.ds = self
        if self.pre_anim_transforms is not None:
            for k,v in self.pre_anim_transforms._modules.items():
                v.ds = self

    def compute_stats(self):
        """Computes statistics on database."""

        logging.info('Computing statistics...')

        pass

    @property
    def mean(self):
        """Returns mean animation pose."""

        meanpose_path = os.path.join(self.stats_path, 'meanpose.npy')

        if self.__mean is not None:
            return self.__mean
        elif os.path.exists(meanpose_path):
            # @FIXME @TODO redo stats computation, this projection from 2d to 3d
            mean2d = torch.from_numpy(np.load(meanpose_path))
            mean3d = torch.stack([mean2d[:,0], mean2d[:,0], mean2d[:,1]], dim=1).type(torch.float32)
            self.__mean = mean3d
            return mean3d
        else:
            raise Exception('Invalid mean path : {}'.format(meanpose_path))

    @property
    def std(self):
        """Returns std animation pose."""

        stdpose_path = os.path.join(self.stats_path, 'stdpose.npy')

        if self.__std is not None:
            return self.__std
        elif os.path.exists(stdpose_path):
            # @FIXME @TODO redo stats computation, this projection from 2d to 3d
            std2d = torch.from_numpy(np.load(stdpose_path))
            std3d = torch.stack([std2d[:,0], std2d[:,0], std2d[:,1]], dim=1).type(torch.float32)
            self.__std = std3d
            return std3d
        else:
            raise Exception('Invalid std path : {}'.format(stdpose_path))

    def __len__(self):
        if self._frames is None:
            return 0
        else:
            return len(self._frames) // self.batch_size

    def __getitem__(self, index):
        """Get data."""

        if self._frames is None:
            raise Exception('Animations not loaded!')

        data = self._frames[index:index+1].clone()

        if self.transforms is not None:
            data = self.transforms(data)

        data = data.squeeze()
        
        return data



class MixamoDataset(AnimDataset):

    def __init__(self, config=None, *args, **kwargs):
        """Mixamo animation dataset constructor.
        
        """
        super(MixamoDataset, self).__init__(*args, **kwargs)

        # Batch size should be handled by dataloader for validation dataset
        self.batch_size = 1
        
        if config is None:
            rot_axes = np.array([0,0,0])
        else:
            rot_axes = config.rotation_axes

        x_angles = VIEWS if rot_axes[0] else np.array([0])
        z_angles = VIEWS if rot_axes[1] else np.array([0])
        y_angles = VIEWS if rot_axes[2] else np.array([0])
        x_angles, z_angles, y_angles = np.meshgrid(x_angles, z_angles, y_angles)
        angles = torch.from_numpy(np.stack([x_angles.flatten(), z_angles.flatten(), y_angles.flatten()], axis=1))
        self.views = angles

        # Creates our pipelines
        self._generate_preprocessing_pipeline()
        self._generate_2d_pipeline()
        self._generate_finalsteps_pipeline()
        self._generate_data_augmentations()
        self.update_transforms()

    def __getitem__(self, index):
        """Get data."""

        # Pick two motions
        #mi = [np.random.choice(len(self.anim_names), size=2, replace=False) for _ in range(self.batch_size)]
        #m_2 = self.anim_names[mi_1], self.anim_names[mi_2]
        mi_1, mi_2 = np.random.choice(len(self.anim_names), size=2, replace=False)
        m_1, m_2 = self.anim_names[mi_1], self.anim_names[mi_2]

        # Pick two characters
        ci_1, ci_2 = np.random.choice(len(self.characters), size=2, replace=False)
        c_1, c_2 = self.characters[ci_1], self.characters[ci_2]

        # Pick two views
        vi_1, vi_2 = np.random.choice(len(self.views), size=2, replace=False)
        v_1, v_2 = self.views[vi_1], self.views[vi_2]

        # Load indices for each character-motion combo
        ind_11 = np.random.randint(self._anim_structure[c_1][m_1][0], self._anim_structure[c_1][m_1][1])
        ind_12 = np.random.randint(self._anim_structure[c_1][m_2][0], self._anim_structure[c_1][m_2][1])
        ind_21 = np.random.randint(self._anim_structure[c_2][m_1][0], self._anim_structure[c_2][m_1][1])
        ind_22 = np.random.randint(self._anim_structure[c_2][m_2][0], self._anim_structure[c_2][m_2][1])

        data_11 = self._frames[ind_11:ind_11+1, ...]
        data_12 = self._frames[ind_12:ind_12+1, ...]
        data_21 = self._frames[ind_21:ind_21+1, ...]
        data_22 = self._frames[ind_22:ind_22+1, ...]

        # optional data augmentation
        if self.transforms is not None:
            data_111 = self.transforms(data_11)
            data_121 = self.transforms(data_12)
            data_112 = self.transforms(data_11)
            data_122 = self.transforms(data_12)
            data_211 = self.transforms(data_21)
            data_221 = self.transforms(data_22)
            data_212 = self.transforms(data_21)
            data_222 = self.transforms(data_22)
        else:
            data_111 = data_11.clone()
            data_121 = data_12.clone()
            data_112 = data_11.clone()
            data_122 = data_12.clone()
            data_211 = data_21.clone()
            data_221 = data_22.clone()
            data_212 = data_21.clone()
            data_222 = data_22.clone()

        # Preprocessing
        data_111 = self.preprocess({'x':data_111, 'view_angles':v_1}).squeeze()
        data_121 = self.preprocess({'x':data_121, 'view_angles':v_1}).squeeze()
        data_112 = self.preprocess({'x':data_112, 'view_angles':v_2}).squeeze()
        data_122 = self.preprocess({'x':data_122, 'view_angles':v_2}).squeeze()
        data_211 = self.preprocess({'x':data_211, 'view_angles':v_1}).squeeze()
        data_221 = self.preprocess({'x':data_221, 'view_angles':v_1}).squeeze()
        data_212 = self.preprocess({'x':data_212, 'view_angles':v_2}).squeeze()
        data_222 = self.preprocess({'x':data_222, 'view_angles':v_2}).squeeze()

        # Preprocessing
        data_111 = self.preprocess({'x':data_111, 'view_angles':v_1}).squeeze()
        data_121 = self.preprocess({'x':data_121, 'view_angles':v_1}).squeeze()
        data_112 = self.preprocess({'x':data_112, 'view_angles':v_2}).squeeze()
        data_122 = self.preprocess({'x':data_122, 'view_angles':v_2}).squeeze()
        data_211 = self.preprocess({'x':data_211, 'view_angles':v_1}).squeeze()
        data_221 = self.preprocess({'x':data_221, 'view_angles':v_1}).squeeze()
        data_212 = self.preprocess({'x':data_212, 'view_angles':v_2}).squeeze()
        data_222 = self.preprocess({'x':data_222, 'view_angles':v_2}).squeeze()

        # Creates 2D versions
        data_111_2d = self.to_2d(data_111).squeeze()
        data_121_2d = self.to_2d(data_121).squeeze()
        data_112_2d = self.to_2d(data_112).squeeze()
        data_122_2d = self.to_2d(data_122).squeeze()
        data_211_2d = self.to_2d(data_211).squeeze()
        data_221_2d = self.to_2d(data_221).squeeze()
        data_212_2d = self.to_2d(data_212).squeeze()
        data_222_2d = self.to_2d(data_222).squeeze()

        # Final steps
        data_111 = self.final_steps(data_111).squeeze()
        data_121 = self.final_steps(data_121).squeeze()
        data_112 = self.final_steps(data_112).squeeze()
        data_122 = self.final_steps(data_122).squeeze()
        data_211 = self.final_steps(data_211).squeeze()
        data_221 = self.final_steps(data_221).squeeze()
        data_212 = self.final_steps(data_212).squeeze()
        data_222 = self.final_steps(data_222).squeeze()

        return {"X_a": data_111, "X_b": data_222,
                "X_aab": data_112, "X_bba": data_221,
                "X_aba": data_121, "X_bab": data_212,
                "X_abb": data_122, "X_baa": data_211,
                "x_a": data_111_2d, "x_b": data_222_2d,
                "x_aab": data_112_2d, "x_bba": data_221_2d,
                "x_aba": data_121_2d, "x_bab": data_212_2d,
                "x_abb": data_122_2d, "x_baa": data_211_2d,
        }
        
        return data


    def test_process(char, mot):
        """Same as get_item, but only for a given char/motion. Used at test-time."""

        rng = ds._anim_structure[char][mot]
        data_11 = self._frames[rng[0]:rng[1], ...]

    def test_unprocess(char, mot):
        pass


    def _generate_preprocessing_pipeline(self):
        self.preprocess = torch.nn.Sequential(
            ToBasis(parallel_out=True, dataset=self),
            RotateBasis(parallel_out=True, dataset=self),
            RotateAnimWithBasis(pass_along=False, dataset=self),
            ReplaceJoint('Mid_hip', ['R_hip', 'L_hip'], dataset=self),
            ReplaceJoint('Neck', ['R_shoulder', 'L_shoulder'], dataset=self),
            FlipAxis('x'),
            FlipAxis('z'),
            #ScaleAnim(128),
            LocalReferenceFrame(dataset=self),
            NormalizeAnim(mean=self.mean, std=self.std, pass_along=False),
        )

    def _generate_2d_pipeline(self):
        self.to_2d = torch.nn.Sequential(
            To2D(clone=True),
            ToFeatureVector(),
            Permute([0,2,1]),
            Detach()
        )

    def _generate_finalsteps_pipeline(self):
        self.final_steps = torch.nn.Sequential(
            ToFeatureVector(),
            Permute([0,2,1]),
            Detach()
        )

    def _generate_data_augmentations(self):
        self.transforms = torch.nn.Sequential(
            RandomRotateAnim(clone=True),
            RandomScaleAnim()
        )




class MixamoLimbScaleDataset(AnimDataset):

    def __init__(self, config, *args, **kwargs):
        """Mixamo animation dataset constructor.

        This dataset supports the batch_size parameters. Pass the batch_size to
        the dataset constructer and don't pass it to the dataloader. 
        
        """
        super(MixamoLimbScaleDataset, self).__init__(
            *args, **kwargs
        )

        x_angles = VIEWS if config.rotation_axes[0] else np.array([0])
        z_angles = VIEWS if config.rotation_axes[1] else np.array([0])
        y_angles = VIEWS if config.rotation_axes[2] else np.array([0])
        x_angles, z_angles, y_angles = np.meshgrid(x_angles, z_angles, y_angles)
        angles = np.stack([x_angles.flatten(), z_angles.flatten(), y_angles.flatten()], axis=1)
        self.views = angles

        # Creates our pipelines
        self._generate_preprocessing_pipeline()
        self._generate_limbscale_pipeline()
        self._generate_finalsteps_pipeline()
        self._generate_data_augmentations()
        self.update_transforms()

    def __getitem__(self, index):
        """Get data."""

        # Pick random character
        #mi = np.random.choice(len(self.anim_names), size=self.batch_size)
        #m = self.anim_names[mi]

        ## Pick random animation
        #ci = np.random.choice(len(self.characters), size=self.batch_size)
        #c = self.characters[ci]

        # Pick random anim
        ai = np.random.choice(self._anim_ranges.shape[0], size=self.batch_size)
        anim_rngs = np.take(self._anim_ranges, ai, axis=0)
        anim_indices = np.random.randint(anim_rngs[:,0], anim_rngs[:,1])

        # Pick view
        vi = np.random.choice(len(self.views), size=self.batch_size)
        v = np.take(self.views, vi)

        # Load indices for character motion combo
        #indices = self._anim_structure[c][m]
        #indices = self.

        # Pick window
        #if (indices[1] - indices[0] > 0):
        #    ri = np.random.randint(indices[0],indices[1])
        #else:
        #    ri = indices[0]

        #data = self._frames[ri:ri+1, ...].clone()
        data = np.take(self._frames, anim_indices, axis=0).clone()

        # optional data augmentation
        if self.transforms is not None:
            data = self.transforms(data).squeeze()

        #preprocess = torch.nn.Sequential(
        #    ToBasis(parallel_out=True, dataset=self),
        #    RotateAnimWithBasis(pass_along=False, dataset=self),
        #    ReplaceJoint('Mid_hip', ['R_hip', 'L_hip'], dataset=self),
        #    ReplaceJoint('Neck', ['R_shoulder', 'L_shoulder'], dataset=self),
        #    FlipAxis('x'),
        #    FlipAxis('y'),
        #    ScaleAnim(128),
        #    To2D(pass_along=False),
        #)
        anim = self.preprocess(data)

        #limb_scale = torch.nn.Sequential(
        #    IK(clone=True, dataset=self),
        #    LimbScale(),
        #    FK(dataset=self),
        #)


        anim_scaled = self.limb_scale(anim)

        # generate two views
        #final_steps = torch.nn.Sequential(
        #    LocalReferenceFrame(dataset=self),
        #    NormalizeAnim(mean=self.mean, std=self.std),
        #    ToFeatureVector()
        #)

        anim = self.final_steps(anim)
        anim_scaled = self.final_steps(anim_scaled)

        return {
            'x':anim,
            'x_s':anim_scaled
        }

    def _generate_preprocessing_pipeline(self):
        self.preprocess = torch.nn.Sequential(
            ToBasis(parallel_out=True, dataset=self),
            RotateAnimWithBasis(pass_along=False, dataset=self),
            ReplaceJoint('Mid_hip', ['R_hip', 'L_hip'], dataset=self),
            ReplaceJoint('Neck', ['R_shoulder', 'L_shoulder'], dataset=self),
            FlipAxis('x'),
            FlipAxis('z'),
            #ScaleAnim(128),
            To2D(pass_along=False),
        )

    def _generate_limbscale_pipeline(self):
        self.limb_scale = torch.nn.Sequential(
            RandomScaleAnim(dataset=self, rng=(0.5, 2.0), clone=True),
            IK(dataset=self, clone=True),
            LimbScale(dataset=self),
            FK(dataset=self),
        )

    def _generate_finalsteps_pipeline(self):
        self.final_steps = torch.nn.Sequential(
            LocalReferenceFrame(dataset=self),
            NormalizeAnim(mean=self.mean, std=self.std),
            ToFeatureVector(),
            Permute([0,2,1])
        )

    def _generate_data_augmentations(self):
        self.transforms = torch.nn.Sequential(
            RandomRotateAnim(),
            RandomScaleAnim()
        )



class SoloDanceDataset(AnimDataset):

    def __init__(self, config, *args, **kwargs):
        """Solo dance animation dataset constructor.

        This dataset supports the batch_size parameters. Pass the batch_size to
        the dataset constructer and don't pass it to the dataloader. 
        
        """
        super(SoloDanceDataset, self).__init__(
            *args, **kwargs
        )

        self.__mean = None
        self.__std = None

        # Creates our pipelines
        self._generate_preprocessing_pipeline()
        self._generate_limbscale_pipeline()
        self._generate_finalsteps_pipeline()
        self._generate_data_augmentations()
        self.update_transforms()

    def __getitem__(self, index):
        """Get data."""

        # Pick random anim
        ai = np.random.choice(self._anim_ranges.shape[0], size=self.batch_size)
        anim_rngs = np.take(self._anim_ranges, ai, axis=0)
        anim_indices = np.random.randint(anim_rngs[:,0], anim_rngs[:,1])

        data = np.take(self._frames, anim_indices, axis=0).clone()

        # optional data augmentation
        if self.transforms is not None:
            data = self.transforms(data).squeeze()

        anim = self.preprocess(data)
        anim_scaled = self.limb_scale(anim)

        anim = self.final_steps(anim)
        anim_scaled = self.final_steps(anim_scaled)

        return {
            'x':anim,
            'x_s':anim_scaled,
            'meanpose': self.mean,
            'stdpose': self.std,
        }

    def _generate_preprocessing_pipeline(self):
        self.preprocess = torch.nn.Sequential(
            ReplaceJoint('Mid_hip', ['R_hip', 'L_hip'], dataset=self),
            ReplaceJoint('Neck', ['R_shoulder', 'L_shoulder'], dataset=self),
        )

    def _generate_limbscale_pipeline(self):
        self.limb_scale = torch.nn.Sequential(
            RandomScaleAnim(dataset=self, rng=(0.5, 2.0), clone=True),
            IK(dataset=self, clone=True),
            LimbScale(dataset=self),
            FK(dataset=self),
        )

    def _generate_finalsteps_pipeline(self):
        self.final_steps = torch.nn.Sequential(
            LocalReferenceFrame(dataset=self),
            NormalizeAnim(mean=self.mean, std=self.std),
            ToFeatureVector(),
            Permute([0,2,1])
        )

    def _generate_data_augmentations(self):
        self.transforms = torch.nn.Sequential(
        )



if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('data.log'),
            logging.StreamHandler()
        ]
    )

    #pre_transforms = torch.nn.Sequential(
    #    ReplaceJoint('Mid_hip', ['R_hip', 'L_hip']),
    #    ReplaceJoint('Neck', ['R_shoulder', 'L_shoulder']),
    #    SlidingWindow(64, 32)
    #)    
    pre_transforms = torch.nn.Sequential(
        #SlidingWindow(config.seq_len, config.stride),
        SlidingWindow(64, 32),
    )
    pre_anim = torch.nn.Sequential(
        #InputData('basis', torch.eye(3), parallel_out=True),
        #InputRandomData('view_angles', ((0,0,0), (0,0,2*np.pi)), parallel_out=True),
        #RotateBasis(parallel_out=True),
        #RotateAnimWithBasis(pass_along=False ),
    )
    
    #ds = AnimDataset(
    #    pre_transforms=pre_transforms,
    #    #pre_anim_transforms=pre_anim,
    #)
    #ds = MixamoLimbScaleDataset(
    #ds = MixamoDataset(
    #    config=get_config('configs/transmomo.yaml'),
    #    pre_transforms=pre_transforms,
    #    pre_anim_transforms=pre_anim,
    #    batch_size=64
    #)
    ds = SoloDanceDataset(
        config=get_config('configs/transmomo.yaml'),
        pre_transforms=pre_transforms,
        batch_size=64
    )


    #transformations = torch.nn.Sequential(
    #    #IK(),
    #    #LimbScale(std=0.05),
    #    #FK(),
    #    #To2D(),
    #    RandomRotateAnim(),
    #    RandomScaleAnim(),
    #    #ToBasis(parallel_out=True),
    #    #RotateAnimWithBasis(pass_along=False),
    #    #LocalReferenceFrame(pass_along=False),
    #    #NormalizeAnim(mean=ds.mean, std=ds.std)
    #)


    #ds.transforms = transformations
    #ds.update_transforms()


    #ds.save()


    #dl = DataLoader(ds, batch_size=64, drop_last=True, num_workers=8)
    #dl = DataLoader(ds, batch_size=64, drop_last=True)
    #dl = DataLoader(ds, batch_size=None, num_workers=8)
    dl = DataLoader(ds, batch_size=None)
    _ = ds.mean
    _ = ds.std

    print('starting loop')
    import time
    start = time.time()
    for i, pose in enumerate(dl):
        #if i == 0:
        #    start = time.time()
        #if i >= 10:
        #    #print((time.time() - start) / 3600.0)
        #    print('%f' % ((time.time() - start) / 3600.0))
        #    raise Exception

        #continue
        visualize_mpl(tst['x_s'][0], show_basis=False, ds=ds)
