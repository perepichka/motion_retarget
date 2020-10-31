"""Motion transformations and augmentations."""

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

from visualization import pose2im_all

PARENTS = [-1, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13]

## Helper functions
#
#def fk(local_pos, local_rot, parents):
#    """Simple forward kinematics function."""
#
#    pass
#
#def ik(data, parents):
#    """Simple inverse kinematics function."""
#
#    for index, parent_index in enumerate(parents):
#        joints[...,parent_index, :]-=
#
#    pass

# Pytorch 

def build_transforms(transforms):
    """Helper function for building transformations.

    """

    for transform_info in transforms:
        transform = get(__globals__(), transform_name)

class LocalReferenceFrame(nn.Module):

    def __init__(self, *args, **kwargs):
        """Local reference frame transform."""
        super().__init__()

    def forward(self, x):

        if type(x) == tuple:
            x = x[0]
        pass

class To2D(nn.Module):

    def __init__(self, *args, **kwargs):
        """Zeros out root."""
        super().__init__()

    def forward(self, x):

        x[..., 0] = 0
        return x


class ZeroRoot(nn.Module):

    def __init__(self, *args, **kwargs):
        """Zeros out root."""
        super().__init__()

    def forward(self, x):

        if type(x) == tuple:
            x = x[0]
        return x - x[0:1, :]


class LimbScale(nn.Module):

    def __init__(self, mean=0, std=1, per_batch=False, *args, **kwargs):
        """Scales limbs. Note that data should be in local reference frame.

        :param mean: Mean of normal distribution for scaling.
        :param std: Std of normal distribution for scaling.
        :param per_batch: Whether to sample a single scaling factor per-batch.
        
        """


        super().__init__()

        # Scaling parameters
        self.mean = mean
        self.std = std

        # Optional par
        self.per_batch = per_batch

        if self.per_batch:
            self.noise = None

        self.ranges = None


    def forward(self, x, *args, **kwargs):
        """Transformation function.

        :param x: Input animation data.

        """

        if self.ranges is not None:

            noise = torch.empty_like(x)
            for r in self.ranges:
                means = self.mean * torch.ones_like(x[r[0]:r[1]])
                stds = self.std * torch.ones_like(means)
                noise[r[0]:r[1]] = torch.normal(means, stds)

            noise = torch.normal(means, stds)

        elif self.per_batch:
            if self.noise is None:
                # Creates arrays of random scaling
                means = self.mean*torch.ones_like(x)
                stds = self.std*torch.ones_like(x)

                noise = torch.normal(means, stds)
                self.noise = noise
            else:
                noise = self.noise
        else:
            # Creates arrays of random scaling
            means = self.mean*torch.ones_like(x)
            stds = self.std*torch.ones_like(x)

            noise = torch.normal(means, stds)

        # Dont scale root
        noise[..., 0, :] = 0
                
        return x + noise

class IK(nn.Module):

    def __init__(self, parents=None, *args, **kwargs):
        """Inverse kinematics transform."""

        super().__init__()

        if parents is None:
            self.parents = PARENTS
        else:
            self.parents = parents

    def forward(self, x):

        if type(x) == tuple:
            x = x[0]

        local_x = torch.empty(x.shape)
        local_x[..., 0, :] = x[..., 0, :]
        
        for index, parent_index in enumerate(self.parents):
            if parent_index != -1:
                local_x[..., index, :] = x[..., index, :] - x[..., parent_index, :] 
        return local_x

class FK(nn.Module):

    def __init__(self, parents=None, *args, **kwargs):
        """Forward kinematics transform."""

        super().__init__()

        if parents is None:
            self.parents = PARENTS
        else:
            self.parents = parents

    def forward(self, x):

        if type(x) == tuple:
            x = x[0]

        global_x = torch.empty(x.shape)
        global_x[..., 0, :] = x[..., 0, :]
        
        for index, parent_index in enumerate(self.parents):
            if parent_index != -1:
                global_x[..., index, :] = x[..., index, :] + global_x[..., parent_index, :] 
        return global_x

