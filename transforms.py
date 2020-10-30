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

class LocalReferenceFrame(nn.Module):

    def __init__(self, *args, **kwargs):
        """Local reference frame transform."""
        super().__init__()

    def forward(self, x):
        pass

class ZeroRoot(nn.Module):

    def __init__(self, *args, **kwargs):
        """Zeros out root."""
        super().__init__()

    def forward(self, x):
        return x - x[0:1, :]


class LimbScale(nn.Module):

    def __init__(self, mean=0, std=1, per_batch=True, *args, **kwargs):
        """Scales limbs. Note that data should be in local reference frame.

        :param mean: Mean of normal distribution for scaling.
        :param std: Std of normal distribution for scaling.
        :param per_batch: Whether to sample a single scaling factor per-batch.
        
        """
        super().__init__()
        self.mean = mean
        self.std = std

        self.per_batch = per_batch
        self.noise = None

    def forward(self, x):
        
        if not self.per_batch:
            # Creates arrays of random scaling
            means = self.mean*torch.ones_like(x)
            stds = self.std*torch.ones_like(x)

        else:
            if self.noise is None:
                # Creates arrays of random scaling
                means = self.mean*torch.ones_like(x)
                stds = self.std*torch.ones_like(x)

                noise = torch.normal(means, stds)
                self.noise = noise
            else:
                noise = self.noise

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

        global_x = torch.empty(x.shape)
        global_x[..., 0, :] = x[..., 0, :]
        
        for index, parent_index in enumerate(self.parents):
            if parent_index != -1:
                global_x[..., index, :] = x[..., index, :] + global_x[..., parent_index, :] 
        return global_x

