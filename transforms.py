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

DEFAULT_REF_JOINTS = ['R_shoulder','L_shoulder','R_hip','L_hip']

AXES = ['x', 'y', 'z']
PARENTS = [-1, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13]


class _AnimTransform(nn.Module):

    def __init__(self, *args, **kwargs):
        """Base animation transform."""
        super().__init__()
        self._anim_ranges = None
        self._anim_info = None
        self.joint_names = None
        self.parent_indices = None
        self.root_joint = None
        self.num_joints = None
        self.num_frames = None

        for k,v in kwargs.items():
            setattr(self, key, value)


class LocalReferenceFrame(_AnimTransform):

    def __init__(self, *args, **kwargs):
        """Local reference frame transform."""
        super().__init__(*args, **kwargs)

    def forward(self, x):

        if type(x) == tuple:
            x = x[0]
        pass


class To2D(nn.Module):

    def __init__(self, keep_dim=False, *args, **kwargs):
        """Projects 3D animation to 2D.

        :param keem_dim: Keep the third dimmension, with values being 0.
        
        """
        super().__init__(*args, **kwargs)
        self.keep_dim = keep_dim

    def forward(self, x):

        if self.keep_dim:
            return x[..., [1,2]]
        else:
            x[..., 0] = 0
            return x

class ToBasis(nn.Module):

    def __init__(self, ref_joints=DEFAULT_REF_JOINTS, *args, **kwargs):
        """Convert motion to basis vectors.

        :param ref_joints: Reference joints.
        
        """
        super().__init__(*args, **kwargs)

        self.ref_joints = ref_joints

    def forward(self, x):

        ref_joint_indices = [self.joint_names.index(n) for n in self.ref_joints]

        if self.keep_dim:
            return x[..., [1,2]]
        else:
            x[..., 0] = 0
            return x


class ZeroRoot(_AnimTransform):

    def __init__(self, *args, **kwargs):
        """Zeros out root."""
        super().__init__(*args, **kwargs)

    def forward(self, x):

        if type(x) == tuple:
            x = x[0]
        return x - x[0:1, :]


class LimbScale(_AnimTransform):

    def __init__(self, mean=0, std=1, per_batch=False, *args, **kwargs):
        """Scales limbs. Note that data should be in local reference frame.

        :param mean: Mean of normal distribution for scaling.
        :param std: Std of normal distribution for scaling.
        :param per_batch: Whether to sample a single scaling factor per-batch.
        
        """

        super().__init__(*args, **kwargs)

        # Scaling parameters
        self.mean = mean
        self.std = std

        # Optional par
        self.per_batch = per_batch

        if self.per_batch:
            self.noise = None


    def forward(self, x, *args, **kwargs):
        """Transformation function.

        :param x: Input animation data.

        """
        if self._anim_ranges is not None:

            noise = torch.empty_like(x)
            for r in self._anim_ranges:
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


class IK(_AnimTransform):

    def __init__(self, *args, **kwargs):
        """Inverse kinematics transform."""
        super().__init__(*args, **kwargs)


    def forward(self, x):
        """Transformation function.

        :param x: Input animation data.

        """

        if type(x) == tuple:
            x = x[0]

        local_x = torch.empty(x.shape)
        local_x[..., 0, :] = x[..., 0, :]
        
        for index, parent_index in enumerate(self.parent_indices):
            if parent_index != -1:
                local_x[..., index, :] = x[..., index, :] - x[..., parent_index, :] 
        return local_x

class FK(_AnimTransform):

    def __init__(self, *args, **kwargs):
        """Forward kinematics transform."""
        super().__init__(*args, **kwargs)
        
    def forward(self, x):
        """Transformation function.

        :param x: Input animation data.

        """


        if type(x) == tuple:
            x = x[0]

        global_x = torch.empty(x.shape)
        global_x[..., 0, :] = x[..., 0, :]
        
        for index, parent_index in enumerate(self.parent_indices):
            if parent_index != -1:
                global_x[..., index, :] = x[..., index, :] + global_x[..., parent_index, :] 
        return global_x



class ReplaceJoint(_AnimTransform):

    def __init__(self, rep_joint, ref_joints, *args, **kwargs):
        """Replace a joint by the average of multiple other joints.
        :param rep_joint: Name of joint to replace.
        :param ref_joints: List of names of reference joints.
        
        """
        super().__init__(*args, **kwargs)

        self.rep_joint = rep_joint
        self.ref_joints = ref_joints
        
    def forward(self, x):
        """Transformation function.

        :param x: Input animation data.

        """

        try:
            rep_joint_index = self.joints.index(rep_joint)
            ref_joint_indices = torch.zeros(self.num_joints).type(torch.bool)

            for ref_joint in ref_joints:
                ref_joint_indices[self.joints.index(ref_joint)] = True
        except Exception as e:
            logging.error('Invalid reference/replacement joint specified!')
            raise e

        x[..., rep_joint_index, :]= torch.mean(
            x[..., ref_joint_indices, :], dim=-2, keepdim=True
        )

        return x


class FlipAxis(_AnimTransform):

    def __init__(self, axis, *args, **kwargs):
        """Replace a joint by the average of multiple other joints.

        :param axis: Axis to flip.
        
        """
        super().__init__(*args, **kwargs)

        if type(axis) == int:
            self.axis = axis
        elif type(axis) == str:
            self.axis = AXIS[axis]

    def forward(self, x):
        """Transformation function.

        :param x: Input animation data.

        """
        x[..., self.axis] = -x[..., self.axis]
        return x


class ScaleAnim(_AnimTransform):

    def __init__(self, amount, *args, **kwargs):
        """Scale an animation.

        :param amount: Amount to scale by.
        
        """
        super().__init__(*args, **kwargs)
        self.amount = amount


    def forward(self, x):
        """Transformation function.

        :param x: Input animation data.

        """
        x = self.amount * x
        return x


class NormalizeAnim(_AnimTransform):

    def __init__(self, mean=0, std=1, *args, **kwargs):
        """Normalize an animation.

        :param amount: Amount to scale by.
        
        """
        super().__init__(*args, **kwargs)
        self.mean = mean
        self.std = std


    def forward(self, x):
        """Transformation function.

        :param x: Input animation data.

        """
        return (x - self.mean) / self.std


class DenormalizeAnim(_AnimTransform):

    def __init__(self, mean=0, std=1, *args, **kwargs):
        """Denormalize an animation.

        :param amount: Amount to scale by.
        
        """
        super().__init__(*args, **kwargs)
        self.mean = mean
        self.std = std


    def forward(self, x):
        """Transformation function.

        :param x: Input animation data.

        """
        return (x * self.std) + self.mean


