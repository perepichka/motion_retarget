"""Motion transformations and augmentations."""

import sys, os, glob

from tqdm import tqdm
import logging

import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEFAULT_REF_JOINTS = ['R_shoulder', 'L_shoulder', 'R_hip', 'L_hip']

AXES = ['x', 'y', 'z']
PARENTS = [-1, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13]

DEFAULT_ANGLES = torch.tensor(
    [i * np.pi / 6 for i in range(-3, 4)], dtype=torch.float32
)
DEFAULT_AXES = torch.tensor([0, 0, 1])


class _AnimTransform(nn.Module):

    def __init__(self, dataset=None, *args, **kwargs):
        """Generic base animation transform."""
        super().__init__()
        self.ds = dataset

        for k, v in kwargs.items():
            setattr(self, k, v)


class LocalReferenceFrame(_AnimTransform):

    def __init__(self, *args, **kwargs):
        """Local reference frame transform."""
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass


class To2D(nn.Module):

    def __init__(self, keep_dim=False, *args, **kwargs):
        """Projects 3D animation to 2D.

        :param keem_dim: Keep the third dimmension, with values being 0.
        
        """
        super().__init__(*args, **kwargs)
        self.keep_dim = keep_dim

    def forward(self, x):
        if not self.keep_dim:
            return x[..., [1, 2]]
        else:
            x[..., 0] = 0
            return x

class ToFeatureVector(nn.Module):

    def __init__(self, *args, **kwargs):
        """To feature vector.
        
        """
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = x.reshape(x.shape[:-2] + (x.shape[-2] * x.shape[-1],))
        return x



class RotateBasis(_AnimTransform):

    def __init__(self, angles=DEFAULT_ANGLES, axes=DEFAULT_AXES, *args, **kwargs):
        """Rotates basis by angles specified.

        :param angles: Angles of rotation, shaped [3,3]
        :param angles: Axes of rotation, shaped [3]

        """
        super().__init__(*args, **kwargs)

        x_angles = angles if axes[0] else torch.tensor([0], dtype=torch.float32)
        z_angles = angles if axes[1] else torch.tensor([0], dtype=torch.float32)
        y_angles = angles if axes[2] else torch.tensor([0], dtype=torch.float32)
        x_angles, z_angles, y_angles = torch.meshgrid(x_angles, z_angles, y_angles)
        angles = torch.stack([x_angles.flatten(), z_angles.flatten(), y_angles.flatten()], dim=1)
        rand_int = torch.randint(0, angles.shape[0], (1,))[0]
        self.view_angles = angles[rand_int]

    def forward(self, data):
        cx, cy, cz = torch.cos(self.view_angles)
        sx, sy, sz = torch.sin(self.view_angles)

        x = data[0]
        x_cpm = torch.tensor([
            [0, -x[2], x[1]],
            [x[2], 0, -x[0]],
            [-x[1], x[0], 0]
        ], dtype=torch.float32)
        x = x.reshape(-1, 1)
        mat33_x = cx * torch.eye(3) + sx * x_cpm + (1.0 - cx) * torch.matmul(x, x.T)

        mat33_z = torch.tensor([
            [cz, sz, 0],
            [-sz, cz, 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        out = data @ mat33_x.T @ mat33_z

        return out


class ToBasis(_AnimTransform):

    def __init__(self, ref_joints=DEFAULT_REF_JOINTS, *args, **kwargs):
        """Convert motion to basis vectors.

        :param ref_joints: Reference joints.
        
        """
        super().__init__(*args, **kwargs)

        self.ref_joints = ref_joints

        assert len(self.ref_joints) == 4, "Unsupported amount of ref joints!"

    def forward(self, x):
        ind = [self.ds.joint_names.index(n) for n in self.ref_joints]

        horiz = (x[..., ind[0], :] - x[..., ind[1], :] + x[..., ind[2], :] - x[..., ind[3], :]) / 2

        while len(horiz.shape) > 1:
            horiz = torch.mean(horiz, dim=0)

        horiz = horiz / torch.norm(horiz)

        z = torch.tensor([0, 0, 1], device=x.device, dtype=x.dtype)
        # z = z[None, ...].repeat_interleave(x.shape[0], dim=0)

        # print(z.shape)
        y = torch.cross(horiz, z)
        y = y / torch.norm(y)  # [..., None, :]
        x = torch.cross(y, z)

        out = torch.stack([x, y, z], dim=1).detach()

        return out


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
        if self.ds is not None:

            noise = torch.empty_like(x)
            for r in self.ds._anim_ranges:
                means = self.mean * torch.ones_like(x[r[0]:r[1]])
                stds = self.std * torch.ones_like(means)
                noise[r[0]:r[1]] = torch.normal(means, stds)

            noise = torch.normal(means, stds)

        elif self.per_batch:
            if self.noise is None:
                # Creates arrays of random scaling
                means = self.mean * torch.ones_like(x)
                stds = self.std * torch.ones_like(x)

                noise = torch.normal(means, stds)
                self.noise = noise
            else:
                noise = self.noise
        else:
            # Creates arrays of random scaling
            means = self.mean * torch.ones_like(x)
            stds = self.std * torch.ones_like(x)

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

        for index, parent_index in enumerate(self.ds.parent_indices):
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
            rep_joint_index = self.ds.joint_names.index(self.rep_joint)
            ref_joint_indices = torch.zeros(self.ds.num_joints).type(torch.bool)

            for ref_joint in self.ref_joints:
                ref_joint_indices[self.ds.joint_names.index(ref_joint)] = True
        except Exception as e:
            logging.error('Invalid reference/replacement joint specified!')
            raise e

        x[..., rep_joint_index, :] = torch.mean(
            x[..., ref_joint_indices, :], dim=-2, keepdim=False
        )

        return x


class SlidingWindow(_AnimTransform):

    def __init__(self, window_size, stride, *args, **kwargs):
        """Transforms animation data to sliding windows of data.

        :param window_size: Size of window in frames.
        :param stride: Size of stride in frames.
        
        """
        super().__init__(*args, **kwargs)
        self.window_size = window_size
        self.stride = stride


    def forward(self, x):
        """Transformation function.

        :param x: Input animation data.

        """

        assert self.ds != None, "Need dataset for this transformation!"

        assert x.shape[0] == self.ds._frames.shape[0], "This transformation can only be done as a pre-transform"

        num_windows = 0
        for r in self.ds._anim_ranges:

            seq_length = (r[1] - r[0])
            out_size = math.floor((seq_length - self.window_size)/self.stride)+1
            num_windows += out_size

        x_new = torch.empty([num_windows, self.window_size, self.ds.num_joints, 3])

        index = 0
        for r in self.ds._anim_ranges:
            length = r[1] - r[0] 
            if length < self.window_size:
                continue
            for start in range(0, length-self.window_size, self.stride):
                seq = x[r[0]+start:r[0]+start+self.window_size]
                x_new[index] = seq
                index+=1
        return x_new


class FlipAxis(_AnimTransform):

    def __init__(self, axis, *args, **kwargs):
        """Flips an axis of the animation.

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


class RotateAnim(_AnimTransform):

    def __init__(self, basis, *args, **kwargs):
        """Rotates an animation around axes.

        :param basis: Change of basis parameter [3,3].

        """
        self.basis = basis
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = self.basis @ x
        return x
