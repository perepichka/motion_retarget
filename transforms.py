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

from copy import deepcopy


DEFAULT_REF_FRAME_JOINT = 'Mid_hip'

DEFAULT_REF_JOINTS = ['R_shoulder', 'L_shoulder', 'R_hip', 'L_hip']

AXES = ['x', 'y', 'z']
PARENTS = [-1, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13]

DEFAULT_ANGLES = torch.tensor(
    [i * np.pi / 6 for i in range(-3, 4)], dtype=torch.float32
)
DEFAULT_AXES = torch.tensor([0, 0, 1])


def generate_preprocessing_pipeline(view_angle, params=None):

    pipeline = []
    
    if params is not None:
        #@TODO add data augmentation here
        pass

    #@TODO Figure out basis here

    pipeline.append(
        ReplaceJoint('Mid_hip', ['R_hip', 'L_hip']),
        ReplaceJoint('Neck', ['R_shoulder', 'L_shoulder']),
        FlipAxis('x'),
        FlipAxis('y'),
        ScaleAnim(128),
    )

    #pipeline.append(
    # RotateAnim()
    #)


    return torch.nn.Sequential(pipeline)




class _AnimTransform(nn.Module):

    def __init__(self, parallel_out=False, pass_along=True, dataset=None, clone=False, *args, **kwargs):
        """Generic base animation transform.

        :param parallel_out: Output the basis alongside the original data.
        :param pass_along: Pass along all data this transform gets to subsequent transforms.
        :param clone: Clone the data before passing it to transform.
        :param dataset: (optional) Animation dataset.
        
        """
        super().__init__()
        self.ds = dataset
        self.parallel_out = parallel_out
        self.pass_along = pass_along
        self.clone = clone

        for k, v in kwargs.items():
            setattr(self, k, v)

    def forward(self, data):

        if type(data) == dict:
            x = data['x']
            kwargs = {k:v for k,v in data.items() if k != 'x'}
        else:
            x = data
            kwargs = {}

        if self.clone:
            x = x.clone()

        if len(x.shape) == 2:
            x = x[None, ...]


        out = self.transform(x, **kwargs)

        if type(out) != dict:
            out = {'x': out}
        
        if self.pass_along:
            for k,v in out.items():
                kwargs[k] = v

            if len(kwargs) == 1:
                return kwargs['x']
            else:
                return kwargs
        else:
            if len(out) == 1:
                return out['x']
            else:
                return out


    def transform(self, x, *args, **kwargs):
        raise NotImplementedError


class EmptyTransform(_AnimTransform):
    def __init__(self, *args, **kwargs):
        """Transform that does nothing, for test purposes."""
        super().__init__(*args, **kwargs)

    def transform(self, x, *args, **kwargs):
        return x


class InputData(_AnimTransform):
    def __init__(self, name, value, *args, **kwargs):
        """Transform that inputs data into a variable."""
        self.name = name
        self.value = value
        super().__init__(*args, **kwargs)

    def transform(self, x, *args, **kwargs):
        if self.parallel_out:
            out = {}
            out['x'] = x
            out[self.name] = self.value
            return out
        else:
            return value

class InputRandomData(_AnimTransform):
    def __init__(self, name, rng, *args, **kwargs):
        """Transform that inputs data into a variable."""
        self.name = name
        self.rng = rng
        super().__init__(*args, **kwargs)

    def transform(self, x, *args, **kwargs):
        rand_data = torch.from_numpy(np.random.uniform(self.rng[0], self.rng[1]))
        if self.parallel_out:
            out = {}
            out['x'] = x
            out[self.name] = rand_data
            return out
        else:
            return rand_data


class Detach(_AnimTransform):
    def __init__(self, *args, **kwargs):
        """Transform that detaches the data."""
        super().__init__(*args, **kwargs)

    def transform(self, x, *args, **kwargs):
        return x.detach()


class To2D(_AnimTransform):

    def __init__(self, keep_dim=False, *args, **kwargs):
        """Projects 3D animation to 2D.

        :param keem_dim: Keep the third dimmension, with values being 0.
        
        """
        super().__init__(*args, **kwargs)
        self.keep_dim = keep_dim

    def transform(self, x, *args, **kwargs):
        if not self.keep_dim:
            #return x[..., [1, 2]]
            return x[..., [0, 2]]
        else:
            x[..., 0] = 0
            return x

class Permute(_AnimTransform):

    def __init__(self, new_dims, *args, **kwargs):
        """Implements permute operation.

        :param new_dims: New dimmensions.
        
        """
        super().__init__(*args, **kwargs)
        self.new_dims = new_dims

    def transform(self, x, *args, **kwargs):
        return x.permute(self.new_dims)

class ToFeatureVector(_AnimTransform):

    def __init__(self, *args, **kwargs):
        """To feature vector.
        
        """
        super().__init__(*args, **kwargs)

    def transform(self, x, *args, **kwargs):
        x = x.reshape(x.shape[:-2] + (x.shape[-2] * x.shape[-1],))
        return x


class FromFeatureVector(_AnimTransform):

    def __init__(self, *args, **kwargs):
        """Back from feature vector.
        
        """
        super().__init__(*args, **kwargs)

    def transform(self, x, *args, **kwargs):
        if self.ds is None:
            raise Exception('Need dataset for this transform!')
        x = x.reshape(x.shape[:-1], + (self.ds.num_joints, x.shape[-1]/self.ds.num_joints))
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

    def transform(self, x, basis=None, view_angles=None, *args, **kwargs):

        if basis is None:
            basis = x

        if view_angles is None:
            view_angles = self.view_angles
            
        cx, cy, cz = torch.cos(view_angles)
        sx, sy, sz = torch.sin(view_angles)

        basis = basis.squeeze()
        basis_0 = basis.squeeze()
        basis_0 = basis[0]

        basis_cpm = torch.tensor([
            [0, -basis_0[2], basis_0[1]],
            [basis_0[2], 0, -basis_0[0]],
            [-basis_0[1], basis_0[0], 0]
        ], dtype=torch.float)

        #x_cpm = torch.cat([x_cpm_0, x_cpm_1, x_cpm_2, 
            
        basis_0 = basis_0.reshape(-1, 1)
        mat33_x = cx * torch.eye(3) + sx * basis_cpm + (1.0 - cx) * torch.matmul(basis_0, basis_0.T)

        mat33_z = torch.tensor([
            [cz, sz, 0],
            [-sz, cz, 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        out = basis @ mat33_x.T @ mat33_z

        if self.parallel_out:
            return {'x': x, 'basis': out}
        else:
            return out

class RandomRotateBasis(RotateBasis):

    def __init__(self, rng=((-np.pi/9, -np.pi/9, -np.pi/6), (np.pi/9, np.pi/9, np.pi/6)), *args, **kwargs):
        """Random animation rotation.

        :param rng: Range of uniform distribution to generate rotation factor.
        
        """
        amount = torch.from_numpy(np.random.uniform(rng[0], rng[1]))
        super().__init__(amount, *args, **kwargs)



class ToBasis(_AnimTransform):

    def __init__(self, ref_joints=DEFAULT_REF_JOINTS, *args, **kwargs):
        """Convert motion to basis vectors.

        :param ref_joints: Reference joints.
        
        """
        super().__init__(*args, **kwargs)
        
        self.ref_joints = ref_joints

        assert len(self.ref_joints) == 4, "Unsupported amount of ref joints!"

    def transform(self, x, *args, **kwargs):

        ind = [self.ds.joint_names.index(n) for n in self.ref_joints]

        horiz = (x[..., ind[0], :] - x[..., ind[1], :] + x[..., ind[2], :] - x[..., ind[3], :]) / 2
    

        while len(horiz.shape) > 1:
            horiz = torch.mean(horiz, dim=0)

        horiz = horiz / torch.norm(horiz)

        z = torch.tensor([0, 0, 1], device=x.device, dtype=x.dtype)

        y = torch.cross(horiz, z)
        y = y / torch.norm(y)  # [..., None, :]
        x2 = torch.cross(y, z)

        out = torch.stack([x2, y, z], dim=1).detach()

        if torch.any(torch.isnan(out)):
            print('BASIS IS NAN!')
            print('------')
            print(x.shape)
            print(x.mean())
            raise Exception

        if self.parallel_out:
            return {'x': x, 'basis': out}
        else:
            return out


class ZeroRoot(_AnimTransform):

    def __init__(self, *args, **kwargs):
        """Zeros out root."""
        super().__init__(*args, **kwargs)

    def transform(self, x, *args, **kwargs):
        if type(x) == tuple:
            x = x[0]
        return x - x[0:1, :]


class LimbScale(_AnimTransform):

    def __init__(self, rng=[0.5, 2.0], symmetric=False, mean=0, std=1, *args, **kwargs):
        """Scales limbs. Note that data should be in local reference frame.

        :param mean: Mean of normal distribution for scaling.
        :param std: Std of normal distribution for scaling.
        :param per_batch: Whether to sample a single scaling factor per-batch.
        
        """

        super().__init__(*args, **kwargs)

        self.rng = rng
        self.symmetric = symmetric

        # Scaling parameters
        self.mean = mean
        self.std = std

    def transform(self, x, *args, **kwargs):
        """Transformation function.

        :param x: Input animation data.

        """
 
        # Creates arrays of random scaling
        means = self.mean * torch.ones_like(x)
        stds = self.std * torch.ones_like(x)

        #scales = (self.rng[1] - self.rng[0]) * torch.normal(means, stds) + self.rng[0]

        scales = (self.rng[1] - self.rng[0]) * torch.rand([x.shape[0], 1, x.shape[-2], 1]) + self.rng[0]
        # Dont scale root
        scales[..., 0, :] = 0

        x = scales * x

        return x


class IK(_AnimTransform):

    def __init__(self, *args, **kwargs):
        """Inverse kinematics transform."""
        super().__init__(*args, **kwargs)
    
    def transform(self, x, *args, **kwargs):
        """Transformation function.

        :param x: Input animation data.

        """

        if type(x) == tuple:
            x = x[0]

        local_x = torch.empty(x.shape)
        local_x[..., 0, :] = x[..., 0, :]

        for index, parent_index in enumerate(self.ds.parent_indices):
            if parent_index != -1:
                local_x[..., index, :] = x[..., index, :] - x[..., parent_index, :]
        return local_x


class FK(_AnimTransform):

    def __init__(self, *args, **kwargs):
        """Forward kinematics transform."""
        super().__init__(*args, **kwargs)

    def transform(self, x, *args, **kwargs):
        """Transformation function.

        :param x: Input animation data.

        """

        global_x = torch.empty(x.shape)
        global_x[..., 0, :] = x[..., 0, :]

        for index, parent_index in enumerate(self.ds.parent_indices):
            if parent_index != -1:
                global_x[..., index, :] = x[..., index, :] + global_x[..., parent_index, :]
        return global_x


class LocalReferenceFrame(_AnimTransform):
    def __init__(self, ref_joint=DEFAULT_REF_FRAME_JOINT,*args, **kwargs):
        """Gets a local reference frame wrt some joint. Adds velocities.

        :param ref_joint: Joint of reference.
        
        """
        self.ref_joint = ref_joint
        super().__init__(*args, **kwargs)

    def transform(self, x, *args, **kwargs):
        """Transformation function.

        :param x: Input animation data.

        """

        rji = self.ds.joint_names.index(self.ref_joint)

        centers = x[...,rji:rji+1, :]
        x = x - centers

        traj = torch.cat([
            torch.zeros(centers.shape[:-3] + (1,) + centers.shape[-2:]),
            centers[..., 1:, :, :] - centers[..., :1, :, :]],
            dim=-3
        )
        x = torch.cat([x[...,:rji,:], traj, x[...,rji+1:,:]], dim=-2)
        return x

class GlobalReferenceFrame(_AnimTransform):
    def __init__(self, ref_joint=DEFAULT_REF_FRAME_JOINT,*args, **kwargs):
        """Opposite of local reference frame.

        :param ref_joint: Joint of reference.
        
        """
        self.ref_joint = ref_joint
        super().__init__(*args, **kwargs)

    def transform(self, x, *args, **kwargs):
        """Transformation function.

        :param x: Input animation data.

        """

        raise NotImplemented


class ReplaceJoint(_AnimTransform):

    def __init__(self, rep_joint, ref_joints, *args, **kwargs):
        """Replace a joint by the average of multiple other joints.
        :param rep_joint: Name of joint to replace.
        :param ref_joints: List of names of reference joints.
        
        """
        super().__init__(*args, **kwargs)

        self.rep_joint = rep_joint
        self.ref_joints = ref_joints

    def transform(self, x, *args, **kwargs):
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


    def transform(self, x, *args, **kwargs):
        """Transformation function.

        :param x: Input animation data.

        """

        assert self.ds != None, "Need dataset for this transformation!"

        assert x.shape[0] == self.ds._frames.shape[0], "This transformation can only be done as a pre-transform"

        num_windows = 0
        num_dims = x.shape[-1]
        for r in self.ds._anim_ranges:
            seq_length = (r[1] - r[0])
            if seq_length < self.window_size:
                continue
            elif seq_length == self.window_size:
                num_windows += 1
            else:
                out_size = math.floor((seq_length - self.window_size)/self.stride)+1
                num_windows += out_size

        x_new = torch.empty([num_windows, self.window_size, self.ds.num_joints, num_dims])

        # New meta-info
        anim_ranges_new = self.ds._anim_ranges.copy()
        anim_structure_new = deepcopy(self.ds._anim_structure)
        anim_info_new = deepcopy(self.ds._anim_info)

        # Prune missing anims/characters
        for i in range(len(anim_ranges_new) - 1, -1, -1):
            length = anim_ranges_new[i][1] - anim_ranges_new[i][0] 
            if length < self.window_size:
                cn = anim_info_new[i]['character']
                an = anim_info_new[i]['anim_name']
                del anim_structure_new[cn][an]
                del anim_ranges_new[i]
                del anim_structure_new[i]


        index = 0
        
        for i, r in enumerate(anim_ranges_new):
            length = r[1] - r[0] 
            start_index = index
            cn = anim_info_new[i]['character']
            an = anim_info_new[i]['anim_name']
            for start in range(0, (length+1)-self.window_size, self.stride):
                seq = x[r[0]+start:r[0]+start+self.window_size]
                x_new[index] = seq
                index+=1

            # Update ranges and other info
            anim_ranges_new[i] = (start_index,index)
            anim_structure_new[cn][an] = (start_index,index)

        assert index == x_new.shape[0], 'Something went wrong, not all windows are filled!'

        self.ds._anim_ranges = np.array(anim_ranges_new)
        self.ds._anim_info = anim_info_new
        self.ds._anim_structure = anim_structure_new

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
            self.axis = AXES.index(axis)

    def transform(self, x, *args, **kwargs):
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

    def transform(self, x, *args, **kwargs):
        """Transformation function.

        :param x: Input animation data.

        """
        x = self.amount * x
        return x


class RandomScaleAnim(ScaleAnim):

    def __init__(self, rng=(0.5, 1.5), *args, **kwargs):
        """Random animation scaling.

        :param rng: Range of uniform distribution to generate scaling factor.
        
        """
        amount = np.random.uniform(rng[0], rng[1])
        super().__init__(amount, *args, **kwargs)



class NormalizeAnim(_AnimTransform):

    def __init__(self, mean=0, std=1, *args, **kwargs):
        """Normalize an animation.

        :param amount: Amount to scale by.
        
        """
        super().__init__(*args, **kwargs)
        self.mean = mean
        self.std = std

    def transform(self, x, *args, **kwargs):
        """Transformation function.

        :param x: Input animation data.

        """
        if x.shape[-1] == 2 and self.mean.shape[-1] == 3:
            return (x - self.mean[..., [0,2]]) / self.std[..., [0,2]]

        else:
            return (x - self.mean) / self.std


class DenormalizeAnim(_AnimTransform):

    def __init__(self, mean=0, std=1, *args, **kwargs):
        """Denormalize an animation.

        :param amount: Amount to scale by.
        
        """
        super().__init__(*args, **kwargs)
        self.mean = mean
        self.std = std

    def transform(self, x, *args, **kwargs):
        """Transformation function.

        :param x: Input animation data.

        """
        return (x * self.std) + self.mean


class RotateAnimWithBasis(_AnimTransform):

    def __init__(self, basis=None, *args, **kwargs):
        """Rotates an animation around axes.

        :param basis (optional): Change of basis parameter [3,3]. 
        If not passsed at construction time, it should be passed later at runtime.


        """
        self.basis = basis
        super().__init__(*args, **kwargs)

    def transform(self, x, basis=None, *args, **kwargs):
        if basis is None:
            if self.basis is None:
                raise Exception('No basis given!')
            basis = self.basis
        if len(x.shape) == 3:
            x = x[None, ...]

        x = x.permute([0, 1, 3, 2])
        x = basis @ x

        x = x.permute([0, 1, 3, 2])
        return x


class RotateAnim(_AnimTransform):

    def __init__(self, euler_angles, *args, **kwargs):
        """Rotates an animation.

        :param euler: Euler angles.

        """
        self.euler_angles = euler_angles
        super().__init__(*args, **kwargs)

    def transform(self, x, *args, **kwargs):
        cx, cy, cz = torch.cos(self.euler_angles)
        sx, sy, sz = torch.sin(self.euler_angles)
        mat33_x = torch.tensor([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx, cx]
        ], dtype=torch.float32)
        mat33_y = torch.tensor([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ], dtype=torch.float32)
        mat33_z = torch.tensor([
            [cz, -sz, 0],
            [sz, cz, 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        x = x.permute([0,2,3,1])
        res = mat33_x @ mat33_y @ mat33_z @ x
        res = res.permute([0,3,1,2])

        return res


class RandomRotateAnim(RotateAnim):

    def __init__(self, rng=((-np.pi/9, -np.pi/9, -np.pi/6), (np.pi/9, np.pi/9, np.pi/6)), *args, **kwargs):
        """Random animation rotation.

        :param rng: Range of uniform distribution to generate rotation factor.
        
        """
        amount = torch.from_numpy(np.random.uniform(rng[0], rng[1]))
        super().__init__(amount, *args, **kwargs)


