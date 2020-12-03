"""Module for visualizing animation data. """

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from transforms import *

PARENTS = [-1, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13]


def visualize_mpl(data, show_basis=False, ds=None):
    """Visualizes animation data using matplotlib.

    :param data: Animation data in format [nframes, njoints, 3]
    :param show_basis: Visualize basis vectors.

    """


    if show_basis:
        bt = ToBasis(dataset=ds)
        basis_vectors = bt(data)

        rb = RotateBasis(dataset=ds)
        rotated_basis = rb(basis_vectors).squeeze()

        if type(data) == torch.Tensor:
            basis_vectors = basis_vectors.clone().numpy().squeeze()
            rotated_basis = rotated_basis.clone().numpy().squeeze()
        else:
            basis_vectors = basis_vectors.copy().squeeze()
            rotated_basis = rotated_basis.copy().squeeze()

        basis_vectors = basis_vectors.swapaxes(0,-1)
        #basis_vectors = rotated_basis.swapaxes(0,-1)

    if type(data) == torch.Tensor:
        data = data.clone().numpy().squeeze()
    else:
        data = data.copy().squeeze()

        
    # Easier to work with in this format
    data = data.copy().swapaxes(0,-1)
    #data = data.copy().swapaxes(0,1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    frame = 0

    # Visualize joints
    joints = ax.scatter(data[0, :, frame], data[1, :, frame], data[2, :, frame])

    # Visualize links
    links = []
    for joint_index, parent_index in zip(range(len(PARENTS)),PARENTS):
        if parent_index != -1:
            links.append(ax.plot3D(
                [data[0,joint_index,frame], data[0,parent_index,frame]],
                [data[1,joint_index,frame], data[1,parent_index,frame]],
                [data[2,joint_index,frame], data[2,parent_index,frame]]
            ))
        else:
            links.append(None)


    if show_basis:
        basis = []
        for b in range(basis_vectors.shape[1]):
            hip_joint = ds.joint_names.index('Mid_hip')
            root_x = 0
            root_y = 0
            root_z = 0
            #root_x = data[0, 0, frame]
            #root_y = data[1, 0, frame]
            #root_z = data[2, 0, frame]
            basis += ax.plot3D(
                [root_x, root_x + basis_vectors[0,b]],
                [root_y, root_y + basis_vectors[1,b]],
                [root_z, root_z + basis_vectors[2,b]]
            )


    update_args = [data, joints, links]

    if show_basis:
        update_args.append(basis)
        update_args.append(basis_vectors)

    anim = animation.FuncAnimation(
        fig, update_mpl_anim, data.shape[-1], fargs=update_args,
        interval=20,
        blit=False
    )

    ax.set_xlim3d(0,1)
    ax.set_ylim3d(0,1)
    ax.set_zlim3d(0,1)

    plt.show()


def update_mpl_anim(frame, data, joints, links, basis=None, basis_vectors=None):

    
    # Update joint positions
    joints._offsets3d = data[:, :, frame]

    # Update links
    for joint_index, parent_index, link in zip(range(len(PARENTS)),PARENTS,links):
        if parent_index != -1:
            link[0].set_data(
                np.array([[data[0,joint_index,frame], data[0,parent_index,frame]],
                [data[1,joint_index,frame], data[1,parent_index,frame]]])
            )
            link[0].set_3d_properties(
                np.array([data[2,joint_index,frame], data[2,parent_index,frame]])
            )

    if basis is not None:
        for b in range(basis_vectors.shape[1]):
            root_x = 0
            root_y = 0
            root_z = 0
            #root_x = data[0, 0, frame]
            #root_y = data[1, 0, frame]
            #root_z = data[2, 0, frame]
            basis[b].set_data(
                np.array([[root_x, root_x + basis_vectors[0,b]],
                [root_y, root_y + basis_vectors[1,b]]])
            )
            basis[b].set_3d_properties(
                np.array([root_z, root_z + basis_vectors[2,b]])
            )



if __name__ == '__main__':
    pass
