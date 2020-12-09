"""Module for visualizing animation data. """

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import imageio
from lib.util.visualization import motion2video

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

PARENTS = [-1, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13]


def visualize_mpl(data, show_basis=False, ds=None, fps=60, j_color='green', l_color='green', b_color='#222222', auto_limits=True, save_path=None):
    """Visualizes animation data using matplotlib.

    :param data: Animation data in format [nframes, njoints, 3]
    :param show_basis: Visualize basis vectors.
    :param ds: Dataset.
    :param fps: Framerate of animation.
    :param j_color: Joint color.
    :param l_color: Link color.
    :param b_color: Background color.
    :param auto_limits: Automatically detect axes limits.
    :param save_path: Path to save animation.

    """

    ndim = data.shape[-1]
    if ndim not in [2,3]:
        raise Exception('Invalid data format')

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

    # Background color
    plt.rcParams['axes.facecolor'] = b_color
    plt.rcParams['figure.facecolor'] = b_color

    fig = plt.figure()

    if ndim == 3:
        ax = fig.add_subplot(111, projection='3d')
    elif ndim == 2:
        ax = fig.add_subplot(111)


    frame = 0

    # Visualize joints
    if ndim == 3:
        joints = ax.scatter(data[0, :, frame], data[1, :, frame], data[2, :, frame], color=j_color)
    elif ndim == 2:
        joints = ax.scatter(data[0, :, frame], data[1, :, frame], color=j_color)

    # Visualize links
    links = []
    for joint_index, parent_index in zip(range(len(PARENTS)),PARENTS):
        if parent_index != -1:
            if ndim == 3:
                links.append(ax.plot3D(
                    [data[0,joint_index,frame], data[0,parent_index,frame]],
                    [data[1,joint_index,frame], data[1,parent_index,frame]],
                    [data[2,joint_index,frame], data[2,parent_index,frame]],
                    color=l_color
                ))
            elif ndim == 2:
                links.append(ax.plot(
                    [data[0,joint_index,frame], data[0,parent_index,frame]],
                    [data[1,joint_index,frame], data[1,parent_index,frame]],
                    color=l_color
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
            if ndim == 3:
                basis += ax.plot3D(
                    [root_x, root_x + basis_vectors[0,b]],
                    [root_y, root_y + basis_vectors[1,b]],
                    [root_z, root_z + basis_vectors[2,b]]
                )
            elif ndim == 2:
                basis += ax.plot(
                    [root_x, root_x + basis_vectors[0,b]],
                    [root_y, root_y + basis_vectors[1,b]]
                )


    update_args = [ndim, data, joints, links]

    if show_basis:
        update_args.append(basis)
        update_args.append(basis_vectors)

    anim = animation.FuncAnimation(
        fig, update_mpl_anim, data.shape[-1], fargs=update_args,
        interval=20,
        blit=False
    )

    if auto_limits:
        x_max = data[0,:].max()
        x_min = data[0,:].min()
        x_std = data[0,:].std()
        y_max = data[1,:].max()
        y_min = data[1,:].min()
        y_std = data[1,:].std()
        if ndim == 3:
            z_max = data[2,:].max()
            z_min = data[2,:].min()
            z_std = data[2,:].std()
    else:
        x_max = 1
        x_min = 0
        x_std = 0
        y_max = 1
        y_min = 0
        y_std = 0
        if ndim == 3:
            z_max = 1
            z_min = 0
            z_std = 0

    if ndim == 3:
        ax.set_xlim3d(x_min-x_std,x_max+x_std)
        ax.set_ylim3d(y_min-y_std,y_max+y_std)
        ax.set_zlim3d(z_min-z_std,z_max+z_std)
    elif ndim == 2:
        ax.set_xlim(x_min-x_std,x_max+x_std)
        ax.set_ylim(y_min-y_std,y_max+y_std)

    plt.axis('off')
    plt.show()

    if save_path is not None:
        plt.rcParams['animation.convert_path'] = r'C:\Program Files\ImageMagick-7.0.10-Q16-HDRI\magick.exe'
        plt.rcParams['animation.ffmpeg_path'] = r'C:\Program Files\ImageMagick-7.0.10-Q16-HDRI\ffmpeg.exe'
        #writer = animation.ImageMagickFileWriter(fps=fps)

        anim.save(save_path, fps=fps, writer='imagemagick')
        #anim.save(save_path, fps=fps)


def update_mpl_anim(frame, ndim, data, joints, links, basis=None, basis_vectors=None):

    # Update joint positions
    if ndim == 3:
        joints._offsets3d = data[:, :, frame]
    elif ndim == 2:
        joints._offsets = data[:, :, frame].swapaxes(0,1)


    # Update links
    for joint_index, parent_index, link in zip(range(len(PARENTS)),PARENTS,links):
        if parent_index != -1:
            link[0].set_data(
                np.array([[data[0,joint_index,frame], data[0,parent_index,frame]],
                [data[1,joint_index,frame], data[1,parent_index,frame]]])
            )
            if ndim == 3:
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
            if ndim == 3:
                basis[b].set_3d_properties(
                    np.array([root_z, root_z + basis_vectors[2,b]])
                )



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('path', help='path to npy animation to visualize')
    parser.add_argument('--b_color', type=str, help='background color', default='white')
    parser.add_argument('--c_color', type=str, help='character color', default='green')
    parser.add_argument('--fps', type=int, help='animation framerate', default=60)
    parser.add_argument('--flip', action='store_true')

    if not os.path.isdir('./visualizations'):
        os.mkdir('./visualizations')

    args = parser.parse_args()
    anim = np.load(args.path)

    if args.flip:
        print('flipping animation...')
        anim[:, 0, :] = -anim[:, 0, :]
        anim[:, 1, :] = -anim[:, 1, :]

    anim = anim.swapaxes(0, -1).swapaxes(1,2)

    basename = os.path.basename(args.path).replace('npy', 'gif')
    
    if basename in ['{}.gif'.format(i) for i in range(1,10)]:
        basename = os.path.dirname(args.path).split(os.path.sep)[-1] + '.gif'

    out_path = os.path.realpath(os.path.join(
        './visualizations',
        basename
    ))

    visualize_mpl(
        anim,
        save_path=out_path,
        fps=args.fps,
        j_color=args.c_color,
        l_color=args.c_color,
        b_color=args.b_color
    )
