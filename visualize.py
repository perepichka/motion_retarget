"""Module for visualizing animation data. """

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

PARENTS = [-1, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13]

def visualize_mpl(data):
    """Visualizes animation data using matplotlib.


    :param data: Animation data in format [njoints, 3, nframes]

    """
    
    # Easier to work with in this format
    data = data.copy().swapaxes(0,1)

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


    anim = animation.FuncAnimation(
        fig, update_mpl_anim, data.shape[-1], fargs=(data, joints, links),
        interval=20,
        blit=False
    )

    #plt.xlim(-1, 3)
    #plt.ylim(-3, 3)
    ax.set_xlim3d(0,1)
    ax.set_ylim3d(0,1)
    ax.set_zlim3d(0,1)

    plt.show()


def update_mpl_anim(frame, data, joints, links):
    
    # Update joint positions
    joints._offsets3d = data[:, :, frame]

    # Update links
    for joint_index, parent_index, link in zip(range(len(PARENTS)),PARENTS,links):
        if parent_index != -1:
            link[0].set_data(
                [[data[0,joint_index,frame], data[0,parent_index,frame]],
                [data[1,joint_index,frame], data[1,parent_index,frame]]]
            )
            link[0].set_3d_properties(
                [data[2,joint_index,frame], data[2,parent_index,frame]]
            )

if __name__ == '__main__':
    pass
