import os
import json
import torch
import argparse
import numpy as np

from models import get_autoencoder
from utils import get_config
from itertools import combinations

from data import MixamoDatasetTest
from transforms import *


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='which config to use.')
    parser.add_argument('--description', type=str, default="data/mixamo/36_800_24/mse_description.json",
                        help="path to the description file which specifies how to run test")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="path to trained model weights")
    parser.add_argument('--data_dir', type=str, default="data/mixamo/36_800_24/test",
                        help="path to the directory storing test data")
    parser.add_argument('--out_dir', type=str, required=True,
                        help="path to output directory")
    args = parser.parse_args()

    config = get_config(args.config)
    ae = get_autoencoder(config)
    ae.load_state_dict(torch.load(args.checkpoint))
    ae.cuda()
    ae.eval()

    pre_trans = torch.nn.Sequential()
    pre_anim = torch.nn.Sequential(
        #InputData('basis', torch.eye(3), parallel_out=True),
        #InputRandomData('view_angles', ((0,0,0), (0,0,2*np.pi)), parallel_out=True),
        #RotateBasis(parallel_out=True),
        #RotateAnimWithBasis(pass_along=False),
    )
    ds = MixamoDatasetTest(
        path=args.data_dir,
        config=config,
        pre_transform=pre_trans,
        pre_anim=pre_anim
    )
    mean_pose, std_pose = ds.mean, ds.std


    description = json.load(open(args.description))
    chars = list(description.keys())

    cnt = 0
    os.makedirs(args.out_dir, exist_ok=True)

    for char1, char2 in combinations(chars, 2):

        motions1 = description[char1]
        motions2 = description[char2]

        for i, mot1 in enumerate(motions1):
            for j, mot2 in enumerate(motions2):

                path1 = os.path.join(args.data_dir, char1, mot1, "{}.npy".format(mot1))
                path2 = os.path.join(args.data_dir, char2, mot2, "{}.npy".format(mot2))

                ############
                # CROSS 2D #
                ############

                out_path1 = os.path.join(args.out_dir, "motion_{}_{}_body_{}_{}.npy".format(char1, i, char2, j))
                out_path2 = os.path.join(args.out_dir, "motion_{}_{}_body_{}_{}.npy".format(char2, j, char1, i))

                data_in = ds.process(char1, mot1, char2, mot2)

                x_a, x_a_start = data_in['data_1'], data_in['start_1'] 
                x_b, x_b_start = data_in['data_2'], data_in['start_2'] 

                x_ab = ae.cross2d(x_a, x_b, x_a)
                x_ba = ae.cross2d(x_b, x_a, x_b)

                data_out = ds.unprocess(x_ab, x_ba, x_a_start, x_b_start)
                x_ab, x_ba = data_out['data_1'], data_out['data_2']

                x_ab_np = x_ab.numpy().squeeze().swapaxes(2,1).swapaxes(-1, 0)
                x_ba_np = x_ba.numpy().squeeze().swapaxes(2,1).swapaxes(-1, 0)

                np.save(out_path1, x_ab_np)
                np.save(out_path2, x_ba_np)

                cnt += 1
                print("computed {} pairs".format(cnt), end="\r")

    print("finished" + " " * 20)


if __name__ == "__main__":
    main()


