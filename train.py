"""Module for training neural net models to predict global positions. """

import argparse
import sys
import shutil
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



import random

#make sure you're getting right classes
import lib.network
from lib.loss import *
from lib.util.general import weights_init, get_model_list, get_scheduler
from lib.network import Discriminator
from lib.operation import rotate_and_maybe_project

from data import get_dataloader

# Define defaults here
DEFAULT_NUM_EPOCHS = 50
DEFAULT_BATCH_SIZE = 8

class Trainer(nn.Module):

    def __init__(self, config):
        super(Trainer, self).__init__(config)

        lr = config.lr
        autoencoder_cls = getattr(lib.network, config.autoencoder.cls)
        self.autoencoder = autoencoder_cls(config.autoencoder)
        self.discriminator = Discriminator(config.discriminator)

        # Setup the optimizers
        beta1 = config.beta1
        beta2 = config.beta2
        dis_params = list(self.discriminator.parameters())
        ae_params = list(self.autoencoder.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=config.weight_decay)
        self.ae_opt = torch.optim.Adam([p for p in ae_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=config.weight_decay)
        self.dis_scheduler = get_scheduler(self.dis_opt, config)
        self.ae_scheduler = get_scheduler(self.ae_opt, config)

        # Network weight initialization
        self.apply(weights_init(config.init))
        self.discriminator.apply(weights_init('gaussian'))

        self.angle_unit = np.pi / (config.K + 1)
        view_angles = np.array([i * self.angle_unit for i in range(1, config.K + 1)])
        x_angles = view_angles if config.rotation_axes[0] else np.array([0])
        z_angles = view_angles if config.rotation_axes[1] else np.array([0])
        y_angles = view_angles if config.rotation_axes[2] else np.array([0])
        x_angles, z_angles, y_angles = np.meshgrid(x_angles, z_angles, y_angles)
        angles = np.stack([x_angles.flatten(), z_angles.flatten(), y_angles.flatten()], axis=1)
        self.angles = torch.tensor(angles).float().cuda()
        self.rotation_axes = torch.tensor(config.rotation_axes).float().cuda()
        self.rotation_axes_mask = [(_ > 0) for _ in config.rotation_axes]

    def train(self):
        # @TODO move transmomo train.py here

    def forward(self, data):
        x_a, x_b = data["x_a"], data["x_b"]
        batch_size = x_a.size(0)
        self.eval()
        body_a, body_b = self.sample_body_code(batch_size)
        motion_a = self.autoencoder.encode_motion(x_a)
        body_a_enc, _ = self.autoencoder.encode_body(x_a)
        motion_b = self.autoencoder.encode_motion(x_b)
        body_b_enc, _ = self.autoencoder.encode_body(x_b)
        x_ab = self.autoencoder.decode(motion_a, body_b)
        x_ba = self.autoencoder.decode(motion_b, body_a)
        self.train()
        return x_ab, x_ba

    def dis_update(self, data, config):

        x_a = data["x"].detach()
        x_s = data["x_s"].detach()  # the limb-scaled version of x_a

        self.dis_opt.zero_grad()

        # encode
        motion_a = self.autoencoder.encode_motion(x_a)
        body_a, body_a_seq = self.autoencoder.encode_body(x_a)
        view_a, view_a_seq = self.autoencoder.encode_view(x_a)

        motion_s = self.autoencoder.encode_motion(x_s)
        body_s, body_s_seq = self.autoencoder.encode_body(x_s)
        view_s, view_s_seq = self.autoencoder.encode_view(x_s)

        # decode (reconstruct, transform)
        inds = random.sample(list(range(self.angles.size(0))), config.K)
        angles = self.angles[inds].clone().detach()  # [K, 3]
        angles += self.angle_unit * self.rotation_axes * torch.randn([3], device=x_a.device)
        angles = angles.unsqueeze(0).unsqueeze(2)  # [B=1, K, T=1, 3]

        X_a_recon = self.autoencoder.decode(motion_a, body_a, view_a)
        x_a_trans = rotate_and_maybe_project(X_a_recon, angles=angles, body_reference=config.autoencoder.body_reference,
                                             project_2d=True)

        x_a_exp = x_a.repeat_interleave(config.K, dim=0)

        self.loss_dis_trans = self.discriminator.calc_dis_loss(x_a_trans.detach(), x_a_exp)

        if config.trans_gan_ls_w > 0:
            X_s_recon = self.autoencoder.decode(motion_s, body_s, view_s)
            x_s_trans = rotate_and_maybe_project(X_s_recon, angles=angles,
                                                 body_reference=config.autoencoder.body_reference, project_2d=True)
            x_s_exp = x_s.repeat_interleave(config.K, dim=0)
            self.loss_dis_trans_ls = self.discriminator.calc_dis_loss(x_s_trans.detach(), x_s_exp)
        else:
            self.loss_dis_trans_ls = 0

        self.loss_dis_total = config.trans_gan_w * self.loss_dis_trans + \
                              config.trans_gan_ls_w * self.loss_dis_trans_ls

        self.loss_dis_total.backward()
        self.dis_opt.step()

    def ae_update(self, data, config):

        x_a = data["x"].detach()
        x_s = data["x_s"].detach()
        self.ae_opt.zero_grad()

        # encode
        motion_a = self.autoencoder.encode_motion(x_a)
        body_a, body_a_seq = self.autoencoder.encode_body(x_a)
        view_a, view_a_seq = self.autoencoder.encode_view(x_a)

        motion_s = self.autoencoder.encode_motion(x_s)
        body_s, body_s_seq = self.autoencoder.encode_body(x_s)
        view_s, view_s_seq = self.autoencoder.encode_view(x_s)

        # invariance loss
        self.loss_inv_v_ls = self.recon_criterion(view_a, view_s) if config.inv_v_ls_w > 0 else 0
        self.loss_inv_m_ls = self.recon_criterion(motion_a, motion_s) if config.inv_m_ls_w > 0 else 0

        # body triplet loss
        if config.triplet_b_w > 0:
            self.loss_triplet_b = triplet_margin_loss(
                body_a_seq, body_s_seq,
                neg_range=config.triplet_neg_range,
                margin=config.triplet_margin)
        else:
            self.loss_triplet_b = 0

        # reconstruction
        X_a_recon = self.autoencoder.decode(motion_a, body_a, view_a)
        x_a_recon = rotate_and_maybe_project(X_a_recon, angles=None, body_reference=config.autoencoder.body_reference,
                                             project_2d=True)

        X_s_recon = self.autoencoder.decode(motion_s, body_s, view_s)
        x_s_recon = rotate_and_maybe_project(X_s_recon, angles=None, body_reference=config.autoencoder.body_reference,
                                             project_2d=True)

        self.loss_recon_x = 0.5 * self.recon_criterion(x_a_recon, x_a) + \
                            0.5 * self.recon_criterion(x_s_recon, x_s)

        # cross reconstruction
        X_as_recon = self.autoencoder.decode(motion_a, body_s, view_s)
        x_as_recon = rotate_and_maybe_project(X_as_recon, angles=None, body_reference=config.autoencoder.body_reference,
                                              project_2d=True)

        X_sa_recon = self.autoencoder.decode(motion_s, body_a, view_a)
        x_sa_recon = rotate_and_maybe_project(X_sa_recon, angles=None, body_reference=config.autoencoder.body_reference,
                                              project_2d=True)

        self.loss_cross_x = 0.5 * self.recon_criterion(x_as_recon, x_s) + 0.5 * self.recon_criterion(x_sa_recon, x_a)

        # apply transformation
        inds = random.sample(list(range(self.angles.size(0))), config.K)
        angles = self.angles[inds].clone().detach()
        angles += self.angle_unit * self.rotation_axes * torch.randn([3], device=x_a.device)
        angles = angles.unsqueeze(0).unsqueeze(2)

        x_a_trans = rotate_and_maybe_project(X_a_recon, angles=angles, body_reference=config.autoencoder.body_reference,
                                             project_2d=True)
        x_s_trans = rotate_and_maybe_project(X_s_recon, angles=angles, body_reference=config.autoencoder.body_reference,
                                             project_2d=True)

        # GAN loss
        self.loss_gan_trans = self.discriminator.calc_gen_loss(x_a_trans)
        self.loss_gan_trans_ls = self.discriminator.calc_gen_loss(x_s_trans) if config.trans_gan_ls_w > 0 else 0

        # encode again
        motion_a_trans = self.autoencoder.encode_motion(x_a_trans)
        body_a_trans, _ = self.autoencoder.encode_body(x_a_trans)
        view_a_trans, view_a_trans_seq = self.autoencoder.encode_view(x_a_trans)

        motion_s_trans = self.autoencoder.encode_motion(x_s_trans)
        body_s_trans, _ = self.autoencoder.encode_body(x_s_trans)

        self.loss_inv_m_trans = 0.5 * self.recon_criterion(motion_a_trans,
                                                           motion_a.repeat_interleave(config.K, dim=0)) + \
                                0.5 * self.recon_criterion(motion_s_trans, motion_s.repeat_interleave(config.K, dim=0))
        self.loss_inv_b_trans = 0.5 * self.recon_criterion(body_a_trans, body_a.repeat_interleave(config.K, dim=0)) + \
                                0.5 * self.recon_criterion(body_s_trans, body_s.repeat_interleave(config.K, dim=0))

        # view triplet loss
        if config.triplet_v_w > 0:
            view_a_seq_exp = view_a_seq.repeat_interleave(config.K, dim=0)
            self.loss_triplet_v = triplet_margin_loss(
                view_a_seq_exp, view_a_trans_seq,
                neg_range=config.triplet_neg_range, margin=config.triplet_margin)
        else:
            self.loss_triplet_v = 0

        # add all losses
        self.loss_total = torch.tensor(0.).float().cuda()
        self.loss_total += config.recon_x_w * self.loss_recon_x
        self.loss_total += config.cross_x_w * self.loss_cross_x
        self.loss_total += config.inv_v_ls_w * self.loss_inv_v_ls
        self.loss_total += config.inv_m_ls_w * self.loss_inv_m_ls
        self.loss_total += config.inv_b_trans_w * self.loss_inv_b_trans
        self.loss_total += config.inv_m_trans_w * self.loss_inv_m_trans
        self.loss_total += config.trans_gan_w * self.loss_gan_trans
        self.loss_total += config.trans_gan_ls_w * self.loss_gan_trans_ls
        self.loss_total += config.triplet_b_w * self.loss_triplet_b
        self.loss_total += config.triplet_v_w * self.loss_triplet_v

        self.loss_total.backward()
        self.ae_opt.step()

    def recon_criterion(self, input, target):
        raise NotImplemented

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.ae_scheduler is not None:
            self.ae_scheduler.step()

    def resume(self, checkpoint_dir, config):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "autoencoder")
        state_dict = torch.load(last_model_name)
        self.autoencoder.load_state_dict(state_dict)
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "discriminator")
        state_dict = torch.load(last_model_name)
        self.discriminator.load_state_dict(state_dict)
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['discriminator'])
        self.ae_opt.load_state_dict(state_dict['autoencoder'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, config, iterations)
        self.ae_scheduler = get_scheduler(self.ae_opt, config, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        ae_name = os.path.join(snapshot_dir, 'autoencoder_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'discriminator_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save(self.autoencoder.state_dict(), ae_name)
        torch.save(self.discriminator.state_dict(), dis_name)
        torch.save({'autoencoder': self.ae_opt.state_dict(), 'discriminator': self.dis_opt.state_dict()}, opt_name)

    def validate(self, data, config):
        re_dict = self.evaluate(self.autoencoder, data, config)
        for key, val in re_dict.items():
            setattr(self, key, val)

    @staticmethod
    def recon_criterion(input, target):
        return torch.mean(torch.abs(input - target))

    @classmethod
    def evaluate(cls, autoencoder, data, config):
        autoencoder.eval()
        x_a, x_b = data["x_a"], data["x_b"]
        x_aba, x_bab = data["x_aba"], data["x_bab"]
        batch_size, _, seq_len = x_a.size()

        re_dict = {}

        with torch.no_grad():  # 2D eval

            x_a_recon = autoencoder.reconstruct2d(x_a)
            x_b_recon = autoencoder.reconstruct2d(x_b)
            x_aba_recon = autoencoder.cross2d(x_a, x_b, x_a)
            x_bab_recon = autoencoder.cross2d(x_b, x_a, x_b)

            re_dict['loss_val_recon_x'] = cls.recon_criterion(x_a_recon, x_a) + cls.recon_criterion(x_b_recon, x_b)
            re_dict['loss_val_cross_body'] = cls.recon_criterion(x_aba_recon, x_aba) + cls.recon_criterion(
                x_bab_recon, x_bab)
            re_dict['loss_val_total'] = 0.5 * re_dict['loss_val_recon_x'] + 0.5 * re_dict['loss_val_cross_body']

        autoencoder.train()
        return re_dict

# ====================================== #
# Code for creating/storing experiments
# ====================================== #
def str2type(v):
    """Convert string to type."""

    if type(v) != str:
        return v

    funcs = [str2bool, int, float, str]
    for func in funcs:
        try:
            v = func(v)
            return v
        except Exception as e:
            pass
    raise Exception('Cannot convert type'.format(v))


def str2bool(v):
    """Convert string to bool for commandline arguments."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    elif v.lower() in ('None', 'none', 'null'):
        return None
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_args(args, path):
    """Load argparse arguments from a saved arguments.txt file.

    :param args: Argparse arguments.
    :param path: Path to arguments.txt.
    :return:
    """

    try:
        with open(path, 'r') as f:
            lines = f.readlines()

            assert len(lines) % 2 == 0, "Unequal num of params/values!"
            keys = lines[::2]
            vals = lines[1::2]

            for k,v in zip(keys, vals):
                setattr(args, k, v)
    # @TODO implement custom exception handling if necessary
    except FileNotFoundError as e:
        raise
    except AttributeError as e:
        raise
    except Exception as e:
        raise

    return args

def create_experiment(args):
    """Creates an experiment directory with arguments.

    :param args: Arguments to store in the directory.
    :return:
    """

    # Generate experiment name/folder
    if not os.path.exists('experiments'):
        os.mkdir('experiments')

    if args.name is not None:
        exp_name = args.name
    else:
        exp_name = '{}_{}_{}_{}'.format(
            os.path.splitext(os.path.basename(args.database))[0],
            args.model, args.optimizer, args.loss
        )
        # Add number of experiment at end
        for i in range(1000):
            num = '_{}'.format(i)
            if not os.path.exists(
                    os.path.join('experiments', exp_name + num)
            ):
                logging.info(os.path.join('experiments', exp_name + num))
                exp_name = exp_name + num
                break

    # Create directory for experiment
    exp_dir = os.path.join('experiments', exp_name)
    os.mkdir(exp_dir)

    # Store arguments used to run the model
    argsdict = args.__dict__.copy()

    model_name = argsdict.pop('model')
    db_name = argsdict.pop('database')

    with open(os.path.join(exp_dir, 'arguments.txt'), 'w+') as f:
        f.write(model_name + '\n')
        f.write(db_name + '\n')
        for k in sorted(argsdict):
            f.write('--' + k + '\n' + str(argsdict[k]) + '\n')

    return exp_dir



def create_args():
    """Creates arguments for training."""

    parser = argparse.ArgumentParser(
        description='Train a model to predict global positions.',
        fromfile_prefix_chars='@'
    )
    # Model parameters
    parser.add_argument(
        'config',
        type=str,
        help='Path to config file.'
    )
    parser.add_argument(
        '--use_gpu',
        type=str2bool,
        help='Attempt to use GPU to run model. Will fallback to CPU if not '
             'available.',
        default=True
    )
    parser.add_argument(
        '--name',
        help='Name of the experiment. If not specified, will be '
             'auto-generated',
        default=None
    )

    # Database parameters
    parser.add_argument(
        'dataset',
        type=str,
        help='Dataset to use'
    )
    
   
    parser.add_argument(
        '--shuffle_data', '--do_shuffle',
        type=str2bool,
        help='Whether or not to shuffle the data',
        default=True
    )

    # Training hyperparameters
    parser.add_argument(
        '--epochs',
        type=str,
        help='Number of epochs to run for.',
        default=DEFAULT_NUM_EPOCHS
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Size of a single batch.',
        default=DEFAULT_BATCH_SIZE
    )
    parser.add_argument(
        '--normalize',
        type=str2bool,
        help='Normalize the input data',
        default=True
    )
    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate to use.',
        default=1e-3
    )
    parser.add_argument(
        '--momentum',
        type=float,
        help='Momentum to use (if applicable).',
        default=0.0
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        help='Type of optimizer to use',
        default='SGD'
    )
    parser.add_argument(
        '--loss',
        type=str,
        help='Type of loss to use',
        default='MSE'
    )
    parser.add_argument(
        '--loss_reduction',
        type=str,
        help='Type of loss reduction to use',
        default='mean'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Seed for numpy and pytorch random functions. Will default to '
             'random if not set',
        default=1
    )

    # Saving/visualizing parameters
    parser.add_argument(
        '--save',
        type=str2bool,
        help='Whether or not to save the model at each epoch.',
        default=True
    )
    parser.add_argument(
        '--load',
        type=str,
        help='Path to saved model to load.',
        default=None
    )

    parser.add_argument(
        '--visualize',
        type=str2bool,
        help='Whether or not to visualize the model.',
        default=True
    )

    # Logging level
    parser.add_argument(
        '--logging',
        type=str,
        help='Choice of logging level (none, warning, info, error, critical)',
        default='info'
    )

    return parser

if __name__ == '__main__':

    # Parse arguments from command-line
    parser = create_args()

    # Parse arguments
    args = parser.parse_args()

    # Make sure arguments of correct type
    # (this is only an issue when loading arguments from file)
    argsdict = args.__dict__
    for k, v in argsdict.items():
        argsdict[k] = str2type(v)

    # Check if loading is requested
    model_path = None
    if args.load is not None:
        if os.path.isdir(args.load):
            # Tries to load best.pt file
            best_path = os.path.join(args.load, 'best.pt')
            if os.path.isfile(best_path):
                logging.debug('Found model path at {}'.format(best_path))
                model_path = best_path
            else:
                logging.debug('Did not find model path at {}'.format(
                    best_path))
                raise Exception('Unkown model path!')

        elif os.path.isfile(args.load):
            logging.debug('Found mOdel path at {}'.format(model_path))
            model_path = args.load

    # If mode is valid(ation) and epochs hasn't been explicitly set, set num
    # epochs to 1
    if args.mode == 'valid' and args.epochs == DEFAULT_NUM_EPOCHS:
        args.epochs = 1

    # Create an experiment folder
    exp_dir = create_experiment(args)

    # Sets up logging for experiment
    handler = setup_logging(args.logging, os.path.join(exp_dir, 'log.txt'))

    # Tries to create a trainer object
    try:
        trainer = Trainer(
            #*args
        )

    except Exception as e:
        # Error in creating trainer

        # Shutdown logging first
        handler.close()
        #formatter.close()

        # Cleanup created folder
        shutil.rmtree(exp_dir)
        raise e

    # Train the model
    trainer()
