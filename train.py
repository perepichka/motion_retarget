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

# Define defaults here
DEFAULT_NUM_EPOCHS = 50
DEFAULT_BATCH_SIZE = 8

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
        'model',
        type=str,
        help='Type of model to train.'
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
