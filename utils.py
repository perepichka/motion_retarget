"""Module for general utilities."""
import sys, os, glob

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from easydict import EasyDict as edict

import yaml


def get_config(config_path):
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    config = edict(config)
    _, config_filename = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_filename)
    config.name = config_name
    return config
