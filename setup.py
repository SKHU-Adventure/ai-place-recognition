#!/usr/bin/python

import sys
import os
from configparser import ConfigParser
import ast
import random
import numpy as np
import torch


class Configuration(object):
    def __init__(self, *file_names):
        parser = ConfigParser()
        parser.optionxform = str
        found = parser.read(file_names)
        if not found:
            raise ValueError('No config file found!')
        for name in parser.sections():
            self.__dict__.update({item[0]: ast.literal_eval(item[1]) for item in parser.items(name)})


config = Configuration('setup.ini', os.path.join(sys.argv[1], 'setup.ini'))
config.base_dir = os.path.abspath(sys.argv[1])
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["PYTHONHASHSEED"] = str(config.seed)


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
