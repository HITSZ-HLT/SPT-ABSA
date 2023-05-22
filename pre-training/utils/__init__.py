import os
import ujson as json
import numpy as np
# from collections import defaultdict



def mkdir_if_not_exist(path):
    dir_name, file_name = os.path.split(path)
    if dir_name:
        _mkdir_if_not_exist(dir_name)


def _mkdir_if_not_exist(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def load_json(file_name):
    with open(file_name, mode='r', encoding='utf-8-sig') as f:
        return json.load(f)


def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


def yield_data_file(data_dir):
    for file_name in os.listdir(data_dir):
        yield os.path.join(data_dir, file_name)