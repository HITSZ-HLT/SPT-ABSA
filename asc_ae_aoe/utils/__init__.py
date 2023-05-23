import os
import numpy as np
# import ujson as json
import json



def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()



def load_json(file_name):
    with open(file_name, mode='r', encoding='utf-8-sig') as f:
        return json.load(f)



def load_line_json(file_name):
    with open(file_name, mode='r', encoding='utf-8-sig') as f:
        for line in f:
            yield json.loads(line)



def mkdir_if_not_exist(path):
    dir_name, file_name = os.path.split(path)
    if dir_name:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def save_json(json_obj, file_name):
    mkdir_if_not_exist(file_name)
    with open(file_name, mode='w', encoding='utf-8-sig') as f:
        json.dump(json_obj, f, indent=4, cls=NpEncoder)




def append_json(file_name, obj, mode='a'):
    mkdir_if_not_exist(file_name)
    with open(file_name, mode=mode, encoding='utf-8') as f:
        if type(obj) is dict:
            string = json.dumps(obj)
        elif type(obj) is list:
            string = ' '.join([str(item) for item in obj])
        elif type(obj) is str:
            string = obj
        else:
            raise Exception()

        string = string + '\n'
        f.write(string)


def append_new_line(file_name, string):
    mkdir_if_not_exist(file_name)
    with open(file_name, mode='a', encoding='utf-8') as f:
        string = string + '\n'
        f.write(string)