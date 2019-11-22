import os
from .mobilenetv3_seg import *
from .mobilenetv3_liteflow_seg import get_mobilenet_v3_large_liteflow_seg, get_mobilenet_v3_small_liteflow_seg

from collections import OrderedDict
def get_segmentation_model(model, **kwargs):
    models = {
        'mobilenetv3_small': get_mobilenet_v3_small_seg,
        'mobilenetv3_large': get_mobilenet_v3_large_seg,
        'mobilenetv3_large_liteflow': get_mobilenet_v3_large_liteflow_seg,
        'mobilenetv3_small_liteflow': get_mobilenet_v3_small_liteflow_seg
    }
    return models[model](**kwargs)


def get_model_file(name, root='./runs/mobilenetv3/2019_07_13_17_07_37_b16_lr0.01'):
    root = os.path.expanduser(root)
    
    file_path = os.path.join(root, name + '.pth')
    file_path = './runs/mobilenetv3/2019_08_11_16_37_55_b16_lr0.015/mobilenetv3_large_yuanqu_best_model.pth'
    if os.path.exists(file_path):
        return file_path
    else:
        raise ValueError('Model file is not found. Downloading or trainning.')
        
def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
        module state_dict inplace
        :param state_dict is the loaded DataParallel model_state
    """
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict
