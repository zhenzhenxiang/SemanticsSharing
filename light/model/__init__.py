import os
from .mobilenetv3_pwcnet_seg import get_mobilenet_v3_large_pwcflow_seg, get_mobilenet_v3_small_pwcflow_seg

def get_segmentation_model(model, **kwargs):
    models = {
        'mobilenetv3_large_pwcflow': get_mobilenet_v3_large_pwcflow_seg,
        'mobilenetv3_small_pwcflow': get_mobilenet_v3_small_pwcflow_seg
    }
    return models[model](**kwargs)



