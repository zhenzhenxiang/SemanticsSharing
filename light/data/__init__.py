"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .yuanqu import yuanquSegmentation
datasets = {
    'yuanqu': yuanquSegmentation
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)