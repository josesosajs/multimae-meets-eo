import os
import h5py
import json
import torch
import utils
import random
import os.path
import numpy as np
import torch.nn.functional as F

from PIL import Image
from copy import deepcopy
from torchvision import datasets, transforms
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
from .data_constants import S2_MEAN, S2_STD, DEPTH_MEAN, DEPTH_STD

def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None):
    
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances

class CustomMultiTaskDatasetFolder(VisionDataset):

    def __init__(
            self,
            root: str,
            tasks: List[str],
            classes_def: Optional[str] = None, 
            splits_path: Optional[str] = None, 
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            prefixes: Optional[Dict[str,str]] = None,
            max_images: Optional[int] = None,
            split: Optional[str] = 'train'):
        
        super(CustomMultiTaskDatasetFolder, self).__init__(root, transform=transform, 
            target_transform=target_transform)
        
        self.root = root    
        self.classes_def = classes_def
        self.tasks = tasks
        
        classes, class_to_idx = self._find_classes(self.classes_def)

        self.indices = json.load(open(splits_path, 'r'))[split]
        
        prefixes = {} if prefixes is None else prefixes
        prefixes.update({task: '' for task in tasks if task not in prefixes})

        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = {}        
        self.cache = {}

    def _open_hdf5(self, root):
        self.data = h5py.File(root, "r")

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        
        with open(dir) as json_file: class_to_idx = json.load(json_file)

        classes = list(class_to_idx.keys())
        
        return classes, class_to_idx

    def __getitem__(self, index: int):
        if not hasattr(self, "data_full"): self._open_hdf5(self.root)
        
        self.samples = {'rgb' : self.data['sentinel2'], 
                        'ired' : self.data['sentinel2'],
                        'sired' : self.data['sentinel2'],
                        'ebands' : self.data['sentinel2'],
                        'semseg' : self.data['esa_worldcover'], 
                        'depth' : self.data['aster']}

        if index in self.cache: sample_dict, target = deepcopy(self.cache[index])
        
        else:
            sample_dict = {}
            
            #Getting image class, not used in this case
            target = np.where(self.data['biome'][self.indices[index]] == 1)[0][0]
            
            for task in self.tasks:
                #Getting only infrared channels (red edges bands) from sentinel2 data
                if task == 'rgb':
                    data = self.samples[task][self.indices[index]]
                    data = (data - np.array(S2_MEAN)[:, None, None]) / np.array(S2_STD)[:, None, None]
                    sample = data[[3, 2, 1], :, :]

                #Getting only infrared channels (red edges bands) from sentinel2 data
                elif task == 'ired':
                    data = self.samples[task][self.indices[index]]
                    data = (data - np.array(S2_MEAN)[:, None, None]) / np.array(S2_STD)[:, None, None]
                    sample = data[[4, 5, 6], :, :]

                #Getting only short wave infrared channels from sentinel2 data
                elif task == 'sired':
                    data = self.samples[task][self.indices[index]]
                    data = (data - np.array(S2_MEAN)[:, None, None]) / np.array(S2_STD)[:, None, None]
                    sample = data[[11, 12], :, :]

                #Getting extrabands from sentinel2 data
                elif task == 'ebands':
                    data = self.samples[task][self.indices[index]]
                    data = (data - np.array(S2_MEAN)[:, None, None]) / np.array(S2_STD)[:, None, None]
                    sample = data[[7, 8], :, :]

                elif task == 'semseg':
                    old_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 255]
                    new_values = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]

                    data = self.samples[task][self.indices[index]]
                    
                    for old, new in zip(old_values, new_values):
                        data = np.where(data == old, new, data)

                    sample = data

                elif task == 'depth':                     
                    data = self.samples[task][self.indices[index]][[0], :, :]   
                    data = (data - DEPTH_MEAN[0]) / DEPTH_STD[0]  

                    sample = data
                
                else: sample = 'No sample'
                
                sample_dict[task] = sample

        if self.transform is not None:
            sample_dict = self.transform(sample_dict)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample_dict, target

    def __len__(self):
        return len(self.indices)
