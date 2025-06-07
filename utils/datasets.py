import os
import torch
import random
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from utils import create_transform
from .data_constants import IMAGE_TASKS
from torchvision import datasets, transforms
from .dataset_folder import CustomMultiTaskDatasetFolder

class DataAugmentationForMultiMAE(object):
    def __init__(self, args):
        self.input_size = args.input_size
        self.hflip = args.hflip

    def __call__(self, task_dict):
        flip = random.random() < self.hflip
        
        for task in task_dict:
            task_dict[task] = TF.to_tensor(task_dict[task]).permute(1,2,0).to(torch.float)

        # Flip all tasks randomly, but consistently for all tasks
        for task in task_dict:
            if task not in IMAGE_TASKS: continue

            task_dict[task] = F.interpolate(task_dict[task].unsqueeze(0), 
                                    size=(self.input_size, self.input_size), 
                                    mode='bilinear')
            
            if flip: task_dict[task] = TF.hflip(task_dict[task])
                
        for task in task_dict:
            if task in ['depth']: 
                img = task_dict[task].squeeze(0)
            elif task in ['rgb', 'ired', 'sired', 'ebands']: 
                img = task_dict[task].squeeze(0)
            elif task in ['semseg']:
                scale_factor = 0.25
                img = F.interpolate(task_dict[task], 
                                size=(int(self.input_size*scale_factor), int(self.input_size*scale_factor)), 
                                mode='bilinear').to(torch.long).squeeze(0).squeeze(0)
            task_dict[task] = img
        
        return task_dict

    def __repr__(self):
        repr = "(DataAugmentationForMultiMAE,\n"
        repr += ")"
        return repr


def build_multimae_custom_pretraining_dataset(args):
    transform = DataAugmentationForMultiMAE(args)
    return CustomMultiTaskDatasetFolder(args.data_path, args.all_domains, classes_def=args.classes_def, 
                                        splits_path=args.splits_path ,transform=transform)
