import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from pytorch_grad_cam import GradCAM
from skimage.util import view_as_windows
import os


def get_significant_areas_coords(model, images, n=8):
   
    if torch.isnan(images).any() or torch.isinf(images).any():
        raise ValueError("")
    
    target_layers = [model.layer4[-1]]  
    cam = GradCAM(model=model, target_layers=target_layers)

    try:
        grayscale_cams = cam(input_tensor=images, targets=None, eigen_smooth=False)
    except RuntimeError as e:
        raise RuntimeError(e)
    # grayscale_cams = cam(input_tensor=images, targets=None, eigen_smooth=False)
    coords_list = []

    for idx, grayscale_cam in enumerate(grayscale_cams):

        if np.isnan(grayscale_cam).any() or np.isinf(grayscale_cam).any():
            raise ValueError(idx)
        window_shape = (n, n)
        windows = view_as_windows(grayscale_cam, window_shape)
        window_sums = windows.sum(axis=(2, 3))
        max_window_coords = np.unravel_index(window_sums.argmax(), window_sums.shape)
        coords_list.append((max_window_coords[0], max_window_coords[1]))
    
    return coords_list


def get_unsignificant_areas_coords(model, images, n=8):

    target_layers = [model.layer4[-1]]  
    cam = GradCAM(model=model, target_layers=target_layers)
    
    grayscale_cams = cam(input_tensor=images, targets=None, eigen_smooth=False)
    coords_list = []

    for idx, grayscale_cam in enumerate(grayscale_cams):
        window_shape = (n, n)
        windows = view_as_windows(grayscale_cam, window_shape)
        window_sums = windows.sum(axis=(2, 3))
        max_window_coords = np.unravel_index(window_sums.argmin(), window_sums.shape)
        coords_list.append((max_window_coords[0], max_window_coords[1]))
    
    return coords_list
