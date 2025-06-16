import os
import torch
import numpy as np
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def canny_edge_detector(image, low_threshold=20, high_threshold=100, kernel_size=7, binary=False, binary_thresh=50):
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
 
    magnitude = cv2.Canny(blurred_image, low_threshold, high_threshold)
    
    magnitude = normalize_mri_percentile_to_255(magnitude)
    
    if binary:
        magnitude[magnitude > binary_thresh] = 255
        magnitude[magnitude != 255] = 0
 
    return magnitude

def normalize_mri_percentile_to_255(mri_data, lower_percentile=0, upper_percentile=100):
    lower_bound = np.percentile(mri_data, lower_percentile)
    upper_bound = np.percentile(mri_data, upper_percentile)
    mri_data_normalized = (mri_data - lower_bound) / (upper_bound - lower_bound)
    mri_data_scaled = mri_data_normalized * 255
    mri_data_scaled = np.round(mri_data_scaled).astype(np.uint8)

    return mri_data_scaled
    
def process_mri_volume(volume, **kwargs):
    processed_slices = []
    for slice_ in volume:
        processed_slice = canny_edge_detector(slice_, **kwargs)
        processed_slices.append(processed_slice)
    return np.array(processed_slices)

def extract_feature(numpy_array, seg_numpy_array, mode="tissue"):
    if isinstance(numpy_array, torch.Tensor):
        numpy_array = numpy_array.to('cpu').detach().numpy()
    if isinstance(seg_numpy_array, torch.Tensor):
        seg_numpy_array = seg_numpy_array.to('cpu').detach().numpy()
    if numpy_array.shape != seg_numpy_array.shape:
        raise ValueError("numpy_array and seg_numpy_array must have the same shape")
    
    mask = seg_numpy_array == 0
    
    if mode == "breast":
        enhancement_factor = 10
        numpy_array[mask] = (numpy_array[mask] * enhancement_factor).astype('float64')
    
    elif mode == "tissue":
        seg_numpy_array = cv2.dilate(seg_numpy_array, np.ones((10, 10), np.uint8), iterations=1)
        mask = seg_numpy_array == 0
        numpy_array[mask] = 0

    else:
        raise ValueError("mode must be 'tissue' or 'breast'")
    return numpy_array

def extract_feature_ssim(numpy_array, seg_numpy_array, mode="tissue"):
    if isinstance(numpy_array, torch.Tensor):
        numpy_array = numpy_array.to('cpu').detach().numpy()
    if isinstance(seg_numpy_array, torch.Tensor):
        seg_numpy_array = seg_numpy_array.to('cpu').detach().numpy()
    if numpy_array.shape != seg_numpy_array.shape:
        raise ValueError("numpy_array and seg_numpy_array must have the same shape")
    
    mask = seg_numpy_array == 0
    
    if mode == "breast":
        enhancement_factor = 10
        numpy_array[mask] = (numpy_array[mask] * enhancement_factor).astype('float64')
    
    elif mode == "tissue":
        mask = seg_numpy_array == 0
        numpy_array[mask] = 0

    else:
        raise ValueError("mode must be 'tissue' or 'breast'")
    return numpy_array

def get_edge_ssim(tensor_array, tensor_array_seg):
    if not isinstance(tensor_array, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")

    if not isinstance(tensor_array_seg, torch.Tensor):
        tensor_array_seg = torch.from_numpy(tensor_array_seg)
        
    batch, channel, depth, width, height = tensor_array.shape
    edges_array = torch.zeros_like(tensor_array)

    if tensor_array.dim() == 3:
        tensor_array = tensor_array.unsqueeze(0).unsqueeze(0)
    if tensor_array_seg.dim() == 3:
        tensor_array_seg = tensor_array_seg.unsqueeze(0).unsqueeze(0) 
        
    for b in range(batch):
        for c in range(channel):
            slice_ = tensor_array[b, c, :, :, :]
            slice_seg_ = tensor_array_seg[b, c, :, :, :]
            slice_ = extract_feature_ssim(slice_, slice_seg_, mode = "tissue")
            slice_ = normalize_mri_percentile_to_255(slice_)
            edges_array[b, c, :, :, :] =  torch.from_numpy(slice_).to(torch.uint8)
    
    return edges_array

def get_edge(tensor_array, tensor_array_seg):
    if not isinstance(tensor_array, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")

    if not isinstance(tensor_array_seg, torch.Tensor):
        tensor_array_seg = torch.from_numpy(tensor_array_seg)
        
    batch, channel, depth, width, height = tensor_array.shape
    edges_array = torch.zeros_like(tensor_array)

    if tensor_array.dim() == 3:
        tensor_array = tensor_array.unsqueeze(0).unsqueeze(0)
    if tensor_array_seg.dim() == 3:
        tensor_array_seg = tensor_array_seg.unsqueeze(0).unsqueeze(0) 
        
    for b in range(batch):
        for c in range(channel):
            slice_ = tensor_array[b, c, :, :, :]
            slice_seg_ = tensor_array_seg[b, c, :, :, :]
            slice_ = extract_feature(slice_, slice_seg_, mode = "tissue")
            slice_ = normalize_mri_percentile_to_255(slice_)
            edges_array[b, c, :, :, :] =  torch.from_numpy(slice_).to(torch.uint8)
    
    return edges_array

def apply_gaussian_filter(volume, sigma=1):
    volume_np = volume.detach().cpu().numpy()
    
    for c in range(volume_np.shape[1]):
        for z in range(volume_np.shape[2]):
            volume_np[0, c, z, :, :] = gaussian_filter(volume_np[0, c, z, :, :], sigma=sigma)
    
    smoothed_volume = torch.from_numpy(volume_np).to(volume.device)
    return smoothed_volume