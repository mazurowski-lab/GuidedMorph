import numpy as np
import os
import matplotlib.pyplot as plt
import csv
import SimpleITK as sitk
import pandas as pd
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim
import random
import torch
import skimage.feature
import skimage.measure
from sklearn.preprocessing import minmax_scale
from skimage.filters import frangi
import cv2
import time


def save_visualization_slices(x, y, x_def, count, results_dir='results'):
    os.makedirs(results_dir, exist_ok=True)
    slices = [1/3, 1/2, 2/3]

    # Creating a 3x3 grid for subfigures, with each row representing one of x, y, x_def
    fig, axs = plt.subplots(3, 3, figsize=(15, 15)) 

    for col, slice_frac in enumerate(slices):
        slice_idx = int(x.shape[0] * slice_frac)  # Adjusting slice index based on the fraction

        # Plotting slices for x
        x_slice = x[slice_idx, :, :]
        axs[0, col].imshow(x_slice, cmap='gray')
        axs[0, col].set_title(f'X Slice {int(slice_frac*100)}%')
        
        # Plotting slices for y
        y_slice = y[slice_idx, :, :]
        axs[1, col].imshow(y_slice, cmap='gray')
        axs[1, col].set_title(f'Y Slice {int(slice_frac*100)}%')

        # Plotting slices for x_def
        x_def_slice = x_def[slice_idx, :, :]
        axs[2, col].imshow(x_def_slice, cmap='gray')
        axs[2, col].set_title(f'X_def Slice {int(slice_frac*100)}%')

    # Turn off axis visibility for all subplots
    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    # Save the figure with a unique name based on 'count'
    fig.savefig(os.path.join(results_dir, f"visual_{count}.png"))
    plt.close(fig)  # Close the figure to free memory

def analyze_and_save_tensor(tensor, variable_name):
    tensor = tensor.detach().cpu()
    tensor_sum = torch.sum(tensor).item()
    tensor_max = torch.max(tensor).item()
    tensor_min = torch.min(tensor).item()

    print(f"Sum of {variable_name}: {tensor_sum}")
    print(f"Max of {variable_name}: {tensor_max}")
    print(f"Min of {variable_name}: {tensor_min}")

    if tensor.dim() == 5:
        depth_dimension = 2
    elif tensor.dim() == 3:
        depth_dimension = 0
    else:
        raise ValueError("Tensor dimension not supported. Expected 3D or 5D tensor.")

    indices = [int(tensor.size(depth_dimension) * i / 5) for i in range(1, 5)]

    fig, axs = plt.subplots(1, len(indices), figsize=(20, 5))

    for i, index in enumerate(indices):
        if tensor.dim() == 5:
            tensor_slice = tensor[0, 0, index, :, :]
        elif tensor.dim() == 3:
            tensor_slice = tensor[index, :, :]
        
        tensor_slice_np = tensor_slice.numpy()
        
        ax = axs[i]
        cax = ax.imshow(tensor_slice_np, cmap='viridis')
        fig.colorbar(cax, ax=ax)
        ax.set_title(f"{variable_name} {i+1}/5")
    
    plt.tight_layout()
    plt.savefig(f"{variable_name}_combined_slices.png")
    plt.close(fig)

def divide_volume_with_overlap(volume, num_sub_vols_x=2, num_sub_vols_y=10, num_sub_vols_z=10):
    sub_volumes = []
    positions = []
    
    # Calculate dimensions of each sub-volume
    sub_vol_dz = int(volume.shape[0] // (num_sub_vols_z * 1/2 + 1/2))
    sub_vol_dy = int(volume.shape[1] // (num_sub_vols_y * 1/2 + 1/2))
    sub_vol_dx = int(volume.shape[2] // (num_sub_vols_x * 1/2 + 1/2))
    
    # Calculate step size (50% overlap)
    step_z = sub_vol_dz // 2
    step_y = sub_vol_dy // 2
    step_x = sub_vol_dx // 2
    
    # Iterate over each dimension to create sub-volumes
    for z in range(num_sub_vols_z):
        for y in range(num_sub_vols_y):
            for x in range(num_sub_vols_x):
                start_z = z * step_z
                start_y = y * step_y
                start_x = x * step_x
                end_z = min(start_z + sub_vol_dz, volume.shape[0])
                end_y = min(start_y + sub_vol_dy, volume.shape[1])
                end_x = min(start_x + sub_vol_dx, volume.shape[2])
                
                # Extract the sub-volume
                sub_volume = volume[start_z:end_z, start_y:end_y, start_x:end_x]
                sub_volumes.append(sub_volume)
                positions.append((start_z, end_z, start_y, end_y, start_x, end_x))

    return sub_volumes, positions, (sub_vol_dz, sub_vol_dy, sub_vol_dx)

def mask_volume_except_sub_volume(volume, position):
    start_z, end_z, start_y, end_y, start_x, end_x = position
    masked_volume = np.zeros_like(volume)
    masked_volume[start_z:end_z, start_y:end_y, start_x:end_x] = volume[start_z:end_z, start_y:end_y, start_x:end_x]
    return masked_volume

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def combine_metrics_to_score(features):
    # Convert the features dictionary values to a NumPy array
    features = np.array(list(features.values()))
    weights = np.array([0, 0, 0, 0, 1])
    
    # Calculate the weighted sum using the dot product
    combined_score = np.dot(weights, features)
    
    return combined_score

def fuse_positions(positions):
    # Initialize the bounding coordinates
    min_z = min(pos[0] for pos in positions)
    max_z = max(pos[1] for pos in positions)
    min_y = min(pos[2] for pos in positions)
    max_y = max(pos[3] for pos in positions)
    min_x = min(pos[4] for pos in positions)
    max_x = max(pos[5] for pos in positions)

    # Return the bounding rectangle (sub-volume)
    fused_position = (min_z, max_z, min_y, max_y, min_x, max_x)
    return fused_position

def calculate_clinical_metrics(sub_volumes, positions, n=10):
    scores = []
    for i, (sub_volume, position) in enumerate(zip(sub_volumes, positions)):
        slice_features = []
        
        properties = ['contrast', 'correlation', 'energy', 'homogeneity', 'entropy']
        noise_level = np.std(sub_volume)

        for slice_idx in range(sub_volume.shape[0]):
            slice_data = sub_volume[slice_idx, :, :]

            glcm = skimage.feature.graycomatrix(slice_data, distances=[30], angles=[0], levels=256, symmetric=True)
            haralick_features = []
            for prop in properties:
                if prop == 'entropy':
                    entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))
                    haralick_features.append(entropy)
                else:
                    feature_value = skimage.feature.graycoprops(glcm, prop)[0, 0]
                    haralick_features.append(feature_value)
            
            slice_features.append(haralick_features)
        
        mean_features = np.mean(slice_features, axis=0)

        features = dict(zip(properties, mean_features))
        features['noise'] = noise_level
        features['sub_volume'] = sub_volume
        features['position'] = position
        scores.append(features)

    # Sort the scores based on correlation
    sorted_scores = sorted(scores, key=lambda x: x['correlation'])

    # Get the top n sub-volumes with the lowest correlation
    top_n_scores = sorted_scores[:n]

    top_n_positions = [score['position'] for score in top_n_scores]
    fused_position = fuse_positions(top_n_positions)

    return top_n_scores, fused_position


def calculate_entropy(sub_volume, sigma = 2):
    data = smoothed_volume.flatten()
    
    # Calculate histogram of data with bins ranging from 0 to 1
    histogram, _ = np.histogram(data, bins=256, range=(0, 1))
    
    # Normalize the histogram to sum to 1 (probability distribution)
    histogram_normalized = histogram / histogram.sum()
    
    # Calculate entropy using the normalized histogram
    e = entropy(histogram_normalized)
    
    # Calculate the mean intensity of the sub_volume
    mean_intensity = data.mean()
    return e * 10 + mean_intensity

def find_neighboring_sub_volumes(volume, position):
    start_z, end_z, start_y, end_y, start_x, end_x = position
    step_z = end_z - start_z
    step_y = end_y - start_y
    step_x = end_x - start_x
    neighboring_sub_volumes = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                new_start_z = max(start_z + dz * step_z // 8, 0)
                new_end_z = min(new_start_z + step_z, volume.shape[0])
                new_start_y = max(start_y + dy * step_y // 8, 0)
                new_end_y = min(new_start_y + step_y, volume.shape[1])
                new_start_x = max(start_x + dx * step_x // 8, 0)
                new_end_x = min(new_start_x + step_x, volume.shape[2])
                if new_end_z > new_start_z and new_end_y > new_start_y and new_end_x > new_start_x:  # Check if the volume is valid
                    sub_volume = volume[new_start_z:new_end_z, new_start_y:new_end_y, new_start_x:new_end_x]
                    neighboring_sub_volumes.append((sub_volume, (new_start_z, new_end_z, new_start_y, new_end_y, new_start_x, new_end_x)))
    return neighboring_sub_volumes

def find_most_similar_sub_volume(sub_volume1, neighboring_sub_volumes):
    highest_ssim = -1
    most_similar_sub_volume = None
    most_similar_position = None
    for sub_volume2, position in neighboring_sub_volumes:
        mid_slice1 = sub_volume1[sub_volume1.shape[0] // 2]
        mid_slice2 = sub_volume2[sub_volume2.shape[0] // 2]
        
        # Check if dimensions match, if not, pad the smaller image
        if mid_slice1.shape != mid_slice2.shape:
            # Assuming mid_slice1 is the reference, pad mid_slice2
            padded_mid_slice2 = np.zeros(mid_slice1.shape)
            min_z, min_y = min(mid_slice1.shape[0], mid_slice2.shape[0]), min(mid_slice1.shape[1], mid_slice2.shape[1])
            padded_mid_slice2[:min_z, :min_y] = mid_slice2[:min_z, :min_y]
            mid_slice2 = padded_mid_slice2  # Use the padded image for SSIM calculation
        
        current_ssim = ssim(mid_slice1, mid_slice2)
        if current_ssim > highest_ssim:
            highest_ssim = current_ssim
            most_similar_sub_volume = sub_volume2
            most_similar_position = position
    return most_similar_sub_volume, most_similar_position

def ensure_5d(volume):
    if volume.ndim == 3:
        volume = volume[np.newaxis, np.newaxis, ...]  # Add batch and channel dimensions
    return volume

def visualize_fused_sub_volume(volume, fused_position):
    min_z, max_z, min_y, max_y, min_x, max_x = fused_position
    fused_sub_volume = volume[min_z:max_z, min_y:max_y, min_x:max_x]
    return fused_sub_volume

def frangi_filter(volume_data):
    volume_data = (volume_data - np.min(volume_data)) / (np.max(volume_data) - np.min(volume_data)) * 255
    frangi_volume = np.zeros_like(volume_data)
    for i in range(volume_data.shape[0]):
        frangi_volume[i, :, :] = frangi(volume_data[i, :, :], sigmas=1, black_ridges = False)
    
    # Normalize the Frangi filter output
    frangi_volume = (frangi_volume - np.min(frangi_volume)) / (np.max(frangi_volume) - np.min(frangi_volume)) * 255
    threshold = np.percentile(frangi_volume[frangi_volume > 0], 90)
    frangi_volume[frangi_volume < threshold] = 0
    return frangi_volume

def feature_extract(volume1_tensor, volume2_tensor):
    volume1 = volume1_tensor.detach().cpu().numpy()
    volume2 = volume2_tensor.detach().cpu().numpy()
    
    # Function ensure_5d is assumed to be defined elsewhere
    volume1 = ensure_5d(volume1)
    volume2 = ensure_5d(volume2)

    output_volume1 = np.zeros_like(volume1)
    output_volume2 = np.zeros_like(volume2)
    
    # Assuming functions divide_volume_with_overlap, find_most_informative_sub_volume,
    # find_neighboring_sub_volumes, and mask_volume_except_sub_volume are defined elsewhere
    
    for batch_idx in range(volume1.shape[0]):
        for channel_idx in range(volume1.shape[1]):
            batch_channel_volume1 = volume1[batch_idx, channel_idx].squeeze()
            batch_channel_volume2 = volume2[batch_idx, channel_idx].squeeze()
            
            # Process volume
            mask1 = frangi_filter(batch_channel_volume1)
            mask2 = frangi_filter(batch_channel_volume2)
            mask1 = cv2.dilate(mask1.astype(np.uint8), np.ones((10, 10), np.uint8), iterations=1)
            mask2 = cv2.dilate(mask2.astype(np.uint8), np.ones((10, 10), np.uint8), iterations=1)
            masked_volume1 = batch_channel_volume1 * (mask1 > 0)
            masked_volume2 = batch_channel_volume2 * (mask2 > 0)
            
            # Corrected: Move these lines inside the loop to ensure they execute for each batch and channel
            output_volume1[batch_idx, channel_idx, :, :, :] = np.expand_dims(masked_volume1, axis=0)  # Re-add the squeezed dimension
            output_volume2[batch_idx, channel_idx, :, :, :] = np.expand_dims(masked_volume2, axis=0)

    # Convert the output numpy arrays back to torch tensors with the appropriate device and dtype
    output_volume1_tensor = torch.from_numpy(output_volume1).to(device=volume1_tensor.device, dtype=volume1_tensor.dtype)
    output_volume2_tensor = torch.from_numpy(output_volume2).to(device=volume2_tensor.device, dtype=volume2_tensor.dtype)
    
    return output_volume1_tensor, output_volume2_tensor

def feature_extract_torch(volume1_tensor, volume2_tensor):
    device = volume1_tensor.device
    
    def frangi_torch(volume, sigmas=[1], alpha=0.5, beta=0.5, black_ridges=True):
        """
        Frangi vesselness filter implementation in PyTorch.
        Args:
            volume: Input tensor of shape [B, C, D, H, W]
            sigmas: List of scales to analyze
            alpha, beta: Frangi vesselness parameters
            black_ridges: If True, detect black ridges; if False, detect white ridges
        """        
        if not black_ridges:
            volume = -volume
            
        B, C, D, H, W = volume.shape
        volume_reshaped = volume.view(B * C * D, 1, H, W)
        filtered_max = torch.zeros_like(volume_reshaped)
        
        for sigma in sigmas:
            # Create Gaussian kernel
            kernel_size = int(4 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            coords = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2
            gaussian = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            gaussian = gaussian / gaussian.sum()
            kernel_2d = gaussian.unsqueeze(0) * gaussian.unsqueeze(1)
            kernel = kernel_2d.unsqueeze(0).unsqueeze(0)
            
            # Compute Hessian
            padded = torch.nn.functional.pad(volume_reshaped, (kernel_size//2,)*4, mode='reflect')
            smoothed = torch.nn.functional.conv2d(padded, kernel, padding=0)
            
            # Calculate Hessian elements
            padded = torch.nn.functional.pad(smoothed, (1, 1, 1, 1), mode='reflect')
            dxx = padded[:, :, 1:-1, 2:] - 2 * padded[:, :, 1:-1, 1:-1] + padded[:, :, 1:-1, :-2]
            dyy = padded[:, :, 2:, 1:-1] - 2 * padded[:, :, 1:-1, 1:-1] + padded[:, :, :-2, 1:-1]
            dxy = (padded[:, :, 2:, 2:] - padded[:, :, 2:, :-2] - 
                  padded[:, :, :-2, 2:] + padded[:, :, :-2, :-2]) / 4
            
            # Scale Hessian elements
            dxx = sigma ** 2 * dxx
            dyy = sigma ** 2 * dyy
            dxy = sigma ** 2 * dxy
            
            # Compute eigenvalues
            trace = dxx + dyy
            det = dxx * dyy - dxy * dxy
            sqrt_term = torch.sqrt(trace * trace - 4 * det + 1e-10)
            
            # Calculate initial eigenvalues
            lambda_1 = (trace + sqrt_term) / 2
            lambda_2 = (trace - sqrt_term) / 2
            
            # Sort eigenvalues per pixel
            lambdas = torch.stack([lambda_1, lambda_2], dim=0)
            lambdas_sorted, _ = torch.sort(lambdas, dim=0)
            
            # Get sorted eigenvalues
            lambda1 = lambdas_sorted[0]  # smaller eigenvalue
            lambda2 = lambdas_sorted[1]  # larger eigenvalue
            
            r_b = torch.abs(lambda1) / (torch.abs(lambda2) + 1e-10)
            s = torch.sqrt(lambda1**2 + lambda2**2)
            
            gamma = s.max() / 2
            gamma = torch.where(gamma == 0, torch.ones_like(gamma), gamma)
            
            vesselness = torch.exp(-(r_b**2) / (2 * beta**2))
            vesselness *= (1 - torch.exp(-(s**2) / (2 * gamma**2)))
            vesselness[lambda2 < 0] = 0  # Suppress dark to bright transitions
            
            filtered_max = torch.maximum(filtered_max, vesselness)
        
        result = filtered_max.view(B, C, D, H, W)
        # Debug: Print final result stats
        # print(f"Final result stats - min: {result.min():.4f}, max: {result.max():.4f}, mean: {result.mean():.4f}")
        return result

    sigmas = [1]  # Multiple scales
    mask1 = frangi_torch(volume1_tensor, sigmas=sigmas, black_ridges=False)  # Changed to False for bright vessels
    mask2 = frangi_torch(volume2_tensor, sigmas=sigmas, black_ridges=False)  # Changed to False for bright vessels

    threshold = 0.1  # Adjusted threshold
    mask1_binary = (mask1 > threshold).float()
    mask2_binary = (mask2 > threshold).float()

    kernel_size = 5
    padding = kernel_size // 2
    
    B, C, D, H, W = mask1_binary.shape
    mask1_reshaped = mask1_binary.view(B * C * D, 1, H, W)
    mask2_reshaped = mask2_binary.view(B * C * D, 1, H, W)
    
    mask1_dilated = torch.nn.functional.max_pool2d(
        mask1_reshaped, kernel_size=kernel_size, stride=1, padding=padding
    ).view(B, C, D, H, W)
    
    mask2_dilated = torch.nn.functional.max_pool2d(
        mask2_reshaped, kernel_size=kernel_size, stride=1, padding=padding
    ).view(B, C, D, H, W)

    return volume1_tensor * mask1_dilated, volume2_tensor * mask2_dilated

def feature_extract_torch_optimized(volume1_tensor, volume2_tensor):
    """
    Optimized feature extraction with batch processing.
    Args:
        volume1_tensor, volume2_tensor: Input tensors of shape [B, C, D, H, W]
    """
    device = volume1_tensor.device
    
    @torch.no_grad()  # Disable gradient computation for inference
    def frangi_torch_optimized(volume, sigmas=[1], black_ridges=False):
        if not black_ridges:
            volume = -volume
            
        B, C, D, H, W = volume.shape
        
        # Process all batches at once
        volume_reshaped = volume.reshape(-1, 1, H, W)
        filtered_max = torch.zeros_like(volume_reshaped)
        
        # Pre-compute gaussian kernels for all sigmas
        kernels = {}
        for sigma in sigmas:
            kernel_size = int(4 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            coords = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2
            gaussian = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            gaussian = gaussian / gaussian.sum()
            kernel_2d = gaussian.unsqueeze(0) * gaussian.unsqueeze(1)
            kernels[sigma] = kernel_2d.unsqueeze(0).unsqueeze(0)

        for sigma in sigmas:
            kernel = kernels[sigma]
            kernel_size = kernel.shape[-1]
            
            # Compute Hessian efficiently
            padded = torch.nn.functional.pad(volume_reshaped, (kernel_size//2,)*4, mode='reflect')
            smoothed = torch.nn.functional.conv2d(padded, kernel.to(device), padding=0)
            
            # Calculate Hessian elements in batch
            padded = torch.nn.functional.pad(smoothed, (1, 1, 1, 1), mode='reflect')
            dxx = padded[:, :, 1:-1, 2:] - 2 * padded[:, :, 1:-1, 1:-1] + padded[:, :, 1:-1, :-2]
            dyy = padded[:, :, 2:, 1:-1] - 2 * padded[:, :, 1:-1, 1:-1] + padded[:, :, :-2, 1:-1]
            dxy = (padded[:, :, 2:, 2:] - padded[:, :, 2:, :-2] - 
                  padded[:, :, :-2, 2:] + padded[:, :, :-2, :-2]) / 4
            
            # Scale Hessian elements
            dxx = sigma ** 2 * dxx
            dyy = sigma ** 2 * dyy
            dxy = sigma ** 2 * dxy
            
            # Compute eigenvalues efficiently
            trace = dxx + dyy
            det = dxx * dyy - dxy * dxy
            sqrt_term = torch.sqrt(trace * trace - 4 * det + 1e-10)
            
            lambda1 = (trace - sqrt_term) / 2  # smaller eigenvalue
            lambda2 = (trace + sqrt_term) / 2  # larger eigenvalue
            
            # Compute vesselness
            r_b = torch.abs(lambda1) / (torch.abs(lambda2) + 1e-10)
            s = torch.sqrt(lambda1**2 + lambda2**2)
            
            gamma = 0.5 * s.max()
            gamma = torch.where(gamma == 0, torch.ones_like(gamma), gamma)
            
            beta = 0.5
            vesselness = torch.exp(-(r_b**2) / (2 * beta**2))
            vesselness *= (1 - torch.exp(-(s**2) / (2 * gamma**2)))
            vesselness[lambda2 < 0] = 0
            
            filtered_max = torch.maximum(filtered_max, vesselness)
        
        # Reshape back to original dimensions
        return filtered_max.view(B, C, D, H, W)

    # Process both volumes together
    combined_volume = torch.cat([volume1_tensor, volume2_tensor], dim=0)
    mask = frangi_torch_optimized(combined_volume, sigmas=[1], black_ridges=False)
    
    # Split results back
    mask1, mask2 = torch.split(mask, [volume1_tensor.shape[0], volume2_tensor.shape[0]], dim=0)

    # Apply threshold and create binary masks
    threshold = 0.1
    mask1_binary = (mask1 > threshold).float()
    mask2_binary = (mask2 > threshold).float()

    # Efficient dilation using max_pool2d
    kernel_size = 5
    padding = kernel_size // 2
    
    B, C, D, H, W = mask1_binary.shape
    masks_reshaped = torch.cat([
        mask1_binary.view(B * C * D, 1, H, W),
        mask2_binary.view(B * C * D, 1, H, W)
    ], dim=0)
    
    masks_dilated = torch.nn.functional.max_pool2d(
        masks_reshaped, 
        kernel_size=kernel_size, 
        stride=1, 
        padding=padding
    )
    
    # Reshape back to original dimensions
    mask1_dilated, mask2_dilated = torch.split(
        masks_dilated.view(-1, C, D, H, W),
        [B, B],
        dim=0
    )

    return (volume1_tensor * mask1_dilated, 
            volume2_tensor * mask2_dilated)

def profile_feature_extract():
    """Compare performance between original and optimized versions"""
    # Create sample data
    batch_size = 2
    sample_data = torch.randn(batch_size, 1, 128, 256, 256, device='cuda')
    
    print("\nTesting original feature_extract_torch:")
    # Warm up
    for _ in range(3):
        feature_extract_torch(sample_data, sample_data)
    
    # Profile original
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        feature_extract_torch(sample_data, sample_data)
        torch.cuda.synchronize()
    original_time = (time.time() - start) / 10
    
    print(f"Average time per iteration (original): {original_time:.3f}s")
    
    print("\nTesting optimized feature_extract_torch_optimized:")
    # Warm up
    for _ in range(3):
        feature_extract_torch_optimized(sample_data, sample_data)
    
    # Profile optimized
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        feature_extract_torch_optimized(sample_data, sample_data)
        torch.cuda.synchronize()
    optimized_time = (time.time() - start) / 10
    
    print(f"Average time per iteration (optimized): {optimized_time:.3f}s")
    print(f"Speedup: {original_time/optimized_time:.2f}x\n")