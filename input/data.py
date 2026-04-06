"""
FILE: data.py
API: VOLUME PROVIDER (DATA ABSTRACTION)
---------------------------------------
Note: Updated to 'GPU-Resident' mode. Data is moved to the target device (MPS/CUDA)
once during initialization to maximize training throughput.
"""

import numpy as np
import torch
import nibabel as nib
from typing import Dict, Tuple


class BaseVolumeProvider:
    """Abstract Interface for volumetric data sources."""
    def get_slice(self, axis='z', index=0):
        """Returns (2D Array/Tensor, Metadata Dict)"""
        raise NotImplementedError

    def get_total_slices(self, axis='z'):
        """Returns the number of available slices along an axis."""
        raise NotImplementedError

class NiftiVolumeProvider(BaseVolumeProvider):
    """
    Loads real medical NIfTI data. 
    Handles orientation standardization, robust normalization, 
    and affine-aware coordinate mapping.
    """
    def __init__(self, file_path: str, device: str = "cpu"):
        self.device = device
        
        # 1. Load and force to Canonical Orientation (RAS+)
        img = nib.load(file_path)
        img = nib.as_closest_canonical(img)
        data = img.get_fdata().astype(np.float32)

        # 2. Robust Intensity Normalization
        p_low = np.percentile(data, 0.5)
        p_high = np.percentile(data, 99.5)
        data = np.clip(data, p_low, p_high)
        data = (data - p_low) / (p_high - p_low + 1e-8)
        data[data < 0.02] = 0
        
        # 3. GEOMETRIC METADATA
        self.spacing = img.header.get_zooms()
        self.shape = data.shape
        
        # --- THE PERFORMANCE BOOST ---
        # Move the entire 3D volume to the GPU once.
        self.volume_tensor = torch.from_numpy(data).to(self.device)
        print(f"Loaded NIfTI to {self.device}: {self.shape}")

    def get_total_slices(self, axis: str = 'z') -> int:
        return self.shape[2]

    def get_slice(self, axis: str = 'z', index: int = 0) -> Tuple[torch.Tensor, Dict]:
        # Slicing is now done directly on the GPU memory
        slice_data = self.volume_tensor[:, :, index]
        
        depth = self.shape[2]
        z_pos = -1.0 + (index / (depth - 1)) * 2.0
        normalized_thickness = 2.0 / depth 
        
        metadata = {
            'z_center': z_pos,
            'z_range': (z_pos - normalized_thickness/2, z_pos + normalized_thickness/2),
            'thickness': normalized_thickness,
            'mm_spacing': self.spacing[2]
        }
        return slice_data, metadata

class SyntheticVolumeProvider(BaseVolumeProvider):
    def __init__(self, resolution=64, device="cpu"):
        self.res = resolution
        self.device = device
        grid_z, grid_y, grid_x = np.mgrid[-1:1:complex(resolution), 
                                          -1:1:complex(resolution), 
                                          -1:1:complex(resolution)]
        dist = np.sqrt(grid_x**2 + grid_y**2 + grid_z**2)
        data = (dist < 0.5).astype(np.float32)
        self.volume_tensor = torch.from_numpy(data).to(self.device)

    def get_total_slices(self, axis='z'): return self.res

    def get_slice(self, axis='z', index=0):
        slice_data = self.volume_tensor[index, :, :]
        z_center = -1.0 + (index / (self.res - 1)) * 2.0
        thickness = 2.0 / self.res
        return slice_data, {'z_center': z_center, 'z_range': (z_center-thickness/2, z_center+thickness/2), 'thickness': thickness}

class ManualVolumeProvider(BaseVolumeProvider):
    def __init__(self, list_of_slices, device="cpu"):
        self.device = device
        # Convert list of slices to a single 3D GPU Tensor
        stacked = np.stack(list_of_slices, axis=-1).astype(np.float32)
        self.volume_tensor = torch.from_numpy(stacked).to(self.device)
        self.num_slices = self.volume_tensor.shape[2]

    def get_total_slices(self, axis='z'): return self.num_slices

    def get_slice(self, axis='z', index=0):
        slice_data = self.volume_tensor[:, :, index]
        z_center = -1.0 + (index / (self.num_slices - 1)) * 2.0 if self.num_slices > 1 else 0.0
        thickness = 2.0 / self.num_slices
        return slice_data, {'z_center': z_center, 'z_range': (z_center-thickness/2, z_center+thickness/2), 'thickness': thickness}

class PhantomProvider(BaseVolumeProvider):
    def __init__(self, res=64, device="mps"):
        self.res = res
        self.device = device
        z, y, x = np.ogrid[-1:1:res*1j, -1:1:res*1j, -1:1:res*1j]
        volume = (x**2 + (y/1.2)**2 + (z/1.4)**2 < 0.9).astype(np.float32) * 0.2
        volume += (x**2 + (y/1.1)**2 + (z/1.3)**2 < 0.7).astype(np.float32) * 0.4
        volume -= ((x-0.3)**2 + (y-0.2)**2 + (z)**2 < 0.05).astype(np.float32) * 0.6
        volume -= ((x+0.3)**2 + (y-0.2)**2 + (z)**2 < 0.05).astype(np.float32) * 0.6
        volume += ((x)**2 + (y+0.4)**2 + (z+0.2)**2 < 0.02).astype(np.float32) * 0.3
        volume = np.clip(volume, 0, 1)
        self.volume_tensor = torch.from_numpy(volume).to(self.device)

    def get_total_slices(self, axis='z'): return self.res

    def get_slice(self, axis='z', index=0):
        slice_data = self.volume_tensor[index, :, :]
        z_pos = -1.0 + (index / (self.res - 1)) * 2.0
        thickness = 2.0 / self.res
        return slice_data, {'z_center': z_pos, 'z_range': (z_pos-thickness/2, z_pos+thickness/2), 'thickness': thickness}