"""
FILE: data.py
API: VOLUME PROVIDER (DATA ABSTRACTION)
---------------------------------------
Role: 
    The 'Source of Truth'. Abstractly handles loading medical data (NIfTI) 
    or generating synthetic phantoms (Cubes/Spheres).

Main Classes:
    - BaseVolumeProvider: Abstract interface for all volumetric data.
    - SyntheticVolumeProvider: Generates procedural 3D shapes for debugging.
    - NiftiVolumeProvider: Loads and normalizes MRI/CT brain files (.nii.gz).

Metadata Contract:
    Every slice must return a dict with:
    - 'z_center': The normalized [-1, 1] position of the slice.
    - 'z_range':  (z_start, z_end) defining the physical thickness for Ray Sampling.
    - 'spacing':  The distance between slices.
"""

import numpy as np
import torch
import nibabel as nib

class BaseVolumeProvider:
    """Abstract Interface for volumetric data sources."""
    def get_slice(self, axis='z', index=0):
        """Returns (2D Numpy Array, Metadata Dict)"""
        raise NotImplementedError

    def get_total_slices(self, axis='z'):
        """Returns the number of available slices along an axis."""
        raise NotImplementedError

class SyntheticVolumeProvider(BaseVolumeProvider):
    """
    Generates a 3D volume procedurally (e.g., a sphere inside a cube).
    Use this to verify your Trainer/Sampler math without real-world noise.
    """
    def __init__(self, resolution=64):
        self.res = resolution
        # Create a simple 3D sphere: (x^2 + y^2 + z^2 < r^2)
        grid_z, grid_y, grid_x = np.mgrid[-1:1:complex(resolution), 
                                          -1:1:complex(resolution), 
                                          -1:1:complex(resolution)]
        dist = np.sqrt(grid_x**2 + grid_y**2 + grid_z**2)
        self.volume = (dist < 0.5).astype(np.float32) # Sphere with radius 0.5

    def get_total_slices(self, axis='z'):
        return self.res

    def get_slice(self, axis='z', index=0):
        # Slice the volume along the chosen axis
        if axis == 'z': slice_data = self.volume[index, :, :]
        elif axis == 'y': slice_data = self.volume[:, index, :]
        else: slice_data = self.volume[:, :, index]

        # Calculate normalized metadata
        thickness = 2.0 / self.res
        z_center = -1.0 + (index / (self.res - 1)) * 2.0
        
        metadata = {
            'z_center': z_center,
            'z_range': (z_center - thickness/2, z_center + thickness/2),
            'thickness': thickness
        }
        return slice_data, metadata

class NiftiVolumeProvider(BaseVolumeProvider):
    """
    Loads real neuro-imaging data. Handles normalization and affine mapping.
    """
    def __init__(self, file_path):
        # Load NIfTI file (NIfTI stores data in [W, H, D] or [H, W, D])
        img = nib.load(file_path)
        data = img.get_fdata().astype(np.float32)

        # 1. Intensity Normalization [0, 1]
        self.volume = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
        
        # 2. Get geometric info from the NIfTI header (Voxel Spacing)
        # pixdim[1:4] are usually (dx, dy, dz) in mm
        self.spacing = img.header.get_zooms()
        self.shape = self.volume.shape

    def get_total_slices(self, axis='z'):
        # Usually, axis 'z' is the 3rd dimension in medical data
        return self.shape[2]

    def get_slice(self, axis='z', index=0):
        # Assuming Z is the 3rd dimension
        slice_data = self.volume[:, :, index]
        
        # Calculate normalized Z [-1, 1]
        depth = self.shape[2]
        z_center = -1.0 + (index / (depth - 1)) * 2.0
        
        # thickness is based on normalized space relative to total depth
        normalized_thickness = 2.0 / depth 
        
        metadata = {
            'z_center': z_center,
            'z_range': (z_center - normalized_thickness/2, z_center + normalized_thickness/2),
            'thickness': normalized_thickness,
            'real_spacing_mm': self.spacing[2]
        }
        return slice_data, metadata
    
class ManualVolumeProvider(BaseVolumeProvider):
    """
    Allows the user to provide a list of hand-drawn 2D slices.
    Ideal for the 'Hollow Cube' test.
    """
    def __init__(self, list_of_slices):
        # list_of_slices: List of 2D Numpy arrays [H, W]
        self.slices = [s.astype(np.float32) for s in list_of_slices]
        self.num_slices = len(self.slices)
        self.h, self.w = self.slices[0].shape

    def get_total_slices(self, axis='z'):
        return self.num_slices

    def get_slice(self, axis='z', index=0):
        slice_data = self.slices[index]
        
        # Space the slices evenly in [-1, 1]
        z_center = -1.0 + (index / (self.num_slices - 1)) * 2.0 if self.num_slices > 1 else 0.0
        thickness = 2.0 / self.num_slices
        
        metadata = {
            'z_center': z_center,
            'z_range': (z_center - thickness/2, z_center + thickness/2),
            'thickness': thickness
        }
        return slice_data, metadata
    
class PhantomProvider(BaseVolumeProvider):
    """
    Generates a 3D 'Shepp-Logan' style phantom.
    Consists of nested ellipsoids representing a skull, brain, and ventricles.
    """
    def __init__(self, res=64):
        self.res = res
        # 1. Create a 3D coordinate grid [-1, 1]
        z, y, x = np.ogrid[-1:1:res*1j, -1:1:res*1j, -1:1:res*1j]
        
        # 2. Build the volume (nested ellipsoids)
        # Intensity values: Skull (0.2), Brain (0.5), Ventricles (0.0)
        
        # Outer Skull
        self.volume = (x**2 + (y/1.2)**2 + (z/1.4)**2 < 0.9).astype(np.float32) * 0.2
        
        # Brain Matter
        self.volume += (x**2 + (y/1.1)**2 + (z/1.3)**2 < 0.7).astype(np.float32) * 0.4
        
        # Left Ventricle (Hole)
        self.volume -= ((x-0.3)**2 + (y-0.2)**2 + (z)**2 < 0.05).astype(np.float32) * 0.6
        
        # Right Ventricle (Hole)
        self.volume -= ((x+0.3)**2 + (y-0.2)**2 + (z)**2 < 0.05).astype(np.float32) * 0.6
        
        # Add a "Tumor" or specific structure to test high-detail detection
        self.volume += ((x)**2 + (y+0.4)**2 + (z+0.2)**2 < 0.02).astype(np.float32) * 0.3

        # Clip to ensure intensities stay in [0, 1]
        self.volume = np.clip(self.volume, 0, 1)

    def get_total_slices(self, axis='z'):
        return self.res

    def get_slice(self, axis='z', index=0):
        """Returns a 2D slice and metadata for the trainer."""
        # Assume Z-axis slicing for standard training
        slice_data = self.volume[index, :, :]
        
        # Calculate normalized position
        z_pos = -1.0 + (index / (self.res - 1)) * 2.0
        thickness = 2.0 / self.res
        
        metadata = {
            'z_center': z_pos,
            'z_range': (z_pos - thickness/2, z_pos + thickness/2),
            'thickness': thickness
        }
        return slice_data, metadata