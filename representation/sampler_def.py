"""
FILE: sampler_def.py
API: 2D-TO-3D COORDINATE STRATEGY (SAMPLING INTERFACE)
-----------------------------------------------------
Role: 
    An implementation of the Strategy Pattern to translate 2D pixels into 3D space.
    Decouples the physics of the MRI scan from the training engine.

Main Classes:
    - BaseSampler:  Abstract interface for all sampling strategies.
    - PointSampler: Strategy for high-res 'thin' slices; 1 pixel = 1 3D point.
    - RaySlabSampler: Strategy for clinical 'thick' slices; 1 pixel = N points 
                      integrated along a mini-ray (Partial Volume Effect).

Methods:
    - sample(slice_2d, metadata, batch_size) -> (coords_3d, target_intensities)
"""
import torch

class BaseSampler:
    """Interface for sampling 3D points from 2D slices."""
    def sample(self, slice_data, metadata, batch_size):
        raise NotImplementedError

class PointSampler(BaseSampler):
    """Strategy 1: Treat every pixel as a discrete 3D point."""
    def sample(self, slice_2d, metadata, batch_size):
        # Pick random (x, y) indices
        idx_h = torch.randint(0, slice_2d.shape[0], (batch_size,))
        idx_w = torch.randint(0, slice_2d.shape[1], (batch_size,))
        
        # Map to [-1, 1] based on metadata (z-position)
        z = metadata['z_center'] 
        y = (idx_h / slice_2d.shape[0]) * 2 - 1
        x = (idx_w / slice_2d.shape[1]) * 2 - 1
        
        coords = torch.stack([x, y, torch.full_as(x, z)], dim=-1)
        targets = slice_2d[idx_h, idx_w].unsqueeze(-1)
        return coords, targets

class RaySlabSampler(BaseSampler):
    """Strategy 2: Treat every pixel as an integral of a 'thick' slice (The NeRF Way)."""
    def __init__(self, num_samples_per_ray=8):
        self.n_samples = num_samples_per_ray

    def sample(self, slice_2d, metadata, batch_size):
        # 1. Pick random pixels
        idx_h = torch.randint(0, slice_2d.shape[0], (batch_size,))
        idx_w = torch.randint(0, slice_2d.shape[1], (batch_size,))
        
        # 2. Define the 'Slab' (z_start to z_end)
        z_start, z_end = metadata['z_range']
        
        # 3. Create a mini-ray for every pixel piercing the thickness
        t_vals = torch.linspace(z_start, z_end, self.n_samples)
        
        # Generate 3D points for the whole batch [Batch, Samples, 3]
        y = (idx_h / slice_2d.shape[0]) * 2 - 1
        x = (idx_w / slice_2d.shape[1]) * 2 - 1
        
        # Broadcast x,y across the z-samples
        coords = torch.stack([
            x.repeat_interleave(self.n_samples),
            y.repeat_interleave(self.n_samples),
            t_vals.repeat(batch_size)
        ], dim=-1)
        
        targets = slice_2d[idx_h, idx_w].unsqueeze(-1)
        return coords, targets # Trainer will average these coords later