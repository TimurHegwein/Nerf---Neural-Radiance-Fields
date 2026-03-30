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
import numpy as np

class BaseSampler:
    """Interface for sampling 3D points from 2D slices."""
    def sample(self, slice_data, metadata, batch_size):
        raise NotImplementedError

class PointSampler(BaseSampler):
    """Strategy 1: Treat every pixel as a discrete 3D point."""
    def __init__(self, device="cpu"):
        self.device = device

    def sample(self, slice_2d, metadata, batch_size):
        # 1. Type Conversion: Ensure we are working with Tensors on the right device
        if isinstance(slice_2d, np.ndarray):
            slice_2d = torch.from_numpy(slice_2d).to(self.device)
            
        # 2. Pick random (x, y) indices
        idx_h = torch.randint(0, slice_2d.shape[0], (batch_size,), device=self.device)
        idx_w = torch.randint(0, slice_2d.shape[1], (batch_size,), device=self.device)
        
        # 3. Map to [-1, 1] 
        # Note: We divide by (shape - 1) so that the last pixel index maps exactly to 1.0
        z = metadata['z_center'] 
        y = (idx_h.float() / (slice_2d.shape[0] - 1)) * 2 - 1
        x = (idx_w.float() / (slice_2d.shape[1] - 1)) * 2 - 1
        
        # 4. Create coordinate vector [Batch, 3]
        # We ensure z is a tensor on the same device
        z_tensor = torch.full_as(x, z)
        coords = torch.stack([x, y, z_tensor], dim=-1)
        
        # 5. Get targets (now unsqueeze works because slice_2d is a Tensor)
        targets = slice_2d[idx_h, idx_w].unsqueeze(-1)
        
        return coords, targets

class RaySlabSampler(BaseSampler):
    """Strategy 2: Treat every pixel as an integral of a 'thick' slice (The NeRF Way)."""
    def __init__(self, num_samples_per_ray=8, device="cpu"):
        self.n_samples = num_samples_per_ray
        self.device = device

    def sample(self, slice_2d, metadata, batch_size):
        # Ensure slice_2d is a torch tensor on the correct device
        if isinstance(slice_2d, np.ndarray):
            slice_2d = torch.from_numpy(slice_2d).to(self.device)
        
        # 1. Pick random pixels
        # Ensure indices are on the same device as the data
        idx_h = torch.randint(0, slice_2d.shape[0], (batch_size,), device=self.device)
        idx_w = torch.randint(0, slice_2d.shape[1], (batch_size,), device=self.device)
        
        # 2. Define the 'Slab' (z_start to z_end)
        z_start, z_end = metadata['z_range']
        
        # 3. Create a mini-ray for every pixel piercing the thickness
        t_vals = torch.linspace(z_start, z_end, self.n_samples, device=self.device)
        
        # Generate 3D points for the whole batch
        y = (idx_h.float() / slice_2d.shape[0]) * 2 - 1
        x = (idx_w.float() / slice_2d.shape[1]) * 2 - 1
        
        # Broadcast x,y across the z-samples
        # coords shape: [batch_size * n_samples, 3]
        coords = torch.stack([
            x.repeat_interleave(self.n_samples),
            y.repeat_interleave(self.n_samples),
            t_vals.repeat(batch_size)
        ], dim=-1)
        
        # 4. Get ground truth targets
        # Now slice_2d is a tensor, so unsqueeze(-1) will work!
        targets = slice_2d[idx_h, idx_w].unsqueeze(-1)
        
        return coords, targets