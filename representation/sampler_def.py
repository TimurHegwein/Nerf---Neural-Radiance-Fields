import torch
import numpy as np

class BaseSampler:
    """
    INTERFACE: 2D-to-3D COORDINATE PROJECTION
    -----------------------------------------
    Abstract base class defining the contract for sampling 3D points from 2D slices.
    Subclasses define the 'physics' of how a pixel relates to the 3D volume.
    """
    def sample(self, slice_data, metadata, batch_size):
        raise NotImplementedError

class PointSampler(BaseSampler):
    """
    STRATEGY 1: DISCRETE POINT SUPERVISION
    --------------------------------------
    Treats each MRI pixel as an infinitely small point in 3D space.
    Best for: High-resolution isotropic scans with negligible slice thickness.
    """
    def __init__(self, device="mps"):
        self.device = device

    def sample(self, slice_2d, metadata, batch_size):
        slice_2d = torch.as_tensor(slice_2d, device=self.device) 
        # 1. Stochastic Pixel Selection
        # We sample random indices to break spatial correlation during training.
        idx_h = torch.randint(0, slice_2d.shape[0], (batch_size,), device=self.device)
        idx_w = torch.randint(0, slice_2d.shape[1], (batch_size,), device=self.device)
        
        # 2. Coordinate Mapping [Indices -> Normalized World Space]
        res_h, res_w = slice_2d.shape
        y = (idx_h.float() / (res_h - 1)) * 2 - 1
        x = (idx_w.float() / (res_w - 1)) * 2 - 1
        
        # 3. ANTI-ALIASING JITTER
        # Logic: If we only sample the exact pixel centers, the MLP learns a 'grid' 
        # of dots. By adding a random shift within +/- 0.5 pixels, we force the 
        # model to learn the continuous area represented by the pixel.
        y += (torch.rand_like(y) - 0.5) * (2.0 / res_h)
        x += (torch.rand_like(x) - 0.5) * (2.0 / res_w)
        
        # 4. Final 3D Coordinate Assembly
        z = metadata['z_center'] 
        z_tensor = torch.full_like(x, z)
        coords = torch.stack([x, y, z_tensor], dim=-1)
        
        # 5. Extract Ground Truth Intensity
        targets = slice_2d[idx_h, idx_w].unsqueeze(-1)
        return coords, targets

class RaySlabSampler(BaseSampler):
    """
    STRATEGY 2: VOLUMETRIC SLAB INTEGRATION
    ---------------------------------------
    Treats each MRI pixel as a 'thick' column of tissue. 
    Implements the 'Inverse Rendering' logic for Partial Volume Effect correction.
    """
    def __init__(self, num_samples_per_ray=8, device="mps"):
        self.n_samples = num_samples_per_ray
        self.device = device

    def sample(self, slice_2d, metadata, batch_size):
        slice_2d = torch.as_tensor(slice_2d, device=self.device)

        # 1. Stochastic Ray Selection
        idx_h = torch.randint(0, slice_2d.shape[0], (batch_size,), device=self.device)
        idx_w = torch.randint(0, slice_2d.shape[1], (batch_size,), device=self.device)
        
        # 2. Volumetric Domain (Z-Axis)
        # Defines the segment of the ray that exists 'inside' this thick slice.
        z_start, z_end = metadata['z_range']
        t_vals = torch.linspace(z_start, z_end, self.n_samples, device=self.device)
        
        # 3. Coordinate Mapping with Jitter
        res_h, res_w = slice_2d.shape
        y = (idx_h.float() / (res_h - 1)) * 2 - 1
        x = (idx_w.float() / (res_w - 1)) * 2 - 1
        
        # Anti-Aliasing shift in X-Y plane
        y += (torch.rand_like(y) - 0.5) * (2.0 / res_h)
        x += (torch.rand_like(x) - 0.5) * (2.0 / res_w)
        
        # 4. Ray-Point Expansion [Batch, N_Samples, 3]
        # We broadcast the X, Y coords across N Z-samples to create the 'mini-ray'.
        coords = torch.stack([
            x.repeat_interleave(self.n_samples),
            y.repeat_interleave(self.n_samples),
            t_vals.repeat(batch_size)
        ], dim=-1)
        
        targets = slice_2d[idx_h, idx_w].unsqueeze(-1)
        return coords, targets