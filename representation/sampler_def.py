import torch
import numpy as np

class BaseSampler:
    """Interface for sampling 3D points from 2D slices."""
    def sample(self, slice_data, metadata, batch_size):
        raise NotImplementedError

class PointSampler(BaseSampler):
    """Strategy 1: Treat every pixel as a discrete 3D point + Jitter."""
    def __init__(self, device="mps"):
        self.device = device

    def sample(self, slice_2d, metadata, batch_size):
        if isinstance(slice_2d, np.ndarray):
            slice_2d = torch.from_numpy(slice_2d).to(self.device)
            
        # 1. Pick random pixel indices
        idx_h = torch.randint(0, slice_2d.shape[0], (batch_size,), device=self.device)
        idx_w = torch.randint(0, slice_2d.shape[1], (batch_size,), device=self.device)
        
        # 2. Map to [-1, 1] 
        res_h, res_w = slice_2d.shape
        y = (idx_h.float() / (res_h - 1)) * 2 - 1
        x = (idx_w.float() / (res_w - 1)) * 2 - 1
        
        # --- ADD JITTER (Anti-Aliasing) ---
        # Shift coords randomly within +/- half a pixel to force continuous learning
        y += (torch.rand_like(y) - 0.5) * (2.0 / res_h)
        x += (torch.rand_like(x) - 0.5) * (2.0 / res_w)
        
        # 3. Create coordinate vector [Batch, 3]
        z = metadata['z_center'] 
        z_tensor = torch.full_like(x, z)
        coords = torch.stack([x, y, z_tensor], dim=-1)
        
        targets = slice_2d[idx_h, idx_w].unsqueeze(-1)
        return coords, targets

class RaySlabSampler(BaseSampler):
    """Strategy 2: Treat every pixel as an integral of a 'thick' slice + Jitter."""
    def __init__(self, num_samples_per_ray=8, device="mps"):
        self.n_samples = num_samples_per_ray
        self.device = device

    def sample(self, slice_2d, metadata, batch_size):
        if isinstance(slice_2d, np.ndarray):
            slice_2d = torch.from_numpy(slice_2d).to(self.device)
        
        # 1. Pick random pixel indices
        idx_h = torch.randint(0, slice_2d.shape[0], (batch_size,), device=self.device)
        idx_w = torch.randint(0, slice_2d.shape[1], (batch_size,), device=self.device)
        
        # 2. Define the 'Slab' (z_start to z_end)
        z_start, z_end = metadata['z_range']
        t_vals = torch.linspace(z_start, z_end, self.n_samples, device=self.device)
        
        # 3. Map to [-1, 1]
        res_h, res_w = slice_2d.shape
        y = (idx_h.float() / (res_h - 1)) * 2 - 1
        x = (idx_w.float() / (res_w - 1)) * 2 - 1

        # --- ADD JITTER (Anti-Aliasing) ---
        y += (torch.rand_like(y) - 0.5) * (2.0 / res_h)
        x += (torch.rand_like(x) - 0.5) * (2.0 / res_w)
        
        # 4. Broadcast x,y across the z-samples
        # coords shape: [batch_size * n_samples, 3]
        coords = torch.stack([
            x.repeat_interleave(self.n_samples),
            y.repeat_interleave(self.n_samples),
            t_vals.repeat(batch_size)
        ], dim=-1)
        
        targets = slice_2d[idx_h, idx_w].unsqueeze(-1)
        return coords, targets