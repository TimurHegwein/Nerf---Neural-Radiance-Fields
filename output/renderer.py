"""
FILE: renderer.py
API: INFERENCE & VOLUME VISUALIZATION
-------------------------------------
Role: 
    Queries the trained NeuralField to generate images or 3D volumes.
    Demonstrates 'Super-Resolution' by rendering at higher density than training.

Main Class: NeuroRenderer
    - Method: render_slice(z_pos, res) -> 2D Image
    - Method: render_volume(res) -> 3D Numpy Array
"""

import torch
import numpy as np

class NeuroRenderer:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval() # Set to evaluation mode

    @torch.no_grad()
    def render_slice(self, z_pos=0.0, resolution=128):
        """
        Renders a single 2D cross-section at a specific Z depth.
        z_pos: Normalized position in [-1, 1]
        """
        # 1. Create a grid of coordinates for the slice
        grid_coords = torch.linspace(-1, 1, resolution)
        y, x = torch.meshgrid(grid_coords, grid_coords, indexing='ij')
        
        # 2. Flatten and add the Z dimension
        # Shape: [Res*Res, 3]
        coords_3d = torch.stack([
            x.flatten(), 
            y.flatten(), 
            torch.full_as(x.flatten(), z_pos)
        ], dim=-1).to(self.device)

        # 3. Query the Neural Scene (The 'Pickle')
        intensities = self.model(coords_3d)
        
        # 4. Reshape back to image
        return intensities.cpu().reshape(resolution, resolution).numpy()

    @torch.no_grad()
    def render_volume(self, resolution=64):
        """Generates a full 3D voxel grid from the Neural Field."""
        volume = np.zeros((resolution, resolution, resolution))
        # We render slice by slice to avoid GPU memory overflow
        z_coords = np.linspace(-1, 1, resolution)
        for i, z in enumerate(z_coords):
            volume[i, :, :] = self.render_slice(z_pos=z, resolution=resolution)
        return volume