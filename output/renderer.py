"""
FILE: output/renderer.py
API: INFERENCE, VOLUME VISUALIZATION & COMPARISON
-------------------------------------
Role: 
    Queries the trained NeuralField to generate images or 3D volumes.
    Provides comparison plots between Ground Truth (Input) and Neural Field (Output),
    including 'In-Between' slices to demonstrate 3D interpolation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# We import BaseVolumeProvider only for type hinting
from input.data import BaseVolumeProvider

class NeuroRenderer:
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def render_slice(self, z_pos: float = 0.0, resolution: int = 128) -> np.ndarray:
        """
        Renders a single 2D cross-section at a specific Z depth.
        :return: 2D Numpy array of the reconstructed slice.
        """
        grid_coords = torch.linspace(-1, 1, resolution, device=self.device)
        y, x = torch.meshgrid(grid_coords, grid_coords, indexing='ij')
        
        coords_3d = torch.stack([
            x.flatten(), 
            y.flatten(), 
            torch.full_like(x.flatten(), float(z_pos))
        ], dim=-1)

        # Query the MLP and move result back to CPU for visualization
        intensities = self.model(coords_3d)
        return intensities.cpu().reshape(resolution, resolution).numpy()

    def plot_comparison(self, volume_provider: BaseVolumeProvider, num_slices: int = 5, resolution: int = 128) -> None:
        """
        Creates a 3-row dashboard comparing Ground Truth, Reconstruction, and Interpolation.
        """
        total_available = volume_provider.get_total_slices()
        indices = np.linspace(0, total_available - 1, num_slices, dtype=int)
        idx_step = (total_available - 1) / (num_slices - 1) if num_slices > 1 else 0

        fig, axes = plt.subplots(3, num_slices, figsize=(num_slices * 3, 10))
        
        for i, idx in enumerate(indices):
            # 1. Fetch Ground Truth (Now a GPU tensor!)
            gt_slice, metadata = volume_provider.get_slice(axis='z', index=idx)
            z_pos = metadata['z_center']
            
            # --- PERFORMANCE FIX ---
            # Matplotlib cannot plot GPU tensors; move to CPU and convert to Numpy
            if torch.is_tensor(gt_slice):
                gt_slice = gt_slice.detach().cpu().numpy()

            # 2. Render NeRF at same Z (Reconstruction)
            recon_slice = self.render_slice(z_pos=z_pos, resolution=resolution)

            # 3. Calculate and Render 'In-Between' Z (The Gap/Interpolation)
            if i < num_slices - 1:
                mid_idx = idx + (idx_step / 2)
            else:
                mid_idx = idx - (idx_step / 2)
            
            z_mid = -1.0 + (mid_idx / (total_available - 1)) * 2.0
            interp_slice = self.render_slice(z_pos=z_mid, resolution=resolution)

            # --- PLOTTING ---
            # Row 0: Ground Truth
            axes[0, i].imshow(gt_slice, cmap='bone', vmin=0, vmax=1)
            axes[0, i].set_title(f"GT Slice {idx}\n(Z={z_pos:.2f})")
            axes[0, i].axis('off')

            # Row 1: NeRF Reconstruction
            axes[1, i].imshow(recon_slice, cmap='bone', vmin=0, vmax=1)
            axes[1, i].set_title(f"NeRF Recon\n(Seen Z)")
            axes[1, i].axis('off')

            # Row 2: NeRF Interpolation (THE GAP)
            axes[2, i].imshow(interp_slice, cmap='bone', vmin=0, vmax=1)
            axes[2, i].set_title(f"NeRF Interp\n(Unseen Z={z_mid:.2f})")
            axes[2, i].axis('off')

        plt.tight_layout()
        plt.suptitle("Neural Field Analysis: Reconstruction vs. Interpolation", fontsize=16, y=1.02)
        plt.show()

    @torch.no_grad()
    def render_volume(self, resolution: int = 64) -> np.ndarray:
        """
        Generates a full 3D voxel grid from the Neural Field.
        :return: 3D Numpy array [Z, H, W].
        """
        volume = np.zeros((resolution, resolution, resolution))
        z_coords = np.linspace(-1, 1, resolution)
        for i, z in enumerate(z_coords):
            volume[i, :, :] = self.render_slice(z_pos=z, resolution=resolution)
        return volume