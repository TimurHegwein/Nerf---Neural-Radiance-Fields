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

class NeuroRenderer:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def render_slice(self, z_pos=0.0, resolution=128):
        """Renders a single 2D cross-section at a specific Z depth."""
        grid_coords = torch.linspace(-1, 1, resolution, device=self.device)
        y, x = torch.meshgrid(grid_coords, grid_coords, indexing='ij')
        
        coords_3d = torch.stack([
            x.flatten(), 
            y.flatten(), 
            torch.full_like(x.flatten(), float(z_pos))
        ], dim=-1)

        intensities = self.model(coords_3d)
        return intensities.cpu().reshape(resolution, resolution).numpy()

    def plot_comparison(self, volume_provider, num_slices=5, resolution=128):
        """
        Creates a 3-row dashboard:
        Row 1: Original Input Slices (Ground Truth)
        Row 2: NeRF Reconstruction (At the exact same Z as GT)
        Row 3: NeRF Interpolation (At Z-positions BETWEEN the GT slices)
        """
        total_available = volume_provider.get_total_slices()
        indices = np.linspace(0, total_available - 1, num_slices, dtype=int)
        
        # Calculate the distance between indices to find the 'Midpoint'
        idx_step = (total_available - 1) / (num_slices - 1) if num_slices > 1 else 0

        # Change to 3 rows
        fig, axes = plt.subplots(3, num_slices, figsize=(num_slices * 3, 10))
        
        for i, idx in enumerate(indices):
            # 1. Fetch Ground Truth
            gt_slice, metadata = volume_provider.get_slice(axis='z', index=idx)
            z_pos = metadata['z_center']

            # 2. Render NeRF at same Z
            recon_slice = self.render_slice(z_pos=z_pos, resolution=resolution)

            # 3. Calculate and Render 'In-Between' Z (The Gap)
            # We take the midpoint between current slice and the next one
            # Unless it's the last slice, then we look backwards
            if i < num_slices - 1:
                mid_idx = idx + (idx_step / 2)
            else:
                mid_idx = idx - (idx_step / 2)
            
            # Map mid_idx to [-1, 1] coordinate space
            z_mid = -1.0 + (mid_idx / (total_available - 1)) * 2.0
            interp_slice = self.render_slice(z_pos=z_mid, resolution=resolution)

            # --- Plotting ---
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
    def render_volume(self, resolution=64):
        """Generates a full 3D voxel grid from the Neural Field."""
        volume = np.zeros((resolution, resolution, resolution))
        z_coords = np.linspace(-1, 1, resolution)
        for i, z in enumerate(z_coords):
            volume[i, :, :] = self.render_slice(z_pos=z, resolution=resolution)
        return volume