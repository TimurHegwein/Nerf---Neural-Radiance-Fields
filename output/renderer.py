"""
FILE: output/renderer.py
API: INFERENCE, VOLUME VISUALIZATION & COMPARISON
-------------------------------------
Role: 
    Queries the trained NeuralField to generate images or 3D volumes.
    Provides comparison plots between ground truth (Input) and Neural Field (Output).
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
        # Wichtig: grid_coords direkt auf dem Device erstellen
        grid_coords = torch.linspace(-1, 1, resolution, device=self.device)
        y, x = torch.meshgrid(grid_coords, grid_coords, indexing='ij')
        
        coords_3d = torch.stack([
            x.flatten(), 
            y.flatten(), 
            torch.full_like(x.flatten(), z_pos)
        ], dim=-1)

        intensities = self.model(coords_3d)
        return intensities.cpu().reshape(resolution, resolution).numpy()

    def plot_comparison(self, volume_provider, num_slices=5, resolution=128):
        """
        Erstellt ein Dashboard: 
        Obere Reihe: Original Slices (Ground Truth)
        Untere Reihe: NeRF Rekonstruktion
        """
        # Indizes für die Slices auswählen (gleichmäßig verteilt)
        total_available = volume_provider.get_total_slices()
        indices = np.linspace(0, total_available - 1, num_slices, dtype=int)

        fig, axes = plt.subplots(2, num_slices, figsize=(num_slices * 3, 7))
        
        for i, idx in enumerate(indices):
            # 1. Ground Truth holen
            gt_slice, metadata = volume_provider.get_slice(axis='z', index=idx)
            z_pos = metadata['z_center']

            # 2. NeRF Rekonstruktion rendern
            recon_slice = self.render_slice(z_pos=z_pos, resolution=resolution)

            # --- Plotten ---
            # Oben: Ground Truth
            ax_gt = axes[0, i]
            ax_gt.imshow(gt_slice, cmap='bone', vmin=0, vmax=1)
            ax_gt.set_title(f"GT Slice {idx}\n(Z={z_pos:.2f})")
            ax_gt.axis('off')

            # Unten: NeRF
            ax_recon = axes[1, i]
            ax_recon.imshow(recon_slice, cmap='bone', vmin=0, vmax=1)
            ax_recon.set_title(f"NeRF Recon\n({resolution}x{resolution})")
            ax_recon.axis('off')

        plt.tight_layout()
        plt.suptitle("Comparison: Ground Truth (Input) vs. Neural Field (Output)", fontsize=16, y=1.05)
        plt.show()

    @torch.no_grad()
    def render_volume(self, resolution=64):
        """Generates a full 3D voxel grid from the Neural Field."""
        volume = np.zeros((resolution, resolution, resolution))
        z_coords = np.linspace(-1, 1, resolution)
        for i, z in enumerate(z_coords):
            volume[i, :, :] = self.render_slice(z_pos=z, resolution=resolution)
        return volume