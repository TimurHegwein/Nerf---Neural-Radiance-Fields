"""
FILE: output/visualizer.py
API: 3D MESH EXTRACTION & INTERACTIVE VIEWING
---------------------------------------------
Role:
    Converts the implicit 'Neural Scene' (Pickle) into an explicit 3D Mesh.
    Uses the Marching Cubes algorithm to find the surface boundary.

Input:  .pth file containing the trained NeuralField weights and config.
Output: An interactive 3D window showing the reconstructed geometry.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

from representation.model import NeuralField


def load_neural_field(model_path: str, device: str = "cpu") -> NeuralField:
    """Rebuild a NeuralField from a checkpoint, using the saved config when available.

    Falls back to legacy plain-state_dict checkpoints that don't carry a config —
    in that case the caller must ensure their NeuralField defaults match the saved
    architecture, otherwise load_state_dict will fail.
    """
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        config = ckpt.get("config") or {}
        model = NeuralField(**config)
        model.load_state_dict(ckpt["state_dict"])
    else:
        # Legacy: bare state_dict with no config metadata.
        model = NeuralField()
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model


class NeuroVisualizer:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = load_neural_field(model_path, device=device)

    @torch.no_grad()
    def extract_mesh(self, resolution=64, threshold=0.5):
        print(f"Sampling 3D grid at {resolution}^3 resolution...")

        grid_coords = torch.linspace(-1, 1, resolution, device=self.device)
        z, y, x = torch.meshgrid(grid_coords, grid_coords, grid_coords, indexing='ij')
        coords_3d = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=-1)

        intensities = []
        chunk_size = 64**2
        for i in range(0, coords_3d.shape[0], chunk_size):
            chunk = coords_3d[i : i + chunk_size]
            intensities.append(self.model(chunk))

        volume = torch.cat(intensities).reshape(resolution, resolution, resolution).cpu().numpy()

        print("Running Marching Cubes...")
        verts, faces, normals, values = measure.marching_cubes(volume, level=threshold)
        verts = (verts / (resolution - 1)) * 2 - 1
        return verts, faces

    def show(self, resolution=64, threshold=0.5):
        verts, faces = self.extract_mesh(resolution, threshold)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        mesh = Poly3DCollection(verts[faces], alpha=0.7)
        mesh.set_facecolor([0.5, 0.5, 1.0])
        mesh.set_edgecolor('black')
        mesh.set_linewidth(0.1)
        ax.add_collection3d(mesh)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_title(f"3D Neural Field Reconstruction (Threshold: {threshold})")

        print("Opening 3D Viewer...")
        plt.show()


if __name__ == "__main__":
    viz = NeuroVisualizer("checkpoints/brain_0.pth")
    viz.show(resolution=80, threshold=0.4)
