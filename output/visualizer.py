"""
FILE: output/visualizer.py
API: 3D MESH EXTRACTION & INTERACTIVE VIEWING
---------------------------------------------
Role: 
    Converts the implicit 'Neural Scene' (Pickle) into an explicit 3D Mesh.
    Uses the Marching Cubes algorithm to find the surface boundary.

Input:  .pth (Pickle) file containing the trained NeuralField weights.
Output: An interactive 3D window showing the reconstructed geometry.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

# Import your model definition
from representation.model import NeuralField

class NeuroVisualizer:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        # 1. Initialize and Load Model
        self.model = NeuralField(encoding_type="standard", num_freqs=12)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def extract_mesh(self, resolution=64, threshold=0.5):
        """
        Samples the Neural Field on a grid and runs Marching Cubes.
        """
        print(f"Sampling 3D grid at {resolution}^3 resolution...")
        
        # 1. Create 3D Grid
        grid_coords = torch.linspace(-1, 1, resolution, device=self.device)
        z, y, x = torch.meshgrid(grid_coords, grid_coords, grid_coords, indexing='ij')
        
        coords_3d = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=-1)
        
        # 2. Query model (in chunks to avoid OOM)
        intensities = []
        chunk_size = 64**2 # Process slice by slice
        for i in range(0, coords_3d.shape[0], chunk_size):
            chunk = coords_3d[i : i + chunk_size]
            intensities.append(self.model(chunk))
        
        volume = torch.cat(intensities).reshape(resolution, resolution, resolution).cpu().numpy()

        # 3. Marching Cubes to find the surface at 'threshold'
        # Returns vertices, faces, normals, values
        print("Running Marching Cubes...")
        verts, faces, normals, values = measure.marching_cubes(volume, level=threshold)
        
        # Scale vertices back to [-1, 1] range
        verts = (verts / (resolution - 1)) * 2 - 1
        
        return verts, faces

    def show(self, resolution=64, threshold=0.5):
        """Displays the mesh using Matplotlib 3D."""
        verts, faces = self.extract_mesh(resolution, threshold)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create 3D PolyCollection
        mesh = Poly3DCollection(verts[faces], alpha=0.7)
        face_color = [0.5, 0.5, 1.0] # Light Blue
        mesh.set_facecolor(face_color)
        mesh.set_edgecolor('black')
        mesh.set_linewidth(0.1)
        
        ax.add_collection3d(mesh)

        # Set limits
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_title(f"3D Neural Field Reconstruction (Threshold: {threshold})")
        
        print("Opening 3D Viewer...")
        plt.show()

if __name__ == "__main__":
    # You can run this file directly to view your saved scene
    viz = NeuroVisualizer("brain_scene.pth")
    viz.show(resolution=80, threshold=0.4)