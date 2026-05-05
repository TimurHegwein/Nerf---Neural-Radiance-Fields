"""
FILE: output/visualizer.py
API: 3D MESH EXTRACTION & INTERACTIVE VIEWING
---------------------------------------------
Role:
    Converts the implicit 'Neural Scene' into an explicit 3D Mesh via
    Marching Cubes, then renders it as an interactive Plotly figure
    (rotate / zoom / pan with the mouse, inline in Jupyter/Colab).

Input:  .pth file containing the trained NeuralField weights and config.
Output: Interactive Plotly Mesh3d figure.
"""

import torch
import numpy as np
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
        model = NeuralField()
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model


class NeuroVisualizer:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model = load_neural_field(model_path, device=device)

    @torch.no_grad()
    def extract_mesh(self, resolution: int = 64, threshold: float = 0.5):
        """Sample the implicit field on a dense grid and run Marching Cubes."""
        print(f"Sampling 3D grid at {resolution}^3 resolution...")

        grid_coords = torch.linspace(-1, 1, resolution, device=self.device)
        z, y, x = torch.meshgrid(grid_coords, grid_coords, grid_coords, indexing='ij')
        coords_3d = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=-1)

        intensities = []
        chunk_size = resolution ** 2  # one slice per chunk to keep memory bounded
        for i in range(0, coords_3d.shape[0], chunk_size):
            chunk = coords_3d[i : i + chunk_size]
            intensities.append(self.model(chunk))

        volume = torch.cat(intensities).reshape(resolution, resolution, resolution).cpu().numpy()

        print("Running Marching Cubes...")
        verts, faces, _, _ = measure.marching_cubes(volume, level=threshold)
        # Scale vertices from voxel grid back to [-1, 1] world coordinates.
        verts = (verts / (resolution - 1)) * 2 - 1
        return verts, faces

    def show(self, resolution: int = 64, threshold: float = 0.5,
             color: str = "lightblue", opacity: float = 0.85):
        """Render the mesh as an interactive Plotly figure.

        On Jupyter/Colab the figure is shown inline and is fully rotatable.
        Outside notebooks (plain `python visualizer.py`) it opens in the
        default browser via fig.show().
        """
        try:
            import plotly.graph_objects as go
        except ImportError as e:
            raise ImportError(
                "plotly is required for the 3D viewer — install with `pip install plotly`."
            ) from e

        verts, faces = self.extract_mesh(resolution, threshold)
        x, y, z = verts.T
        i, j, k = faces.T

        fig = go.Figure(data=[go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            color=color, opacity=opacity,
            flatshading=True,
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3, roughness=0.5),
            lightposition=dict(x=100, y=200, z=150),
            name="brain",
        )])
        fig.update_layout(
            title=f"3D Neural Field Reconstruction (threshold={threshold}, res={resolution})",
            scene=dict(
                xaxis=dict(range=[-1, 1], title="x"),
                yaxis=dict(range=[-1, 1], title="y"),
                zaxis=dict(range=[-1, 1], title="z"),
                aspectmode="cube",
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            width=800, height=800,
        )
        fig.show()
        return fig


if __name__ == "__main__":
    viz = NeuroVisualizer("checkpoints/brain_0.pth")
    viz.show(resolution=80, threshold=0.4)
