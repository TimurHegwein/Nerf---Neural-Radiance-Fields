"""
FILE: inference_report.py
API: POST-TRAINING VISUALIZATION
-------------------------------
Role: 
    Loads a saved .pth (Neural Field) and generates the Side-by-Side 
    comparison between the Input (Provider) and the Output (NeRF).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# Imports from your tree
from input.data import PhantomProvider
from representation.model import NeuralField
from output.renderer import NeuroRenderer

def run_inference(model_path="brain_scene.pth"):
    # 0. Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Setup Provider (Must be the same settings as training)
    # We use 64 resolution to match your last PoC
    provider = PhantomProvider(res=64)

    # 2. Reconstruct Model Architecture
    # Ensure num_freqs matches what you used in main.py (likely 12)
    model = NeuralField(encoding_type="standard", num_freqs=12).to(device)
    
    # 3. Load the 'Pickle' Weights
    print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Initialize the New Renderer
    renderer = NeuroRenderer(model, device=device)

    # 5. Generate the Side-by-Side Comparison
    print("Generating High-Resolution Comparison...")
    # resolution=256 shows the 'Super-Resolution' capability of the NeRF
    renderer.plot_comparison(provider, num_slices=6, resolution=256)

if __name__ == "__main__":
    run_inference("brain_scene.pth")