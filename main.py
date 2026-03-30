import torch
import numpy as np
import matplotlib.pyplot as plt

# Adjusted Imports to match your tree
from input.data import ManualVolumeProvider
from representation.model import NeuralField
from representation.sampler_def import RaySlabSampler
from representation.trainer_def import NeuroTrainer
from representation.train_loop import run_training
from output.renderer import NeuroRenderer

def create_layered_cube_slices(res=64):
    """
    User Logic: 
    3 layers 0s | 2 layers 1s | 3 layers 0s | 2 layers 1s | 3 layers 0s
    Total: 13 slices.
    """
    empty = np.zeros((res, res))
    
    # Let's make the '1s' an outline of a square (Hollow Cube)
    solid_layer = np.zeros((res, res))
    solid_layer[16:48, 16:48] = 1.0 # Outer square
    solid_layer[20:44, 20:44] = 0.0 # Inner hollow
    
    # Construct the stack
    slices = ( [empty]*3 + [solid_layer]*2 + [empty]*3 + [solid_layer]*2 + [empty]*3 )
    return slices

def main():
    # --- 1. DATA ---
    slices = create_layered_cube_slices(res=64)
    provider = ManualVolumeProvider(slices)
    print(f"Created {len(slices)} layers for training.")

    # --- 2. MODEL & TRAINER ---
    model = NeuralField(encoding_type="standard", num_freqs=12) # High freqs for sharp edges
    sampler = RaySlabSampler(num_samples_per_ray=12) # Higher sampling for thick slabs
    trainer = NeuroTrainer(model, sampler, lr=1e-3)

    # --- 3. TRAINING ---
    # We increase epochs because 13 layers is more complex than 3
    run_training(provider, trainer, epochs=2000, batch_size=1024)

    # --- 4. RENDER & VERIFY ---
    renderer = NeuroRenderer(model)
    
    # We sample 10 positions across the whole Z-range [-1, 1]
    test_z = np.linspace(-1, 1, 10)
    
    plt.figure(figsize=(20, 4))
    for i, z in enumerate(test_z):
        img = renderer.render_slice(z_pos=z, resolution=128)
        plt.subplot(1, 10, i+1)
        plt.imshow(img, cmap='bone')
        plt.title(f"Z={z:.2f}")
        plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    main()