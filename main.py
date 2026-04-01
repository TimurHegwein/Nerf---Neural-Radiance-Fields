import torch
import numpy as np
import matplotlib.pyplot as plt

from input.data import PhantomProvider
from representation.model import NeuralField
from representation.sampler_def import RaySlabSampler
from representation.trainer_def import NeuroTrainer
from representation.train_loop import run_training
from output.renderer import NeuroRenderer

def main():
    # 0. Device erkennen (MPS für Mac, CUDA für Nvidia, sonst CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    # 1. DATA (res=64 als Zahl übergeben!)
    provider = PhantomProvider(res=64)

    # 2. MODEL auf das Device schieben
    model = NeuralField(encoding_type="standard", num_freqs=12).to(device)
    
    # 3. SAMPLER & TRAINER
    sampler = RaySlabSampler(num_samples_per_ray=8, device=device)
    trainer = NeuroTrainer(model, sampler, lr=1e-3)

    # 4. TRAINING
    run_training(provider, trainer, epochs=1000, batch_size=2048)

    # 5. RENDER
    renderer = NeuroRenderer(model, device=device)
    renderer.plot_comparison(provider, num_slices=6, resolution=256)

    test_z = np.linspace(-1, 1, 10)
    
    plt.figure(figsize=(20, 4))
    for i, z in enumerate(test_z):
        img = renderer.render_slice(z_pos=z, resolution=128)
        plt.subplot(1, 10, i+1)
        plt.imshow(img, cmap='bone', vmin=0, vmax=1)
        plt.title(f"Z={z:.2f}")
        plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    main()