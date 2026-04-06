import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Imports basierend auf deiner Verzeichnisstruktur
from input.data import PhantomProvider, NiftiVolumeProvider
from representation.model import NeuralField
from representation.sampler_def import RaySlabSampler
from representation.trainer_def import NeuroTrainer
from representation.train_loop import run_training
from output.renderer import NeuroRenderer

def main() -> None:
    # 0. Device Setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Running on: {device}")
    os.makedirs("checkpoints", exist_ok=True)

    # 1. DATA: Erzeugt das 3D-Phantom (Skull, Brain, Ventricles, Tumor)
    # res=64 erzeugt ein Volumen von 64^3 Voxeln
    # provider = PhantomProvider(res=64)
    provider = NiftiVolumeProvider("brains/brain_0.nii.gz", device=device)

    # 2. MODEL: Definition der Neural Scene
    # 12 Frequenzen erlauben es, scharfe Kanten (Skull/Tumor) zu lernen
    model = NeuralField(encoding_type="standard", num_freqs=14, hidden_dim=512).to(device)
    
    # 3. SAMPLER & TRAINER
    # RaySlabSampler simuliert die Schichtdicke für physikalisch korrekte 3D-Interpolation
    sampler = RaySlabSampler(num_samples_per_ray=8, device=device)
    
    # tv_weight sorgt dafür, dass die Ventrikel und das Gewebe glatt bleiben
    trainer = NeuroTrainer(model, sampler, lr=1e-3, tv_weight=1e-6)

    # 4. TRAINING: Startet die optimierte Loop
    # Wir nutzen 10% der Slices zur Validierung (Generalization Check)
    print("Starting Training Loop...")
    model = run_training(
        volume_provider=provider, 
        trainer=trainer, 
        epochs=1200, 
        batch_size=8192, 
        val_ratio=0.1,
        save_path="checkpoints/brain_0.pth",
        log_dir="runs/brain_0_experiment"
    )

    # 5. RENDER & COMPARISON: Das Kernstück der Evaluation
    print("\nGenerating Analysis Dashboard...")
    renderer = NeuroRenderer(model, device=device)
    
    # plot_comparison zeigt uns: 
    # 1. Was das Modell gesehen hat (GT)
    # 2. Wie gut es das gelernte rekonstruiert (Recon)
    # 3. Wie gut es die Lücken dazwischen füllt (Interpolation)
    renderer.plot_comparison(provider, num_slices=6, resolution=256)

    # Zusätzlicher Slice-Check für die Präsentation
    test_z = np.linspace(-1, 1, 10)
    plt.figure(figsize=(20, 4))
    for i, z in enumerate(test_z):
        img = renderer.render_slice(z_pos=z, resolution=128)
        plt.subplot(1, 10, i+1)
        plt.imshow(img, cmap='bone', vmin=0, vmax=1)
        plt.title(f"Z={z:.2f}")
        plt.axis('off')
    
    plt.suptitle("Neural Field: Latent Brain Representation (Continuous Slicing)")
    plt.show()

if __name__ == "__main__":
    main()