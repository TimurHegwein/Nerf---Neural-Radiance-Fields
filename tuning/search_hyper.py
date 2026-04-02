"""
FILE: tuning/random_search.py
API: STOCHASTIC HYPERPARAMETER OPTIMIZATION
-------------------------------------------
Role: 
    An 'Outer Loop' that samples random configurations and executes 
    training runs using the core NeRF engine.
"""

import random
import os
import torch
from representation.model import NeuralField
from representation.trainer_def import NeuroTrainer
from representation.train_loop import run_training
from input.data import ManualVolumeProvider

def sample_hparams():
    """Defines the search space and samples one configuration."""
    return {
        "lr": 10 ** random.uniform(-4, -2),       # Log-scale: 0.0001 to 0.01
        "num_freqs": random.randint(6, 14),       # Frequency resolution
        "hidden_dim": random.choice([128, 256]),  # Network width
        "num_layers": random.randint(3, 6),       # Network depth
        "batch_size": random.choice([1024, 2048]) # Batch size
    }

def run_experiments(data_slices, num_trials=10):
    print(f"--- Starting Random Search ({num_trials} trials) ---")
    
    for i in range(num_trials):
        hp = sample_hparams()
        
        # Create a unique ID for TensorBoard and Checkpoints
        exp_id = f"trial_{i}_freq{hp['num_freqs']}_lr{hp['lr']:.4f}_dim{hp['hidden_dim']}"
        print(f"\n[Trial {i+1}/{num_trials}] Configuration: {hp}")

        # 1. Initialize Model with sampled params
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = NeuralField(
            num_freqs=hp['num_freqs'], 
            hidden_dim=hp['hidden_dim'], 
            num_layers=hp['num_layers']
        ).to(device)

        # 2. Setup Provider & Trainer
        # We reuse your existing RaySlabSampler (or PointSampler)
        from representation.sampler_def import RaySlabSampler
        sampler = RaySlabSampler(num_samples_per_ray=8, device=device)
        trainer = NeuroTrainer(model, sampler, lr=hp['lr'])

        # 3. Call your existing 'Inner Loop'
        # All logs go to a specific folder for TensorBoard comparison
        run_training(
            volume_provider=ManualVolumeProvider(data_slices),
            trainer=trainer,
            epochs=300,  # Short runs for tuning
            batch_size=hp['batch_size'],
            save_path=f"checkpoints/{exp_id}.pth",
            log_dir=f"runs/random_search/{exp_id}",
            early_stop_threshold=1e-6
        )

if __name__ == "__main__":
    from main import create_layered_cube_slices
    slices = create_layered_cube_slices(res=64)
    run_experiments(slices, num_trials=5)