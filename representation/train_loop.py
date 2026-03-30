"""
FILE: train_loop.py
API: TRAINING ORCHESTRATION (WORKFLOW)
-------------------------------------
Role: 
    Iterates through the VolumeProvider to feed slices into the NeuroTrainer.
    Manages the lifecycle of the training process (Epochs, Logging, Saving).

Main Function: run_training
    - Input: VolumeProvider, NeuroTrainer, Epochs, Sparse_Factor.
    - Output: Trained NeuralField (the 'Pickle' weights).
"""

import torch
import time

def run_training(volume_provider, trainer, epochs=100, batch_size=1024, sparse_factor=1, save_path="outputs/brain_scene.pth"):
    """
    Orchestrates the training of the Neural Field.
    
    :param volume_provider: The data source (NIfTI or Phantom).
    :param trainer: The NeuroTrainer instance.
    :param epochs: Number of complete passes over the available slices.
    :param batch_size: Number of rays/points sampled per training step.
    :param sparse_factor: If > 1, only every n-th slice is used for training.
    :param save_path: File path to save the converged weights.
    """
    
    # 1. Slice Selection (Research Parameter)
    # Define which slices the model is allowed to 'see' during training.
    total_slices = volume_provider.get_total_slices()
    train_indices = list(range(0, total_slices, sparse_factor))
    
    print(f"Starting Training: {len(train_indices)}/{total_slices} slices (Factor: {sparse_factor})")
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # 2. Iterate through the allowed slices
        for slice_idx in train_indices:
            # Fetch slice data and its physical metadata from the provider
            slice_2d, metadata = volume_provider.get_slice(axis='z', index=slice_idx)
            
            # Execute one optimization step (Coordinate Projection -> MLP -> Loss -> Update)
            loss = trainer.train_step(slice_2d, metadata, batch_size=batch_size)
            epoch_loss += loss

        # 3. Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = epoch_loss / len(train_indices)
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch+1}/{epochs}] | Avg Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s")

    # 4. Export the Neural Scene
    # Save the weights (the 'Pickle') which now represent the continuous 3D volume.
    torch.save(trainer.model.state_dict(), save_path)
    print(f"Training complete. Scene saved to {save_path}")
    
    return trainer.model