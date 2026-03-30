"""
FILE: representation/train_loop.py
API: TRAINING ORCHESTRATION (WORKFLOW)
-------------------------------------
Role: 
    Iterates through the VolumeProvider to feed slices into the NeuroTrainer.
    Manages the lifecycle of the training process (Epochs, Logging, Saving).
    Now includes Early Stopping to prevent redundant computation.
"""

import torch
import time

def run_training(volume_provider, trainer, epochs=2000, batch_size=1024, 
                 sparse_factor=1, save_path="brain_scene.pth", 
                 early_stop_threshold=1e-7):
    """
    Orchestrates the training of the Neural Field.
    
    :param early_stop_threshold: Stop training if average loss falls below this value.
    """
    
    total_slices = volume_provider.get_total_slices()
    train_indices = list(range(0, total_slices, sparse_factor))
    
    print(f"Starting Training: {len(train_indices)}/{total_slices} slices (Factor: {sparse_factor})")
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for slice_idx in train_indices:
            slice_2d, metadata = volume_provider.get_slice(axis='z', index=slice_idx)
            loss = trainer.train_step(slice_2d, metadata, batch_size=batch_size)
            epoch_loss += loss

        avg_loss = epoch_loss / len(train_indices)

        # --- EARLY STOPPING LOGIC ---
        if avg_loss < early_stop_threshold:
            elapsed = time.time() - start_time
            print(f"\n[EARLY STOP] Epoch {epoch+1}: Loss {avg_loss:.8f} < Threshold {early_stop_threshold}")
            print(f"Convergence reached in {elapsed:.1f} seconds.")
            break

        # Logging every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch+1}/{epochs}] | Avg Loss: {avg_loss:.8f} | Time: {elapsed:.1f}s")

    # Save the 'Pickle' weights
    torch.save(trainer.model.state_dict(), save_path)
    print(f"Training complete. Scene saved to {save_path}")
    
    return trainer.model