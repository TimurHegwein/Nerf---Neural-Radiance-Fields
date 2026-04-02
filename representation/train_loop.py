"""
FILE: representation/train_loop.py
API: TRAINING ORCHESTRATION & MONITORING
---------------------------------------
Role: 
    Iterates through the VolumeProvider to feed slices into the NeuroTrainer.
    Manages the lifecycle of the training process, including:
    - Best Model Checkpointing (Saving weights with minimum loss).
    - TensorBoard Logging (Visualizing loss curves).
    - Early Stopping.
"""

import torch
import time
import copy
from torch.utils.tensorboard import SummaryWriter

def run_training(volume_provider, trainer, epochs=2000, batch_size=1024, 
                 sparse_factor=1, save_path="brain_scene.pth", 
                 early_stop_threshold=1e-7, log_dir="runs/neuro_nerf_exp"):
    """
    Orchestrates the training and tracks the BEST model weights.
    
    :param log_dir: Directory where TensorBoard logs will be saved.
    """
    
    # 1. Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir=log_dir)
    
    total_slices = volume_provider.get_total_slices()
    train_indices = list(range(0, total_slices, sparse_factor))
    
    print(f"Starting Training: {len(train_indices)}/{total_slices} slices (Factor: {sparse_factor})")
    print(f"Logging to: {log_dir}")
    
    # --- BEST MODEL TRACKING ---
    best_loss = float('inf')
    best_model_state = None
    start_time = time.time()

    try:
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Training Step
            for slice_idx in train_indices:
                slice_2d, metadata = volume_provider.get_slice(axis='z', index=slice_idx)
                loss = trainer.train_step(slice_2d, metadata, batch_size=batch_size)
                epoch_loss += loss

            avg_loss = epoch_loss / len(train_indices)

            # --- TENSORBOARD LOGGING ---
            writer.add_scalar('Loss/train', avg_loss, epoch)
            # You could also log the learning rate if it's dynamic
            # writer.add_scalar('Params/LR', trainer.optimizer.param_groups[0]['lr'], epoch)

            # --- CHECKPOINT LOGIC ---
            if avg_loss < best_loss:
                best_loss = avg_loss
                # Deep copy ensures we don't just store a reference to the changing weights
                best_model_state = copy.deepcopy(trainer.model.state_dict())

            # --- LOGGING TO CONSOLE ---
            if (epoch + 1) % 10 == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.8f} | Best: {best_loss:.8f} | {elapsed:.1f}s")

            # --- EARLY STOPPING ---
            if avg_loss < early_stop_threshold:
                print(f"\n[EARLY STOP] Epoch {epoch+1}: Loss {avg_loss:.8f} < {early_stop_threshold}")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving best model found so far...")

    # --- FINALIZATION ---
    # Load the best weights back into the model before returning/saving
    if best_model_state is not None:
        trainer.model.load_state_dict(best_model_state)
        torch.save(best_model_state, save_path)
        print(f"Final: Best model saved to {save_path} (Loss: {best_loss:.8f})")
    else:
        torch.save(trainer.model.state_dict(), save_path)
        print(f"Final: Last model state saved to {save_path}")
    
    writer.close() # Clean up TensorBoard
    return trainer.model