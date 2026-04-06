"""
FILE: representation/train_loop.py
API: TRAINING ORCHESTRATION & MONITORING
---------------------------------------
Role: 
    Iterates through the VolumeProvider to feed slices into the NeuroTrainer.
    Splits data into Training and Validation sets to monitor 3D generalization.
    Manages the lifecycle of the training process, including:
    - Best Model Checkpointing (Saving weights with minimum Validation Loss).
    - TensorBoard Logging (Loss and PSNR curves).
    - Early Stopping based on Validation performance.
"""

import torch
import torch.nn as nn
import time
import copy
import numpy as np
from typing import List, Tuple, Dict, Optional
from torch.utils.tensorboard import SummaryWriter

# Import-Typen für Type Hinting
from input.data import BaseVolumeProvider
from representation.trainer_def import NeuroTrainer

def run_training(
    volume_provider: BaseVolumeProvider, 
    trainer: NeuroTrainer, 
    epochs: int = 2000, 
    batch_size: int = 1024, 
    val_ratio: float = 0.1,
    save_path: str = "brain_scene.pth", 
    early_stop_threshold: float = 1e-7, 
    log_dir: str = "runs/neuro_nerf_exp",
    cnt_treshold: int = 100
) -> nn.Module:
    """
    Orchestrates the training and tracks the BEST model weights based on Validation.
    
    :param val_ratio: Fraction of slices to hold out for validation (e.g., 0.1 for 10%).
    :return: The trained NeuralField model with optimized weights.
    """
    
    # 1. Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # 2. Data Splitting (Train/Val Split)
    # Wir nehmen jeden n-ten Slice für die Validierung, um die Interpolation zu prüfen
    total_slices = volume_provider.get_total_slices()
    val_step = int(1 / val_ratio) if val_ratio > 0 else total_slices + 1
    
    val_indices = list(range(0, total_slices, val_step))
    train_indices = [i for i in range(total_slices) if i not in val_indices]
    
    print(f"Starting Training: {len(train_indices)} Train-Slices, {len(val_indices)} Val-Slices")
    print(f"Logging to: {log_dir}")
    
    # --- BEST MODEL TRACKING ---
    best_val_loss = float('inf')
    best_model_state: Optional[Dict] = None
    start_time = time.time()

    # Counting runs without model improvement for early stoppage
    cnt = 0

    try:
        for epoch in range(epochs):
            # --- TRAINING PHASE ---
            epoch_train_loss, epoch_train_psnr = 0.0, 0.0
            for slice_idx in train_indices:
                slice_2d, metadata = volume_provider.get_slice(axis='z', index=slice_idx)
                loss, psnr = trainer.train_step(slice_2d, metadata, batch_size=batch_size)
                epoch_train_loss += loss
                epoch_train_psnr += psnr

            avg_train_loss = epoch_train_loss / len(train_indices)
            avg_train_psnr = epoch_train_psnr / len(train_indices)

            # --- VALIDATION PHASE ---
            epoch_val_loss, epoch_val_psnr = 0.0, 0.0
            if val_indices:
                for slice_idx in val_indices:
                    slice_2d, metadata = volume_provider.get_slice(axis='z', index=slice_idx)
                    v_loss, v_psnr = trainer.eval_step(slice_2d, metadata, batch_size=batch_size)
                    epoch_val_loss += v_loss
                    epoch_val_psnr += v_psnr
                
                avg_val_loss = epoch_val_loss / len(val_indices)
                avg_val_psnr = epoch_val_psnr / len(val_indices)
            else:
                avg_val_loss, avg_val_psnr = avg_train_loss, avg_train_psnr

            # --- TENSORBOARD LOGGING ---
            # Wir gruppieren Train und Val in einem Graphen für direkten Vergleich
            writer.add_scalars('Loss/Combined', {'train': avg_train_loss, 'val': avg_val_loss}, epoch)
            writer.add_scalars('PSNR/Combined', {'train': avg_train_psnr, 'val': avg_val_psnr}, epoch)

            # --- CHECKPOINT LOGIC (Always based on Validation) ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(trainer.model.state_dict())
                cnt = 0
            else:
                cnt += 1

            # --- LOGGING TO CONSOLE ---
            if (epoch + 1) % 10 == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(f"Ep [{epoch+1:04d}/{epochs}] | Train PSNR: {avg_train_psnr:.2f}dB | Val PSNR: {avg_val_psnr:.2f}dB | Best Val Loss: {best_val_loss:.8f}")

            # --- EARLY STOPPING ---
            if avg_val_loss < early_stop_threshold or cnt > cnt_treshold:
                print(f"\n[EARLY STOP] Epoch {epoch+1}: Val Loss {avg_val_loss:.8f} < {early_stop_threshold}")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving best model found so far...")

    # --- FINALIZATION ---
    if best_model_state is not None:
        trainer.model.load_state_dict(best_model_state)
        torch.save(best_model_state, save_path)
        print(f"Final: Best model saved to {save_path} (Val Loss: {best_val_loss:.8f})")
    else:
        torch.save(trainer.model.state_dict(), save_path)
        print(f"Final: Last model state saved to {save_path}")
    
    writer.close()
    return trainer.model