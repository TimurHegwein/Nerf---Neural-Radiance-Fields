"""
FILE: representation/train_loop.py
API: TRAINING ORCHESTRATION & MONITORING
---------------------------------------
Role:
    Iterates through the VolumeProvider to feed slices into the NeuroTrainer.
    Splits data into Training and Validation sets to monitor 3D generalization.
    Manages the lifecycle of the training process, including:
    - Best Model Checkpointing (Saving weights with minimum Validation Loss).
    - TensorBoard Logging (Loss, PSNR, and Learning Rate curves).
    - Early Stopping based on Validation performance or lack of improvement.
    - Learning Rate Scheduling (Annealing).
"""

import torch
import torch.nn as nn
import time
import copy
import numpy as np
from typing import Dict, Optional
from torch.utils.tensorboard import SummaryWriter

from input.data import BaseVolumeProvider
from representation.trainer_def import NeuroTrainer


def _split_indices(total: int, val_ratio: float, seed: int):
    """Stratified split: divides slices into N equally-sized bins along Z and
    samples one validation slice per bin. This guarantees uniform coverage of
    the volume (no clustered val regions) while staying random within bins.
    Reproducible via the given seed."""
    if val_ratio <= 0:
        return list(range(total)), []

    n_val = max(1, int(round(total * val_ratio)))
    rng = np.random.default_rng(seed)

    # Bin edges split the index range into n_val contiguous chunks.
    edges = np.linspace(0, total, n_val + 1, dtype=int)
    val_idx = []
    for i in range(n_val):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:  # degenerate bin (more bins than slices) — skip
            continue
        val_idx.append(int(rng.integers(lo, hi)))

    val_set = set(val_idx)
    val_idx = sorted(val_set)
    train_idx = [i for i in range(total) if i not in val_set]
    return train_idx, val_idx


def _save_checkpoint(path: str, model: nn.Module, state_dict: Dict, best_val_loss: float,
                     best_epoch: int, split_seed: int, val_indices) -> None:
    """Save best-so-far state with full reproducibility metadata."""
    torch.save({
        "state_dict": state_dict,
        "config": getattr(model, "config", None),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "split_seed": split_seed,
        "val_indices": val_indices,
    }, path)

def run_training(
    volume_provider: BaseVolumeProvider,
    trainer: NeuroTrainer,
    epochs: int = 2000,
    batch_size: int = 1024,
    val_ratio: float = 0.1,
    save_path: str = "brain_scene.pth",
    early_stop_threshold: float = 1e-7,
    log_dir: str = "runs/neuro_nerf_exp",
    cnt_treshold: int = 100,
    split_seed: int = 42,
    ckpt_every_n_epochs: int = 25,
    slices_per_step: int = 10,
) -> nn.Module:
    """
    Orchestrates the training and tracks the BEST model weights based on Validation.

    :param val_ratio: Fraction of slices to hold out for validation.
    :param cnt_treshold: Patience counter for early stopping.
    :param split_seed: Seed for the random train/val split.
    :return: The trained NeuralField model with optimized weights.
    """
    writer = SummaryWriter(log_dir=log_dir)

    total_slices = volume_provider.get_total_slices()
    train_indices, val_indices = _split_indices(total_slices, val_ratio, split_seed)

    print(f"Starting Training: {len(train_indices)} Train-Slices, {len(val_indices)} Val-Slices (seed={split_seed})")
    print(f"Logging to: {log_dir}")

    best_val_loss = float('inf')
    best_model_state: Optional[Dict] = None
    best_epoch = 0
    pending_save = False  # True iff in-memory best is newer than the file on disk
    start_time = time.time()
    cnt = 0

    try:
        for epoch in range(epochs):
            # --- TRAINING PHASE (multi-slice batching) ---
            # Reshuffle order each epoch so chunks see different slice combinations.
            order = list(train_indices)
            np.random.shuffle(order)

            epoch_train_loss, epoch_train_psnr, n_steps = 0.0, 0.0, 0
            for start in range(0, len(order), slices_per_step):
                chunk = order[start : start + slices_per_step]
                slices_with_meta = [
                    volume_provider.get_slice(axis='z', index=idx) for idx in chunk
                ]
                loss, psnr = trainer.train_step_multi(slices_with_meta,
                                                      rays_per_slice=batch_size)
                epoch_train_loss += loss
                epoch_train_psnr += psnr
                n_steps += 1

            avg_train_loss = epoch_train_loss / max(n_steps, 1)
            avg_train_psnr = epoch_train_psnr / max(n_steps, 1)

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

            current_lr = trainer.step_scheduler()

            writer.add_scalars('Loss/Combined', {'train': avg_train_loss, 'val': avg_val_loss}, epoch)
            writer.add_scalars('PSNR/Combined', {'train': avg_train_psnr, 'val': avg_val_psnr}, epoch)
            writer.add_scalar('Params/LearningRate', current_lr, epoch)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(trainer.model.state_dict())
                best_epoch = epoch
                cnt = 0
                pending_save = True
            else:
                cnt += 1

            # Periodic flush so a session crash never wipes more than ckpt_every_n_epochs of progress.
            if pending_save and (epoch + 1) % ckpt_every_n_epochs == 0 and best_model_state is not None:
                _save_checkpoint(save_path, trainer.model, best_model_state,
                                 best_val_loss, best_epoch, split_seed, val_indices)
                pending_save = False

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Ep [{epoch+1:04d}/{epochs}] | Train PSNR: {avg_train_psnr:.2f}dB | Val PSNR: {avg_val_psnr:.2f}dB | LR: {current_lr:.6f} | Patience: {cnt}/{cnt_treshold}")

            if avg_val_loss < early_stop_threshold:
                print(f"\n[EARLY STOP] Threshold reached at Epoch {epoch+1}")
                break

            if cnt > cnt_treshold:
                print(f"\n[EARLY STOP] No improvement for {cnt_treshold} epochs.")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving best model found so far...")

    # --- FINALIZATION: save state_dict + config so the model is reproducible. ---
    if best_model_state is not None:
        trainer.model.load_state_dict(best_model_state)
        _save_checkpoint(save_path, trainer.model, best_model_state,
                         best_val_loss, best_epoch, split_seed, val_indices)
        print(f"Final: Best model saved to {save_path} (Best Val Loss: {best_val_loss:.8f} @ epoch {best_epoch+1})")
    else:
        _save_checkpoint(save_path, trainer.model, trainer.model.state_dict(),
                         best_val_loss, best_epoch, split_seed, val_indices)
        print(f"Final: Last model state saved to {save_path}")

    writer.close()
    return trainer.model
