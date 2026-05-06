"""
FILE: main.py
Entry point for training a NeuralField on a NIfTI brain volume.

Two configuration modes:
  - Default: hardcoded sensible hyperparameters
  - --use-best-config: load winner from tuning/best_config.json
    (produced by `python -m tuning.search_hyper`)

Other CLI overrides (lr, num_layers, ...) are also supported — all default
to None so the hardcoded values stay in effect unless you ask otherwise.
"""

import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from input.data import NiftiVolumeProvider
from output.renderer import NeuroRenderer
from representation.model import NeuralField
from representation.sampler_def import RaySlabSampler
from representation.train_loop import run_training
from representation.trainer_def import NeuroTrainer

SEED = 42

# Defaults — overridden by CLI flags or by --use-best-config.
DEFAULTS = {
    "num_freqs": 12,
    "hidden_dim": 512,
    "num_layers": 6,
    "lr": 1e-3,
    "tv_weight": 1e-7,
    "epochs": 500,
    "batch_size": 8192,
    "slices_per_step": 10,
    "val_ratio": 0.1,
    "patience": 100,
}


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch for reproducibility.
    Note: MPS does not yet guarantee full determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--use-best-config", action="store_true",
                   help="Load hyperparameters from tuning/best_config.json")
    p.add_argument("--brain", default="brains/brain_0.nii.gz")
    p.add_argument("--save-path", default="checkpoints/brain_0.pth")
    p.add_argument("--log-dir", default="runs/brain_0_experiment")
    # Hyperparameter overrides (default to None -> use DEFAULTS)
    for k in ("num_freqs", "hidden_dim", "num_layers", "epochs",
              "batch_size", "slices_per_step", "patience"):
        p.add_argument(f"--{k.replace('_', '-')}", type=int, default=None)
    for k in ("lr", "tv_weight", "val_ratio"):
        p.add_argument(f"--{k.replace('_', '-')}", type=float, default=None)
    return p.parse_args()


def resolve_config(args) -> dict:
    cfg = dict(DEFAULTS)
    if args.use_best_config:
        try:
            with open("tuning/best_config.json") as f:
                winner = json.load(f)
            cfg.update(winner["hparams"])
            print(f"[config] using best_config from tuning/best_config.json")
            print(f"         (val PSNR={winner.get('best_val_psnr', '?'):.2f} dB "
                  f"@ epoch {winner.get('best_epoch', '?')+1 if isinstance(winner.get('best_epoch'), int) else '?'})")
        except FileNotFoundError:
            print("[config] best_config.json not found — falling back to defaults")
    # Apply explicit CLI overrides last (highest priority)
    for k in DEFAULTS:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    return cfg


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)
    set_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Running on: {device} | seed={SEED}")
    print(f"Config: {cfg}")
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    # 1. DATA
    provider = NiftiVolumeProvider(args.brain, device=device)

    # 2. MODEL
    model = NeuralField(
        encoding_type="standard",
        num_freqs=cfg["num_freqs"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # 3. SAMPLER + TRAINER
    sampler = RaySlabSampler(num_samples_per_ray=8, device=device)
    trainer = NeuroTrainer(model, sampler,
                           lr=cfg["lr"], tv_weight=cfg["tv_weight"])
    if trainer.amp_enabled:
        print("AMP (mixed precision) enabled — FP16 forward on CUDA")

    # 4. TRAIN
    print("Starting Training Loop...")
    model = run_training(
        volume_provider=provider,
        trainer=trainer,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        val_ratio=cfg["val_ratio"],
        save_path=args.save_path,
        log_dir=args.log_dir,
        cnt_treshold=cfg["patience"],
        split_seed=SEED,
        slices_per_step=cfg["slices_per_step"],
    )

    # 5. RENDER & COMPARISON
    print("\nGenerating Analysis Dashboard...")
    renderer = NeuroRenderer(model, device=device)
    renderer.plot_comparison(provider, num_slices=6, resolution=256)

    # Continuous Z-slicing for the presentation
    test_z = np.linspace(-1, 1, 10)
    plt.figure(figsize=(20, 4))
    for i, z in enumerate(test_z):
        img = renderer.render_slice(z_pos=float(z), resolution=128)
        plt.subplot(1, 10, i + 1)
        plt.imshow(img, cmap='bone', vmin=0, vmax=1)
        plt.title(f"Z={z:.2f}")
        plt.axis('off')

    plt.suptitle("Neural Field: Latent Brain Representation (Continuous Slicing)")
    plt.show()


if __name__ == "__main__":
    main()
