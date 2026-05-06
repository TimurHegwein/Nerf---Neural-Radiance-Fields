"""
FILE: tuning/search_hyper.py
API: RANDOM HYPERPARAMETER SEARCH
---------------------------------
Role:
    Outer-loop random search over architecture and optimizer
    hyperparameters. Each trial trains for a short number of
    epochs and reports the best validation PSNR. The winning
    config is dumped to JSON for use by main.py.

Usage:
    python -m tuning.search_hyper --trials 8 --epochs 50

Outputs:
    runs/hypersearch/<trial_id>/   TensorBoard logs per trial
    tuning/best_config.json        winning hyperparameters
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from input.data import NiftiVolumeProvider
from representation.model import NeuralField
from representation.sampler_def import RaySlabSampler
from representation.trainer_def import NeuroTrainer
from representation.train_loop import run_training, _split_indices

SEARCH_SPACE = {
    "lr":         lambda r: 10 ** r.uniform(-3.5, -2.5),  # ~3e-4 .. 3e-3
    "num_freqs":  lambda r: r.choice([10, 12, 14]),
    "num_layers": lambda r: r.choice([4, 5, 6, 7]),
    "hidden_dim": lambda r: r.choice([384, 512]),
    "tv_weight":  lambda r: 10 ** r.uniform(-8, -6),
}


def sample_hparams(rng: random.Random) -> dict:
    return {k: sampler(rng) for k, sampler in SEARCH_SPACE.items()}


def evaluate_trial(model: torch.nn.Module, sampler, provider, val_indices,
                   batch_size: int, device: str) -> float:
    """Compute mean Val MSE over the held-out slices, return PSNR."""
    import torch.nn as nn
    criterion = nn.MSELoss()
    model.eval()
    losses = []
    with torch.no_grad():
        for idx in val_indices:
            slice_2d, meta = provider.get_slice(axis='z', index=idx)
            coords, targets = sampler.sample(slice_2d, meta, batch_size)
            preds = model(coords)
            if isinstance(sampler, RaySlabSampler):
                preds = preds.reshape(batch_size, sampler.n_samples).mean(dim=1, keepdim=True)
            losses.append(criterion(preds, targets).item())
    avg_mse = float(np.mean(losses))
    psnr = -10.0 * np.log10(max(avg_mse, 1e-12))
    return psnr


def run_search(brain_path: str, trials: int, epochs: int, batch_size: int,
               base_seed: int, val_ratio: float, out_dir: str, ckpt_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"[hypersearch] device={device} | trials={trials} | epochs/trial={epochs}")

    provider = NiftiVolumeProvider(brain_path, device=device)
    total_slices = provider.get_total_slices()
    train_idx, val_idx = _split_indices(total_slices, val_ratio, base_seed)
    print(f"[hypersearch] {len(train_idx)} train / {len(val_idx)} val slices "
          f"(stratified, seed={base_seed})")

    rng = random.Random(base_seed)
    results = []
    t_start = time.time()

    for trial in range(trials):
        hp = sample_hparams(rng)
        trial_id = (f"t{trial:02d}_lr{hp['lr']:.4f}_l{hp['num_layers']}"
                    f"_h{hp['hidden_dim']}_f{hp['num_freqs']}_tv{hp['tv_weight']:.0e}")
        print(f"\n[Trial {trial+1}/{trials}] {trial_id}")
        print(f"  hparams: {hp}")

        # Re-seed the global RNGs per trial so weight init / sampling are
        # comparable while the hparams differ.
        torch.manual_seed(base_seed + trial)
        np.random.seed(base_seed + trial)

        model = NeuralField(
            encoding_type="standard",
            num_freqs=hp["num_freqs"],
            hidden_dim=hp["hidden_dim"],
            num_layers=hp["num_layers"],
        ).to(device)

        sampler = RaySlabSampler(num_samples_per_ray=8, device=device)
        trainer = NeuroTrainer(model, sampler,
                               lr=hp["lr"], tv_weight=hp["tv_weight"])

        ckpt_path = os.path.join(ckpt_dir, f"{trial_id}.pth")
        log_dir = os.path.join(out_dir, trial_id)
        t0 = time.time()
        run_training(
            volume_provider=provider,
            trainer=trainer,
            epochs=epochs,
            batch_size=batch_size,
            val_ratio=val_ratio,
            save_path=ckpt_path,
            log_dir=log_dir,
            early_stop_threshold=1e-7,
            cnt_treshold=epochs,        # disable early-stop inside the short trial
            split_seed=base_seed,        # share split across trials for fairness
            ckpt_every_n_epochs=epochs,  # only flush at end (no per-trial Drive overhead)
        )
        elapsed = time.time() - t0

        # Reload best checkpoint and score it on val
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        best_val_psnr = evaluate_trial(model, sampler, provider, val_idx,
                                       batch_size, device)
        results.append({
            "trial_id": trial_id,
            "hparams": hp,
            "best_val_loss": ckpt["best_val_loss"],
            "best_val_psnr": best_val_psnr,
            "best_epoch": ckpt["best_epoch"],
            "trial_seconds": elapsed,
        })
        print(f"  -> Best Val PSNR: {best_val_psnr:.2f} dB | took {elapsed:.0f}s")

    total_time = time.time() - t_start
    print(f"\n[hypersearch] total wall time: {total_time/60:.1f} min")

    # Sort and report
    results.sort(key=lambda r: r["best_val_psnr"], reverse=True)
    print("\n=== Leaderboard (best Val PSNR) ===")
    for rank, r in enumerate(results, 1):
        print(f"  {rank}. {r['best_val_psnr']:.2f} dB  {r['trial_id']}")

    # Save full results + winner
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_dir, "leaderboard.json"), "w") as f:
        json.dump(results, f, indent=2)

    winner = results[0]
    with open("tuning/best_config.json", "w") as f:
        json.dump(winner, f, indent=2)
    print(f"\nWinner saved to tuning/best_config.json")
    print(f"  config: {winner['hparams']}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--brain", default="brains/brain_0.nii.gz")
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50,
                   help="Epochs per trial (short, just enough to differentiate configs)")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="runs/hypersearch")
    p.add_argument("--ckpt-dir", default="checkpoints/hypersearch")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    run_search(
        brain_path=args.brain,
        trials=args.trials,
        epochs=args.epochs,
        batch_size=args.batch_size,
        base_seed=args.seed,
        val_ratio=args.val_ratio,
        out_dir=args.out_dir,
        ckpt_dir=args.ckpt_dir,
    )
