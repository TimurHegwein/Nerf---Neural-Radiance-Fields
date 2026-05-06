"""
FILE: representation/trainer_def.py
API: NEURAL FIELD CONVERGENCE ENGINE (OPTIMIZER)
-----------------------------------------------
Role:
    Orchestrates the training and evaluation steps. Connects the
    Sampling Strategy to the Neural Field Model. Implements
    Total Variation (TV) Regularization, LR scheduling, and
    optional FP16 mixed-precision training (AMP) on CUDA.

Metrics:
    - MSE: Mean Squared Error (core optimization objective)
    - PSNR: Peak Signal-to-Noise Ratio (human-readable quality)
    - TV Loss: local smoothness regulariser
    - LR: current learning rate
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, List
from torch.optim.lr_scheduler import ExponentialLR

from representation.sampler_def import RaySlabSampler


class NeuroTrainer:
    def __init__(self, model: nn.Module, sampler: object,
                 lr: float = 1e-3, tv_weight: float = 1e-6,
                 use_amp: bool = True):
        """
        :param model: NeuralField MLP [f(x,y,z) -> Intensity].
        :param sampler: Strategy mapping 2D pixels to 3D coordinates.
        :param lr: Adam learning rate.
        :param tv_weight: smoothness regulariser strength.
        :param use_amp: enable mixed-precision (FP16 forward) on CUDA. No-op on
            MPS / CPU. Typically 1.5-2x faster on T4 with negligible quality loss.
        """
        self.model = model
        self.sampler = sampler
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.998)
        self.criterion = nn.MSELoss()
        self.tv_weight = tv_weight

        device_type = next(model.parameters()).device.type
        self.device_type = device_type
        self.amp_enabled = bool(use_amp and device_type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda") if self.amp_enabled else None

    # -------- metric helpers --------

    def calculate_psnr(self, mse: torch.Tensor) -> torch.Tensor:
        """PSNR = -10 * log10(MSE) for [0, 1] normalised data."""
        if mse.item() < 1e-12:
            return torch.tensor(100.0)
        return -10.0 * torch.log10(mse)

    def _compute_tv_loss(self, n_probes: int) -> torch.Tensor:
        """Local smoothness prior: random points and their nearby neighbours
        should produce similar predictions."""
        device = next(self.model.parameters()).device
        pts = torch.rand((n_probes, 3), device=device) * 2 - 1
        pts_neighbor = pts + torch.randn_like(pts) * 0.01
        return torch.mean(torch.abs(self.model(pts) - self.model(pts_neighbor)))

    # -------- core forward/backward (shared by single- and multi-slice paths) --------

    def _step(self, coords: torch.Tensor, targets: torch.Tensor,
              n_rays: int, tv_probes: int) -> Tuple[float, float]:
        """One forward + backward + optimizer step. Returns (total_loss, psnr)."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(self.device_type, enabled=self.amp_enabled):
            preds = self.model(coords)
            if isinstance(self.sampler, RaySlabSampler):
                preds = preds.reshape(n_rays, self.sampler.n_samples).mean(dim=1, keepdim=True)
            mse_loss = self.criterion(preds, targets)
            tv_loss = self._compute_tv_loss(tv_probes)
            total_loss = mse_loss + (self.tv_weight * tv_loss)

        if self.amp_enabled:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        # Metric uses pure MSE (not the regularised loss) so PSNR stays comparable
        # across runs with different tv_weight values.
        psnr = self.calculate_psnr(mse_loss.detach().float())
        return total_loss.item(), psnr.item()

    # -------- single-slice path (legacy) --------

    def train_step(self, slice_2d, metadata: Dict, batch_size: int = 1024) -> Tuple[float, float]:
        """Single-slice training step — kept for backwards compatibility."""
        coords, targets = self.sampler.sample(slice_2d, metadata, batch_size)
        return self._step(coords, targets, n_rays=batch_size, tv_probes=batch_size)

    # -------- multi-slice path (fast) --------

    def train_step_multi(self, slices_with_metadata: List, rays_per_slice: int) -> Tuple[float, float]:
        """Train on K slices in a single forward/backward — much fewer kernel launches.

        :param slices_with_metadata: list of K (slice_tensor, metadata_dict) tuples
        :param rays_per_slice: rays sampled per slice; total rays in the step is
            K * rays_per_slice.
        """
        coords, targets = self.sampler.sample_multi(slices_with_metadata, rays_per_slice)
        n_rays = len(slices_with_metadata) * rays_per_slice
        # TV probes scale with the work we're doing per step but cap at 8192 to
        # keep TV cost bounded — TV uses random points, not training data.
        tv_probes = min(n_rays, 8192)
        return self._step(coords, targets, n_rays=n_rays, tv_probes=tv_probes)

    # -------- scheduler --------

    def step_scheduler(self) -> float:
        self.scheduler.step()
        return self.optimizer.param_groups[0]['lr']

    # -------- evaluation (single-slice; val set is small) --------

    @torch.no_grad()
    def eval_step(self, slice_2d, metadata: Dict, batch_size: int = 1024) -> Tuple[float, float]:
        self.model.eval()
        coords, targets = self.sampler.sample(slice_2d, metadata, batch_size)
        with torch.amp.autocast(self.device_type, enabled=self.amp_enabled):
            preds = self.model(coords)
            if isinstance(self.sampler, RaySlabSampler):
                preds = preds.reshape(batch_size, self.sampler.n_samples).mean(dim=1, keepdim=True)
            mse_loss = self.criterion(preds, targets)
        mse_f32 = mse_loss.detach().float()
        return mse_f32.item(), self.calculate_psnr(mse_f32).item()
