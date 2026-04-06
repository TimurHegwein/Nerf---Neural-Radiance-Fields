"""
FILE: representation/trainer_def.py
API: NEURAL FIELD CONVERGENCE ENGINE (OPTIMIZER)
-----------------------------------------------
Role: 
    Orchestrates the training and evaluation steps. Connects the 
    Sampling Strategy to the Neural Field Model. Implements 
    Total Variation (TV) Regularization and Learning Rate Scheduling.
    
Metrics:
    - MSE: Mean Squared Error (Core optimization objective).
    - PSNR: Peak Signal-to-Noise Ratio (Human-readable quality metric).
    - TV Loss: Measures local smoothness of the continuous volume.
    - LR: Current Learning Rate (Tracked for convergence analysis).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from torch.optim.lr_scheduler import ExponentialLR
from representation.sampler_def import RaySlabSampler

class NeuroTrainer:
    def __init__(self, model: nn.Module, sampler: object, lr: float = 1e-3, tv_weight: float = 1e-6):
        """
        Initializes the optimization engine.
        :param model: The NeuralField MLP [f(x,y,z) -> Intensity].
        :param sampler: Strategy to map 2D pixels to 3D coordinates.
        :param lr: Learning rate for the Adam optimizer.
        :param tv_weight: Strength of the smoothness regularization.
        """
        self.model = model
        self.sampler = sampler
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # SCHEDULER: Decays the learning rate exponentially.
        # gamma=0.998 means LR drops by 0.2% every time step_scheduler is called.
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.998)
        
        self.criterion = nn.MSELoss()
        self.tv_weight = tv_weight

    def calculate_psnr(self, mse: torch.Tensor) -> torch.Tensor:
        """
        Calculates Peak Signal-to-Noise Ratio.
        Formula: -10 * log10(MSE) for data normalized in [0, 1].
        """
        if mse == 0:
            return torch.tensor(100.0)
        return -10.0 * torch.log10(mse)

    def _compute_tv_loss(self, batch_size: int) -> torch.Tensor:
        """
        Total Variation Regularization (3D Smoothness):
        Probes random points to enforce spatial consistency.
        """
        device = next(self.model.parameters()).device
        pts = torch.rand((batch_size, 3), device=device) * 2 - 1
        eps = 0.01 
        pts_neighbor = pts + torch.randn_like(pts) * eps
        
        pred_pts = self.model(pts)
        pred_neighbor = self.model(pts_neighbor)
        
        return torch.mean(torch.abs(pred_pts - pred_neighbor))

    def train_step(self, slice_2d: np.ndarray, metadata: Dict, batch_size: int = 1024) -> Tuple[float, float]:
        """
        Executes a single optimization iteration (Backprop loop).
        :return: (Total Loss value, PSNR value)
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 1. Forward Pass
        coords, targets = self.sampler.sample(slice_2d, metadata, batch_size)
        preds = self.model(coords)
        
        if isinstance(self.sampler, RaySlabSampler):
            preds = preds.reshape(batch_size, self.sampler.n_samples).mean(dim=1, keepdim=True)
            
        # 2. Combined Loss
        mse_loss = self.criterion(preds, targets)
        tv_loss = self._compute_tv_loss(batch_size)
        total_loss = mse_loss + (self.tv_weight * tv_loss)
        
        # 3. Backpropagation
        total_loss.backward()
        
        # GRADIENT CLIPPING: Prevents exploding gradients in deep MLPs.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 4. Metrics Logging
        psnr = self.calculate_psnr(mse_loss.detach())
        
        return total_loss.item(), psnr.item()

    def step_scheduler(self) -> float:
        """
        Updates the learning rate. Call this at the end of each epoch.
        :return: The new learning rate value.
        """
        self.scheduler.step()
        return self.optimizer.param_groups[0]['lr']

    @torch.no_grad()
    def eval_step(self, slice_2d: np.ndarray, metadata: Dict, batch_size: int = 1024) -> Tuple[float, float]:
        """
        Evaluates the model on a slice without updating weights (Validation).
        """
        self.model.eval()
        coords, targets = self.sampler.sample(slice_2d, metadata, batch_size)
        preds = self.model(coords)
        
        if isinstance(self.sampler, RaySlabSampler):
            preds = preds.reshape(batch_size, self.sampler.n_samples).mean(dim=1, keepdim=True)
            
        mse_loss = self.criterion(preds, targets)
        psnr = self.calculate_psnr(mse_loss)
        
        return mse_loss.item(), psnr.item()