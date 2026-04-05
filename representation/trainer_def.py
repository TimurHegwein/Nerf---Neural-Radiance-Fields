"""
FILE: representation/trainer_def.py
API: NEURAL FIELD CONVERGENCE ENGINE (OPTIMIZER)
-----------------------------------------------
Role: 
    Orchestrates the training and evaluation steps. Connects the 
    Sampling Strategy to the Neural Field Model.
    
Metrics:
    - MSE: Mean Squared Error (used for optimization).
    - PSNR: Peak Signal-to-Noise Ratio (used for human-readable quality).
"""

import torch
import torch.nn as nn
from representation.sampler_def import RaySlabSampler

class NeuroTrainer:
    def __init__(self, model, sampler, lr=1e-3):
        """
        Initializes the optimization engine.
        :param model: The NeuralField MLP [f(x,y,z) -> Intensity].
        :param sampler: Strategy to map 2D pixels to 3D coordinates.
        :param lr: Learning rate for the Adam optimizer.
        """
        self.model = model
        self.sampler = sampler
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def calculate_psnr(self, mse):
        """
        Calculates Peak Signal-to-Noise Ratio.
        Formula: 20 * log10(MAX_I / sqrt(MSE))
        Since our intensities are normalized to [0, 1], MAX_I = 1.0.
        """
        if mse == 0:
            return 100.0  # Perfect reconstruction
        return -10.0 * torch.log10(mse)

    def train_step(self, slice_2d, metadata, batch_size=1024):
        """
        Executes a single optimization iteration (Update loop).
        Returns: (loss_value, psnr_value)
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 1. Coordinate Generation & Forward Pass
        coords, targets = self.sampler.sample(slice_2d, metadata, batch_size)
        preds = self.model(coords)
        
        # 2. Volume Integration (Aggregation)
        if isinstance(self.sampler, RaySlabSampler):
            preds = preds.reshape(batch_size, self.sampler.n_samples).mean(dim=1, keepdim=True)
            
        # 3. Loss & Backpropagation
        loss = self.criterion(preds, targets)
        loss.backward()
        self.optimizer.step()
        
        # 4. Metrics
        psnr = self.calculate_psnr(loss.detach())
        
        return loss.item(), psnr.item()

    @torch.no_grad()
    def eval_step(self, slice_2d, metadata, batch_size=1024):
        """
        Evaluates the model on a slice without updating weights.
        Used for the Validation Set to check generalization.
        """
        self.model.eval()
        
        # 1. Coordinate Generation & Forward Pass
        coords, targets = self.sampler.sample(slice_2d, metadata, batch_size)
        preds = self.model(coords)
        
        # 2. Volume Integration (Aggregation)
        if isinstance(self.sampler, RaySlabSampler):
            preds = preds.reshape(batch_size, self.sampler.n_samples).mean(dim=1, keepdim=True)
            
        # 3. Calculate Metrics
        loss = self.criterion(preds, targets)
        psnr = self.calculate_psnr(loss)
        
        return loss.item(), psnr.item()