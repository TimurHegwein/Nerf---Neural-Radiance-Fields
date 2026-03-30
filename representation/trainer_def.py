"""
FILE: trainer.py
API: NEURAL FIELD CONVERGENCE ENGINE (OPTIMIZER)
-----------------------------------------------
Role: 
    Orchestrates the training loop by connecting the Sampler (Strategy) 
    to the NeuralField (Model). It is agnostic to the specific physics 
    used to generate 3D points.

Main Class: NeuroTrainer
    - Input: NeuralField (The MLP), BaseSampler (The Point/Ray Strategy).
    - Logic: Minimizes the Mean Squared Error (MSE) between the 
             reconstructed neural volume and 2D MRI slice data.
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

    def train_step(self, slice_2d, metadata, batch_size=1024):
        """
        Executes a single optimization iteration on a provided MRI slice.

        """
        self.optimizer.zero_grad()
        
        # 1. Coordinate Generation
        # Translate 2D pixel locations into 3D world-space probes.
        # coords: [Batch * N_Samples, 3], targets: [Batch, 1]
        coords, targets = self.sampler.sample(slice_2d, metadata, batch_size)
        
        # 2. Forward Pass (Inference)
        # Query the Neural Field for the intensity at each 3D probe location.
        preds = self.model(coords)
        
        # 3. Volume Integration (Aggregation)
        # If using Rays, we represent a 'thick' pixel by averaging multiple 
        # points along the Z-axis. This simulates the 'Partial Volume Effect'.
        # Reshape from [Batch * N, 1] -> [Batch, N] then mean -> [Batch, 1].
        if isinstance(self.sampler, RaySlabSampler):
            preds = preds.view(batch_size, self.sampler.n_samples).mean(dim=1, keepdim=True)
            
        # 4. Error Calculation & Backpropagation
        # Compare the integrated Neural Field prediction against the real MRI pixel.
        loss = self.criterion(preds, targets)
        loss.backward()
        
        # 5. Weights Update
        # Update the MLP weights to move the 'Neural Scene' closer to the data.
        self.optimizer.step()
        
        return loss.item()
    
    