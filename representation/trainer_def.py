import torch
from sampler_def import RaySlabSampler

class NeuroTrainer:
    def __init__(self, model, sampler, lr=1e-3):
        self.model = model
        self.sampler = sampler
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_step(self, slice_2d, metadata, batch_size=1024):
        self.optimizer.zero_grad()
        
        # 1. Get coordinates and ground truth from the strategy
        coords, targets = self.sampler.sample(slice_2d, metadata, batch_size)
        
        # 2. Forward pass
        preds = self.model(coords)
        
        # 3. Aggregation Logic (Crucial for Rays)
        # If we sampled multiple points per pixel (Rays), we average them
        if isinstance(self.sampler, RaySlabSampler):
            preds = preds.view(batch_size, self.sampler.n_samples).mean(dim=1, keepdim=True)
            
        # 4. Backprop
        loss = torch.nn.MSELoss()(preds, targets)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()