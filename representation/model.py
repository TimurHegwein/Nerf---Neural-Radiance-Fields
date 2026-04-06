import torch
import torch.nn as nn
import numpy as np

class SineEncoding(nn.Module):
    def __init__(self, in_features=3, num_frequencies=10):
        super().__init__()
        # Register as buffer to ensure it stays on the correct device (MPS/CUDA)
        freqs = 2**torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer('freqs', freqs) 

    def forward(self, x):
        # x shape: [Batch, 3]
        # Using torch.pi instead of np.pi for better Torch optimization
        x_expanded = x.unsqueeze(-1) 
        weighted_coords = x_expanded * torch.pi * self.freqs 
        
        sin_coords = torch.sin(weighted_coords)
        cos_coords = torch.cos(weighted_coords)
        
        # Flatten to [Batch, in_features * 2 * num_frequencies]
        return torch.cat([sin_coords.flatten(start_dim=1), 
                          cos_coords.flatten(start_dim=1)], dim=-1)

class NeuralField(nn.Module):
    def __init__(self, encoding_type: str = "standard", num_freqs: int = 16, hidden_dim: int = 512, num_layers: int = 8):
        super().__init__()
        
        # 1. Modular Encoding
        if encoding_type == "standard":
            self.encoder = SineEncoding(in_features=3, num_frequencies=num_freqs)
            self.input_dim = 3 * 2 * num_freqs
        else:
            self.encoder = nn.Identity()
            self.input_dim = 3

        # 2. Architecture Construction
        self.layers = nn.ModuleList()
        # First Layer
        self.layers.append(nn.Linear(self.input_dim, hidden_dim))
        
        # Hidden Layers (using num_layers - 1)
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.LayerNorm(hidden_dim))
            
        # 3. Skip Connection Projector (Defined once in init)
        # We project input_dim to hidden_dim so we can add them together
        self.skip_proj = nn.Linear(self.input_dim, hidden_dim)
        self.skip_layer_idx = (num_layers // 2) * 2 # Aim for the middle linear layer
        
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_enc = self.encoder(x)
        
        h = x_enc
        for i, layer in enumerate(self.layers):
            h = layer(h)
            
            # Apply activation and potential skip connection
            if isinstance(layer, nn.Linear) and i > 0:
                # We apply activation AFTER linear, but before the next LayerNorm
                pass 
            elif isinstance(layer, nn.LayerNorm):
                h = self.activation(h)
                
                # Apply Skip Connection halfway through the network
                if i == self.skip_layer_idx:
                    h = h + self.skip_proj(x_enc)
        
        return torch.sigmoid(self.output_layer(h))