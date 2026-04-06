"""
This file specifies which deep learning model architecture and which embedding approach is used:

Currently a classic SineEncoding is implemented
"""
import torch
import torch.nn as nn
import numpy as np

class SineEncoding(nn.Module):
    def __init__(self, in_features=3, num_frequencies=10):
        super().__init__()
        # Wir registrieren freqs als Buffer, damit es mit .to(device) verschoben wird
        freqs = 2**torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer('freqs', freqs) 

    def forward(self, x):
        # x shape: [Batch, 3]
        # Wir müssen x für die Multiplikation vorbereiten
        x_expanded = x.unsqueeze(-1) # [Batch, 3, 1]
        weighted_coords = x_expanded * np.pi * self.freqs # [Batch, 3, Freqs]
        
        sin_coords = torch.sin(weighted_coords)
        cos_coords = torch.cos(weighted_coords)
        
        # Alles flachklatschen für den MLP-Input
        return torch.cat([sin_coords.flatten(start_dim=1), 
                          cos_coords.flatten(start_dim=1)], dim=-1)

class NeuralField(nn.Module):
    def __init__(self, num_freqs: int = 16, hidden_dim: int = 512, num_layers: int = 8):
        super().__init__()
        
        # 1. Encoding
        self.num_freqs = num_freqs
        self.freqs = 2**torch.linspace(0, num_freqs - 1, num_freqs)
        input_dim = 3 * 2 * num_freqs

        # 2. Deep MLP with LayerNorm
        self.layers = nn.ModuleList()
        
        # Input Layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden Layers
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            # We add LayerNorm after every linear layer
            self.layers.append(nn.LayerNorm(hidden_dim))
            
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.activation = nn.SiLU() # SiLU is smoother than ReLU for NeRFs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sine Encoding
        out = []
        for freq in self.freqs:
            out.append(torch.sin(x * np.pi * freq))
            out.append(torch.cos(x * np.pi * freq))
        x_enc = torch.cat(out, dim=-1)
        
        # Forward through layers
        h = x_enc
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if isinstance(layer, nn.LayerNorm):
                h = self.activation(h)
                # Optional: Skip connection (Add input to every 2nd layer)
                if i == 6: # Standard NeRF 'injection' point
                    h = h + torch.nn.functional.interpolate(x_enc.unsqueeze(0), size=h.shape[-1]).squeeze(0)
        
        return torch.sigmoid(self.output_layer(h))