"""
FILE: representation/model.py
API: COORDINATE-TO-INTENSITY MAPPING (NEURAL FIELD)
--------------------------------------------------
Role: 
    Defines the continuous implicit representation of the volume.
    Replaces discrete voxels with a differentiable MLP.
"""

import torch
import torch.nn as nn
import numpy as np

class SineEncoding(nn.Module):
    """Standard NeRF Positional Encoding to capture high-frequency details."""
    def __init__(self, in_features=3, num_frequencies=10):
        super().__init__()
        self.freqs = 2**torch.linspace(0, num_frequencies - 1, num_frequencies)

    def forward(self, x):
        out = []
        for freq in self.freqs:
            out.append(torch.sin(x * np.pi * freq))
            out.append(torch.cos(x * np.pi * freq))
        return torch.cat(out, dim=-1)

class NeuralField(nn.Module):
    """The MLP that represents the 3D volume."""
    def __init__(self, encoding_type="standard", num_freqs=10):
        super().__init__()
        
        # 1. Setup Encoding
        if encoding_type == "standard":
            self.encoding = SineEncoding(in_features=3, num_frequencies=num_freqs)
            input_dim = 3 * 2 * num_freqs # 3 coords * (sin+cos) * freqs
        else:
            # We will implement GridCellEncoding here later for your research!
            self.encoding = nn.Identity() 
            input_dim = 3

        # 2. The Network (The 'Pickle' Weights)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1), # Output: Intensity
            nn.Sigmoid()       # Constrain output to [0, 1]
        )

    def forward(self, x):
        x_encoded = self.encoding(x)
        return self.net(x_encoded)