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
    def __init__(self, encoding_type="standard", num_freqs=10):
        super().__init__()
        
        if encoding_type == "standard":
            self.encoding = SineEncoding(in_features=3, num_frequencies=num_freqs)
            input_dim = 3 * 2 * num_freqs
        else:
            self.encoding = nn.Identity() 
            input_dim = 3

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_encoded = self.encoding(x)
        return self.net(x_encoded)