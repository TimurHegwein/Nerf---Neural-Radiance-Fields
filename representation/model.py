import torch
import torch.nn as nn

class SineEncoding(nn.Module):
    def __init__(self, in_features=3, num_frequencies=10):
        super().__init__()
        freqs = 2 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        # x: [Batch, 3]
        x_expanded = x.unsqueeze(-1)
        weighted = x_expanded * torch.pi * self.freqs

        sin_c = torch.sin(weighted)
        cos_c = torch.cos(weighted)

        return torch.cat([sin_c.flatten(start_dim=1),
                          cos_c.flatten(start_dim=1)], dim=-1)


class NeuralField(nn.Module):
    """
    NeRF-style coordinate MLP with proper concat skip connection in the middle layer.
    Architecture:  [Linear -> LayerNorm -> SiLU] x num_layers, with concat-skip at num_layers // 2.
    """
    def __init__(self, encoding_type: str = "standard", num_freqs: int = 16,
                 hidden_dim: int = 512, num_layers: int = 8):
        super().__init__()

        # 1. Encoding
        if encoding_type == "standard":
            self.encoder = SineEncoding(in_features=3, num_frequencies=num_freqs)
            self.input_dim = 3 * 2 * num_freqs
        else:
            self.encoder = nn.Identity()
            self.input_dim = 3

        # 2. Skip position (in der Mitte)
        self.skip_at = num_layers // 2

        # 3. Build hidden stack. Layer 0 takes input_dim. Layer at skip_at
        #    takes hidden_dim + input_dim (because we concat the encoded input).
        linears = []
        norms = []
        for i in range(num_layers):
            if i == 0:
                in_dim = self.input_dim
            elif i == self.skip_at:
                in_dim = hidden_dim + self.input_dim
            else:
                in_dim = hidden_dim
            linears.append(nn.Linear(in_dim, hidden_dim))
            norms.append(nn.LayerNorm(hidden_dim))

        self.linears = nn.ModuleList(linears)
        self.norms = nn.ModuleList(norms)
        self.activation = nn.SiLU()
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_enc = self.encoder(x)

        h = x_enc
        for i, (lin, norm) in enumerate(zip(self.linears, self.norms)):
            if i == self.skip_at and i != 0:
                h = torch.cat([h, x_enc], dim=-1)
            h = self.activation(norm(lin(h)))

        return torch.sigmoid(self.output_layer(h))
