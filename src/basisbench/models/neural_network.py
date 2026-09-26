from __future__ import annotations

import torch
import torch.nn as nn 


class NeuralNetwork(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if not isinstance(hidden_size, int) or hidden_size <=0:
            raise ValueError(
                'Hidden size must be a positive integer.'
            )
        self.hidden_size = hidden_size
        
        #* create the parameters
        self.w1 = nn.Parameter(torch.empty(hidden_size))
        self.b1 = nn.Parameter(torch.empty(hidden_size))
        self.w2 = nn.Parameter(torch.empty(hidden_size))
        self.b2 = nn.Parameter(torch.empty(1))
        
        #* initialize the parameters
        nn.init.uniform_(self.w1, -0.001, 0.001)
        nn.init.uniform_(self.b1, -0.001, 0.001)
        nn.init.uniform_(self.w2, -0.001, 0.001)
        nn.init.uniform_(self.b2, -0.001, 0.001)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        if x.ndim != 1:
            raise ValueError(
                'Input must be 1-D tensor of shape (N,)'
            )
        z = x[:, None] * self.w1 + self.b1
        h = torch.tanh(z)
        return h @ self.w2 + self.b2
    