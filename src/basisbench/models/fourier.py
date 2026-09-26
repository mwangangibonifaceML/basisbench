from __future__ import annotations

import torch
import torch.nn as nn 

class Fourier(nn.Module):
    def __init__(self, max_frequency: int) -> None:
        super().__init__()
        if not isinstance(max_frequency, int) or max_frequency <=0:
            raise ValueError(
                'Max frequency of a Fourier series must be a positive interger'
            )
            
        self.max_frequency = max_frequency
        
        #* define the parameters
        self.cos_coefficients = nn.Parameter(torch.empty(max_frequency))
        self.sin_coefficients = nn.Parameter(torch.empty(max_frequency))
        self.bias = nn.Parameter(torch.empty(1, dtype=torch.float32))
        
        #* initialize the parameters
        nn.init.uniform_(self.cos_coefficients, -0.001, 0.001)
        nn.init.uniform_(self.sin_coefficients, -0.001, 0.001)
        nn.init.uniform_(self.bias, -0.001, 0.001)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = self.bias
        
        for k in range(1,self.max_frequency+1):
            angles = k*X
            output = sum(
                (self.cos_coefficients[k-1] * torch.cos(angles),
                self.sin_coefficients[k-1] * torch.sin(angles)),
                start= output
            )
        return output
        
        
if __name__ == '__main__':
    f = 3
    x = [i/10 for i in range(10)]
    
    x = torch.tensor([ix for ix in x], dtype=torch.float32)
    angles = f*x
    y = torch.cos(angles) + torch.sin(angles)
    
    model = Fourier(max_frequency=f)
    output = model(x)
    print(output)