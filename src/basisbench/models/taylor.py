from __future__ import annotations

import torch
import torch.nn as nn 
import pandas as pd

class Taylor(nn.Module):
    """
    Taylor Approximation Function

    ## Model:
    For polynomial degree d:

    ŷ = w₀ + w₁x + w₂x² + ... + w_dx^d

    """
    def __init__(self, degree: int) -> None:
        super().__init__()
        if degree <0:
            raise ValueError(
                'degree of a polynomial must be greater than or equal to zer'
            )
        self.degree = degree
        self.coefficients = nn.Parameter(torch.empty(degree + 1))
        
        #* initialize the parameters
        nn.init.uniform_(
            self.coefficients,
            -0.001, 0.001
        )
        
    def _design_matrix(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 1:
            raise ValueError(
                f'X must be a 1D tensor, got shape {tuple(x.shape)}'
            )
        return torch.vander(x, N=self.degree + 1, increasing=True)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        matrix = self._design_matrix(X)
        return (matrix * self.coefficients).sum(dim=-1)
    
    
if __name__ == '__main__':
    lr = 0.01
    epochs = 100
    
    model = Taylor(degree=3)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    train_df = pd.read_csv('C:\\Users\\User\\Desktop\\basisbench\\data\\proccessed\\train_set.csv')
    val_df = pd.read_csv('C:\\Users\\User\\Desktop\\basisbench\\data\\proccessed\\val_set.csv')
    test_df = pd.read_csv('C:\\Users\\User\\Desktop\\basisbench\\data\\proccessed\\test_set.csv')
    
    x_train, y_train = train_df['timenormalized'], train_df['norm_cnt']
    x_val, y_val = val_df['timenormalized'], val_df['norm_cnt']
    x_test, y_test = test_df['timenormalized'], test_df['norm_cnt']
    
    x_train = torch.tensor(x_train.to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    
    for epoch in range(epochs):
        #* kill previuos gradients
        optimizer.zero_grad()
        
        #* forward pass
        model.train()
        predictions = model(x_train)
        train_loss = loss_fn(predictions, y_train)
        
        #* backward pass
        train_loss.backward()
        
        #* update parametes
        optimizer.step()
        model.eval()
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch} training loss {train_loss.item()}')
            
    # print(tn.coefficients)