from __future__ import annotations

import torch
import torch.nn as nn


class Taylor(nn.Module):
    """
    Polynomial regression model:

    y = w0 + w1*x + w2*x^2 + ... + wd*x^d
    """

    def __init__(self, degree: int) -> None:
        super().__init__()

        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
            raise ValueError("degree must be a non-negative integer")

        self.degree = degree
        self.coefficients = nn.Parameter(torch.empty(degree + 1))

        nn.init.uniform_(self.coefficients, -0.001, 0.001)

    def _design_matrix(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 1:
            raise ValueError(f"X must be a 1D tensor, got shape {tuple(x.shape)}")

        return torch.vander(x, N=self.degree + 1, increasing=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        matrix = self._design_matrix(x)
        return matrix @ self.coefficients
