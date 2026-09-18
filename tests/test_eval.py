from __future__ import annotations

import numpy as np
from basisbench.evaluation.metrics import (mean_absolute_error,
                                        root_mean_squared_error,
                                        mean_square_error)

def test_mse():
    ypred = np.array([1,2,3,4])
    ytrue = np.array([1,2,3,4])
    np.isclose(mean_square_error(ypred, ytrue), 1/3)
    
def test_rmse():
    ypred = np.array([1,2,3,4])
    ytrue = np.array([1,2,3,4])
    np.isclose(root_mean_squared_error(ypred, ytrue), 1/3)
    
def test_mae():
    ypred = np.array([1,2,3,4])
    ytrue = np.array([1,2,3,4])
    np.isclose(mean_absolute_error(ypred, ytrue), 1/3)