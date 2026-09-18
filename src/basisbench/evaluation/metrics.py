from __future__ import annotations

import numpy as np
import pandas as pd

from numpy.typing import NDArray

def mean_square_error(ypred: NDArray, ytrue: NDArray) -> np.float:
    error = ypred - ytrue
    mse = np.mean(error * error)
    return mse

def root_mean_squared_error(ypred: NDArray, ytrue: NDArray) -> np.float:
    mse = mean_square_error(ypred, ytrue)
    return mse ** 0.5

def mean_absolute_error(ypred: NDArray, ytrue: NDArray) -> np.float:
    return np.mean(np.abs(ypred - ytrue))