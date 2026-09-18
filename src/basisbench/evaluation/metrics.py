from __future__ import annotations

import numpy as np
import pandas as pd

from numpy.typing import NDArray


def mean_square_error(ypred: NDArray, ytrue: NDArray) -> np.float:
    """Calculate the mean squared error between predictions and observations.

    The mean squared error (MSE) is the arithmetic mean of the squared
    differences between each predicted value and its corresponding true value.
    Because the errors are squared, larger errors have a greater influence on
    the result. A value of zero indicates perfect predictions.

    Args:
        ypred: Array containing predicted values.
        ytrue: Array containing the corresponding observed or target values.

    Returns:
        The mean squared error as a scalar numeric value.

    Raises:
        ValueError: If ``ypred`` and ``ytrue`` cannot be broadcast together.
    """
    error = ypred - ytrue
    mse = np.mean(error * error)
    return mse


def root_mean_squared_error(ypred: NDArray, ytrue: NDArray) -> np.float:
    """Calculate the root mean squared error between predictions and observations.

    The root mean squared error (RMSE) is the square root of the mean squared
    error. It penalizes larger errors more strongly than smaller errors while
    remaining in the same units as the predicted and observed values. A value
    of zero indicates perfect predictions.

    Args:
        ypred: Array containing predicted values.
        ytrue: Array containing the corresponding observed or target values.

    Returns:
        The root mean squared error as a scalar numeric value.

    Raises:
        ValueError: If ``ypred`` and ``ytrue`` cannot be broadcast together.
    """
    mse = mean_square_error(ypred, ytrue)
    return mse ** 0.5


def mean_absolute_error(ypred: NDArray, ytrue: NDArray) -> np.float:
    """Calculate the mean absolute error between predictions and observations.

    The mean absolute error (MAE) is the arithmetic mean of the absolute
    differences between each predicted value and its corresponding true value.
    Unlike squared-error metrics, MAE weights errors linearly and is therefore
    generally less sensitive to outliers. A value of zero indicates perfect
    predictions.

    Args:
        ypred: Array containing predicted values.
        ytrue: Array containing the corresponding observed or target values.

    Returns:
        The mean absolute error as a scalar numeric value.

    Raises:
        ValueError: If ``ypred`` and ``ytrue`` cannot be broadcast together.
    """
    return np.mean(np.abs(ypred - ytrue))
