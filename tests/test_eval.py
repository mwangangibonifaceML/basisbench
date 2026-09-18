from __future__ import annotations

import numpy as np
import pytest

from basisbench.evaluation.metrics import (
    mean_absolute_error,
    mean_square_error,
    root_mean_squared_error,
)


@pytest.mark.parametrize(
    "ypred, ytrue, expected_mse, expected_rmse, expected_mae",
    [
        (
            np.array([1, 2, 3, 4]),
            np.array([0, 2, 5, 4]),
            1.25,
            np.sqrt(1.25),
            0.75,
        ),
        (
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 3, 4]),
            0.0,
            0.0,
            0.0,
        ),
    ],
)
def test_metrics(ypred, ytrue, expected_mse, expected_rmse, expected_mae):
    assert np.isclose(mean_square_error(ypred, ytrue), expected_mse)
    assert np.isclose(root_mean_squared_error(ypred, ytrue), expected_rmse)
    assert np.isclose(mean_absolute_error(ypred, ytrue), expected_mae)
