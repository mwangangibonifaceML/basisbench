from __future__ import annotations

import pandas as pd
import numpy as np
import torch
import argparse
import torch.nn as nn

from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class TimeScaler:
    """Affine transform that maps a time column to approximately [-1, 1]."""
    
    minimum: float
    maximum: float
    
    def transform(self, values: np.ndarray) -> np.ndarray:
        gap = self.maximum - self.minimum
        if gap <=0:
            raise ValueError(
                'The training time column must have atleast two values'
            )
        return 2.0 * (values - self.minimum) / gap - 1
    
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=Path("C:\\Users\\User\\Desktop\\basisbench\\data\\raw\\hour.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("C:\\Users\\User\\Desktop\\basisbench\\data\\processed"))
    parser.add_argument("--train", type=Path, default=Path("C:\\Users\\User\\Desktop\\basisbench\\data\\processed\\train_set.csv"))
    parser.add_argument("--validation", type=Path, default=Path("C:\\Users\\User\\Desktop\\basisbench\\data\\processed\\val_set.csv"))
    parser.add_argument("--test", type=Path, default=Path("C:\\Users\\User\\Desktop\\basisbench\\data\\processed\\test_set.csv"))
    parser.add_argument(
        "--time-column",
        default=None,
        help="Numeric time column. Defaults to timenormalized, or instant if unavailable.",
    )
    parser.add_argument("--target-column", default="cnt")
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--max-frequency", type=int, default=5)
    parser.add_argument('--hidden-size', type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/taylor"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd", "adam"])
    return parser.parse_args()

def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module
) -> tuple[float, float, float, float]:
    model.eval()
    
    num_samples = 0
    total_absolute_error = 0.0
    total_mean_squared_error = 0.0
    total_root_mean_squared_error = 0.0
    total_log_sqared_error = 0.0
    
    with torch.no_grad():
        for x, y_log in dataloader:
            prediction_log = model(x)

            prediction = torch.expm1(prediction_log).clamp_min(0.0)
            actual = torch.expm1(y_log)
            
            total_absolute_error += torch.sum(torch.abs(prediction - actual)).item()
            total_mean_squared_error += torch.sum((prediction - actual) ** 2).item()
            total_root_mean_squared_error += torch.sum((prediction - actual) ** 2).item()
            total_log_sqared_error += torch.sum((prediction_log - y_log) ** 2).item()
            # total_loss += loss_fn(prediction_log, y_log).item() * x.size(0)
            num_samples += x.size(0)    
            
    mse_log = total_log_sqared_error / num_samples
    mae = total_absolute_error / num_samples
    mse = total_mean_squared_error / num_samples
    rmse = torch.sqrt(torch.tensor(total_root_mean_squared_error / num_samples)).item()

    return mse_log,mse, rmse, mae

def load_data(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.validation)
    test_df = pd.read_csv(args.test)
    return train_df, val_df, test_df