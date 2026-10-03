from __future__ import annotations

import pandas as pd

def split_data(
    df: pd.DataFrame,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15
) -> tuple[pd.DataFrame]:
    "split time-series chronologically"
    
    if not (0 < train_fraction < 1):
        raise ValueError("train_fraction must be between 0 and 1")
    if not (0 < val_fraction < 1):
        raise ValueError("val_fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be less than 1")
    
    n = len(df)
    
    train_len = int(n*train_fraction)
    val_len = int(n*(train_fraction + val_fraction))

    train_set = df.iloc[:train_len].copy()
    val_set = df.iloc[train_len:val_len].copy()
    test_set = df.iloc[val_len:].copy()
    
    return (
        train_set,
        val_set,
        test_set
    )