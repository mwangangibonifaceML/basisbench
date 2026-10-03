from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from basisbench.data.splitting import split_data

def load_raw_data(path: str | Path) -> pd.DataFrame:
    "Load the raw Bike sharing dataset"
    return pd.read_csv(path)

def create_time_stamps_and_timehours_columns(dataframe: pd.DataFrame):
    dataframe['datetime'] = (
            pd.to_datetime(dataframe['dteday']) +
            pd.to_timedelta(dataframe['hr'], unit='h')
        )
    dataframe['timehour'] = (
            dataframe['datetime'] - dataframe['datetime'].min()      
    ).dt.total_seconds() / 3600
    return dataframe

def preprocess_data(
    input_path: str | Path,
    output_path: str | Path) -> pd.DataFrame:
    
    output_path = Path(output_path)
    input_path = Path(input_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    #* load the data and create time stamps creating 
    #* a new column with proper datetime column
    dataframe = load_raw_data(input_path)
    dataframe = create_time_stamps_and_timehours_columns(dataframe)
    
    #* split the data and save the splits
    train_df, val_df, test_df = split_data(dataframe)
    test_df.to_csv(output_path/'test_set.csv', index=False)
    train_df.to_csv(output_path/'train_set.csv', index=False)
    val_df.to_csv(output_path/'val_set.csv', index=False)

def choose_time_column(frame: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"time column {requested!r} is not present in the CSV")
        return requested

    for candidate in ("timehours", "timehour"):
        if candidate in frame.columns:
            return candidate

    raise ValueError(
        "could not find a time column; pass --time-column with a numeric column "
        "such as 'instant or datetime'"
    )
    
def numeric_column(frame: pd.DataFrame, column: str, split_name: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"{column!r} is missing from the {split_name} CSV")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{column!r} contains missing or non-numeric values in {split_name}")
    return values

    


