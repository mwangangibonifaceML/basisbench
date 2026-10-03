from __future__ import annotations

import torch
import pandas as pd
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from basisbench.data.proccess_data import numeric_column, choose_time_column
from basisbench.utils.utils import TimeScaler
from basisbench.utils.utils import parse_args

def make_tensors(
    frame: pd.DataFrame,
    split_name: str,
    time_column: str,
    target_column: str,
    scaler: TimeScaler,
) -> TensorDataset:
    
    time_column = choose_time_column(frame, requested= time_column)
    time = scaler.transform(numeric_column(frame, time_column, split_name))
    target = numeric_column(frame, target_column, split_name)
    if (target < 0).any():
        raise ValueError("cnt must be non-negative when using the log1p target transform")

    #* log1p reduces the effect of the highly skewed bike-count target.
    x = torch.from_numpy(time.astype(np.float32))
    y = torch.from_numpy(np.log1p(target).astype(np.float32))
    
    return TensorDataset(x,y)
    
def create_split_dataloader(
    frame: pd.DataFrame,
    split_name: str,
    scaler: TimeScaler,
    batch_size: int = 32,
    time_column: str = 'timehour',
    target_column: str = 'cnt'
) -> torch.utils.data.DataLoader:
    
    tensor_dataset = make_tensors(
        frame=frame,
        split_name=split_name,
        time_column=time_column,
        target_column=target_column,
        scaler=scaler
    )
    
    return DataLoader(tensor_dataset, batch_size= batch_size, shuffle=(split_name=='train'))