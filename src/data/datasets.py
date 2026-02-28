from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.config.schema import DataConfig


@dataclass
class TaskSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class TextLabelDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int):
        return self.texts[index], torch.tensor(self.labels[index], dtype=torch.long)


def _stratify_or_none(frame: pd.DataFrame, label_col: str):
    return frame[label_col] if frame[label_col].nunique() > 1 else None


def split_by_task(task_frames: Dict[str, pd.DataFrame], data_config: DataConfig) -> Dict[str, TaskSplit]:
    splits: Dict[str, TaskSplit] = {}
    for task_name, frame in task_frames.items():
        if not 0 <= data_config.test_size < 1:
            raise ValueError("test_size must be in [0, 1)")
        if not 0 <= data_config.val_size < 1:
            raise ValueError("val_size must be in [0, 1)")
        if data_config.test_size + data_config.val_size >= 1:
            raise ValueError("val_size + test_size must be < 1")

        if data_config.test_size > 0:
            train_val_df, test_df = train_test_split(
                frame,
                test_size=data_config.test_size,
                random_state=data_config.random_state,
                stratify=_stratify_or_none(frame, data_config.label_col),
            )
        else:
            train_val_df = frame
            test_df = frame.iloc[0:0].copy()

        val_ratio_on_train_val = data_config.val_size / max((1.0 - data_config.test_size), 1e-12)
        if val_ratio_on_train_val > 0:
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=val_ratio_on_train_val,
                random_state=data_config.random_state,
                stratify=_stratify_or_none(train_val_df, data_config.label_col),
            )
        else:
            train_df = train_val_df
            val_df = train_val_df.iloc[0:0].copy()

        splits[task_name] = TaskSplit(
            train=train_df.reset_index(drop=True),
            val=val_df.reset_index(drop=True),
            test=test_df.reset_index(drop=True),
        )
    return splits
