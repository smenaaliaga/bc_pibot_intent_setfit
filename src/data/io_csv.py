from pathlib import Path
from typing import Dict, Tuple
import re

import pandas as pd

from src.config.schema import DataConfig


_EXCLUDED_TOPIC_REGEX = re.compile(
    r"\b(?:ipc|inflacion|tpm|tasa\s+de\s+politica\s+monetaria)\b",
    re.IGNORECASE,
)


def _read_and_validate(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"{path} must include columns: {text_col}, {label_col}")
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(str)
    excluded_mask = df[text_col].str.contains(_EXCLUDED_TOPIC_REGEX)
    if excluded_mask.any():
        df = df.loc[~excluded_mask].copy()
    return df


def load_task_data(data_config: DataConfig) -> Dict[str, pd.DataFrame]:
    data_dir = data_config.data_dir
    return {
        "macro": _read_and_validate(data_dir / data_config.macro_file, data_config.text_col, data_config.label_col),
        "intent": _read_and_validate(data_dir / data_config.intent_file, data_config.text_col, data_config.label_col),
        "context": _read_and_validate(data_dir / data_config.context_file, data_config.text_col, data_config.label_col),
    }


def build_label_maps(df: pd.DataFrame, label_col: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = sorted(df[label_col].unique().tolist())
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    return label2id, id2label
