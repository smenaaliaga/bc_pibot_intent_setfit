from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    data_dir: Path = Path("data")
    macro_file: str = "dataset_macro.csv"
    intent_file: str = "dataset_intent.csv"
    context_file: str = "dataset_context.csv"
    text_col: str = "text"
    label_col: str = "label"
    val_size: float = 0.2
    test_size: float = 0.1
    random_state: int = 42


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    max_length: int = 128
    epochs: int = 3
    batch_size: int = 16
    lr_encoder: float = 2e-5
    lr_heads: float = 1e-3
    weight_decay: float = 0.01
    task_weight_macro: float = 1.0
    task_weight_intent: float = 1.0
    task_weight_context: float = 1.0
    device: str = "cpu"
    output_dir: Path = Path("artifacts")
    seed: int = 42
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    patience: int = 5
    min_delta: float = 0.001


TASKS = ("macro", "intent", "context")
