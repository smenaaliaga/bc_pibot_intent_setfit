from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.schema import TrainConfig
from src.data.datasets import TextLabelDataset
from src.eval.metrics import compute_metrics
from src.model.encoder import SharedEncoder
from src.model.multitask_model import MultiTaskClassifier
from src.train.losses import get_task_weight


@dataclass
class TrainArtifacts:
    encoder: SharedEncoder
    multitask_model: MultiTaskClassifier
    history: Dict[str, List[float]]


class MultiTaskTrainer:
    def __init__(
        self,
        encoder: SharedEncoder,
        model: MultiTaskClassifier,
        train_config: TrainConfig,
    ):
        self.encoder = encoder
        self.model = model
        self.train_config = train_config
        self.device = torch.device(train_config.device)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        encoder_params = list(self.encoder.model.parameters())
        head_params = list(self.model.parameters())

        self.optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": train_config.lr_encoder},
                {"params": head_params, "lr": train_config.lr_heads},
            ],
            weight_decay=train_config.weight_decay,
        )

    def _make_loader(self, dataset: TextLabelDataset, shuffle: bool = True) -> DataLoader:
        return DataLoader(dataset, batch_size=self.train_config.batch_size, shuffle=shuffle)

    def train(
        self,
        train_datasets: Dict[str, TextLabelDataset],
        val_datasets: Dict[str, TextLabelDataset],
    ) -> TrainArtifacts:
        history: Dict[str, List[float]] = {"loss": []}

        for epoch in range(self.train_config.epochs):
            self.model.train()
            epoch_loss = 0.0
            step_count = 0

            loaders = {task: self._make_loader(ds, shuffle=True) for task, ds in train_datasets.items()}
            iterators = {task: iter(loader) for task, loader in loaders.items()}
            max_steps = max(len(loader) for loader in loaders.values())

            loop = tqdm(range(max_steps), desc=f"Epoch {epoch + 1}/{self.train_config.epochs}")
            for _ in loop:
                for task_name in ("macro", "intent", "context"):
                    iterator = iterators[task_name]
                    try:
                        texts, labels = next(iterator)
                    except StopIteration:
                        iterators[task_name] = iter(loaders[task_name])
                        texts, labels = next(iterators[task_name])

                    labels = labels.to(self.device)
                    embeddings = self.encoder.encode_for_training(list(texts))
                    logits = self.model(task_name, embeddings)
                    task_weight = get_task_weight(task_name, self.train_config)
                    loss = self.criterion(logits, labels) * task_weight

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += float(loss.item())
                    step_count += 1

            avg_loss = epoch_loss / max(step_count, 1)
            history["loss"].append(avg_loss)
            val_metrics = self.evaluate(val_datasets)
            print(f"Epoch {epoch + 1} | train_loss={avg_loss:.4f} | val={val_metrics}")

        return TrainArtifacts(encoder=self.encoder, multitask_model=self.model, history=history)

    @torch.no_grad()
    def evaluate(self, val_datasets: Dict[str, TextLabelDataset]) -> Dict[str, Dict[str, float]]:
        self.model.eval()
        metrics_by_task: Dict[str, Dict[str, float]] = {}

        for task_name, dataset in val_datasets.items():
            if len(dataset) == 0:
                metrics_by_task[task_name] = {"accuracy": float("nan"), "f1_macro": float("nan")}
                continue

            loader = self._make_loader(dataset, shuffle=False)
            y_true: List[int] = []
            y_pred: List[int] = []

            for texts, labels in loader:
                labels = labels.to(self.device)
                embeddings = self.encoder.encode(list(texts), convert_to_tensor=True).to(self.device)
                logits = self.model(task_name, embeddings)
                preds = torch.argmax(logits, dim=-1)

                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

            metrics_by_task[task_name] = compute_metrics(y_true, y_pred)

        return metrics_by_task


def build_datasets_from_frames(
    split_frames: Dict[str, object],
    label_maps: Dict[str, Dict[str, int]],
    text_col: str,
    label_col: str,
) -> Tuple[Dict[str, TextLabelDataset], Dict[str, TextLabelDataset], Dict[str, TextLabelDataset]]:
    train_datasets: Dict[str, TextLabelDataset] = {}
    val_datasets: Dict[str, TextLabelDataset] = {}
    test_datasets: Dict[str, TextLabelDataset] = {}

    for task_name, split in split_frames.items():
        label2id = label_maps[task_name]

        train_texts = split.train[text_col].tolist()
        train_labels = [label2id[label] for label in split.train[label_col].tolist()]

        val_texts = split.val[text_col].tolist()
        val_labels = [label2id[label] for label in split.val[label_col].tolist()]

        test_texts = split.test[text_col].tolist()
        test_labels = [label2id[label] for label in split.test[label_col].tolist()]

        train_datasets[task_name] = TextLabelDataset(train_texts, train_labels)
        val_datasets[task_name] = TextLabelDataset(val_texts, val_labels)
        test_datasets[task_name] = TextLabelDataset(test_texts, test_labels)

    return train_datasets, val_datasets, test_datasets
