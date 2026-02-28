from typing import Dict

import torch
from torch import nn

from src.model.heads import ClassificationHead


class MultiTaskClassifier(nn.Module):
    def __init__(self, embedding_dim: int, num_classes_by_task: Dict[str, int]):
        super().__init__()
        self.heads = nn.ModuleDict(
            {task: ClassificationHead(embedding_dim, num_classes) for task, num_classes in num_classes_by_task.items()}
        )

    def forward(self, task: str, embeddings: torch.Tensor) -> torch.Tensor:
        if task not in self.heads:
            raise KeyError(f"Unknown task: {task}")
        return self.heads[task](embeddings)
