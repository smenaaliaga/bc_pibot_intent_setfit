import torch
from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        hidden = in_features // 2
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)
