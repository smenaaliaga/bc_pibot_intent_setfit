import json
from pathlib import Path
from typing import Dict

import torch

from src.model.encoder import SharedEncoder
from src.model.multitask_model import MultiTaskClassifier


def save_artifacts(
    output_dir: Path,
    encoder: SharedEncoder,
    multitask_model: MultiTaskClassifier,
    label2id: Dict[str, Dict[str, int]],
    train_config: Dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_dir = output_dir / "encoder"
    encoder.save(str(encoder_dir))

    torch.save(multitask_model.state_dict(), output_dir / "heads.pt")

    with open(output_dir / "label2id.json", "w", encoding="utf-8") as file:
        json.dump(label2id, file, ensure_ascii=False, indent=2)

    id2label = {
        task: {str(index): label for label, index in task_map.items()} for task, task_map in label2id.items()
    }
    with open(output_dir / "id2label.json", "w", encoding="utf-8") as file:
        json.dump(id2label, file, ensure_ascii=False, indent=2)

    with open(output_dir / "train_config.json", "w", encoding="utf-8") as file:
        json.dump(train_config, file, ensure_ascii=False, indent=2, default=str)


def load_artifacts(artifact_dir: Path, device: str = "cpu"):
    encoder = SharedEncoder.load(str(artifact_dir / "encoder"), device=device)

    with open(artifact_dir / "label2id.json", "r", encoding="utf-8") as file:
        label2id = json.load(file)

    num_classes_by_task = {task: len(task_map) for task, task_map in label2id.items()}
    embedding_dim = encoder.encode(["_probe_"], convert_to_tensor=True).shape[-1]
    multitask_model = MultiTaskClassifier(embedding_dim=embedding_dim, num_classes_by_task=num_classes_by_task)
    multitask_model.load_state_dict(torch.load(artifact_dir / "heads.pt", map_location=device))
    multitask_model.to(device)
    multitask_model.eval()

    with open(artifact_dir / "id2label.json", "r", encoding="utf-8") as file:
        raw_id2label = json.load(file)
    id2label = {
        task: {int(index): label for index, label in task_map.items()}
        for task, task_map in raw_id2label.items()
    }

    return encoder, multitask_model, label2id, id2label
