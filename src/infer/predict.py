from typing import Dict

import torch


def predict_all_tasks(
    text: str,
    encoder,
    multitask_model,
    id2label: Dict[str, Dict[int, str]],
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    embedding = encoder.encode([text], convert_to_tensor=True).to(device)
    results: Dict[str, Dict[str, float]] = {}

    with torch.no_grad():
        for task_name in ("macro", "intent", "context"):
            logits = multitask_model(task_name, embedding)
            probs = torch.softmax(logits, dim=-1)[0]
            pred_id = int(torch.argmax(probs).item())
            score = float(probs[pred_id].item())
            results[task_name] = {
                "label": id2label[task_name][pred_id],
                "score": score,
            }

    return results
