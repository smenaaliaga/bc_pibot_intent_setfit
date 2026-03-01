from typing import Dict, List

import torch


def _extract_top_k(
    probs: torch.Tensor,
    id2label_task: Dict[int, str],
    top_k: int,
) -> List[Dict[str, float]]:
    k = min(max(top_k, 1), int(probs.shape[0]))
    top_probs, top_ids = torch.topk(probs, k=k, dim=-1)
    ranked: List[Dict[str, float]] = []
    for prob, class_id in zip(top_probs.tolist(), top_ids.tolist()):
        ranked.append({"label": id2label_task[int(class_id)], "score": float(prob)})
    return ranked


def predict_all_tasks_detailed(
    text: str,
    encoder,
    multitask_model,
    id2label: Dict[str, Dict[int, str]],
    device: str = "cpu",
    top_k: int = 3,
) -> Dict[str, Dict[str, object]]:
    embedding = encoder.encode([text], convert_to_tensor=True).to(device)
    results: Dict[str, Dict[str, object]] = {}

    with torch.no_grad():
        for task_name in ("macro", "intent", "context"):
            logits = multitask_model(task_name, embedding)
            probs = torch.softmax(logits, dim=-1)[0]
            ranked = _extract_top_k(probs=probs, id2label_task=id2label[task_name], top_k=top_k)
            best = ranked[0]
            results[task_name] = {
                "label": str(best["label"]),
                "score": float(best["score"]),
                "top_k": ranked,
            }

    return results


def predict_all_tasks(
    text: str,
    encoder,
    multitask_model,
    id2label: Dict[str, Dict[int, str]],
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    detailed = predict_all_tasks_detailed(
        text=text,
        encoder=encoder,
        multitask_model=multitask_model,
        id2label=id2label,
        device=device,
        top_k=1,
    )
    return {
        task_name: {"label": str(values["label"]), "score": float(values["score"])}
        for task_name, values in detailed.items()
    }
