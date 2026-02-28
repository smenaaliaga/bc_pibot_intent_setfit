from typing import Dict


TASK_WEIGHT_KEYS = {
    "macro": "task_weight_macro",
    "intent": "task_weight_intent",
    "context": "task_weight_context",
}


def get_task_weight(task_name: str, config: object) -> float:
    key = TASK_WEIGHT_KEYS[task_name]
    return float(getattr(config, key))


def get_task_weights(config: object) -> Dict[str, float]:
    return {task: get_task_weight(task, config) for task in TASK_WEIGHT_KEYS}
