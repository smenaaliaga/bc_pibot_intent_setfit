from src.infer.predict import predict_all_tasks, predict_all_tasks_detailed


class DummyEncoder:
    def encode(self, texts, convert_to_tensor=True):
        import torch

        return torch.ones((len(texts), 4), dtype=torch.float)


class DummyModel:
    def __call__(self, task, embedding):
        import torch

        if task == "macro":
            return torch.tensor([[0.1, 0.9]], dtype=torch.float)
        if task == "intent":
            return torch.tensor([[0.8, 0.2]], dtype=torch.float)
        return torch.tensor([[0.3, 0.7]], dtype=torch.float)


def test_predict_all_tasks_smoke():
    id2label = {
        "macro": {0: "0", 1: "1"},
        "intent": {0: "value", 1: "methodology", 2: "other"},
        "context": {0: "standalone", 1: "followup"},
    }
    result = predict_all_tasks("hola", DummyEncoder(), DummyModel(), id2label)
    assert set(result.keys()) == {"macro", "intent", "context"}
    assert "label" in result["macro"] and "score" in result["macro"]


def test_predict_all_tasks_detailed_smoke():
    id2label = {
        "macro": {0: "0", 1: "1"},
        "intent": {0: "value", 1: "methodology", 2: "other"},
        "context": {0: "standalone", 1: "followup"},
    }
    result = predict_all_tasks_detailed("hola", DummyEncoder(), DummyModel(), id2label, top_k=3)
    assert set(result.keys()) == {"macro", "intent", "context"}
    assert "top_k" in result["intent"]
    assert len(result["intent"]["top_k"]) >= 1
