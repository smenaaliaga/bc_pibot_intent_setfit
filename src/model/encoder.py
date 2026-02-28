from typing import List

import torch
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging


class SharedEncoder:
    def __init__(self, model_name: str, device: str = "cpu"):
        hf_logging.set_verbosity_error()
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.model.to(device)

    def encode(self, texts: List[str], convert_to_tensor: bool = True) -> torch.Tensor:
        return self.model.encode(texts, convert_to_tensor=convert_to_tensor)

    def encode_for_training(self, texts: List[str]) -> torch.Tensor:
        tokenized = self.model.tokenize(texts)
        tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
        output = self.model(tokenized)
        return output["sentence_embedding"]

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        hf_logging.set_verbosity_error()
        instance = cls.__new__(cls)
        instance.device = device
        instance.model = SentenceTransformer(path, device=device)
        instance.model.to(device)
        return instance
