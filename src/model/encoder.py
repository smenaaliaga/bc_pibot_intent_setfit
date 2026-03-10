from typing import List, Optional

import torch
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging


class SharedEncoder:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        hf_logging.set_verbosity_error()
        self.device = device
        self.use_lora = use_lora
        self.model = SentenceTransformer(model_name, device=device)
        self.model.to(device)
        
        if use_lora:
            self._apply_lora(lora_r, lora_alpha, lora_dropout)

    def _apply_lora(self, r: int, alpha: int, dropout: float) -> None:
        """Aplicar LoRA al transformer subyacente del modelo."""
        # Acceder al transformer dentro de SentenceTransformer
        # SentenceTransformer es un Sequential, el primer módulo es el transformer
        transformer_module = None
        for module in self.model.modules():
            if hasattr(module, 'auto_model'):
                transformer_module = module
                break
        
        if transformer_module is None:
            print("Advertencia: No se encontró transformer module, intentando directamente")
            transformer_module = self.model[0]
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["query", "value"],  # Aplicar a query y value projections en BERT
            lora_dropout=dropout,
            bias="none",
            task_type=None,
        )
        
        # Aplicar LoRA al modelo
        transformer_module.auto_model = get_peft_model(transformer_module.auto_model, lora_config)
        transformer_module.auto_model.print_trainable_parameters()

    def encode(self, texts: List[str], convert_to_tensor: bool = True) -> torch.Tensor:
        return self.model.encode(texts, convert_to_tensor=convert_to_tensor)

    def encode_for_training(self, texts: List[str]) -> torch.Tensor:
        tokenized = self.model.tokenize(texts)
        tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
        output = self.model(tokenized)
        return output["sentence_embedding"]

    def save(self, path: str) -> None:
        if self.use_lora:
            # Guardar solo los pesos de LoRA
            for module in self.model.modules():
                if hasattr(module, 'auto_model') and hasattr(module.auto_model, 'save_pretrained'):
                    module.auto_model.save_pretrained(f"{path}/lora_weights")
                    break
        self.model.save(path)

    @classmethod
    def load(
        cls,
        path: str,
        device: str = "cpu",
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        hf_logging.set_verbosity_error()
        instance = cls.__new__(cls)
        instance.device = device
        instance.use_lora = use_lora
        instance.model = SentenceTransformer(path, device=device)
        instance.model.to(device)
        
        if use_lora:
            instance._apply_lora(lora_r, lora_alpha, lora_dropout)
            # Cargar pesos de LoRA si existen
            try:
                from pathlib import Path as PathlibPath
                lora_path = PathlibPath(path) / "lora_weights"
                if lora_path.exists():
                    from peft import PeftModel
                    for module in instance.model.modules():
                        if hasattr(module, 'auto_model'):
                            module.auto_model = PeftModel.from_pretrained(
                                module.auto_model, str(lora_path)
                            )
                            break
            except Exception as e:
                print(f"Advertencia: No se pudieron cargar pesos de LoRA: {e}")
        
        return instance
