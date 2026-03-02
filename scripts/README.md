---
language:
- es
license: mit
library_name: pytorch
tags:
- text-classification
- multitask-learning
- sentence-transformers
- spanish
- intent-classification
pipeline_tag: text-classification
---

# PIBot Intent Router (Multitarea)

Modelo multitarea para clasificación de texto en español usando un encoder compartido (`sentence-transformers`) y 3 cabezas clasificadoras:

- `macro`: `"1"` o `"0"`
- `intent`: `"value"` , `"method"`, `"other"`
- `context`: `"standalone"` o `"followup"`

Este repositorio contiene artefactos de inferencia:

- `encoder/`
- `heads.pt`
- `label2id.json`
- `id2label.json`
- `train_config.json`

## Uso rápido

### 1) Instalar dependencias

```bash
pip install torch sentence-transformers huggingface-hub
```

Si usarás el código de este proyecto para inferencia local:

```bash
pip install -r requirements.txt
```

### 2) Descargar artefactos desde Hugging Face

```python
from huggingface_hub import snapshot_download

artifact_dir = snapshot_download(repo_id="TU_USUARIO/TU_REPO")
print(artifact_dir)
```

### 3) Inferencia con el código del proyecto

```python
from pathlib import Path

from src.serialization.artifacts import load_artifacts
from src.infer.predict import predict_all_tasks

artifact_dir = Path("RUTA_DESCARGADA_DESDE_SNAPSHOT")
encoder, multitask_model, _, id2label = load_artifacts(artifact_dir, device="cpu")

text = "quiero pagar mi factura"
output = predict_all_tasks(
    text=text,
    encoder=encoder,
    multitask_model=multitask_model,
    id2label=id2label,
    device="cpu",
)
print(output)
```

Salida esperada (ejemplo):

```json
{
  "macro": {"label": "1", "score": 0.97},
  "intent": {"label": "method", "score": 0.92},
  "context": {"label": "standalone", "score": 0.88}
}
```

## Uso por CLI

Con este proyecto clonado, también puedes probar:

```bash
python -m src.main test --artifact-dir artifacts --text "quiero pagar mi factura" --device cpu
```

## Notas

- El modelo está diseñado para inferencia de 3 tareas simultáneas.
- `score` corresponde a la probabilidad de la clase predicha por cada tarea.
- Para endpoint administrado en HF, se recomienda agregar `handler.py` y `requirements.txt` orientados al entorno de despliegue.
