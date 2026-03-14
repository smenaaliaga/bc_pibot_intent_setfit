# bc_pibot_intent_setfit

Clasificador multitarea para routing de preguntas macroeconómicas. Usa un **encoder compartido** con **3 cabezas clasificadoras** independientes entrenadas de forma conjunta.

## Arquitectura

```
Texto → [Encoder multilingual-mpnet-base-v2 (278M params, 768 dim)]
              │
              ├── macro_cls:   Linear(768,384) → ReLU → Dropout(0.3) → Linear(384,2)   → {1, 0}
              ├── intent_cls:  Linear(768,384) → ReLU → Dropout(0.3) → Linear(384,3)   → {value, methodology, other}
              └── context_cls: Linear(768,384) → ReLU → Dropout(0.3) → Linear(384,2)   → {standalone, followup}
```

| Tarea | Clases | Descripción |
|-------|--------|-------------|
| `macro` | `1`, `0` | ¿Es una pregunta macroeconómica? |
| `intent` | `value`, `methodology`, `other` | ¿Pide un dato, una explicación metodológica, o es ambiguo? |
| `context` | `standalone`, `followup` | ¿La pregunta es independiente o depende de la anterior? |

## Datasets

En `data/` deben existir 3 CSVs con columnas `text` y `label`:

| Archivo | Ejemplos | Distribución |
|---------|----------|--------------|
| `dataset_macro.csv` | 794 | 610 macro=1, 184 macro=0 |
| `dataset_intent.csv` | 1169 | 748 value, 266 methodology, 155 other |
| `dataset_context.csv` | 1054 | 535 standalone, 519 followup |

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### GPU (opcional)

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Entrenamiento

### Comando recomendado (fine-tuning completo, sin LoRA)

```bash
python -m src.main train \
  --model-name sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
  --data-dir data \
  --output-dir models/artifacts \
  --device cuda \
  --val-size 0.1 \
  --test-size 0.1 \
  --epochs 25 \
  --batch-size 16 \
  --lr-encoder 8e-6 \
  --lr-heads 4e-4 \
  --weight-decay 0.05 \
  --task-weight-macro 1.2 \
  --task-weight-intent 1.5 \
  --task-weight-context 0.8 \
  --patience 5 \
  --seed 42
```

### Parámetros clave

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--model-name` | `paraphrase-multilingual-MiniLM-L12-v2` | Encoder sentence-transformers |
| `--epochs` | `3` | Máximo de épocas (early stopping puede cortar antes) |
| `--lr-encoder` | `2e-5` | Learning rate del encoder |
| `--lr-heads` | `1e-3` | Learning rate de las cabezas |
| `--weight-decay` | `0.01` | Regularización L2 |
| `--task-weight-intent` | `1.0` | Peso de la tarea intent (subir para clases desbalanceadas) |
| `--patience` | `5` | Épocas sin mejora antes de parar (0 = desactivado) |
| `--min-delta` | `0.001` | Mejora mínima en val F1 para resetear patience |

### Early stopping

El trainer monitorea el **promedio de F1-macro** de las 3 tareas sobre validación. Si no mejora en `--patience` épocas consecutivas:
1. Detiene el entrenamiento
2. Restaura los pesos del mejor epoch

Esto permite usar `--epochs 20` sin riesgo de sobreajuste: el modelo se queda con la mejor versión.

### LoRA (opcional)

Para entrenar solo adaptadores LoRA en lugar de fine-tuning completo:

```bash
python -m src.main train --data-dir data --output-dir models/artifacts --device cuda --use-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 --epochs 20 --patience 5
```

Sin el flag `--use-lora`, se hace fine-tuning completo (recomendado con alta capacidad de cómputo).

## Evaluación

```bash
python -m src.main evaluate --artifact-dir models/artifacts --data-dir data --split test --device cuda
```

## Test por texto

```bash
python -m src.main test --artifact-dir models/artifacts --text "cuanto crecio el pib" --device cuda
```

Múltiples textos:

```bash
python -m src.main test --artifact-dir models/artifacts --text "cuanto crecio el pib" --text "como se calcula el imacec"
```

Desde archivo (una línea por texto):

```bash
python -m src.main test --artifact-dir models/artifacts --texts-file data/texts_for_test.txt
```

## Modo interactivo

```bash
python -m src.main interactive --artifact-dir models/artifacts --device cuda
```

## Subir a Hugging Face

### Setup (una vez)

```bash
pip install huggingface-hub
huggingface-cli login
```

### Upload

```bash
python -m src.main upload --artifact-dir models/artifacts --repo-id BCCh/pibot-intent-router --clean-repo
```

Con token explícito (CI):

```bash
python -m src.main upload --artifact-dir models/artifacts --repo-id BCCh/pibot-intent-router --hf-token $HF_TOKEN --clean-repo
```

`--clean-repo` elimina archivos previos en HF que no estén en la subida actual.

## Artefactos generados

En `models/artifacts/`:

```
encoder/              # Encoder sentence-transformers serializado
heads.pt              # Pesos de las 3 cabezas clasificadoras
label2id.json         # Mapeo label → id por tarea
id2label.json         # Mapeo id → label por tarea
train_config.json     # Configuración usada en entrenamiento
```

## Despliegue en endpoint

Para Hugging Face Inference Endpoints, además de los artefactos se necesita un `handler.py` y `requirements.txt` en el repo para que el endpoint sepa inicializar y ejecutar la predicción multitarea.
