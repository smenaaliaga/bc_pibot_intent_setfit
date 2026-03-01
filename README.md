# bc_pibot_intent_setfit

Proyecto end-to-end para entrenar **3 cabezas clasificadoras multitarea** con un **sentence embedding compartido**:

- `macro_cls`: `{1, 0}`
- `intent_cls`: `{value, methodology, other}`
- `context_mode_cls`: `{standalone, followup}`

Incluye:
- Entrenamiento multitarea real (`train`)
- Evaluación (`evaluate`)
- Test por texto (`test`)
- Modo interactivo (`interactive`)
- Modo QA por consola (`qa`)
- Subida a Hugging Face (`upload` o `scripts/upload_to_hf.py`)

## 1) Estructura esperada de datos

En `data/` deben existir:

- `dataset_macro.csv`
- `dataset_intent.csv`
- `dataset_context.csv`

Cada CSV debe tener columnas obligatorias:

- `text`
- `label`

## 2) Instalación

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Opcional: habilitar GPU (NVIDIA)

Si quieres entrenar en GPU, instala PyTorch con CUDA:

```bash
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verificación rápida:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

## 3) Entrenamiento

```bash
python -m src.main train --data-dir data --output-dir artifacts --device cpu
```

Para GPU:

```bash
python -m src.main train --data-dir data --output-dir artifacts --device cuda
```

Por defecto el split es `train/val/test` con `--val-size 0.2` y `--test-size 0.1`.
Si quieres ajustar proporciones:

```bash
python -m src.main train --data-dir data --output-dir artifacts --device cuda --val-size 0.1 --test-size 0.1
```

Durante `train` se imprime:
- tamaño por tarea para `train`, `val`, `test`
- evaluación por época sobre `val`
- evaluación final explícita sobre `val` y `test`

### Configuración recomendada (según tus datasets actuales)

Con los datasets actuales (macro más desbalanceado y context más pequeño), una configuración robusta es:

- `--device cuda`
- `--val-size 0.1`
- `--test-size 0.1`
- `--epochs 6`
- `--batch-size 16`
- `--max-length 32`
- `--lr-encoder 2e-5`
- `--lr-heads 8e-4`
- `--task-weight-macro 1.2`
- `--task-weight-intent 1.0`
- `--task-weight-context 1.6`
- `--seed 42`

Comando recomendado:

```bash
python -m src.main train --data-dir data --output-dir artifacts --device cuda --val-size 0.15 --test-size 0.15 --epochs 8 --batch-size 16 --max-length 24 --lr-encoder 2e-5 --lr-heads 7e-4 --task-weight-macro 1.0 --task-weight-intent 1.0 --task-weight-context 1.8 --seed 42
```

También puedes parametrizar nombres de archivos y columnas:

```bash
python -m src.main train --data-dir data --macro-file dataset_macro.csv --intent-file dataset_intent.csv --context-file dataset_context.csv --text-col text --label-col label
```

Parámetros importantes:
- `--model-name`
- `--epochs`
- `--batch-size`
- `--lr-encoder`
- `--lr-heads`
- `--task-weight-macro`
- `--task-weight-intent`
- `--task-weight-context`
- `--test-size`
- `--val-size`
- `--seed`

## 4) Evaluación

```bash
python -m src.main evaluate --artifact-dir artifacts --data-dir data --device cpu
```

Puedes elegir split de evaluación:

```bash
python -m src.main evaluate --artifact-dir artifacts --data-dir data --split test --val-size 0.1 --test-size 0.1
```

## 5) Test por texto

Predice las **3 tareas a la vez** con score:

```bash
python -m src.main test --artifact-dir artifacts --text "quiero pagar mi factura" --device cpu
```

Múltiples textos en la misma ejecución:

```bash
python -m src.main test --artifact-dir artifacts --text "hola" --text "necesito ayuda"
```

O desde archivo (una línea por texto):

```bash
python -m src.main test --artifact-dir artifacts --texts-file data/texts_for_test.txt
```

## 6) Modo interactivo

```bash
python -m src.main interactive --artifact-dir artifacts --device cuda
```

Escribe `exit` para terminar.

## 6.1) Modo QA por consola

Muestra clasificación por tarea y ranking top-k por cada una:

```bash
python -m src.main qa --artifact-dir artifacts --device cpu --top-k 3
```

Escribe `exit` para terminar.

## 7) Subir a Hugging Face

### 7.1) Prerrequisitos

- Python 3.10+ con entorno virtual activo.
- Dependencia instalada:

```bash
pip install huggingface-hub
```

- Cuenta en https://huggingface.co con token de escritura (`write`) en Settings → Access Tokens.

### 7.2) Login en Hugging Face (una sola vez)

```bash
huggingface-cli login
```

Pega el token cuando se solicite. Queda guardado en cache local de Hugging Face.

### 7.3) Subir los artefactos del modelo

Después de `huggingface-cli login`, puedes subir **sin** pasar token explícito:

### Opción A: Desde `main.py` (sin token)

```bash
python -m src.main upload --artifact-dir artifacts --repo-id TU_USUARIO/TU_REPO --clean-repo
```

### Opción B: Script dedicado (sin token)

```bash
python scripts/upload_to_hf.py --artifact-dir artifacts --repo-id TU_USUARIO/TU_REPO --clean-repo
```

Si prefieres pasar token explícito (por ejemplo en CI), usa variable de entorno:

PowerShell (Windows):

```bash
$env:HF_TOKEN="hf_xxx"
```

### Opción A: Desde `main.py` (con token)

```bash
python -m src.main upload --artifact-dir artifacts --repo-id TU_USUARIO/TU_REPO --hf-token $env:HF_TOKEN --clean-repo
```

### Opción B: Script dedicado (con token)

```bash
python scripts/upload_to_hf.py --artifact-dir artifacts --repo-id TU_USUARIO/TU_REPO --hf-token $env:HF_TOKEN --clean-repo
```

Ambas opciones crean (si no existe) el repo `TU_USUARIO/TU_REPO` y suben el contenido de `artifacts/` + `README.md` en la raíz del repo en HF.

Notas útiles:

- `--clean-repo` elimina archivos previos en HF que no estén en la subida actual (evita "archivos viejos").
- Se usa siempre `scripts/README.md` como model card y se publica en HF como `README.md`.
- `--private` es opcional (solo si quieres crear/subir a repositorio privado).

### 7.4) Verificación rápida

Abre en navegador:

```bash
https://huggingface.co/TU_USUARIO/TU_REPO
```

Debes ver, al menos:

- `encoder/`
- `heads.pt`
- `label2id.json`
- `id2label.json`
- `train_config.json`

### 7.5) Uso posterior en endpoint

Con estos comandos se suben los artefactos del modelo, que sirven para descargar y usar inferencia desde código.

Si quieres desplegar un **Inference Endpoint** administrado en Hugging Face con carga automática del repo, además de `artifacts/` necesitarás publicar también un `handler.py` y dependencias (`requirements.txt`) para que el endpoint sepa cómo inicializar y ejecutar la predicción multitarea.

## 8) Artefactos generados

En `artifacts/` se guarda:
- `encoder/`
- `heads.pt`
- `label2id.json`
- `id2label.json`
- `train_config.json`

## 9) Notas

- El sistema está parametrizado por CLI para entrenamiento, evaluación, test y upload.
- El modo `test` y `interactive` devuelven `label` y `score` para `macro`, `intent`, `context`.
- El modo `qa` además muestra el ranking top-k por tarea.
