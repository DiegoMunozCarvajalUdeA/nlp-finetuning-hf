# Proyecto 2: Fine-tuning para tareas de PLN

Proyecto del curso de Procesamiento de Lenguaje Natural (UdeA): fine-tuning de modelos de Hugging Face para clasificación de secuencias, etiquetado de secuencias y sequence-to-sequence.

Todo el desarrollo está en [`notebooks/proyecto2_finetuning.ipynb`](notebooks/proyecto2_finetuning.ipynb).

## Tareas

| Tarea | Dataset | Modelo | Métrica(s) |
|---|---|---|---|
| Clasificación de secuencias | pendiente | pendiente | pendiente |
| Etiquetado de secuencias | pendiente | pendiente | pendiente |
| Sequence-to-sequence | pendiente | pendiente | pendiente |

## Instalación local (uv)

Necesitas [uv](https://docs.astral.sh/uv/). uv descarga la versión de Python del proyecto (3.13, la misma de Colab) si no la tienes.

```bash
uv sync                        # crea .venv con las versiones exactas de uv.lock
cp .env.example .env           # y pega tu token de Hugging Face en HF_TOKEN
uv run python -m ipykernel install --user --name proyecto2-nlp --display-name "Python 3.13 (proyecto2-nlp)"
uv run python scripts/check_env.py
```

Después abre el notebook en VS Code o con `uv run jupyter lab` y escoge el kernel `proyecto2-nlp`.

## Abrir en Google Colab

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiegoMunozCarvajalUdeA/nlp-finetuning-hf/blob/main/notebooks/proyecto2_finetuning.ipynb)

También sirve Archivo > Abrir cuaderno > GitHub en Colab.

Antes de correrlo:

1. Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU T4.
2. En el panel de Secrets (icono de llave) crea `HF_TOKEN` con tu token de Hugging Face y dale acceso al notebook.
3. Corre todo desde arriba. La primera celda instala lo que falta sin tocar el torch de Colab, y la configuración monta Google Drive: los modelos y checkpoints quedan en `MyDrive/proyecto2-nlp/outputs`.

## Estructura

```
notebooks/proyecto2_finetuning.ipynb   notebook principal
scripts/check_env.py                   revisa que el entorno quedó bien
outputs/                               modelos, checkpoints y figuras (no se sube a git)
pyproject.toml, uv.lock                dependencias
```
