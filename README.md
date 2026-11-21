# Proyecto 2: Fine-tuning para tareas de PLN

Proyecto del curso de Procesamiento de Lenguaje Natural (UdeA): fine-tuning de modelos de Hugging Face para clasificación de secuencias, etiquetado de secuencias y sequence-to-sequence.

Todo el desarrollo está en [`notebooks/proyecto2_finetuning.ipynb`](notebooks/proyecto2_finetuning.ipynb). Cómo escogí los datasets y modelos está en [`docs/seleccion_datasets.md`](docs/seleccion_datasets.md).

## Tareas

| Tarea | Dataset | Modelo | Métrica principal |
|---|---|---|---|
| Clasificación de secuencias | [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) | [`distilbert/distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased) | F1 macro |
| Etiquetado de secuencias (NER) | [`BramVanroy/conll2002`](https://huggingface.co/datasets/BramVanroy/conll2002), config `es` | [`dccuchile/bert-base-spanish-wwm-cased`](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) | F1 de seqeval |
| Sequence-to-sequence (resumen de diálogos) | [`knkarthick/dialogsum`](https://huggingface.co/datasets/knkarthick/dialogsum) | [`facebook/bart-base`](https://huggingface.co/facebook/bart-base) | ROUGE |

## Resultados en Colab (T4)

| Tarea | Métrica | Validación | Test | Learning rate | Época escogida |
|---|---|---|---|---|---|
| 1. Emoción | F1 macro | 0.9230 | 0.8874 | 5e-5 | 3 de 3 |
| 2. NER | F1 de seqeval | 0.8643 | 0.8772 | 5e-5 | 3 de 4 |

El learning rate y la época se escogieron con validación. Las salidas de Colab y su análisis están en el notebook; la Tarea 3 todavía no tiene resultados de la T4.

## Instalación local (uv)

Necesitas [uv](https://docs.astral.sh/uv/). uv descarga la versión de Python del proyecto (3.13, la misma de Colab) si no la tienes.

```bash
uv sync                        # crea .venv con las versiones exactas de uv.lock
cp .env.example .env           # y pega tu token de Hugging Face en HF_TOKEN
uv run python -m ipykernel install --user --name proyecto2-nlp --display-name "Python 3.13 (proyecto2-nlp)"
uv run python scripts/check_env.py
```

Después abre el notebook en VS Code o con `uv run jupyter lab` y escoge el kernel `proyecto2-nlp`.

Sin GPU CUDA, como en un Mac, los entrenamientos corren en modo rápido: usan unos cientos de ejemplos para comprobar que el código funciona y guardan todo en `outputs/prueba_rapida`. Esos números no sirven como resultado; los de verdad salen de Colab.

## Abrir en Google Colab

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiegoMunozCarvajalUdeA/nlp-finetuning-hf/blob/main/notebooks/proyecto2_finetuning.ipynb)

También sirve Archivo > Abrir cuaderno > GitHub en Colab.

Antes de correrlo:

1. Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU T4.
2. En el panel de Secrets (icono de llave) crea `HF_TOKEN` con tu token de Hugging Face y dale acceso al notebook.
3. Corre todo desde arriba. La primera celda instala lo que falta sin tocar el torch de Colab, y la configuración monta Google Drive: los modelos, tablas y figuras quedan en `MyDrive/proyecto2-nlp/outputs`.

## Estructura

```
notebooks/proyecto2_finetuning.ipynb   notebook principal
docs/seleccion_datasets.md             candidatos, medidas y decisión de datasets
scripts/check_env.py                   revisa que el entorno quedó bien
outputs/                               modelos, checkpoints y figuras (no se sube a git)
pyproject.toml, uv.lock                dependencias
```
