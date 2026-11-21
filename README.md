# Proyecto 2: Fine-tuning para tareas de PLN

Proyecto del curso de Procesamiento de Lenguaje Natural (UdeA): fine-tuning de modelos de Hugging Face para clasificación de secuencias, etiquetado de secuencias y sequence-to-sequence. En cada tarea describo el dataset, hago un análisis exploratorio, busco el learning rate con una muestra del train, entreno con todo el train y evalúo el mejor modelo en test.

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
| 3. Resumen | ROUGE-L | 0.4122 | 0.4502 | 1e-4 | 4 de 5 |

El learning rate y la época se escogieron con validación; el test se evaluó al final. En la Tarea 2, el F1 de test queda 0.71 puntos por debajo del 88.43 que reporta el paper de BETO, que lo más probable es que sea de test; yo entrené con una sola semilla. En la Tarea 3, el ROUGE-L de test es 0.4502 si cada predicción se compara con la mejor de sus tres referencias y 0.3674 si se promedian las tres; la línea base de las primeras 2 intervenciones da 0.2535 y 0.2068.

En la T4, la búsqueda y el entrenamiento tardaron 8.3 min en la Tarea 1, 8.6 min en la Tarea 2 y 40.1 min en la Tarea 3, contando en esta la generación final de validación y test.

## Abrir en Google Colab

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiegoMunozCarvajalUdeA/nlp-finetuning-hf/blob/main/notebooks/proyecto2_finetuning.ipynb)

1. Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU T4.
2. Opcional: en el panel de Secrets (icono de llave) crea `HF_TOKEN` con tu token de Hugging Face. Los modelos y datasets son públicos, así que el notebook corre sin él; solo haría falta para subir modelos al Hub, y esas celdas están apagadas.
3. Corre todo desde arriba. La primera celda instala lo que falta sin tocar el torch de Colab y la configuración monta Google Drive: los modelos, tablas y figuras quedan en `MyDrive/proyecto2-nlp/outputs`. Si ya están ahí, el notebook los carga en vez de entrenar otra vez.

## Instalación local (uv)

Necesitas [uv](https://docs.astral.sh/uv/). uv descarga la versión de Python del proyecto (3.13, la misma de Colab) si no la tienes.

```bash
uv sync                        # crea .venv con las versiones exactas de uv.lock
cp .env.example .env           # opcional: tu token de Hugging Face en HF_TOKEN
uv run python -m ipykernel install --user --name proyecto2-nlp --display-name "Python 3.13 (proyecto2-nlp)"
uv run python scripts/check_env.py
```

Después abre el notebook en VS Code o con `uv run jupyter lab` y escoge el kernel `proyecto2-nlp`. Sin GPU CUDA, como en un Mac, los entrenamientos corren en modo rápido con unos cientos de ejemplos y guardan todo en `outputs/prueba_rapida`. Esos números solo sirven para comprobar que el código funciona.

## Estructura

```
notebooks/proyecto2_finetuning.ipynb   notebook principal
docs/seleccion_datasets.md             candidatos, medidas y decisión de datasets
scripts/check_env.py                   revisa que el entorno quedó bien
outputs/                               modelos, checkpoints y figuras (no se sube a git)
pyproject.toml, uv.lock                dependencias
```

Licencias de los datos: emotion es solo para uso educativo y de investigación, conll2002 no declara licencia ("unknown") y DialogSum es CC BY-NC-SA 4.0. Los detalles están en [`docs/seleccion_datasets.md`](docs/seleccion_datasets.md).
