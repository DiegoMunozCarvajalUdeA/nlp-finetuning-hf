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
| 3. Resumen | ROUGE-L | 0.4122 | 0.4502 | 1e-4 | 4 de 5 |

El learning rate y la época se escogieron con validación. En la Tarea 3, validación tiene una referencia por diálogo; el test de la tabla usa la mejor de tres referencias para cada predicción y métrica. Al promediar las evaluaciones contra cada referencia por separado, el ROUGE-L de test es 0.3674. La línea base de las primeras 2 intervenciones da 0.2535 y 0.2068 en esas dos versiones, respectivamente. Las salidas de Colab, su análisis y las Conclusiones están en el notebook.

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

## Correr solo la Tarea 3 en Colab

1. Abre la versión de `main` con el enlace de Colab y escoge GPU T4.
2. Ejecuta la sección 0 para instalar, montar Drive y configurar las utilidades.
3. Ejecuta la Tarea 3 desde sus constantes y la carga del dataset hasta la demo de 3.5. No necesita ejecutar las tareas 1 y 2.
4. Deja `REENTRENAR_T3 = False`: carga los pasos vigentes y completa los que falten. Si cambian el learning rate ganador, las épocas o la generación, repite los pasos que ya no correspondan. El notebook avisa si el ganador está en un borde de la grilla.
5. Revisa que el ROUGE-L de validación de 3.5 coincida con el de la mejor época. Deja `SUBIR_AL_HUB_T3 = False` y descarga el notebook con las salidas para incorporarlas al repo.

La corrida de Colab entrenó 5 épocas y escogió la 4 por ROUGE-L de validación. Los tiempos registrados suman 40.1 min para búsqueda, entrenamiento y generación final de validación y test: 669.4 s, 1633.5 s y 102.15 s, respectivamente. No es el tiempo total de la sesión: descargar, tokenizar, calcular el ROUGE final y escribir en Drive suma tiempo. Con la línea base y la demo, la suma registrada es 40.7 min. El presupuesto previo era de 37 a 55 min; el cálculo y las medidas están en [`docs/seleccion_datasets.md`](docs/seleccion_datasets.md#tiempos-reales-en-la-t4).

En `MyDrive/proyecto2-nlp/outputs/tarea3/` deben quedar el mejor modelo y tokenizer (`modelo/`, incluida su `generation_config.json`), `busqueda_lr.csv`, `config_busqueda.json`, los historiales por learning rate y del entrenamiento completo, `entrenamiento.json`, `linea_base.json`, `predicciones.jsonl`, `metricas_val_test.json`, `demo.json` y las tablas de comparación, longitud y distribución de ROUGE-L. Las figuras quedan en `outputs/figuras/`. Los checkpoints intermedios usan `/content/checkpoints/` y se eliminan al terminar; no van a Drive.

La Tarea 3 se probó en copias del notebook en `/tmp`, con MPS y `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5`: una corrida completa en modo rápido, otra reutilizando todo sin entrenar ni generar y una tercera cambiando el ganador de la búsqueda para comprobar que repite el entrenamiento y el test. En las tres se comprobó la coincidencia del ROUGE-L de validación con la época guardada. En las pruebas locales no se ejecutaron las celdas del Hub; en Colab la subida permaneció apagada. Después se corrigió la colisión del nombre `titulo` y se probó el EDA seguido de las gráficas de búsqueda y entrenamiento. El notebook del repo conserva las salidas completas de Colab de las tres tareas.

## Estructura

```
notebooks/proyecto2_finetuning.ipynb   notebook principal
docs/seleccion_datasets.md             candidatos, medidas y decisión de datasets
scripts/check_env.py                   revisa que el entorno quedó bien
outputs/                               modelos, checkpoints y figuras (no se sube a git)
pyproject.toml, uv.lock                dependencias
```
