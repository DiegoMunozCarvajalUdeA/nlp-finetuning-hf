# Selección de datasets y modelos

Notas para decidir qué dataset y qué modelo usar en cada tarea: los candidatos que probé, lo que medí y lo que recomendaba.

## Decisión

Me quedé con las tres recomendaciones de este documento:

| Tarea | Dataset | Modelo | Métrica principal |
|---|---|---|---|
| 1. Clasificación de secuencias | `dair-ai/emotion` | `distilbert/distilbert-base-uncased` | F1 macro |
| 2. Etiquetado de secuencias (NER) | `BramVanroy/conll2002`, config `es` | `dccuchile/bert-base-spanish-wwm-cased` | F1 de seqeval |
| 3. Sequence-to-sequence (resumen de diálogos) | `knkarthick/dialogsum` | `facebook/bart-base` | ROUGE |

Las longitudes de este documento salen de muestras de 3000 ejemplos. En el EDA del notebook las medí sobre los splits completos, así que algunos máximos y porcentajes cambian un poco.

Todo lo probé el 2026-11-17 con las versiones del `uv.lock` (datasets 5.2.0, transformers 5.23.0). Los scripts de prueba y los JSON con las medidas quedaron en `outputs/exploracion/`, que no va a git.

## Cómo medí

- Carga: cada candidato lo cargué de verdad con `load_dataset`. "Sí" en la tabla quiere decir que carga directo; si necesita algo más, lo digo.
- Tamaños y splits: los que devuelve `load_dataset`. En XL-Sum el API del visor dio error 500 y los saqué de la metadata del parquet y de la card.
- Clases: distribución sobre todo el train.
- Longitud en tokens: muestra aleatoria de 3000 ejemplos del train (semilla fija), con el tokenizer del modelo sugerido, sin truncar y contando los tokens especiales. El máximo es el de la muestra, no el del split completo. En XL-Sum y wikineural la muestra sale de streaming con un buffer de 10000, así que es menos representativa.
- Tiempo en T4: es una estimación, no lo corrí en una T4. El método está en [Tiempos en T4](#tiempos-en-t4) y por eso doy rangos de X a 2X.
- fp16: hice un forward pass de cada modelo seq2seq en fp32 y en fp16 para ver si las activaciones se salen del rango de fp16. Está en [fp16 en la T4](#fp16-en-la-t4).

## Cargar datasets que tienen script

datasets 5.x ya no ejecuta scripts de carga, por eso fallaban varios (tweet_sentiment_multilingual, eriktks/conll2003, conll2002, xlsum). El Hub guarda una conversión automática a parquet en la rama `refs/convert/parquet` de cada repo, y esa sí carga. La config se pasa como `data_dir`; con el nombre de la config no funciona:

```python
from datasets import load_dataset

ds = load_dataset("csebuetnlp/xlsum", data_dir="spanish", revision="refs/convert/parquet")
```

Esa rama la genera el Hub, no los autores, y puede cambiar o desaparecer. Prefiero datasets con parquet en `main` y uso este truco solo cuando no hay otra opción.

Dos cosas que encontré de paso:

- `BramVanroy/conll2002` tiene parquet en `main` y carga directo. Es el mismo contenido que `eriktks/conll2002` por la rama parquet.
- `universal_dependencies` sin organización ya no existe, pero `universal-dependencies/universal_dependencies` sí carga directo (config `es_ancora`).

## Tarea 1: clasificación de secuencias

| Dataset | Idioma | Tamaño y splits (train / val / test) | Licencia | ¿Carga? | Tokens: media / p95 / máx | Modelo sugerido | Métrica | Tiempo T4 (estimado) | Pros | Contras |
|---|---|---|---|---|---|---|---|---|---|---|
| `dair-ai/emotion` | inglés | 16000 / 2000 / 2000, 6 clases | "for educational and research purposes only" (card) | sí | 22.5 / 46 / 75 (distilbert) | `distilbert/distilbert-base-uncased` o `FacebookAI/roberta-base` | F1 macro + accuracy | distilbert 3 épocas: 1.5 a 3 min; roberta: 3 a 6 min | tuits cortos con 6 emociones, se explica en una frase; card completa | en inglés; desbalanceado (joy 33.5 %, surprise 3.6 %) |
| `mteb/amazon_massive_intent`, config `es` | español (localizado del inglés) | 11514 / 2033 / 2974, 60 intenciones | CC BY 4.0 en el original (`AmazonScience/massive`); la card de mteb dice apache-2.0 | sí | 9.9 / 18 / 37 (BETO uncased) | `dccuchile/bert-base-spanish-wwm-uncased` | F1 macro + accuracy | 5 épocas: 1.4 a 3 min | español; licencia clara; "qué le pide la persona al asistente de voz" se entiende rápido | 60 clases desbalanceadas (la mayor tiene 202 veces más ejemplos que la menor); `label` viene como texto; frases muy cortas |
| `SINAI/EmoEvent` (parte en español) | español | 5723 / 844 / 1656, 7 clases | CC BY-NC-SA 4.0 | sí, con `data_files` | 43.4 / 76 / 102 (robertuito) | `pysentimiento/robertuito-base-uncased` o BETO uncased | F1 macro | 4 épocas: 2.5 a 5 min | emociones en tuits en español, parecido a emotion | muy desbalanceado: others 48.9 %, fear 1.1 % (65 en train); card corta; robertuito no declara licencia y pide preprocesar los tuits |
| `SetFit/amazon_reviews_multi_es` | español | 200000 / 5000 / 5000, 5 estrellas | ninguna en la card; es copia de un dataset retirado | sí | 35.2 / 89 / 404 (BETO uncased) | BETO uncased | F1 macro + accuracy | 200k x 2 épocas: 49 a 97 min; 50k x 3 épocas: 18 a 36 min | balanceado (20 % por clase); reseñas fáciles de entender | el original (`defunct-datasets/amazon_reviews_multi`) se retiró por decisión de los proveedores y sus términos no dejaban republicar; card vacía; toca submuestrear |

Notas:

- emotion trae los nombres de clase (`ClassLabel`). Con `max_length=128` no se corta nada (el máximo de la muestra es 75).
- MASSIVE: antes de entrenar hay que convertir `label` con `class_encode_column("label")`. Hay clases casi vacías (`cooking_query` tiene 4 ejemplos en train y `general_greet` 1 en test), así que el F1 de esas clases va a salir muy ruidoso. El texto es una localización del inglés (SLURP), no habla espontánea. Con `max_length=64` sobra.
- EmoEvent: la config por defecto mezcla inglés y español. Para quedarse con el español:

  ```python
  load_dataset(
      "SINAI/EmoEvent",
      data_files={"train": "splits/es/train.tsv", "validation": "splits/es/dev.tsv", "test": "splits/es/test.tsv"},
      delimiter="\t",
  )
  ```

- Amazon lo descarto por la licencia, aunque en tamaño y balance es el mejor de la tabla.
- También miré `cardiffnlp/tweet_sentiment_multilingual` (spanish): 1839 / 324 / 870, 3 clases balanceadas, CC BY 3.0 más los términos de Twitter. Solo carga por la rama parquet y es muy pequeño. `stanfordnlp/imdb` no tiene split de validación y el 14 % de las reseñas pasa de 512 tokens con distilbert.

Recomendación: `dair-ai/emotion` con `distilbert/distilbert-base-uncased`. Carga directo, trae los tres splits, la card es buena y la tarea se entiende sin explicar nada. Entrena en minutos, así que queda tiempo para la búsqueda de learning rate y para mirar los errores. Si quiero un modelo más fuerte, `roberta-base` también cabe de sobra.

Segunda opción: MASSIVE en español con BETO uncased, si quiero que esta tarea también sea en español. Es igual de barata y la licencia es clara; lo malo son las 60 clases con algunas casi vacías.

Métricas: F1 macro como métrica principal (la de `metric_for_best_model`), porque las clases están desbalanceadas y un modelo que casi nunca predice surprise puede tener buena accuracy. Reporto también accuracy, que es la que todo el mundo entiende, y en el análisis el F1 por clase y la matriz de confusión.

## Tarea 2: etiquetado de secuencias

| Dataset | Idioma | Tamaño y splits (train / val / test) | Licencia | ¿Carga? | Tokens: media / p95 / máx | Modelo sugerido | Métrica | Tiempo T4 (estimado) | Pros | Contras |
|---|---|---|---|---|---|---|---|---|---|---|
| `BramVanroy/conll2002`, config `es` (NER) | español | 8324 / 1916 / 1518; PER, ORG, LOC, MISC | "unknown" en la card | sí | 38.8 / 74 / 341 (BETO cased) | `dccuchile/bert-base-spanish-wwm-cased` | seqeval F1 por entidad | 4 épocas: 3.4 a 7 min | NER clásico en español, anotado a mano; la card de BETO trae una cifra para comparar (88.43 de F1, en validación) | licencia sin declarar; 87.6 % de los tokens son O; MISC es la clase más escasa (11.6 % de las entidades) |
| `universal-dependencies/universal_dependencies`, config `es_ancora` (POS) | español | 14287 / 1654 / 1721; 16 etiquetas UPOS | CC BY 4.0 según `PlanTL-GOB-ES/UD_Spanish-AnCora`; el repo de UD dice "other" | sí | 38.8 / 74 / 149 (BETO cased) | `dccuchile/bert-base-spanish-wwm-cased` | accuracy por token | 3 épocas: 4.3 a 9 min | anotado a mano, licencia clara, etiquetas fáciles de explicar | casi saturado (la card de BETO da 98.97 en POS), poca mejora que mostrar; el split de validación se llama `dev` |
| `eriktks/conll2003` (NER) | inglés | 14041 / 3250 / 3453; PER, ORG, LOC, MISC | "other" (texto de Reuters) | sí, con `refs/convert/parquet` | 21.7 / 47 / 173 (bert-base-cased) | `google-bert/bert-base-cased` | seqeval F1 por entidad | 4 épocas: 3.6 a 7 min | el NER más conocido; trae nombres de etiqueta | en inglés; solo carga por la rama parquet |
| `unimelb-nlp/wikiann`, config `es` (NER) | español | 20000 / 10000 / 10000; PER, ORG, LOC | "unknown" | sí | 13.2 / 22 / 104 (BETO cased) | `dccuchile/bert-base-spanish-wwm-cased` | seqeval F1 por entidad | 3 épocas: 1.8 a 4 min | carga directo y es el más grande | anotación automática; son fragmentos de Wikipedia de 6.4 palabras en promedio, no frases |

Notas:

- Para etiquetar uso `is_split_into_words=True` y pongo la etiqueta solo en el primer subtoken de cada palabra; el resto va con -100.
- conll2002: con `max_length=256` casi no se corta nada (0.2 % de las frases pasan de 128 tokens con BETO). El texto son noticias de la agencia EFE de mayo de 2000. La card de BETO reporta 88.43 de F1 en NER con este dataset, medido sobre validación (el enlace del resultado apunta a un archivo `dev_results`), no sobre test. Sirve para saber si mi resultado en validación tiene sentido.
- AnCora: `upos` viene como texto y hay que pasarlo a ids. Las contracciones vienen partidas ("del" aparece como "de" + "el"). Con `max_length=160` entra todo (máximo 149).
- Descartados: `Babelscape/wikineural` es (76320 / 9540 / 9618, anotación automática, CC BY-NC-SA 4.0, y las etiquetas no traen nombres en las features; el mapeo está en la card) y MultiCoNER v2 en español (33 tipos de entidad, test de 246900 frases, solo por la rama parquet).

Recomendación: `BramVanroy/conll2002` (es) con `dccuchile/bert-base-spanish-wwm-cased`. Es NER en español con anotación manual, carga directo y tengo una cifra de referencia en la card de BETO (88.43 de F1 en validación). Uso la versión cased porque las mayúsculas ayudan a reconocer nombres propios. Lo que no me gusta es que la licencia diga "unknown".

Segunda opción: AnCora (POS) con BETO cased, si quiero una licencia clara. El problema es que ya está cerca de 99 % de accuracy y en el video no habría mucho que mostrar. Si no importa el idioma, `eriktks/conll2003` con `bert-base-cased` también sirve, aunque solo carga por la rama parquet.

Métricas: para NER uso seqeval (precisión, recall y F1 por entidad, como en CoNLL), donde una entidad solo cuenta si coinciden el tipo y los límites. F1 general como principal y F1 por tipo para el análisis. La accuracy por token no sirve en NER porque el 87.6 % de los tokens son O y un modelo que diga O siempre ya saca 87.6 %. Para POS sí uso accuracy por token, que es lo normal en esa tarea.

## Tarea 3: sequence-to-sequence

| Dataset | Idioma | Tamaño y splits (train / val / test) | Licencia | ¿Carga? | Tokens: media / p95 / máx | Modelo sugerido | Métrica | Tiempo T4 (estimado) | Pros | Contras |
|---|---|---|---|---|---|---|---|---|---|---|
| `knkarthick/dialogsum` (resumen de diálogos) | inglés | 12460 / 500 / 1500 (test: 500 diálogos x 3 resúmenes) | CC BY-NC-SA 4.0 | sí | entrada 211.5 / 390 / 1087; resumen 36.7 / 65 / 165 (bart-base) | `facebook/bart-base` | ROUGE-1/2/L/Lsum | 5 épocas: 23 a 45 min, sin contar la generación | resúmenes escritos por personas; diálogos cotidianos fáciles de mostrar; test con 3 referencias; bart-base no tiene problemas en fp16 | en inglés; 1.2 % de los diálogos pasan de 512 tokens |
| `csebuetnlp/xlsum`, config `spanish` (resumen de noticias) | español (BBC Mundo) | 38110 / 4763 / 4763 | CC BY-NC-SA 4.0 | sí, con `refs/convert/parquet` | entrada 1617 / 4283 / 14109; resumen 52.9 / 100 / 252 (BARTO) | `vgaraujov/bart-base-spanish` o `flax-community/spanish-t5-small` | ROUGE con tokenizer propio | BARTO 3 épocas: 57 min a 2 h; spanish-t5-small: 25 a 50 min | en español; noticias reales con resumen de una frase | el 57 % de los artículos pasa de 512 tokens y se corta; solo carga por la rama parquet; BARTO tiene card corta y pocas descargas |
| `knkarthick/samsum` (resumen de chats) | inglés | 14731 / 818 / 819 | CC BY-NC-ND 4.0 | sí | entrada 151.7 / 363 / 857; resumen 27.7 / 55 / 78 (bart-base) | `facebook/bart-base` | ROUGE-1/2/L/Lsum | 5 épocas: 24 a 49 min | chats tipo mensajería, muy usado | licencia ND; una sola referencia en test; chats con jerga y marcadores como `<file_video>` |
| `Helsinki-NLP/opus_books`, config `en-es` (traducción) | inglés a español | solo train: 93470 (hay que partirlo) | según la card, uso personal, educativo y de investigación | sí | origen 36.5 / 96 / 370; destino 39.2 / 102 / 376 (mt5-small) | `google/mt5-small` (en fp32) | sacreBLEU + chrF | mt5 en fp32 con 80 % del train, 3 épocas: 3 a 6 h | traducción se explica sola | no trae val ni test; mT5 da NaN en fp16 y en fp32 se pasa del límite (ver abajo) |

Notas:

- DialogSum: el test tiene 1500 filas pero son 500 diálogos con 3 resúmenes cada uno (ids `test_N_1`, `test_N_2`, `test_N_3`). Para evaluar hay que generar una vez por diálogo y pasar las 3 referencias como lista. El `rouge` de evaluate acepta listas de referencias y se queda con la mejor para cada predicción (lo probé). `max_source_length=512` corta el 1.2 % y `max_target_length=128` no corta casi nada.
- XL-Sum: el 57 % de los artículos pasa de 512 tokens con el tokenizer de BARTO y el 52 % pasa de 1024, que es el máximo de BART, así que hay que cortar sí o sí. En noticias lo principal suele ir al inicio, pero igual hay que decirlo en el informe. Con los 38110 ejemplos y 3 épocas la estimación queda entre 57 min y 2 h; si se pasa, submuestreo el train a unos 20000. `flax-community/spanish-t5-small` (60M, MIT) es la opción más barata. En el Mac lo medí en streaming; en Colab se puede bajar completo.
- Traducción: no la recomiendo, por los modelos:
  - mT5 da NaN en fp16 (lo medí, abajo) y en fp32 la estimación pasa de 2 h.
  - `google-t5/t5-small` no tiene ñ, Ñ, í, ú, ¿ ni ¡ en el vocabulario: salen como `<unk>` (23 por cada 1000 tokens de texto en español). No puede escribir español bien.
  - `Helsinki-NLP/opus-mt-en-es` sí funciona en fp16 y es barato (unos 20 min), pero es un Marian, no T5 ni BART, y ya fue entrenado con datos de OPUS, así que ajustarlo con opus_books u opus-100 es medio trampa.
- Otros que miré: `Helsinki-NLP/opus-100` en-es (1M / 2000 / 2000, licencia "unknown", frases ruidosas), `LeoCordoba/CC-NEWS-ES-titles` (MIT, 370125 / 16092 / 16093, notas de finanzas con plantilla y poca calidad) y mlsum es (carga por la rama parquet pero es enorme).

Recomendación: `knkarthick/dialogsum` con `facebook/bart-base`. Los resúmenes los escribieron personas, los diálogos se entienden sin contexto y se ven bien en un video, el test trae 3 referencias, bart-base es estable en fp16 y el entrenamiento completo queda alrededor de media hora.

Segunda opción: XL-Sum en español con `vgaraujov/bart-base-spanish`, con `max_source_length=512`, `max_target_length=128` y submuestreo si hace falta. Sirve si quiero el seq2seq en español, a cambio de cortar más de la mitad de los artículos, depender de la rama parquet y usar un modelo poco conocido.

Métricas: ROUGE-1, ROUGE-2, ROUGE-L y ROUGE-Lsum, que son las estándar para resumen (comparan n-gramas y subsecuencias con la referencia). Reporto también la longitud media de lo generado para ver si el modelo se queda corto o se alarga. Dos detalles:

- ROUGE-Lsum necesita las oraciones separadas por salto de línea (con nltk) o da lo mismo que ROUGE-L.
- El tokenizer por defecto de `rouge_score` borra los caracteres que no son ASCII. Lo probé con "La información técnica del niño español" y queda `['la', 'informaci', 'n', 't', 'cnica', 'del', 'ni', 'o', 'espa', 'ol']`. En inglés no pasa nada, pero en español hay que pasar un tokenizer propio, por ejemplo `rouge.compute(..., tokenizer=lambda x: x.lower().split())`, y no usar `use_stemmer` (es el stemmer de Porter, para inglés).

Si al final escojo traducción, la métrica sería sacreBLEU más chrF.

## fp16 en la T4

La T4 no tiene bf16, así que en Colab se entrena en fp16. El rango de fp16 llega hasta 65504; si una activación se pasa, sale inf y luego NaN. Con T5 y mT5 esto es un problema conocido: el PR que intentaba arreglarlo en transformers (#10956) se cerró sin fusionar en 2024 y en el foro de Hugging Face hay un hilo de mT5 con fp16 que da loss cero, sin solución. mT5 se preentrenó en bf16.

Lo medí con un forward pass de un batch de ejemplos: activación máxima en fp32 y loss en fp32 y en fp16 (autocast en MPS, en el Mac). No es la T4, pero el desborde depende de los valores y no del hardware.

| Modelo | Activación máxima en fp32 | Loss fp32 / fp16 | Veredicto |
|---|---|---|---|
| `google/mt5-small` | 131394 (27 módulos pasan de 65504) | 10.621 / NaN | no sirve en fp16 |
| `google-t5/t5-small` | 36437 | 3.734 / 3.734 | funciona, pero está a la mitad del límite |
| `facebook/bart-base` | 2114 | 5.850 / 5.852 | sin problema |
| `vgaraujov/bart-base-spanish` | 410 | 4.823 / 4.823 | sin problema |
| `flax-community/spanish-t5-small` | 329 | 3.725 / 3.725 | sin problema |
| `Helsinki-NLP/opus-mt-en-es` | 1197 | 3.220 / 3.221 | sin problema |

El transformers instalado solo recorta los valores en T5 cuando los hidden states ya están en fp16, y en mT5 deja la capa `wo` en fp32; aun así el forward en fp16 dio NaN. Con los modelos que recomiendo (BART) no hace falta nada especial, pero conviene vigilar que la loss no se vuelva `nan` o `0` en los primeros pasos. Los modelos tipo BERT de las tareas 1 y 2 entrenan en fp16 sin problema conocido.

## Tiempos en T4

No tuve una T4 para medir, así que estimé con FLOPs:

- Referencia: bert-base-uncased en SQuAD v2 en una T4, fp16, secuencia 384, batch 8: 43.9 ejemplos/s (repo BERT-FineTuning-SQuAD-CUDA). Eso equivale a unos 9.4 TFLOP/s efectivos.
- Costo de entrenar: 6 x parámetros sin embeddings x tokens. En seq2seq: 6 x (parámetros del encoder x tokens de entrada + (parámetros del decoder + cabeza LM) x tokens de salida).
- Longitud: el p95 de cada dataset, suponiendo padding dinámico.
- Supongo fp32 3 veces más lento que fp16 y multiplico todo por 1.5 por dataloader, optimizador y evaluación.

El resultado es X; lo realista es entre X y 2X. Con secuencias muy cortas (MASSIVE, wikiann) la GPU queda subutilizada y el tiempo real puede ser varias veces la estimación, pero siguen siendo minutos. No incluye descargar el modelo ni tokenizar, y en la tarea 3 tampoco incluye la generación para calcular ROUGE, que no estimé. Para no gastar tiempo ahí: durante el entrenamiento evalúo con la loss o con ROUGE sobre unos 200 ejemplos de validación, y genero el test completo una sola vez al final. Cuando corra la búsqueda de learning rate tengo los it/s reales y recalculo con eso.

| Tarea | Dataset y modelo | Precisión | Min por época | Épocas | Entrenamiento completo | Búsqueda de LR |
|---|---|---|---|---|---|---|
| 1 | emotion + distilbert | fp16 | 0.5 | 3 | 1.5 a 3 min | 4 LR x 4000 ej x 2 ép: 1 a 2 min |
| 1 | emotion + roberta-base | fp16 | 1.0 | 3 | 3 a 6 min | 4 LR x 4000 ej x 2 ép: 2 a 4 min |
| 1 | MASSIVE es + BETO uncased | fp16 | 0.3 | 5 | 1.4 a 3 min | 4 LR x 3000 ej x 3 ép: 1 a 2 min |
| 1 | EmoEvent es + robertuito | fp16 | 0.6 | 4 | 2.5 a 5 min | 4 LR x 2000 ej x 3 ép: 2.7 a 5 min |
| 1 | Amazon es 200k + BETO | fp16 | 24.3 | 2 | 49 a 97 min | 4 LR x 5000 ej x 2 ép: 5 a 10 min |
| 1 | Amazon es 50k + BETO | fp16 | 6.1 | 3 | 18 a 36 min | igual |
| 2 | conll2002 es + BETO cased | fp16 | 0.8 | 4 | 3.4 a 7 min | 4 LR x 2000 ej x 3 ép: 2.4 a 5 min |
| 2 | AnCora POS + BETO cased | fp16 | 1.4 | 3 | 4.3 a 9 min | 4 LR x 2000 ej x 3 ép: 2.4 a 5 min |
| 2 | conll2003 + bert-base-cased | fp16 | 0.9 | 4 | 3.6 a 7 min | 4 LR x 3000 ej x 3 ép: 2.3 a 5 min |
| 2 | wikiann es + BETO cased | fp16 | 0.6 | 3 | 1.8 a 4 min | 4 LR x 3000 ej x 3 ép: 1 a 2 min |
| 3 | DialogSum + bart-base | fp16 | 4.5 | 5 | 23 a 45 min | 4 LR x 2000 ej x 3 ép: 9 a 17 min |
| 3 | SAMSum + bart-base | fp16 | 4.9 | 5 | 24 a 49 min | 4 LR x 2000 ej x 3 ép: 8 a 16 min |
| 3 | XL-Sum es + BARTO (512 / 100) | fp16 | 19.0 | 3 | 57 a 114 min | 3 LR x 3000 ej x 2 ép: 9 a 18 min |
| 3 | XL-Sum es + spanish-t5-small | fp16 | 8.4 | 3 | 25 a 50 min | 3 LR x 3000 ej x 2 ép: 4 a 8 min |
| 3 | XL-Sum es + mt5-small | fp32 | 44.2 | 3 | 2.2 a 4.4 h | no cabe |
| 3 | opus-100 100k + mt5-small | fp32 | 50.8 | 3 | 2.5 a 5 h | no cabe |
| 3 | opus-100 100k + opus-mt-en-es | fp16 | 5.5 | 3 | 17 a 33 min | 3 LR x 5000 ej x 2 ép: 2 a 4 min |
| 3 | opus_books 80 % + opus-mt-en-es | fp16 | 7.3 | 3 | 22 a 44 min | 3 LR x 5000 ej x 2 ép: 3 a 6 min |
| 3 | opus_books 80 % + mt5-small | fp32 | 62.4 | 3 | 3.1 a 6.2 h | no cabe |

Con las recomendaciones (emotion, conll2002 y DialogSum) todo entra sin submuestrear y lejos del límite de 1 a 2 h por tarea. El único candidato que puede necesitar submuestreo es XL-Sum.

## Learning rates y muestra para la búsqueda

La idea es correr una búsqueda corta sobre una muestra del train, escoger el learning rate con la métrica de validación y después entrenar con todo el train.

Para los modelos tipo BERT parto de lo que recomienda el paper de BERT (apéndice A.3): batch 16 o 32, learning rate 5e-5, 3e-5 o 2e-5, entre 2 y 4 épocas, y avisan que los datasets pequeños son más sensibles a los hiperparámetros. Le agrego 1e-5 por abajo. Para BART no tengo una fuente equivalente; centro la búsqueda en 5e-5, que es el valor por defecto de `TrainingArguments`, y pruebo valores entre 2e-5 y 2e-4. Los T5 se suelen ajustar con learning rates más altos (del orden de 1e-4 a 1e-3); eso no lo verifiqué en el paper.

| Tarea | Learning rates | Muestra | Épocas | Se escoge con |
|---|---|---|---|---|
| 1 (emotion + distilbert) | 1e-5, 2e-5, 3e-5, 5e-5 | 4000 ejemplos estratificados | 2 | F1 macro en validación completa |
| 1 (MASSIVE + BETO, si va) | 1e-5, 2e-5, 3e-5, 5e-5 | 3000 estratificados | 3 | F1 macro en validación |
| 2 (conll2002 + BETO cased) | 1e-5, 2e-5, 3e-5, 5e-5 | 2000 frases | 3 | F1 de seqeval en validación |
| 3 (DialogSum + bart-base) | 2e-5, 5e-5, 1e-4, 2e-4 | 2000 diálogos | 3 | ROUGE-L en 200 ejemplos de validación (generación greedy) y loss |
| 3 (XL-Sum + BARTO, si va) | 2e-5, 5e-5, 1e-4 | 3000 artículos | 2 | ROUGE-L en 200 ejemplos de validación |
| 3 (spanish-t5-small, si va) | 1e-4, 3e-4, 1e-3 | 3000 | 2 | ROUGE-L en 200 ejemplos de validación |

Batch 16 en todas (en BART puede tocar 8 con acumulación de gradiente si no cabe con 512 tokens), misma semilla en todas las corridas y warmup corto. En las tareas 1 y 2 la búsqueda completa cuesta unos minutos; en la tarea 3, entre 10 y 20 min estimados.

## Riesgos que veo

- Licencias: conll2002 y wikiann dicen "unknown"; emotion es solo para uso educativo y de investigación; DialogSum, XL-Sum y EmoEvent son NC. Para un proyecto de la universidad sirve, pero hay que citarlas en el informe.
- Los datasets que cargan por `refs/convert/parquet` (XL-Sum, conll2003, tweet_sentiment_multilingual) dependen de una conversión automática del Hub.
- Los tiempos son estimados. Hay que recalcularlos con los it/s reales de la primera búsqueda.
- En la tarea 3 no estimé el tiempo de generación. Generar el test completo con beam search en cada época puede salir caro, por eso lo dejo para el final.
- El ROUGE en español sale mal si no se cambia el tokenizer.
- La card de BETO aclara que CC BY 4.0 describe su intención, pero que no están seguros de que todos los datos de preentrenamiento lo permitan (sobre todo para uso comercial). robertuito no declara licencia.

## Enlaces que consulté

Dataset cards:

- https://huggingface.co/datasets/dair-ai/emotion
- https://huggingface.co/datasets/mteb/amazon_massive_intent
- https://huggingface.co/datasets/AmazonScience/massive
- https://huggingface.co/datasets/SINAI/EmoEvent
- https://huggingface.co/datasets/SetFit/amazon_reviews_multi_es (card vacía)
- https://huggingface.co/datasets/mteb/amazon_reviews_multi (card vacía)
- https://huggingface.co/datasets/defunct-datasets/amazon_reviews_multi
- https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual
- https://huggingface.co/datasets/BramVanroy/conll2002
- https://huggingface.co/datasets/eriktks/conll2002
- https://huggingface.co/datasets/universal-dependencies/universal_dependencies
- https://huggingface.co/datasets/PlanTL-GOB-ES/UD_Spanish-AnCora
- https://huggingface.co/datasets/eriktks/conll2003
- https://huggingface.co/datasets/unimelb-nlp/wikiann
- https://huggingface.co/datasets/Babelscape/wikineural
- https://huggingface.co/datasets/knkarthick/dialogsum
- https://huggingface.co/datasets/knkarthick/samsum
- https://huggingface.co/datasets/csebuetnlp/xlsum
- https://huggingface.co/datasets/Helsinki-NLP/opus_books
- https://huggingface.co/datasets/Helsinki-NLP/opus-100
- https://huggingface.co/datasets/LeoCordoba/CC-NEWS-ES-titles

Model cards:

- https://huggingface.co/distilbert/distilbert-base-uncased
- https://huggingface.co/FacebookAI/roberta-base
- https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased
- https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased
- https://huggingface.co/pysentimiento/robertuito-base-uncased
- https://huggingface.co/facebook/bart-base
- https://huggingface.co/vgaraujov/bart-base-spanish
- https://huggingface.co/flax-community/spanish-t5-small
- https://huggingface.co/google/mt5-small

Otros:

- Paper de BERT, apéndice A.3: https://arxiv.org/pdf/1810.04805
- PR de transformers sobre T5 en fp16: https://github.com/huggingface/transformers/pull/10956
- Hilo del foro sobre mT5 en fp16: https://discuss.huggingface.co/t/mt5-fine-tuning-using-fp16-yields-zero-loss/31043
- Velocidad de BERT en T4 que usé como referencia: https://github.com/nikhildinesharora/BERT-FineTuning-SQuAD-CUDA
