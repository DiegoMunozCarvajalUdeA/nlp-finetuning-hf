"""Revisa que el entorno local del proyecto quedó bien instalado.

Uso (desde la raíz del repo):

    uv run python scripts/check_env.py

Qué hace:
  1. Imprime la versión de Python y de las librerías principales.
  2. Detecta el dispositivo (cuda, mps o cpu).
  3. Carga el tokenizer y el modelo distilbert-base-uncased (cabeza MLM, así usa
     todos los pesos del checkpoint) y hace un forward pass.
  4. Carga las métricas de evaluate (accuracy, f1, seqeval, sacrebleu, rouge)
     y las calcula sobre un ejemplo de juguete.
  5. Compara las versiones fijadas para Colab en el notebook con las del entorno
     local (que salen del uv.lock).
  6. Dice si hay HF_TOKEN en el .env, sin mostrarlo.

Termina con código 1 si algo falla.
"""

import ast
import importlib.metadata as md
import json
import os
import platform
import sys
import traceback
from pathlib import Path

# Igual que en el notebook: en Mac, lo que MPS no soporte cae a CPU
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "proyecto2_finetuning.ipynb"
MODEL_ID = "distilbert-base-uncased"
LIBS = [
    "torch",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "datasets",
    "evaluate",
    "accelerate",
    "sentencepiece",
    "protobuf",
    "scikit-learn",
    "seqeval",
    "sacrebleu",
    "rouge-score",
    "nltk",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "ipywidgets",
    "python-dotenv",
    "ipykernel",
]


def titulo(texto):
    print(f"\n=== {texto} ===")


def versiones():
    titulo("Versiones")
    print(f"{'python':<16} {platform.python_version()} ({sys.executable})")
    faltan = []
    for lib in LIBS:
        try:
            print(f"{lib:<16} {md.version(lib)}")
        except md.PackageNotFoundError:
            print(f"{lib:<16} NO INSTALADA")
            faltan.append(lib)
    if faltan:
        raise RuntimeError(f"faltan librerías: {', '.join(faltan)}")


def dispositivo():
    import torch

    titulo("Dispositivo")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"cuda: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("mps: Apple Silicon")
    else:
        device = torch.device("cpu")
        print("cpu (no se detectó GPU)")
    return device


def forward_pass(device):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    titulo(f"Tokenizer + modelo ({MODEL_ID})")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID).to(device).eval()
    textos = ["The environment is ready.", "Fine-tuning with Hugging Face"]
    batch = tokenizer(textos, padding=True, return_tensors="pt").to(device)
    print(f"tokens: {tokenizer.convert_ids_to_tokens(batch['input_ids'][0])}")
    with torch.no_grad():
        salida = model(**batch)
    logits = salida.logits
    print(f"logits: {tuple(logits.shape)} en {logits.device}")


def metricas():
    import evaluate

    titulo("Métricas de evaluate")
    ejemplos = {
        "accuracy": dict(predictions=[0, 1, 1, 0], references=[0, 1, 0, 0]),
        "f1": dict(predictions=[0, 1, 1, 0], references=[0, 1, 0, 0], average="macro"),
        "seqeval": dict(
            predictions=[["B-PER", "O", "O", "B-LOC"], ["B-LOC", "O"]],
            references=[["B-PER", "I-PER", "O", "B-LOC"], ["B-LOC", "O"]],
        ),
        "sacrebleu": dict(
            predictions=["the cat is on the mat"],
            references=[["the cat sat on the mat"]],
        ),
        "rouge": dict(
            predictions=["the cat is on the mat"],
            references=["the cat sat on the mat"],
        ),
    }
    for nombre, args in ejemplos.items():
        resultado = evaluate.load(nombre).compute(**args)
        if nombre == "seqeval":
            resultado = {k: v for k, v in resultado.items() if k.startswith("overall")}
        if nombre == "sacrebleu":
            resultado = {"score": resultado["score"]}
        resumen = {k: round(float(v), 4) for k, v in resultado.items()}
        print(f"{nombre:<10} {resumen}")


def pins_colab():
    """Compara COLAB_PINS (celda de instalación del notebook) con el entorno local."""
    titulo("Versiones fijadas para Colab vs uv.lock")
    celdas = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    pins = None
    for celda in celdas:
        if celda["cell_type"] != "code":
            continue
        fuente = "".join(celda["source"])
        if "COLAB_PINS" not in fuente:
            continue
        for nodo in ast.walk(ast.parse(fuente)):
            if isinstance(nodo, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "COLAB_PINS" for t in nodo.targets
            ):
                pins = ast.literal_eval(nodo.value)
                break
        if pins is not None:
            break
    if pins is None:
        raise RuntimeError(f"no encontré COLAB_PINS en {NOTEBOOK.name}")
    distintos = []
    for paquete, version in pins.items():
        local = md.version(paquete)
        marca = "ok" if local == version else "DISTINTA"
        print(f"{paquete:<16} notebook {version:<10} local {local:<10} {marca}")
        if local != version:
            distintos.append(paquete)
    if distintos:
        raise RuntimeError(
            f"actualiza COLAB_PINS en el notebook para: {', '.join(distintos)}"
        )


def token():
    from dotenv import dotenv_values

    titulo("Token de Hugging Face")
    env = ROOT / ".env"
    valor = dotenv_values(env).get("HF_TOKEN") if env.exists() else None
    if valor:
        print(f"HF_TOKEN encontrado en {env.name} (no se muestra)")
    elif env.exists():
        print(".env existe pero HF_TOKEN está vacío")
    else:
        print("no hay .env (cp .env.example .env y pega el token)")


def main():
    fallos = []
    device = None
    pasos = [
        ("versiones", versiones),
        ("dispositivo", dispositivo),
        ("forward pass", lambda: forward_pass(device)),
        ("métricas", metricas),
        ("pins de Colab", pins_colab),
        ("token", token),
    ]
    for nombre, paso in pasos:
        try:
            resultado = paso()
            if nombre == "dispositivo":
                device = resultado
        except Exception:
            print(f"\n[FALLO] {nombre}")
            traceback.print_exc()
            fallos.append(nombre)

    titulo("Resultado")
    if fallos:
        print(f"Fallaron: {', '.join(fallos)}")
        sys.exit(1)
    print("Todo OK")


if __name__ == "__main__":
    main()
