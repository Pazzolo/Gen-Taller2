"""
Registro de corridas, para comparar configuraciones (bge-m3 en Ollama local, bge-m3 en la H200,
OpenAI, …) sobre el mismo corpus y el mismo golden set.

Cada script que produce resultados:
  - escribe su salida cruda en `salidas/<etiqueta>/`, con la etiqueta de la configuración, y
  - añade una fila a `experimentos.csv` con `registrar(...)`.

    python experimentos.py      # regenera la tabla de EXPERIMENTOS.md desde experimentos.csv

La etiqueta sale de las variables de entorno que usa `rag_pipeline.py`, así que dos corridas
con la misma configuración escriben en la misma carpeta y se pueden comparar fila a fila.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import csv
import os

CSV = Path("experimentos.csv")
MD = Path("EXPERIMENTOS.md")
INICIO, FIN = "<!-- tabla:inicio -->", "<!-- tabla:fin -->"
COLUMNAS = ["fecha", "etapa", "etiqueta", "fila_embeddings", "modelo_embeddings",
            "host_embeddings", "generador", "chunk_tokens", "overlap_tokens", "n_fragmentos",
            "tokens_indexados", "costo_usd", "segundos", "k", "hit_rate", "mrr",
            "abstencion_correcta", "abstencion_indebida", "salida", "nota"]


def host_embeddings() -> str:
    """Dónde se calculan los embeddings: 'ollama-local', 'h200', 'api-openai' o 'local'."""
    backend = os.getenv("EMBEDDING_BACKEND", "h200").strip().lower()
    if backend == "h200":
        host = urlparse(os.getenv("H200_EMBED_URL", "http://172.28.230.10:11434")).hostname
        return "ollama-local" if host in {"localhost", "127.0.0.1"} else "h200"
    return {"openai": "api-openai", "local": "local"}.get(backend, backend)


def etiqueta() -> str:
    """Nombre corto de la configuración de embeddings: 'bge-m3@ollama-local', 'bge-m3@h200'…"""
    from rag_pipeline import EMBEDDING_MODEL
    return f"{EMBEDDING_MODEL.split('/')[-1]}@{host_embeddings()}"


def carpeta() -> Path:
    c = Path("salidas") / etiqueta()
    c.mkdir(parents=True, exist_ok=True)
    return c


def registrar(etapa: str, **campos) -> None:
    """Añade una fila a experimentos.csv. Los campos de configuración se rellenan solos."""
    from rag_pipeline import EMBEDDING_BACKEND, EMBEDDING_MODEL, FILA_EMBEDDINGS
    fila = {"fecha": datetime.now().isoformat(timespec="seconds"), "etapa": etapa,
            "etiqueta": etiqueta(), "fila_embeddings": FILA_EMBEDDINGS[EMBEDDING_BACKEND],
            "modelo_embeddings": EMBEDDING_MODEL, "host_embeddings": host_embeddings()}
    fila.update({k: v for k, v in campos.items() if k in COLUMNAS})
    nuevo = not CSV.exists()
    with CSV.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        if nuevo:
            w.writeheader()
        w.writerow(fila)


def tabla_markdown() -> str:
    if not CSV.exists():
        return "_Sin corridas todavía._"
    filas = list(csv.DictReader(CSV.open(encoding="utf-8")))
    # Solo las columnas que tienen algún valor, para que la tabla quepa.
    cols = [c for c in COLUMNAS if any(f.get(c) for f in filas)]
    lineas = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for f in filas:
        lineas.append("| " + " | ".join((f.get(c) or "").replace("|", "/") for c in cols) + " |")
    return "\n".join(lineas)


if __name__ == "__main__":
    texto = MD.read_text(encoding="utf-8")
    antes, resto = texto.split(INICIO)
    _, despues = resto.split(FIN)
    MD.write_text(f"{antes}{INICIO}\n{tabla_markdown()}\n{FIN}{despues}", encoding="utf-8")
    print(tabla_markdown())
