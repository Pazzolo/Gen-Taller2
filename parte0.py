"""
Parte 0 del taller: las tres fallas que no fallan, sobre `ejemplos/` y con el MiniLM local.

    EMBEDDING_BACKEND=local QDRANT_URL=":memory:" python parte0.py

Escribe la salida cruda de cada parte en `salidas/parte0{a,b,c}_*.txt`. La salida de la
ingesta de la 0.a sale de `rag_pipeline.py` tal cual (`salidas/parte0a_ingesta.txt`); aquí
se añade lo que esa ingesta habría producido SIN la comprobación de `load_corpus`.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import io
import os

os.environ.setdefault("EMBEDDING_BACKEND", "local")
os.environ.setdefault("QDRANT_URL", ":memory:")

from ingestion import (MIN_CARACTERES_FRAGMENTO, caracteres_utiles,   # noqa: E402
                       fixed_size_chunks_tokens, normalize_text, read_document)
from rag_pipeline import EMBEDDING_BACKEND, RagPipeline   # noqa: E402

EJEMPLOS = Path("ejemplos")
SALIDAS = Path("salidas")
ESCANEADO = EJEMPLOS / "instructivo_escaneado.pdf"
# Texto de 900 palabras para la 0.b: las primeras 900 palabras que pypdf extrae de
# vaswani-2017-attention-is-all-you-need.pdf, guardadas para que la 0.b no dependa del corpus.
# Se corta por palabras, no por tokens: los tokens se cuentan.
TEXTO_0B = SALIDAS / "parte0b_texto.txt"
PREGUNTAS_0C = ["¿cuál es la política de mascotas?",
                "¿qué exige el instructivo de prácticas de campo sobre el seguro de accidentes?"]


def guardar(nombre: str, texto: str) -> None:
    (SALIDAS / nombre).write_text(texto, encoding="utf-8")
    print(f"\n===== {nombre} =====\n{texto}")


def parte_0a(pipeline: RagPipeline) -> str:
    """Lo que `load_corpus` habría indexado del PDF escaneado sin su comprobación."""
    doc = read_document(ESCANEADO)
    texto = normalize_text(doc.text)
    piezas = fixed_size_chunks_tokens(texto, pipeline.encoder.tokenizer, 126, 25)
    lineas = [f"documento: {doc.name} · {doc.pages} página(s) · useful_chars={doc.useful_chars} "
              f"· parece_escaneado={doc.parece_escaneado}",
              f"texto extraído por pypdf (repr): {doc.text!r}",
              f"fragmentos que produciría la fragmentación sin la comprobación: {len(piezas)}"]
    for i, p in enumerate(piezas):
        lineas.append(f"  [{i}] {p!r} · caracteres útiles={caracteres_utiles(p)} "
                      f"(el filtro por fragmento exige {MIN_CARACTERES_FRAGMENTO})")
    return "\n".join(lineas) + "\n"


def parte_0b(pipeline: RagPipeline) -> str:
    """El aviso de chunk_tokens=900 y el coseno entre el texto entero y el truncado."""
    import numpy as np

    avisos = io.StringIO()
    with redirect_stderr(avisos), redirect_stdout(io.StringIO()):
        pipeline.ingest(EJEMPLOS, chunk_tokens=900)
    aviso = next(l for l in avisos.getvalue().splitlines() if "chunk_tokens=900" in l)

    texto = TEXTO_0B.read_text(encoding="utf-8").strip()
    tok = pipeline.encoder.tokenizer
    n_tokens = len(tok(texto, add_special_tokens=False, truncation=False)["input_ids"])
    tope = pipeline.max_seq_tokens
    # Lo que el modelo ve de verdad: sus primeros `tope` tokens, especiales incluidos.
    ids = tok(texto, truncation=True, max_length=tope)["input_ids"]
    recortado = tok.decode(ids, skip_special_tokens=True)
    v_entero, v_recortado = pipeline.encoder.encode([texto, recortado])
    coseno = float(np.dot(v_entero, v_recortado))   # vectores normalizados
    return (f"aviso de la ingesta:\n  {aviso}\n\n"
            f"texto: {TEXTO_0B} ({len(texto.split())} palabras)\n"
            f"tokens del texto (tokenizador del modelo, sin especiales): {n_tokens}\n"
            f"tope del modelo (max_seq_length): {tope}\n"
            f"fracción que llega al vector: {(tope - 2) / n_tokens:.1%} "
            f"({tope - 2} tokens de contenido + 2 especiales)\n"
            f"coseno(vector del texto entero, vector del texto recortado a {tope}): {coseno:.6f}\n")


def parte_0c(pipeline: RagPipeline) -> str:
    """Los tres vecinos de dos preguntas sin respuesta en el índice."""
    lineas = []
    for pregunta in PREGUNTAS_0C:
        lineas.append(f"pregunta: {pregunta}")
        for pos, hit in enumerate(pipeline.retrieve(pregunta, top_k=3), start=1):
            lineas.append(f"  {pos}. {hit['score']:.3f}  {hit['document']} / {hit['chunk_id']}  "
                          f"{hit['text'][:70]}…")
        lineas.append("")
    return "\n".join(lineas)


if __name__ == "__main__":
    if EMBEDDING_BACKEND != "local":
        raise SystemExit("la Parte 0 corre con EMBEDDING_BACKEND=local (el MiniLM de tope 128)")
    SALIDAS.mkdir(exist_ok=True)
    pipeline = RagPipeline()
    guardar("parte0a_sin_comprobacion.txt", parte_0a(pipeline))
    guardar("parte0b_truncado.txt", parte_0b(pipeline))
    with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
        pipeline.index(pipeline.ingest(EJEMPLOS))   # baseline por defecto, como en la 0.a
    guardar("parte0c_indice_no_se_queja.txt", parte_0c(pipeline))
