"""
Parte 1.3–1.5 del taller: indexa el corpus en Qdrant, muestra el top-5 de tres preguntas de
prueba y genera la respuesta de cada una.

    <prefijo de EXPERIMENTOS.md> python demo_top5.py
    <prefijo de EXPERIMENTOS.md> python demo_top5.py --sin-indexar

Indexa contra el Qdrant del contenedor (`QDRANT_URL` del .env), no `:memory:`, para que la
Parte 2 pueda evaluar sobre el mismo índice con `evaluation.py --sin-indexar`. Antes de
vectorizar cuenta los tokens del corpus y estima el costo con el precio de la fila.

Escribe `salidas/<etiqueta>/parte1_top5.txt` y una fila en `experimentos.csv`.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
import argparse
import io
import time

import experimentos

from rag_pipeline import (EMBEDDING_BACKEND, EMBEDDING_MODEL, QDRANT_URL, COLLECTION,
                          RagPipeline)

CORPUS = Path("corpus")
CHUNK_TOKENS, OVERLAP_TOKENS = 512, 102   # justificados en PARTE1.md, 1.2
# Precio de la fila embed_api_economico (text-embedding-3-small), verificado 2026-08-27.
USD_POR_MILLON = {"openai": 0.02, "h200": 0.0, "local": 0.0}
# Tres preguntas de prueba, en inglés como el corpus. No son del golden set: sirven para ver
# que el pipeline funciona de punta a punta, no para medirlo.
PREGUNTAS = [
    "How long did it take to train the big Transformer model, and on what hardware?",
    "Why does DPO not need to train an explicit reward model?",
    "Which quality aspects of a RAG answer does RAGAS evaluate without human references?",
]


def main(indexar: bool) -> tuple[str, dict]:
    pipeline = RagPipeline()
    cifras = {"chunk_tokens": CHUNK_TOKENS, "overlap_tokens": OVERLAP_TOKENS}
    lineas = [f"embeddings: {EMBEDDING_BACKEND} · {EMBEDDING_MODEL} · tope {pipeline.max_seq_tokens} "
              f"· Qdrant {QDRANT_URL} · colección {COLLECTION}"]
    if indexar:
        ingesta = io.StringIO()
        with redirect_stdout(ingesta):
            chunks = pipeline.ingest(CORPUS, chunk_tokens=CHUNK_TOKENS, overlap_tokens=OVERLAP_TOKENS)
        tok = pipeline.encoder.tokenizer
        n = sum(len(tok(c.text, add_special_tokens=False, truncation=False)["input_ids"])
                for c in chunks)
        usd = n / 1e6 * USD_POR_MILLON[EMBEDDING_BACKEND]
        lineas.append(f"indexando {len(chunks)} fragmentos ({CHUNK_TOKENS}/{OVERLAP_TOKENS}) · "
                      f"{n} tokens · costo estimado USD {usd:.4f}")
        t0 = time.perf_counter()
        pipeline.index(chunks)
        segundos = time.perf_counter() - t0
        cifras.update(n_fragmentos=len(chunks), tokens_indexados=n, costo_usd=f"{usd:.4f}",
                      segundos=f"{segundos:.1f}")
        info = pipeline.client.get_collection(COLLECTION)
        lineas.append(f"indexados: {info.points_count} puntos · dimensión "
                      f"{info.config.params.vectors.size} · distancia {info.config.params.vectors.distance} "
                      f"· {segundos:.1f} s")
    for i, pregunta in enumerate(PREGUNTAS, start=1):
        salida = pipeline.answer(pregunta, top_k=5)
        lineas += ["", f"[{i}] {pregunta}"]
        for pos, hit in enumerate(salida["contexts"], start=1):
            texto = " ".join(hit["text"].split())
            lineas.append(f"  {pos}. {hit['score']:.3f}  {hit['document']} / {hit['chunk_id']}  "
                          f"{texto[:90]}…")
        lineas.append(f"  respuesta [{salida['generator']}] abstuvo={salida['abstained']}:")
        lineas += ["    " + l for l in salida["answer"].strip().splitlines()]
        cifras["generador"] = salida["generator"]
    return "\n".join(lineas) + "\n", cifras


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sin-indexar", action="store_true", help="usa la colección ya indexada")
    indexar = not ap.parse_args().sin_indexar
    salida, cifras = main(indexar)
    print(salida)
    ruta = experimentos.carpeta() / "parte1_top5.txt"
    ruta.write_text(salida, encoding="utf-8")
    experimentos.registrar("1.3-1.5 índice + top-5" if indexar else "1.4-1.5 top-5",
                           salida=str(ruta), **cifras)
