"""
Parte 4 del taller: las cifras de la reflexión, sacadas de los CSV crudos.

1. Pregunta 1: el Hit Rate y el MRR «como antes» (negativas en el denominador, contadas como
   fallo) contra los actuales (solo respondibles), con los resultados de la Parte 2.b.
2. Pregunta 2: una pregunta global corrida de verdad por el baseline, para ver qué recupera.

    <prefijo de EXPERIMENTOS.md> python parte4_cifras.py

Escribe `salidas/<etiqueta>/parte4_cifras.txt`.
"""

from __future__ import annotations

from collections import Counter
import csv

import experimentos
from rag_pipeline import RagPipeline

PREGUNTA_GLOBAL = ("What are the main themes that run across all the papers in this corpus, "
                   "and how do they relate to each other?")
TOKENS_CORPUS = 387086   # salidas/bge-m3@ollama-local/parte1_ingesta.txt


def main() -> str:
    carpeta = experimentos.carpeta()
    lineas = ["== Pregunta 1: el denominador del Hit Rate"]
    for k in (3, 5):
        filas = list(csv.DictReader((carpeta / f"resultados_k{k}.csv").open(encoding="utf-8")))
        resp = [f for f in filas if f["respondible"] == "True"]
        neg = [f for f in filas if f["respondible"] != "True"]
        aciertos = sum(f["hit"] == "True" for f in resp)
        rr = sum(float(f["reciprocal_rank"]) for f in resp)
        abst_neg = sum(f["abstuvo"] == "True" for f in neg)
        lineas += [
            f"k={k}: aciertos {aciertos} de {len(resp)} respondibles · {len(neg)} negativas",
            f"  Hit Rate antes (sobre las {len(filas)}): {aciertos / len(filas):.3f} · "
            f"techo de un sistema perfecto: {len(resp) / len(filas):.3f}",
            f"  Hit Rate ahora (sobre las {len(resp)} respondibles): {aciertos / len(resp):.3f}",
            f"  MRR antes: {rr / len(filas):.3f} · MRR ahora: {rr / len(resp):.3f}",
            f"  abstención correcta: {abst_neg} de {len(neg)} negativas"]

    lineas += ["", "== Pregunta 2: una pregunta global por el RAG vectorial",
               f"pregunta: {PREGUNTA_GLOBAL}"]
    pipeline = RagPipeline()
    salida = pipeline.answer(PREGUNTA_GLOBAL, top_k=5)
    docs = Counter(h["document"] for h in salida["contexts"])
    for pos, h in enumerate(salida["contexts"], start=1):
        lineas.append(f"  {pos}. {h['score']:.3f}  {h['chunk_id']}")
    lineas += [f"documentos distintos en el top-5: {len(docs)} de 16 · {dict(docs)}",
               f"contexto: 5 × 512 = 2560 tokens de {TOKENS_CORPUS} "
               f"({2560 / TOKENS_CORPUS:.2%} del corpus)",
               f"respuesta [{salida['generator']}] abstuvo={salida['abstained']}:",
               "  " + " ".join(salida["answer"].split())]
    return "\n".join(lineas) + "\n"


if __name__ == "__main__":
    texto = main()
    print(texto)
    (experimentos.carpeta() / "parte4_cifras.txt").write_text(texto, encoding="utf-8")
