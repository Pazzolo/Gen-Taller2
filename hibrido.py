"""
Parte 3, Opción B: búsqueda híbrida, BM25 + denso fusionados por Reciprocal Rank Fusion.

`PipelineHibrido` hereda de `RagPipeline` y solo reemplaza `retrieve`. El prompt, el
generador, el detector de abstención y el evaluador son los mismos, así que la comparación con
el baseline cambia una sola cosa. BM25 se construye sobre los MISMOS fragmentos del índice de
Qdrant (se leen de la colección, no se re-fragmenta), para que los dos recuperadores vean
exactamente el mismo corpus.

    <prefijo de EXPERIMENTOS.md> python hibrido.py --modo hibrido     # BM25 + denso, con generación
    <prefijo de EXPERIMENTOS.md> python hibrido.py --modo bm25 --sin-generar   # diagnóstico

Escribe `salidas/<etiqueta>/<modo>/resultados_k{3,5}.csv` y una fila por k en experimentos.csv.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

from rank_bm25 import BM25Okapi

import experimentos
from evaluation import _fmt, escribir_csv, evaluate_retrieval
from rag_pipeline import EMBEDDING_MODEL, RagPipeline, _normalizar_texto

N_CANDIDATOS = 20   # de cada recuperador, antes de fusionar
K_RRF = 60          # constante de RRF (Cormack et al., 2009); fijada antes de medir
VACIAS = set("""a an the of to in on for and or with by is are was were be been being it its as
at from that this these those which what who whom how does do did done into their them they
there than then so such not no can could would should will may might also only both each other
more most some any all our we you your he she his her i me my if but about after before over
under between through during while where when why""".split())


def tokenizar(texto: str) -> list[str]:
    """Misma normalización que el evaluador (minúsculas, NFKD, sin puntuación), lo que además
    deshace las ligaduras (ﬂ → fl), y sin palabras vacías en inglés."""
    return [t for t in _normalizar_texto(texto).split() if t not in VACIAS]


class PipelineHibrido(RagPipeline):
    def __init__(self, modo: str = "hibrido", **kw):
        super().__init__(**kw)
        if modo not in {"hibrido", "bm25"}:
            raise ValueError(f"modo {modo!r}: usa hibrido o bm25")
        self.modo = modo
        self.fragmentos = self._leer_coleccion()
        self.bm25 = BM25Okapi([tokenizar(f["text"]) for f in self.fragmentos])

    def _leer_coleccion(self) -> list[dict]:
        puntos, desde = [], None
        while True:
            lote, desde = self.client.scroll(self.collection, limit=256, offset=desde,
                                             with_payload=True)
            puntos += lote
            if desde is None:
                break
        if not puntos:
            raise SystemExit(f"la colección {self.collection} está vacía: indexa primero (demo_top5.py)")
        return [{"chunk_id": p.payload["chunk_id"], "document": p.payload["document"],
                 "text": p.payload["text"]} for p in sorted(puntos, key=lambda p: p.id)]

    def retrieve_bm25(self, question: str, top_k: int) -> list[dict]:
        puntajes = self.bm25.get_scores(tokenizar(question))
        orden = sorted(range(len(puntajes)), key=lambda i: -puntajes[i])[:top_k]
        return [{**self.fragmentos[i], "score": float(puntajes[i])} for i in orden]

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        if self.modo == "bm25":
            return self.retrieve_bm25(question, top_k)
        densos = super().retrieve(question, top_k=N_CANDIDATOS)
        lexicos = self.retrieve_bm25(question, N_CANDIDATOS)
        fusion: dict[str, dict] = {}
        for lista, nombre in ((densos, "rank_denso"), (lexicos, "rank_bm25")):
            for pos, hit in enumerate(lista, start=1):
                f = fusion.setdefault(hit["chunk_id"], {**hit, "score": 0.0,
                                                        "rank_denso": None, "rank_bm25": None})
                f["score"] += 1 / (K_RRF + pos)
                f[nombre] = pos
        return sorted(fusion.values(), key=lambda f: -f["score"])[:top_k]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["hibrido", "bm25"], default="hibrido")
    ap.add_argument("--golden", type=Path, default=Path("golden_set.json"))
    ap.add_argument("--sin-generar", action="store_true")
    args = ap.parse_args()

    pipeline = PipelineHibrido(modo=args.modo)
    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    carpeta = experimentos.carpeta() / args.modo
    for k in (3, 5):
        r = evaluate_retrieval(golden, pipeline, k=k, generar=not args.sin_generar)
        ruta = carpeta / f"resultados_k{k}.csv"
        escribir_csv(r["rows"], ruta, EMBEDDING_MODEL)
        print(f"{args.modo} k={k}: hit={_fmt(r['hit_rate'])} mrr={_fmt(r['mrr'])} "
              f"abst_correcta={_fmt(r['abstencion_correcta'])} "
              f"abst_indebida={_fmt(r['abstencion_indebida'])}")
        experimentos.registrar(
            f"3.B {args.modo}", k=k, chunk_tokens=512, overlap_tokens=102,
            generador=" ".join(sorted({x["generador"] for x in r["rows"] if x["generador"]})),
            hit_rate=_fmt(r["hit_rate"]), mrr=_fmt(r["mrr"]),
            abstencion_correcta=_fmt(r["abstencion_correcta"]),
            abstencion_indebida=_fmt(r["abstencion_indebida"]), salida=str(ruta),
            nota=f"{N_CANDIDATOS} candidatos por lado, RRF k={K_RRF}" if args.modo == "hibrido"
                 else "BM25 solo, diagnóstico" + (" · sin generar" if args.sin_generar else ""))
