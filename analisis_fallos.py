"""
Parte 2.c del taller: evidencia de los peores casos, sacada del CSV crudo y del índice.

Para cada caso muestra lo que la 2.c pide:
  - la salida de la ingesta de sus documentos fuente;
  - los fragmentos recuperados, con su puntaje y con qué palabras clave de la respuesta
    esperada contiene cada uno (para separar «no se recuperó» de «se recuperó y no se usó»);
  - la respuesta generada.

    <prefijo de EXPERIMENTOS.md> python analisis_fallos.py

Escribe `salidas/<etiqueta>/fallos_evidencia.txt`.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
import csv
import io
import json

from qdrant_client import models

import experimentos
from rag_pipeline import RagPipeline, _normalizar_texto

# (id, k, palabras clave de la respuesta esperada que se buscan en cada fragmento recuperado)
CASOS = [
    (6, 5, ["reward model", "reinforcement learning", "PPO", "binary cross entropy",
            "change of variables"]),
    (5, 3, ["dense vector index", "non-parametric", "faithfulness", "answer relevance",
            "context relevance"]),
    (3, 3, ["Leiden", "Community detection (e.g., Leiden"]),
    (4, 5, ["100B", "only yields performance gains when used with models of", "540B"]),
]


def main() -> str:
    pipeline = RagPipeline()
    golden = {g["id"]: g for g in json.loads(Path("golden_set.json").read_text(encoding="utf-8"))}
    with redirect_stdout(io.StringIO()) as ingesta:
        pipeline.ingest(Path("corpus"), chunk_tokens=512, overlap_tokens=102)
    lineas_ingesta = ingesta.getvalue().splitlines()
    carpeta = experimentos.carpeta()
    lineas = []
    for id_, k, claves in CASOS:
        fila = next(r for r in csv.DictReader((carpeta / f"resultados_k{k}.csv").open(encoding="utf-8"))
                    if int(r["id"]) == id_)
        g = golden[id_]
        lineas += [f"===== caso {id_} ({g['tipo']}) · k={k} · posición={fila['posicion'] or '—'} "
                   f"· hit={fila['hit']} · abstuvo={fila['abstuvo']}",
                   f"pregunta: {g['pregunta']}",
                   f"fuente: {g['documentos_fuente']} · frase esperada: {g['fragmento_esperado']!r}",
                   "ingesta:"]
        lineas += ["  " + l for l in lineas_ingesta if any(d in l for d in g["documentos_fuente"])]
        lineas.append("recuperados:")
        ids, puntajes = fila["retrieved_ids"].split(), fila["retrieved_scores"].split()
        for pos, (cid, sc) in enumerate(zip(ids, puntajes), start=1):
            punto = pipeline.client.scroll(
                pipeline.collection, limit=1, with_payload=True,
                scroll_filter=models.Filter(must=[models.FieldCondition(
                    key="chunk_id", match=models.MatchValue(value=cid))]))[0][0]
            texto = _normalizar_texto(punto.payload["text"])
            tiene = [c for c in claves if _normalizar_texto(c) in texto]
            lineas.append(f"  {pos}. {float(sc):.3f}  {cid}  contiene: {tiene or '—'}")
        lineas += ["respuesta:", "  " + " ".join(fila["respuesta"].split()), ""]
    return "\n".join(lineas)


if __name__ == "__main__":
    salida = main()
    print(salida)
    (experimentos.carpeta() / "fallos_evidencia.txt").write_text(salida + "\n", encoding="utf-8")
