"""
Parte 3: compara el baseline denso con BM25 solo y con el híbrido, desde los CSV crudos.

Además de las métricas del enunciado, mide la hipótesis declarada en PARTE3.md: la cobertura
de fuentes en las preguntas multi-documento (qué fracción de sus documentos fuente aparece en
el top-k), que el Hit Rate no ve porque ahí el acierto es a nivel de documento.

    <prefijo de EXPERIMENTOS.md> python comparar_extension.py

Escribe `salidas/<etiqueta>/comparacion_extension.md`.
"""

from __future__ import annotations

from pathlib import Path
import csv
import json

import experimentos
from tabla_metricas import fmt, metricas

VARIANTES = {"denso (baseline)": "", "bm25 solo": "bm25", "híbrido (RRF)": "hibrido"}


def leer(carpeta: Path, k: int) -> list[dict] | None:
    ruta = carpeta / f"resultados_k{k}.csv"
    return list(csv.DictReader(ruta.open(encoding="utf-8"))) if ruta.exists() else None


if __name__ == "__main__":
    base = experimentos.carpeta()
    golden = {g["id"]: g for g in json.loads(Path("golden_set.json").read_text(encoding="utf-8"))}
    multi_doc = [i for i, g in golden.items() if len(g["documentos_fuente"]) > 1]
    lineas = ["| variante | k | Hit Rate@k | MRR | abstención correcta | abstención indebida | "
              + " | ".join(f"cobertura P{i}" for i in multi_doc) + " |",
              "|---|---|---|---|---|---|" + "---|" * len(multi_doc)]
    detalle = {}
    for nombre, sub in VARIANTES.items():
        for k in (3, 5):
            filas = leer(base / sub if sub else base, k)
            if filas is None:
                continue
            m = metricas(filas)
            cob = []
            for i in multi_doc:
                f = next(f for f in filas if int(f["id"]) == i)
                fuentes = set(golden[i]["documentos_fuente"])
                cob.append(f"{len(fuentes & set(f['retrieved_docs'].split()))}/{len(fuentes)}")
            lineas.append(f"| {nombre} | {k} | {fmt(m['hit_rate'])} | {fmt(m['mrr'])} | "
                          f"{fmt(m['abstencion_correcta'])} | {fmt(m['abstencion_indebida'])} | "
                          + " | ".join(cob) + " |")
            detalle[(nombre, k)] = {int(f["id"]): f for f in filas}

    # Pregunta por pregunta: posición del primer acierto y si se abstuvo.
    lineas += ["", "Posición del primer acierto (— sin acierto, n/a negativa) · A = se abstuvo", "",
               "| id | tipo | " + " | ".join(f"{n} k={k}" for n, k in detalle) + " |",
               "|---|---|" + "---|" * len(detalle)]
    for i, g in golden.items():
        celdas = []
        for clave in detalle:
            f = detalle[clave][i]
            pos = f["posicion"] or ("n/a" if f["respondible"] != "True" else "—")
            celdas.append(pos + (" A" if f["abstuvo"] == "True" else ""))
        lineas.append(f"| {i} | {g['tipo']} | " + " | ".join(celdas) + " |")

    texto = "\n".join(lineas) + "\n"
    print(texto)
    (base / "comparacion_extension.md").write_text(texto, encoding="utf-8")
