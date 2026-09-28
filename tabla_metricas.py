"""
Parte 2.b del taller: la tabla de métricas, reconstruida SOLO desde los CSV crudos.

Lee `salidas/<etiqueta>/resultados_k*.csv` (los que escribe `evaluation.py`), recalcula las
cuatro métricas sin usar el evaluador, las compara con lo que el evaluador registró en
`experimentos.csv` y junta las filas de todos los k en `salidas/<etiqueta>/resultados.csv`.

    <prefijo de EXPERIMENTOS.md> python tabla_metricas.py               # solo la configuración
    <prefijo de EXPERIMENTOS.md> python tabla_metricas.py --entregable  # y copia a ./resultados.csv

Solo `--entregable` escribe el `resultados.csv` de la raíz, el entregable crudo del informe,
para que correr otra configuración (la H200, OpenAI) no lo pise sin querer.

Escribe `salidas/<etiqueta>/resultados.csv` y `salidas/<etiqueta>/tabla_metricas.md`.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import shutil

import experimentos


def media(valores: list[float]) -> float | None:
    return sum(valores) / len(valores) if valores else None


def metricas(filas: list[dict]) -> dict:
    """Las mismas definiciones que evaluation.py, desde las columnas del CSV."""
    resp = [f for f in filas if f["respondible"] == "True"]
    neg = [f for f in filas if f["respondible"] != "True"]
    return {
        "n": len(filas), "n_resp": len(resp), "n_neg": len(neg),
        "hit_rate": media([float(f["hit"] == "True") for f in resp]),
        "mrr": media([float(f["reciprocal_rank"]) for f in resp]),
        "abstencion_correcta": media([float(f["abstuvo"] == "True") for f in neg if f["abstuvo"]]),
        "abstencion_indebida": media([float(f["abstuvo"] == "True") for f in resp if f["abstuvo"]]),
    }


def fmt(x: float | None) -> str:
    return "sin medir" if x is None else f"{x:.3f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entregable", action="store_true",
                    help="copia también a ./resultados.csv, el entregable del informe")
    args = ap.parse_args()
    carpeta = experimentos.carpeta()
    por_k = {int(p.stem.split("_k")[1]): list(csv.DictReader(p.open(encoding="utf-8")))
             for p in sorted(carpeta.glob("resultados_k*.csv"))}
    if not por_k:
        raise SystemExit(f"no hay resultados_k*.csv en {carpeta}/: corre evaluation.py primero")

    tabla = ["| k | preguntas | respondibles | negativas | Hit Rate@k | MRR | abstención correcta | abstención indebida |",
             "|---|---|---|---|---|---|---|---|"]
    for k, filas in sorted(por_k.items()):
        m = metricas(filas)
        tabla.append(f"| {k} | {m['n']} | {m['n_resp']} | {m['n_neg']} | {fmt(m['hit_rate'])} | "
                     f"{fmt(m['mrr'])} | {fmt(m['abstencion_correcta'])} | {fmt(m['abstencion_indebida'])} |")

    # Por tipo de pregunta: dónde falla cada k.
    tipos = sorted({f["tipo"] for filas in por_k.values() for f in filas})
    tabla += ["", "| k | tipo | n | Hit Rate | MRR | se abstuvo |", "|---|---|---|---|---|---|"]
    for k, filas in sorted(por_k.items()):
        for t in tipos:
            del_tipo = [f for f in filas if f["tipo"] == t]
            resp = [f for f in del_tipo if f["respondible"] == "True"]
            tabla.append(f"| {k} | {t} | {len(del_tipo)} | "
                         f"{fmt(media([float(f['hit'] == 'True') for f in resp]))} | "
                         f"{fmt(media([float(f['reciprocal_rank']) for f in resp]))} | "
                         f"{fmt(media([float(f['abstuvo'] == 'True') for f in del_tipo if f['abstuvo']]))} |")

    # Comprobación: las cifras recalculadas tienen que coincidir con las del evaluador.
    registradas = [r for r in csv.DictReader(experimentos.CSV.open(encoding="utf-8"))
                   if r["etapa"] == "2.b evaluación" and r["etiqueta"] == experimentos.etiqueta()]
    avisos = []
    for k, filas in por_k.items():
        ultima = next((r for r in reversed(registradas) if r["k"] == str(k)), None)
        m = metricas(filas)
        for c in ["hit_rate", "mrr", "abstencion_correcta", "abstencion_indebida"]:
            if ultima and ultima[c] != fmt(m[c]):
                avisos.append(f"k={k} {c}: CSV {fmt(m[c])} ≠ evaluador {ultima[c]}")
    tabla += ["", "comprobación contra evaluation.py: " + ("coinciden" if not avisos else "; ".join(avisos))]

    texto = "\n".join(tabla) + "\n"
    print(texto)
    (carpeta / "tabla_metricas.md").write_text(texto, encoding="utf-8")

    todas = [f for _, filas in sorted(por_k.items()) for f in filas]
    ruta = carpeta / "resultados.csv"
    with ruta.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(todas[0].keys()))
        w.writeheader()
        w.writerows(todas)
    print(f"{ruta}: {len(todas)} filas ({', '.join(f'k={k}' for k in sorted(por_k))})")
    if args.entregable:
        shutil.copyfile(ruta, "resultados.csv")
        print(f"copiado a ./resultados.csv (entregable del informe)")
