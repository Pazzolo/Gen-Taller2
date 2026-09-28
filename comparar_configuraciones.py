"""
Compara el baseline con bge-m3 en Ollama local contra bge-m3 en la H200 (y cualquier otra
configuración que tenga resultados en `salidas/`).

Tres niveles, del más fuerte al más débil:
  1. Identidad del modelo: el digest de `bge-m3` en los dos Ollama. Si coincide, los pesos son
     los mismos byte a byte, y cualquier diferencia vendría del servidor, no del modelo.
  2. Vectores: los mismos textos (fragmentos del índice y las preguntas del golden set)
     vectorizados en los dos servidores, con la misma petición que el pipeline
     (`truncate: false`, `num_ctx` 8192); coseno por texto.
  3. Recuperación: métricas y top-k por pregunta desde los CSV crudos de cada configuración.

    python comparar_configuraciones.py                 # necesita la VPN (H200) y el Ollama local

Escribe `salidas/comparacion_configuraciones.md`.
"""

from __future__ import annotations

from pathlib import Path
import csv
import json
import urllib.request

import numpy as np
from qdrant_client import QdrantClient

from tabla_metricas import fmt, metricas

SERVIDORES = {"ollama-local": "http://localhost:11434", "h200": "http://172.28.230.10:11434"}
COLECCION_TEXTOS = "taller2_bge_m3_local"   # de aquí se toman los fragmentos a comparar
N_FRAGMENTOS = 40
SALIDAS = Path("salidas")


def pedir(url: str, ruta: str, cuerpo: dict | None = None) -> dict:
    datos = None if cuerpo is None else json.dumps(cuerpo).encode()
    req = urllib.request.Request(url + ruta, data=datos, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def modelo_servido(url: str) -> dict:
    return next(m for m in pedir(url, "/api/tags")["models"] if m["name"].lower().startswith("bge-m3"))


def vectorizar(url: str, nombre: str, textos: list[str]) -> np.ndarray:
    salida = []
    for i in range(0, len(textos), 16):
        r = pedir(url, "/api/embed", {"model": nombre, "input": textos[i:i + 16], "truncate": False,
                                      "options": {"num_ctx": 8192}})
        salida += r["embeddings"]
    v = np.asarray(salida, dtype=np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def textos_de_prueba() -> list[str]:
    cliente = QdrantClient(url="http://localhost:6333")
    puntos, desde = [], None
    while True:
        lote, desde = cliente.scroll(COLECCION_TEXTOS, limit=256, offset=desde, with_payload=True)
        puntos += lote
        if desde is None:
            break
    puntos.sort(key=lambda p: p.id)
    paso = max(1, len(puntos) // N_FRAGMENTOS)
    fragmentos = [p.payload["text"] for p in puntos[::paso][:N_FRAGMENTOS]]
    preguntas = [g["pregunta"] for g in json.loads(Path("golden_set.json").read_text(encoding="utf-8"))]
    return fragmentos + preguntas


def seccion_modelo_y_vectores() -> list[str]:
    modelos = {h: modelo_servido(u) for h, u in SERVIDORES.items()}
    lineas = ["## 1. Identidad del modelo", "",
              "| servidor | nombre | digest | tamaño | cuantización | parámetros |",
              "|---|---|---|---|---|---|"]
    for h, m in modelos.items():
        d = m.get("details", {})
        lineas.append(f"| {h} | `{m['name']}` | `{m['digest'][:16]}…` | {m['size'] / 1e9:.2f} GB | "
                      f"{d.get('quantization_level', '?')} | {d.get('parameter_size', '?')} |")
    iguales = len({m["digest"] for m in modelos.values()}) == 1
    lineas += ["", f"**Digest {'idéntico' if iguales else 'DISTINTO'}** en los dos servidores.", ""]

    textos = textos_de_prueba()
    vec = {h: vectorizar(SERVIDORES[h], modelos[h]["name"], textos) for h in SERVIDORES}
    cos = np.sum(vec["ollama-local"] * vec["h200"], axis=1)
    nf = len(textos) - 10
    lineas += ["## 2. Vectores: mismo texto, dos servidores", "",
               f"{nf} fragmentos del índice (uno de cada {948 // N_FRAGMENTOS}) y las 10 preguntas "
               "del golden set, con la misma petición que usa el pipeline.", "",
               "| textos | n | coseno mínimo | coseno medio | coseno máximo |", "|---|---|---|---|---|",
               f"| fragmentos | {nf} | {cos[:nf].min():.6f} | {cos[:nf].mean():.6f} | {cos[:nf].max():.6f} |",
               f"| preguntas | 10 | {cos[nf:].min():.6f} | {cos[nf:].mean():.6f} | {cos[nf:].max():.6f} |",
               ""]
    return lineas


def leer(carpeta: Path, k: int) -> list[dict] | None:
    ruta = carpeta / f"resultados_k{k}.csv"
    return list(csv.DictReader(ruta.open(encoding="utf-8"))) if ruta.exists() else None


def seccion_recuperacion() -> list[str]:
    configs = sorted(p.name for p in SALIDAS.iterdir() if p.is_dir() and (p / "resultados_k3.csv").exists())
    variantes = [(c, "") for c in configs] + [(c, "hibrido") for c in configs
                                              if (SALIDAS / c / "hibrido" / "resultados_k3.csv").exists()]
    lineas = ["## 3. Recuperación y generación", "",
              "| configuración | variante | k | Hit Rate@k | MRR | abstención correcta | abstención indebida |",
              "|---|---|---|---|---|---|---|"]
    filas_por = {}
    for c, sub in variantes:
        for k in (3, 5):
            filas = leer(SALIDAS / c / sub if sub else SALIDAS / c, k)
            if filas is None:
                continue
            m = metricas(filas)
            lineas.append(f"| {c} | {sub or 'denso'} | {k} | {fmt(m['hit_rate'])} | {fmt(m['mrr'])} | "
                          f"{fmt(m['abstencion_correcta'])} | {fmt(m['abstencion_indebida'])} |")
            filas_por[(c, sub, k)] = {int(f["id"]): f for f in filas}

    # Mismo top-k? Pregunta por pregunta, entre local y H200, en el denso.
    a, b = "bge-m3@ollama-local", "bge-m3@h200"
    for k in (3, 5):
        if (a, "", k) in filas_por and (b, "", k) in filas_por:
            iguales_orden = iguales_conj = 0
            difs = []
            for i, fa in filas_por[(a, "", k)].items():
                fb = filas_por[(b, "", k)][i]
                ra, rb = fa["retrieved_ids"].split(), fb["retrieved_ids"].split()
                iguales_orden += ra == rb
                iguales_conj += set(ra) == set(rb)
                sa = [float(x) for x in fa["retrieved_scores"].split()]
                sb = [float(x) for x in fb["retrieved_scores"].split()]
                difs += [abs(x - y) for x, y in zip(sa, sb)] if ra == rb else []
                cambio_resp = fa["abstuvo"] != fb["abstuvo"]
                if ra != rb or cambio_resp:
                    lineas.append(f"- k={k}, pregunta {i}: top-k {'igual' if ra == rb else 'distinto'}"
                                  f"{' · abstención ' + fa['abstuvo'] + ' → ' + fb['abstuvo'] if cambio_resp else ''}")
            lineas.append("")
            lineas.append(f"**k={k}, local contra H200:** mismo top-k en el mismo orden en "
                          f"{iguales_orden} de 10 preguntas, mismo conjunto en {iguales_conj} de 10; "
                          f"diferencia máxima de puntaje en los top-k iguales: "
                          f"{max(difs) if difs else float('nan'):.4f}.")
            lineas.append("")
    return lineas


if __name__ == "__main__":
    texto = "\n".join(["# Comparación de configuraciones: bge-m3 local contra H200", ""]
                      + seccion_modelo_y_vectores() + seccion_recuperacion()) + "\n"
    print(texto)
    (SALIDAS / "comparacion_configuraciones.md").write_text(texto, encoding="utf-8")
