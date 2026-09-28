"""
Parte 2.a del taller: verifica el golden set contra los fragmentos que de verdad se indexan.

El golden set es el verificador, y un verificador mal anotado falla sin avisar: una
`fragmento_esperado` que cruza el borde de un fragmento, o que tiene una palabra cortada por el
guion de fin de línea, nunca coincide, y la pregunta cuenta como fallo de recuperación sin
serlo. Este script lo comprueba antes de medir, con la misma `load_corpus`, el mismo tokenizador
y la misma normalización (`_normalizar_texto`) que usa `evaluation.py`.

Para cada pregunta respondible comprueba:
  - que sus documentos fuente existen en el corpus;
  - en cuántos fragmentos cabe entera su `fragmento_esperado` (tiene que ser al menos 1);
  - el solapamiento léxico entre la pregunta y la frase esperada. Si es alto, la pregunta copia
    las palabras del documento y el Hit Rate mide coincidencia de palabras, no recuperación.

    <prefijo de EXPERIMENTOS.md> python verificar_golden.py [golden_set.json]

Escribe `salidas/<etiqueta>/golden_verificacion.txt`.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
import io
import json
import sys

import experimentos
from evaluation import es_respondible
from rag_pipeline import RagPipeline, _normalizar_texto

CORPUS = Path("corpus")
CHUNK_TOKENS, OVERLAP_TOKENS = 512, 102
# Palabras vacías en inglés que no cuentan para el solapamiento léxico.
VACIAS = set("a an the of to in on for and or with by is are was were be it its as at from that "
             "this which what how does do did into their them they".split())


def palabras(texto: str) -> set[str]:
    return {p for p in _normalizar_texto(texto).split() if p not in VACIAS}


def verificar(golden: list[dict], pipeline: RagPipeline) -> list[str]:
    with redirect_stdout(io.StringIO()):
        chunks = pipeline.ingest(CORPUS, chunk_tokens=CHUNK_TOKENS, overlap_tokens=OVERLAP_TOKENS)
    documentos = {c.document for c in chunks}
    tipos = [g["tipo"] for g in golden]
    lineas = [f"{len(golden)} preguntas: " + ", ".join(f"{t}={tipos.count(t)}" for t in sorted(set(tipos)))
              + f" · respondibles={sum(map(es_respondible, golden))} · {len(chunks)} fragmentos", ""]
    problemas = 0
    for g in golden:
        if not es_respondible(g):
            lineas.append(f"[{g['id']:2}] {g['tipo']:<11} sin fuente: cuenta como negativa (solo abstención)")
            continue
        faltan = [d for d in g["documentos_fuente"] if d not in documentos]
        frase = (g.get("fragmento_esperado") or "").strip()
        if frase:
            n = sum(_normalizar_texto(frase) in _normalizar_texto(c.text)
                    for c in chunks if c.document in g["documentos_fuente"])
            estado_frase = f"frase en {n} fragmento(s)"
            solape = palabras(g["pregunta"]) & palabras(frase)
            estado_solape = f"solape pregunta/frase {len(solape)}/{len(palabras(frase))} {sorted(solape)}"
        else:
            n = None
            estado_frase = "sin frase: acierto a nivel de documento"
            estado_solape = ""
        mal = bool(faltan) or n == 0
        problemas += mal
        lineas.append(f"[{g['id']:2}] {g['tipo']:<11} {'PROBLEMA ' if mal else ''}"
                      f"{'faltan ' + str(faltan) + ' · ' if faltan else ''}{estado_frase}"
                      f"{' · ' + estado_solape if estado_solape else ''}")
    lineas += ["", "sin problemas" if not problemas else f"{problemas} pregunta(s) con problemas"]
    return lineas


if __name__ == "__main__":
    ruta_golden = Path(sys.argv[1] if len(sys.argv) > 1 else "golden_set.json")
    golden = json.loads(ruta_golden.read_text(encoding="utf-8"))
    salida = "\n".join(verificar(golden, RagPipeline())) + "\n"
    print(salida)
    (experimentos.carpeta() / "golden_verificacion.txt").write_text(salida, encoding="utf-8")
