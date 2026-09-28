"""
Verificación del corpus antes de indexarlo (Parte 1.1 del taller).

Comprueba los requisitos del enunciado —al menos 5 documentos, al menos 50 páginas, capa de
texto— con la misma regla que usa `ingestion.py` para no indexar un PDF escaneado
(`MIN_CARACTERES_UTILES`), y además cuenta las páginas casi vacías de cada documento: un
PDF puede pasar el umbral del documento entero y tener páginas escaneadas en medio.

    python verificar_corpus.py            # escribe salidas/corpus_verificacion.txt
"""

from __future__ import annotations

from pathlib import Path
import logging
import sys

from pypdf import PdfReader

from ingestion import MIN_CARACTERES_UTILES, caracteres_utiles

MIN_DOCUMENTOS = 5
MIN_PAGINAS = 50
MIN_CARACTERES_PAGINA = 50   # por debajo, la página no aporta texto al índice


def verificar(corpus_dir: Path) -> list[str]:
    lineas = [f"{'páginas':>7}  {'caracteres':>10}  {'vacías':>6}  documento"]
    total_paginas, docs_ok = 0, 0
    for path in sorted(corpus_dir.glob("*.pdf")):
        paginas = [p.extract_text() or "" for p in PdfReader(str(path)).pages]
        utiles = [caracteres_utiles(t) for t in paginas]
        vacias = sum(u < MIN_CARACTERES_PAGINA for u in utiles)
        indexable = sum(utiles) >= MIN_CARACTERES_UTILES
        total_paginas += len(paginas)
        docs_ok += indexable
        marca = "" if indexable else "  ← SIN CAPA DE TEXTO: la ingesta no lo indexa"
        lineas.append(f"{len(paginas):7d}  {sum(utiles):10d}  {vacias:6d}  {path.name}{marca}")
    lineas.append("")
    lineas.append(f"documentos indexables: {docs_ok} (mínimo {MIN_DOCUMENTOS}) · "
                  f"páginas: {total_paginas} (mínimo {MIN_PAGINAS})")
    cumple = docs_ok >= MIN_DOCUMENTOS and total_paginas >= MIN_PAGINAS
    lineas.append("cumple los requisitos del enunciado" if cumple else "NO cumple los requisitos")
    return lineas


if __name__ == "__main__":
    logging.disable(logging.WARNING)   # pypdf avisa de fuentes y objetos; no es la verificación
    corpus = Path(sys.argv[1] if len(sys.argv) > 1 else "corpus")
    salida = "\n".join(verificar(corpus))
    print(salida)
    Path("salidas").mkdir(exist_ok=True)
    Path("salidas/corpus_verificacion.txt").write_text(salida + "\n", encoding="utf-8")
