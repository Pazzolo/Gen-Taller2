"""
Parte 1.1 y 1.2 del taller: qué se pierde en la ingesta y cómo se reparte el corpus en tokens.

Usa el tokenizador de `BAAI/bge-m3` (se descarga solo el tokenizador, no hace falta la VPN),
que es el mismo con el que `rag_pipeline.py` fragmenta contra la H200, y la misma
`load_corpus` del andamiaje, así que los fragmentos son los que se van a indexar.

    python inspeccion_ingesta.py [chunk_tokens] [overlap_tokens]    # por defecto 512 102

Escribe `salidas/parte1_ingesta.txt`.
"""

from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
import io
import logging
import re
import statistics
import sys

from transformers import AutoTokenizer

from ingestion import load_corpus, read_document

CORPUS = Path("corpus")
TOPE_BGE_M3 = 8192   # fila embed_local_multilingue, verificada 2026-08-27
# Encabezado de la bibliografía: una línea que es solo «References» o «Bibliography».
REFERENCIAS = re.compile(r"^\s*(\d+\s*)?(references|bibliography)\s*$", re.I | re.M)
GUION_PARTIDO = re.compile(r"[a-z]-\s+[a-z]")   # «col-\nlection»: palabra cortada al final de línea
LIGADURAS = re.compile("[\ufb00-\ufb06]")          # «ﬂexibility»: ﬁ ﬂ ﬀ ﬃ ﬄ como un solo carácter


def lineas_repetidas(paginas: list[str], minimo: float = 0.3) -> list[str]:
    """Líneas que aparecen en al menos `minimo` de las páginas: encabezados y pies de página."""
    cuenta = Counter()
    for texto in paginas:
        cuenta.update({l.strip() for l in texto.splitlines() if len(l.strip()) > 3})
    umbral = max(3, minimo * len(paginas))
    return [l for l, n in cuenta.most_common() if n >= umbral]


def main(chunk_tokens: int, overlap_tokens: int) -> str:
    logging.disable(logging.WARNING)
    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    contar = lambda t: len(tok(t, add_special_tokens=False, truncation=False)["input_ids"])

    silencio = io.StringIO()
    with redirect_stdout(silencio):
        chunks = load_corpus(CORPUS, tokenizer=tok, max_seq_tokens=TOPE_BGE_M3,
                             chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
    lineas = [f"tokenizador: BAAI/bge-m3 · chunk_tokens={chunk_tokens} · "
              f"overlap_tokens={overlap_tokens} · tope {TOPE_BGE_M3}", "",
              "salida de load_corpus:", *("  " + l for l in silencio.getvalue().splitlines()), ""]

    lineas.append(f"{'doc':<48} {'tokens':>7} {'tok/pág':>7} {'tok/pal':>7} "
                  f"{'%refs+ap':>8} {'guiones':>7} {'ligad':>5}  encabezados/pies repetidos")
    por_pagina, ratios, refs_total, tokens_total = [], [], 0, 0
    for path in sorted(CORPUS.glob("*.pdf")):
        doc = read_document(path)
        paginas = re.split(r"\[page=\d+\]", doc.text)[1:]
        n_tokens = contar(doc.text)
        n_palabras = len(doc.text.split())
        m = list(REFERENCIAS.finditer(doc.text))
        refs = contar(doc.text[m[-1].start():]) if m else 0   # desde el último encabezado
        repetidas = lineas_repetidas(paginas)
        por_pagina += [contar(p) for p in paginas]
        ratios.append(n_tokens / n_palabras)
        refs_total += refs
        tokens_total += n_tokens
        muestra = "; ".join(repr(l[:40]) for l in repetidas[:2]) or "—"
        lineas.append(f"{path.name[:48]:<48} {n_tokens:7d} {n_tokens / doc.pages:7.0f} "
                      f"{n_tokens / n_palabras:7.2f} {refs / n_tokens:8.1%} "
                      f"{len(GUION_PARTIDO.findall(doc.text)):7d} {len(LIGADURAS.findall(doc.text)):5d}  {muestra}")

    largos = [contar(c.text) for c in chunks]
    q = statistics.quantiles(por_pagina, n=4)
    lineas += ["",
               f"total: {tokens_total} tokens en {len(por_pagina)} páginas · "
               f"{len(chunks)} fragmentos",
               f"tokens por página: mediana {statistics.median(por_pagina):.0f} "
               f"(p25 {q[0]:.0f}, p75 {q[2]:.0f})",
               f"tokens por palabra: mediana {statistics.median(ratios):.2f} "
               f"(min {min(ratios):.2f}, max {max(ratios):.2f})",
               f"un fragmento de {chunk_tokens} tokens ≈ {chunk_tokens / statistics.median(por_pagina):.2f} "
               f"páginas ≈ {chunk_tokens / statistics.median(ratios):.0f} palabras",
               f"fragmentos: {min(largos)}–{max(largos)} tokens re-tokenizados "
               f"(mediana {statistics.median(largos):.0f})",
               f"bibliografía + apéndices: {refs_total / tokens_total:.1%} de los tokens del corpus "
               f"(desde el último encabezado References/Bibliography hasta el final; incluye los apéndices)"]
    return "\n".join(lineas) + "\n"


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:3]]
    salida = main(*(args or [512, 102]))
    print(salida)
    Path("salidas").mkdir(exist_ok=True)
    Path("salidas/parte1_ingesta.txt").write_text(salida, encoding="utf-8")
