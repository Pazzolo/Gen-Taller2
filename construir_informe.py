"""
Arma el informe del taller: informe/portada.md + PARTE0..4.md → informe/INFORME.md → HTML → PDF.

El PDF se imprime con Chrome sin interfaz, así que no hace falta LaTeX ni pandoc. Las partes
no se copian a mano: se leen de los mismos archivos del repositorio, y cualquier corrección en
una PARTE*.md llega al PDF volviendo a correr este script.

    python construir_informe.py        # escribe informe/INFORME.md, informe.html e informe.pdf
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

import markdown

DIR = Path("informe")
PARTES = ["PARTE0.md", "PARTE1.md", "PARTE2.md", "PARTE3.md", "PARTE4.md"]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 10pt;
       line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 17pt; border-bottom: 2px solid #333; padding-bottom: 4px; margin-top: 0; }
h1.parte { page-break-before: always; }
h2 { font-size: 13pt; margin-top: 18px; border-bottom: 1px solid #ccc; }
h3 { font-size: 11pt; margin-top: 14px; }
h2, h3 { page-break-after: avoid; break-after: avoid; }
code { font-family: Menlo, monospace; font-size: 8.5pt; background: #f3f3f3; padding: 0 2px; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 6px 8px; font-size: 7.6pt;
      line-height: 1.3; white-space: pre-wrap; overflow-wrap: anywhere; }
pre code { background: none; padding: 0; font-size: inherit; }
table { border-collapse: collapse; margin: 8px 0; font-size: 8.8pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3px 6px; text-align: left; vertical-align: top; }
th { background: #eee; }
"""


def main() -> None:
    partes = [(DIR / "portada.md").read_text(encoding="utf-8")]
    partes += [Path(p).read_text(encoding="utf-8") for p in PARTES]
    texto = "\n\n".join(p.strip() for p in partes) + "\n"
    (DIR / "INFORME.md").write_text(texto, encoding="utf-8")

    cuerpo = markdown.markdown(texto, extensions=["tables", "fenced_code"])
    # Cada parte empieza en página nueva, menos la portada.
    cuerpo = re.sub(r"<h1>(Parte )", r'<h1 class="parte">\1', cuerpo)
    html = (f'<!doctype html><html lang="es"><head><meta charset="utf-8">'
            f"<title>Taller 2 — RAG</title><style>{CSS}</style></head><body>{cuerpo}</body></html>")
    ruta_html = DIR / "informe.html"
    ruta_html.write_text(html, encoding="utf-8")

    ruta_pdf = (DIR / "informe.pdf").resolve()
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={ruta_pdf}", ruta_html.resolve().as_uri()],
                   check=True, capture_output=True)
    print(f"{DIR / 'INFORME.md'} · {ruta_html} · {ruta_pdf} ({ruta_pdf.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
