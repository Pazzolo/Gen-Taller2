# Taller 2 — RAG con Vector Search y Evaluación

**Autor:** Paolo A · **Curso:** IA Generativa, USFQ · **Fecha:** 2026-09-27

**Repositorio:** `github.com/Pazzolo/Gen-Taller2`

La regla del taller fue baseline → medir → extender → medir. Este informe junta las cinco
partes. Todas las cifras salen de corridas reales, y cada tabla se puede reconstruir desde los
CSV crudos del repositorio.

## Resumen

| Elemento | Detalle |
|---|---|
| Corpus | Opción B: 16 papers de las semanas 1 y 2, **inglés**, 440 páginas, todos con capa de texto |
| Fragmentación | tamaño fijo, **512 tokens con 102 de solapamiento**, tokens de `BAAI/bge-m3` → 948 fragmentos |
| Embeddings | fila **`embed_local_multilingue`** (BAAI/bge-m3, 1024 dim., 8192 tokens), **verificada 2026-08-27**, servida por **Ollama local** (no la H200: no había VPN) |
| Índice | Qdrant 1.19.1 en contenedor, colección `taller2_bge_m3_local`, coseno |
| Generación | fila **`open_weight_pequeno`** (`qwen3:1.7b`) por Ollama local, temperatura 0 |
| Golden set | 10 preguntas: 4 simples, 3 multi-fragmento, 2 negativas, 1 adversarial (respondible) |
| Baseline | Hit Rate@3 **0,875**, @5 **1,000** · MRR **0,750** / **0,781** · abstención correcta **1,000** · indebida **0,250** / **0,125** |
| Extensión | Opción B, **híbrido BM25 + denso (RRF)**: Hit Rate@3 **1,000** · MRR **0,854** · indebida **0,125** / **0,000** |
| Costo | **USD 0**: ninguna llamada a API de pago |

## Arquitectura

```
 corpus/*.pdf (16)
      │  pypdf, página por página, marcador [page=n]              ingestion.py
      │  descarta documentos con < 200 caracteres útiles (0.a)
      ▼
 texto ──► fragmentos de 512 tokens, solapamiento 102               ingestion.py
      │    (tokenizador BAAI/bge-m3, el mismo que vectoriza)
      ▼
 bge-m3 en Ollama (truncate: false) ──► vectores de 1024          rag_pipeline.py
      │
      ▼
 Qdrant (contenedor) · coseno · 948 puntos                         rag_pipeline.py
      │
 pregunta ──► vector ──► top-k denso ─────────────┐                rag_pipeline.py
      │                                            ├─ RRF (k=60) ──► top-k   hibrido.py (Parte 3)
      └──────► BM25 sobre los mismos 948 ─────────┘
      ▼
 prompt: solo el contexto + frase fija de abstención                rag_pipeline.py
      ▼
 qwen3:1.7b (Ollama) ──► respuesta ──► se_abstuvo()                 rag_pipeline.py
      ▼
 evaluation.py ──► resultados_k{3,5}.csv ──► tabla_metricas.py ──► resultados.csv
```

## Reproducibilidad

- **Versiones:** `requirements.txt` con las versiones exactas del entorno (Python 3.14.7).
- **Resultados crudos:** `resultados.csv` tiene las 20 filas del baseline (10 preguntas × k = 3
  y 5). `tabla_metricas.py` reconstruye la tabla de la Parte 2.b desde ese archivo y comprueba
  que coincide con lo que reportó el evaluador.
- **Registro de corridas:** cada corrida escribe en `salidas/<modelo>@<host>/` y añade una fila
  a `experimentos.csv`. `EXPERIMENTOS.md` tiene el prefijo exacto de cada configuración y la
  tabla de todas las corridas, preparada para repetir el baseline en la H200 y comparar.
- **Credenciales:** ninguna clave en el repositorio. `.env` está en `.gitignore` y no está
  versionado, y una búsqueda del patrón de claves de OpenAI (`sk-…`) en los archivos
  versionados no encuentra nada.

Prefijo de todas las corridas del baseline:

```bash
export H200_EMBED_URL=http://localhost:11434 EMBEDDING_BACKEND=h200 OPENAI_API_KEY= \
       GENERATION_MODEL=qwen3:1.7b QDRANT_COLLECTION=taller2_bge_m3_local
```

## Presupuesto

No se gastó nada. Los embeddings (bge-m3) y la generación (`qwen3:1.7b`) corrieron en la Mac,
por Ollama. Se intentó la ruta de OpenAI (`embed_api_economico` para embeddings y
`propietario_economico`, `gpt-4o-mini`, para generar), pero la clave fue rechazada con un 401
antes de cualquier llamada facturable. Con esa ruta, el costo estimado de indexar habría sido
de unos USD 0,01: 473 866 tokens a USD 0,02 por millón, precio de la fila verificado el
2026-08-27. RAGAS (Opción D) no se corrió, así que no hay estimación de llamadas de juez.

Tiempos en la Mac: indexar 948 fragmentos, 136,6 s; cada evaluación completa (10 preguntas,
con generación), alrededor de 1,5 min por valor de k.
