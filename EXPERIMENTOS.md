# Experiment log

This log records every configuration run on the same corpus, so they can be compared. The
target configuration is bge-m3 on the H200. Until the VPN is available, the baseline runs
bge-m3 **through a local Ollama** and generates with `qwen3:1.7b`. Both use the same model,
through the same code path (`CodificadorH200` talks to any Ollama server).

## How it works

- **Tags:** each configuration gets a tag of the form `<embedding model>@<host>`, for example
  `bge-m3@ollama-local`, `bge-m3@h200` or `text-embedding-3-small@api-openai`. `experimentos.py`
  derives it from the environment variables.
- **Outputs:** raw outputs go to `salidas/<tag>/`, so runs never overwrite each other.
- **Log file:** every script that produces results appends a row to `experimentos.csv`.
- **This table:** `python experimentos.py` regenerates it from the CSV. Don't edit the table by
  hand.

## Configurations

| Tag | Embedding row | Where | Command prefix |
|---|---|---|---|
| `bge-m3@ollama-local` | `embed_local_multilingue` (BAAI/bge-m3, 1024 dimensions, 8192 tokens, verified 2026-08-27) | Ollama on the Mac, `bge-m3:latest` (567M parameters, F16) | `H200_EMBED_URL=http://localhost:11434 EMBEDDING_BACKEND=h200 OPENAI_API_KEY= GENERATION_MODEL=qwen3:1.7b QDRANT_COLLECTION=taller2_bge_m3_local` |
| `bge-m3@h200` | `embed_local_multilingue` | USFQ H200, needs GlobalProtect | `EMBEDDING_BACKEND=h200 QDRANT_COLLECTION=taller2_bge_m3_h200` |
| `text-embedding-3-small@api-openai` | `embed_api_economico` (1536 dimensions, 8192 tokens, $0.02 per million, verified 2026-08-27) | OpenAI API | `EMBEDDING_BACKEND=openai GENERATION_MODEL=gpt-4o-mini QDRANT_COLLECTION=taller2_openai_small` |

Each configuration has its own Qdrant collection, so indexing one never deletes another. `OPENAI_API_KEY=` (set empty) is part of the local prefix on purpose. `rag_pipeline.generate`
prefers OpenAI whenever a key is present, and the key in `.env` is currently rejected. Setting
it empty forces generation through Ollama.

## Runs

<!-- tabla:inicio -->
| fecha | etapa | etiqueta | fila_embeddings | modelo_embeddings | host_embeddings | generador | chunk_tokens | overlap_tokens | n_fragmentos | tokens_indexados | costo_usd | segundos | k | hit_rate | mrr | abstencion_correcta | abstencion_indebida | salida | nota |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-27T23:15:46 | 1.1-1.2 inspección | text-embedding-3-small@api-openai | embed_api_economico | text-embedding-3-small | api-openai |  | 512 | 102 | 930 | 473866 |  |  |  |  |  |  |  | salidas/text-embedding-3-small@api-openai/parte1_ingesta.txt | solo tokeniza, sin llamar al modelo |
| 2026-09-27T23:16:26 | 1.1-1.2 inspección | bge-m3@ollama-local | embed_local_multilingue | BAAI/bge-m3 | ollama-local |  | 512 | 102 | 948 | 479701 |  |  |  |  |  |  |  | salidas/bge-m3@ollama-local/parte1_ingesta.txt | solo tokeniza, sin llamar al modelo |
| 2026-09-27T23:19:56 | 1.3-1.5 índice + top-5 | bge-m3@ollama-local | embed_local_multilingue | BAAI/bge-m3 | ollama-local | ollama:qwen3:1.7b | 512 | 102 | 948 | 479701 | 0.0000 | 136.6 |  |  |  |  |  | salidas/bge-m3@ollama-local/parte1_top5.txt |  |
| 2026-09-27T23:38:11 | 2.b evaluación | bge-m3@ollama-local | embed_local_multilingue | BAAI/bge-m3 | ollama-local | ollama:qwen3:1.7b |  |  |  |  |  |  | 3 | 0.875 | 0.750 | 1.000 | 0.250 | salidas/bge-m3@ollama-local/resultados_k3.csv | golden_set.json · 8 respondibles, 2 negativas |
| 2026-09-27T23:39:56 | 2.b evaluación | bge-m3@ollama-local | embed_local_multilingue | BAAI/bge-m3 | ollama-local | ollama:qwen3:1.7b |  |  |  |  |  |  | 5 | 1.000 | 0.781 | 1.000 | 0.125 | salidas/bge-m3@ollama-local/resultados_k5.csv | golden_set.json · 8 respondibles, 2 negativas |
| 2026-09-27T23:46:57 | 3.B bm25 | bge-m3@ollama-local | embed_local_multilingue | BAAI/bge-m3 | ollama-local |  | 512 | 102 |  |  |  |  | 3 | 0.875 | 0.750 | sin medir | sin medir | salidas/bge-m3@ollama-local/bm25/resultados_k3.csv | BM25 solo, diagnóstico · sin generar |
| 2026-09-27T23:46:57 | 3.B bm25 | bge-m3@ollama-local | embed_local_multilingue | BAAI/bge-m3 | ollama-local |  | 512 | 102 |  |  |  |  | 5 | 0.875 | 0.750 | sin medir | sin medir | salidas/bge-m3@ollama-local/bm25/resultados_k5.csv | BM25 solo, diagnóstico · sin generar |
| 2026-09-27T23:48:23 | 3.B hibrido | bge-m3@ollama-local | embed_local_multilingue | BAAI/bge-m3 | ollama-local | ollama:qwen3:1.7b | 512 | 102 |  |  |  |  | 3 | 1.000 | 0.854 | 1.000 | 0.125 | salidas/bge-m3@ollama-local/hibrido/resultados_k3.csv | 20 candidatos por lado, RRF k=60 |
| 2026-09-27T23:49:37 | 3.B hibrido | bge-m3@ollama-local | embed_local_multilingue | BAAI/bge-m3 | ollama-local | ollama:qwen3:1.7b | 512 | 102 |  |  |  |  | 5 | 1.000 | 0.854 | 1.000 | 0.000 | salidas/bge-m3@ollama-local/hibrido/resultados_k5.csv | 20 candidatos por lado, RRF k=60 |
| 2026-09-28T10:29:26 | 1.3-1.5 índice + top-5 | bge-m3@h200 | embed_local_multilingue | BAAI/bge-m3 | h200 | ollama:qwen3:1.7b | 512 | 102 | 948 | 479701 | 0.0000 | 25.6 |  |  |  |  |  | salidas/bge-m3@h200/parte1_top5.txt |  |
| 2026-09-28T10:30:46 | 2.b evaluación | bge-m3@h200 | embed_local_multilingue | BAAI/bge-m3 | h200 | ollama:qwen3:1.7b |  |  |  |  |  |  | 3 | 0.875 | 0.750 | 1.000 | 0.250 | salidas/bge-m3@h200/resultados_k3.csv | golden_set.json · 8 respondibles, 2 negativas |
| 2026-09-28T10:32:29 | 2.b evaluación | bge-m3@h200 | embed_local_multilingue | BAAI/bge-m3 | h200 | ollama:qwen3:1.7b |  |  |  |  |  |  | 5 | 1.000 | 0.781 | 1.000 | 0.125 | salidas/bge-m3@h200/resultados_k5.csv | golden_set.json · 8 respondibles, 2 negativas |
| 2026-09-28T10:33:54 | 3.B hibrido | bge-m3@h200 | embed_local_multilingue | BAAI/bge-m3 | h200 | ollama:qwen3:1.7b | 512 | 102 |  |  |  |  | 3 | 1.000 | 0.854 | 1.000 | 0.125 | salidas/bge-m3@h200/hibrido/resultados_k3.csv | 20 candidatos por lado, RRF k=60 |
| 2026-09-28T10:35:16 | 3.B hibrido | bge-m3@h200 | embed_local_multilingue | BAAI/bge-m3 | h200 | ollama:qwen3:1.7b | 512 | 102 |  |  |  |  | 5 | 1.000 | 0.854 | 1.000 | 0.000 | salidas/bge-m3@h200/hibrido/resultados_k5.csv | 20 candidatos por lado, RRF k=60 |
| 2026-09-28T10:41:32 | 2.b evaluación | bge-m3@h200 | embed_local_multilingue | BAAI/bge-m3 | h200 | ollama:granite3.3:latest |  |  |  |  |  |  | 3 | 0.875 | 0.750 | 1.000 | 0.000 | salidas/bge-m3@h200/granite3.3/resultados_k3.csv | golden_set.json · 8 respondibles, 2 negativas |
| 2026-09-28T10:41:55 | 2.b evaluación | bge-m3@h200 | embed_local_multilingue | BAAI/bge-m3 | h200 | ollama:granite3.3:latest |  |  |  |  |  |  | 5 | 1.000 | 0.781 | 1.000 | 0.000 | salidas/bge-m3@h200/granite3.3/resultados_k5.csv | golden_set.json · 8 respondibles, 2 negativas |
<!-- tabla:fin -->

## Journal

- **2026-09-27: OpenAI key rejected.** The key in `.env` returns `401 invalid_api_key`, so
  nothing was indexed and nothing was charged. The OpenAI tokenizer-only inspection is kept
  under `salidas/text-embedding-3-small@api-openai/`.
- **2026-09-27: switched to bge-m3 on a local Ollama**, because there's no VPN and no working
  key. Once the VPN is available, re-run the same scripts with the `bge-m3@h200` prefix and
  compare the rows. Retrieval (Hit Rate, MRR) should match if the two servers run the same
  weights. Any gap would point to the quantisation or version of the served model.
- **2026-09-28: ran the `bge-m3@h200` configuration** over the VPN, with generation kept on
  the local `qwen3:1.7b` so embeddings are the only change. The two servers serve the same
  weights (same digest), the vectors agree to a cosine of at least 0.9997, and Hit Rate, MRR and
  both abstention rates match exactly, dense and hybrid, at k=3 and k=5. The only difference is
  the fifth slot of question 8's top-5: `ouyang-2022-instructgpt-0102` locally (0.5357) and
  `ouyang-2022-instructgpt-0084` on the H200 (0.5335). The first four match. Details in
  `salidas/comparacion_configuraciones.md`.
- **2026-09-28: swapped the generator to `granite3.3` on the H200**, keeping the `bge-m3@h200`
  index (dense, k=3 and k=5; output in `salidas/bge-m3@h200/granite3.3/`, so the `qwen3:1.7b`
  results stay intact). Hit Rate and MRR are unchanged, as expected, since retrieval does not
  depend on the generator. Wrong abstentions drop to 0.000 at both k (from 0.250 and 0.125 with
  `qwen3:1.7b`), and both negatives are still refused. Question 6 is a real fix (answered from
  `rafailov-2023-dpo-0003`), but question 5 at k=3 is not: no Lewis et al. chunk is retrieved, so
  `granite3.3` describes RAG from its own knowledge. On negative 9 it abstains but adds a false
  claim ("GPT-4 … Microsoft"), which `se_abstuvo` still counts as a correct abstention. See the
  counter-test in PARTE2.md, 2.c. `granite3.3` was the only generator that loaded on the H200 that day; the
  other large models crashed the Ollama runner. Hybrid was not re-run because `hibrido.py` has
  no output-path option and would overwrite the `qwen3:1.7b` hybrid results.
