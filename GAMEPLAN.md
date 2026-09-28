# Taller 2 gameplan: RAG baseline, evaluation and extension

The rule for the whole taller is **baseline → measure → extend → measure**.

## Workflow: one branch and one pull request per step

Each step gets its own branch, created from `main` after the previous step's pull request is merged. Paolo reviews and merges every pull request.

| Branch | Contents | Status |
|---|---|---|
| `step-0-setup` | `.gitignore`, scaffold, gameplan | Merged (PR #1). The pinned `requirements.txt` moves to Step 3, when everything is installed |
| `step-1-corpus` | `corpus/` plus the page and text-layer check | Merged (PR #3) |
| `step-2-part0` | Part 0: the three silent failures, scripts and raw output | Merged (PR #2) |
| `step-3-baseline` | Part 1: the baseline RAG | In progress |
| `step-4-evaluation` | Part 2: golden set, metrics, `resultados.csv`, worst cases | Not started |
| `step-5-extension` | Part 3: the extension and its measurement | Not started |
| `step-6-report` | Part 4: reflection and the PDF report | Not started |

## Step 0: Set up (≈30 min)

1. ✅ **The scaffold is in the repo root.** Run every command from the root: `corpus/`, `.env`, `golden_set.json` and `resultados.csv` all live there.
2. ✅ **`.gitignore` is in place.** It covers `.env`, `__pycache__/`, `qdrant_storage/`, `*.zip` and `data/`, because a committed API key is a blocking finding.
3. **Install dependencies.** Create a venv and run `pip install -r requirements.txt`. You'll also need `transformers` for the bge-m3 tokenizer. At the end, run `pip freeze > requirements.txt` so the file has exact versions, which is graded.
4. **Configure keys.** `cp .env.example .env`, then add the OpenAI key.
5. **Start Qdrant:**
   ```bash
   docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
   ```

### Two setup problems in the scaffold

- **Model names are missing.** `rag_pipeline.py` looks for the model table at `REPO.parents[3]/fuentes/modelos/modelos-2026-1.json`, which doesn't exist outside the course repo.
  - Embeddings still work, because they have hardcoded fallbacks.
  - Generation doesn't: `fila_de_la_tabla(...).get("model")` returns `None`, so the OpenAI and Ollama calls get `model=None` and fail.
  - **Fix:** set `GENERATION_MODEL=` in `.env` to the model name of the `propietario_economico` row in the course table, or copy the course's `fuentes/` folder in.
  - For the report, take the `verified_at` dates from the annex: 2026-08-27 for embeddings, 2026-09-19 for rerankers.
- **Results get overwritten.** `evaluation.py` rewrites `resultados.csv` on every run.
  - **Fix:** use `--csv resultados_k3.csv` and `--csv resultados_k5.csv`, then concatenate them into `resultados.csv`, which is the deliverable.

## Step 1: Pick the corpus (decide today)

| | Option A (USFQ, Spanish) | Option B (course papers, English) |
|---|---|---|
| Hitting 5 docs and ≥50 pages | Risky, since the reglamento may not be available | Easy |
| Text layer | Likely to include a scan, which is still gradable (0.a) | Almost always present |
| Material for 2.c | Tables and headers | Two columns, equations, references |
| Reranker for Option C | `rerank_local_multilingue`, no published figures | `rerank_local_ingles`, with published figures |

**✅ Decided: Option B.** The corpus is the 16 week 1 and week 2 papers from `Papers/`, copied flat into `corpus/` (the ingestion doesn't read subfolders). It's English only, 440 pages, and every page has a text layer. The evidence is `verificar_corpus.py`, whose output is saved in `salidas/corpus_verificacion.txt`. Declare the language as English in the report.

Original recommendation: **B**, unless you already have the USFQ PDFs and they total 50 pages or more. It's lower risk, and the references sections and two-column layouts will give you real failures to analyse.

- If you go with B, write the golden-set questions in English. `fragmento_esperado` has to be literal English text, and Spanish questions would add a cross-lingual effect that muddies 2.c.
- Once you've chosen, check each PDF's pages and useful characters with `pypdf` before you do anything else.

## Step 2: Part 0, the three silent failures (10%, ≈1 h, no VPN or key)

Run everything with:

```bash
EMBEDDING_BACKEND=local QDRANT_URL=":memory:" CORPUS_DIR=ejemplos
```

Save all raw output to `salidas/parte0_*.txt` for the report.

### 0.a: The PDF that gets indexed empty

1. Run `python rag_pipeline.py "..."`.
2. Paste the raw ingestion output, including the line `AVISO ... 0 caracteres útiles ... NO se indexa`.
3. Write two sentences:
   - The `parece_escaneado` check blocks any document with fewer than `MIN_CARACTERES_UTILES=200` alphanumeric characters. Page markers aren't counted.
   - Without that check, the file would become a single chunk made of `[page=1] [page=2]`. The `MIN_CARACTERES_FRAGMENTO` check might also catch it, and saying so earns points.

### 0.b: The chunk that gets cut at 128

The CLI doesn't accept `chunk_tokens`, so write a small script, `parte0b.py`. It should:

1. Call `pipeline.ingest(Path("ejemplos"), chunk_tokens=900)` and capture the warning.
2. Take a text of about 900 words and count its tokens with `encoder.tokenizer`.
3. Print `max_seq_length`, which should be 128.
4. Compute the cosine between the full text's vector and the vector of the text cut to 128 tokens. It should be about 1.0.
5. End with the `chunk_tokens` you'll use in Part 1 and a one-sentence justification against bge-m3's 8192-token limit.

### 0.c: The index that doesn't complain

1. Ask "¿cuál es la política de mascotas?" and keep the top 3 with their scores.
2. Ask "¿qué exige el instructivo de prácticas de campo sobre el seguro de accidentes?" and keep the top 3 with their scores.
3. Write the sentence: Hit Rate only measures whether the source was retrieved. For those two questions there's no retrievable source, so there's no hit to miss, only healthy-looking scores. That's why the golden set needs negative questions and abstention needs its own metric.

## Step 3: Part 1, the baseline (35%, ≈3–4 h)

1. **Ingestion.**
   - Put the PDFs in `corpus/`, run the ingestion and save its output.
   - Check 2–3 pages per document by hand against the PDF.
   - Write down what degraded: tables, running headers and footers, hyphenation, columns, references.
   - List any documents the ingestion rejected.
2. **Chunking.**
   - Start from `chunk_tokens=512` and `overlap_tokens=100`.
   - Justify the size by your questions, not just by the fact that it fits: roughly one section or paragraph block per chunk, enough for a simple answer, and still small enough that the top-1 match stays specific.
   - Look at the token-length distribution of paragraphs in your corpus and cite it.
3. **Indexing.**
   - Connect GlobalProtect and use `EMBEDDING_BACKEND=h200` with Qdrant in the container.
   - Declare the row as `embed_local_multilingue`: BAAI/bge-m3, 1024 dimensions, 8192 tokens, verified 2026-08-27.
   - If you have no VPN, use `openai` and declare `embed_api_economico` instead.
   - Don't mix the two. If you switch, reindex.
4. **Retrieval.** Write a `demo_top5.py` that prints the top 5 (document, score, snippet) for 3 test questions. Save the output.
5. **Generation.**
   - Leave the prompt and `ABSTENCION` untouched.
   - Run one end-to-end question and check that the generator column shows `openai:<model>`.

## Step 4: Part 2, evaluation (30%)

### 2.a: The golden set (≈2 h, and it's where the thinking goes)

- **Mix:** 4 simple questions, 3 multi-chunk, 2 negatives, 1 adversarial.
- **Paraphrase the questions.** Don't copy the document's wording, or Hit Rate is 1.0 and tells you nothing.
- **Keep `fragmento_esperado` short and distinctive**, 5–10 words.
  - Check that each phrase sits inside a single chunk. A phrase split across a chunk boundary gives a false negative, which is worth mentioning in 2.c if it happens.
- **Make the negatives plausible and on-topic**, close to the corpus but unanswerable. Those are the ones that actually test abstention.
- **Adversarial question:** decide and say which way you went.
  - Suggested: make it answerable (give it a source document) and judge whether the answer ignores the injected instruction.

### 2.b: Metrics (≈30 min, cents in API cost)

```bash
python evaluation.py --k 3 --csv resultados_k3.csv
python evaluation.py --k 5 --csv resultados_k5.csv --sin-indexar
```

- Concatenate both files into `resultados.csv`.
- Build the table with a small `tabla_metricas.py` that reads the CSV. The table has Hit Rate@3, Hit Rate@5, MRR, correct abstention and wrongful abstention.
- Deriving the table from the CSV means it can be rebuilt from the CSV, which is graded.

### 2.c: The three worst cases

1. **Rank the failures** by first-hit position, then wrongful abstention, then any negative the system answered anyway.
2. **Gather evidence for each case:**
   - the ingestion output for that document;
   - the top-k chunks with their scores;
   - the generated answer. The CSV doesn't store answers (and the in-memory field is truncated to 500 characters), so add a `respuesta` column or save a JSON.
3. **Assign each case to a pipeline stage:** ingestion, chunking, retrieval, prompt or generation.
4. **Check for the Part 0 failures.** For each of the three, either show it in your corpus or show evidence that it's absent.

## Step 5: Part 3, one extension (10%)

Choose the extension from what you measured, and write your hypothesis down **before** you implement it.

| Symptom in 2.b | Extension |
|---|---|
| High Hit Rate@5, low MRR (right doc, wrong position) | **C, reranking** (top-20, cross-encoder, top-5) |
| Misses on exact terms, acronyms or names | **B, hybrid** (BM25 + RRF) |
| The phrase is split across chunks, or multi-chunk questions fail | **A, a second chunking strategy** |
| Retrieval is fine but the answers are questionable | **D, RAGAS** (run 1 question first, count the calls, multiply) |

- Measure the extension with the same golden set and CSV, adding a `variante` column.
- If it doesn't help, explain why. That's a valid result.

## Step 6: Part 4 reflection and report (≈2 h)

### Reflection

- **Q1:** use your own numbers.
  - Compute the old metric: hits divided by all 10 questions, with the negatives counted as misses.
  - Put it next to your Hit Rate over answerable questions only, and explain what each one measures.
- **Q2:** write a global question such as "What are the recurring themes across all the papers?" or "How does X evolve across the corpus?".
  - Explain why top-k retrieval can't answer it: it only sees a few fragments.
  - Explain how GraphRAG would help: an entity graph, community summaries, then map-reduce over those summaries.

### The PDF report must include

- the raw Part 0 outputs;
- an architecture diagram;
- the chunking parameters in tokens, with their justification;
- the embedding row with its id and date;
- the metrics table;
- the three worst cases with their evidence;
- the extension and its measurement;
- the reflection answers.

### Final checklist

- [ ] `git grep -i "sk-"` comes back empty.
- [ ] `.env` isn't tracked.
- [ ] `requirements.txt` has pinned versions.
- [ ] `golden_set.json` is committed.
- [ ] `resultados.csv` is committed.

## Suggested order and time

| When | What |
|---|---|
| Day 1 | Setup, corpus choice, Part 0 |
| Day 2 | Part 1 (needs the VPN) and the golden set |
| Day 3 | 2.b, 2.c, Part 3 |
| Day 4 | Part 4, report, reproducibility check |

The only steps that need the VPN are indexing and anything that re-embeds. Batch them together.
