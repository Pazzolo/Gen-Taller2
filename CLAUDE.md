# Gen-Taller2

This is Taller 2 (RAG with vector search and evaluation), built on the Lab-02 scaffold at the repo root. `GAMEPLAN.md` has the plan and the status of each step.

## Git workflow

- **One branch per gameplan step**, named as in the table in `GAMEPLAN.md` (`step-0-setup`, `step-1-corpus`, …).
- **Create each branch from `main`** after the previous step's pull request is merged. Never commit directly to `main`.
- **Commit in small logical pieces** within a branch, and open one pull request per step. Paolo reviews and merges them.
- **Update the status column in `GAMEPLAN.md`** when a step's pull request is opened or merged.

## Credentials

- **No API key goes into the repo, the report or any output file.** Keys live in `.env`, which `.gitignore` excludes.
- **Before each commit, check the staged files for keys.**

## Running

- Run everything from the repo root.
- Part 0 runs with `EMBEDDING_BACKEND=local QDRANT_URL=":memory:"`.
- Everything else uses bge-m3 on the H200 (needs the GlobalProtect VPN) and Qdrant in Docker.
