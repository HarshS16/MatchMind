# Intelligent Job Matching & Candidate Ranking System

An AI-powered semantic recruitment platform that retrieves relevant candidates for a job description using dense vector search, improves ranking through Cross-Encoder and LLM-based re-ranking, and evaluates retrieval quality using standardized Information Retrieval metrics.

> **Core research question:** Does applying Cross-Encoder or LLM-based re-ranking to vector-retrieved resumes significantly improve candidate ranking quality compared with direct cosine similarity ranking?
>
> **Primary metric:** NDCG@10 · **Secondary:** Precision@5, Recall@5, MRR

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Component Deep-Dive](#3-component-deep-dive)
4. [End-to-End Processes](#4-end-to-end-processes)
5. [Technology Decisions & Trade-offs](#5-technology-decisions--trade-offs)
6. [Evaluation Framework](#6-evaluation-framework)
7. [API Reference](#7-api-reference)
8. [Setup & Running](#8-setup--running)
9. [Project Structure](#9-project-structure)
10. [Current Status & Known Limitations](#10-current-status--known-limitations)
11. [Roadmap](#11-roadmap)

---

## 1. Project Overview

### What it does

A recruiter creates a job description. The system:

1. Embeds the JD with a transformer model.
2. Retrieves the top-K most similar resumes from a vector database (semantic search — not keyword matching).
3. Optionally re-ranks those candidates with a Cross-Encoder or an LLM that produces structured, explainable assessments.
4. Stores every run so ranking approaches can be compared fairly on the **same retrieved candidate pool**.
5. Computes IR metrics (P@K, R@K, NDCG@K, MRR) against human relevance labels to measure whether re-ranking actually helps.

### Why it exists

This is a research-driven portfolio project in the AI/ML + Information Retrieval space. Keyword-based ATS tools miss semantically equivalent phrasing ("ML engineer" vs "machine learning practitioner"). Dense embeddings solve retrieval; neural/LLM re-rankers solve precision. The interesting engineering question is whether the extra ranking stage earns its latency cost — which is exactly what the evaluation framework measures.

### Scope

**In scope:** resume ingestion (PDF/DOCX), preprocessing, embeddings, vector retrieval, two independent re-rankers, evaluation metrics, benchmark experiments, local dashboard, Docker deployment.

**Out of scope:** job applications, LinkedIn/Naukri integration, interview scheduling, email automation, multi-tenant ATS features.

---

## 2. System Architecture

### High-level flow

```
React Dashboard ──► FastAPI Backend
                        │
        ┌───────────────┼───────────────────────┐
        ▼               ▼                       ▼
  Resume Ingestion   Job Descriptions      Evaluation Engine
        │               │                       ▲
        ▼               ▼                       │
  Document Parser → Preprocessing → Embeddings   │
        │               │            │          │
        ▼               │            ▼          │
  Local File Storage    │      Qdrant (vectors)  │
        │               │            │          │
        ▼               ▼            ▼          │
                  PostgreSQL (metadata, experiments, labels)
```

### Retrieval + ranking pipeline (two stages)

```
Job Description
      │
      ▼
Embedding (BGE-base-en-v1.5, 768-dim, cosine)
      │
      ▼
Qdrant top-K retrieval  ──────────────►  Top-20 candidate pool
      │                                     │
      │              ┌──────────────────────┼──────────────────────┐
      │              ▼                      ▼                      ▼
      │      Vector ranking          Cross-Encoder           LLM ranker
      │      (cosine order)      (pairwise relevance)   (structured JSON
      │              │            scores per pair)      assessment 0–100)
      │              └──────────────────────┼──────────────────────┘
      ▼                                     ▼
              MatchRun + MatchResult rows (PostgreSQL)
                                     │
                                     ▼
                    Evaluation: P@K / R@K / NDCG@K / MRR vs labels
```

**Design rule:** all ranking methods run against the *same* retrieved pool. This isolates the effect of re-ranking — a reranker can only reorder what retrieval found, it cannot rescue candidates retrieval missed.

### Runtime topology (Docker Compose)

| Container | Role |
|---|---|
| `elip-frontend` | React SPA served by nginx; proxies `/api/` → backend |
| `elip-backend` | FastAPI + all ML services (CPU) |
| `elip-postgres` | Metadata, match runs, labels, experiments |
| `elip-qdrant` | Vector store |
| volumes | `postgres_data`, `qdrant_data`, `models` (HF cache), `./data` (resumes) |

Ollama runs **outside** Docker on the host (`host.docker.internal:11434`) so GPU/CPU LLM inference stays on the host.

---

## 3. Component Deep-Dive

### 3.1 Resume ingestion (`app/services/ingestion/`)

- Accepts PDF (PyMuPDF) and DOCX (python-docx); rejects other types and files > `MAX_UPLOAD_SIZE_MB`.
- Pipeline per file: validate → extract text → clean → **pseudonymize** (emails/phones/URLs → `[email]`/`[phone]`/`[url]`) → store original under `data/resumes/` → extract name/skills/experience-years → embed → index in Qdrant → mark `processing_status`.
- Failures are per-file: one bad resume never blocks the batch (`bulk_ingest_directory` returns succeeded/failed counts).
- Candidate IDs are UUIDs — no PII in identifiers.

### 3.2 Preprocessing (`app/services/preprocessing/text.py`)

- `clean_text` — whitespace/newline normalization.
- `pseudonymize` — regex-based PII stripping **before** storage and embedding.
- `extract_skills` — curated-technology vocabulary matched against resume text (supporting feature; full resume text remains the retrieval source of truth).
- `split_sections` — header-aware chunking (Summary / Experience / Projects / Education / …) producing `DocumentChunk`s for the optional section-level embedding mode.

### 3.3 Embedding service (`app/services/embeddings/service.py`)

- Lazy-loads `SentenceTransformer` on first use; models cached in the `models` Docker volume (`HF_HOME=/models/huggingface`) so restarts don't re-download.
- `encode()` normalizes embeddings → cosine similarity becomes a dot product.
- **Versioned collections**: each model maps to its own Qdrant collection (`resume_embeddings_baai_bge-base-en-v1-5`, `…_all-minilm-l6-v2`, …) so incompatible vector spaces can never be mixed.
- Configurable: primary `BAAI/bge-base-en-v1.5` (768-dim), baseline `all-MiniLM-L6-v2` (384-dim, faster/cheaper comparison point).

### 3.4 Retrieval (`app/services/retrieval/`)

- **Qdrant store**: cosine top-K search. Fetches `4×K` points then deduplicates by `candidate_id` (best section wins) — this makes section-level indexing safe without returning the same resume twice.
- **BM25 baseline** (`bm25.py`): in-memory `rank_bm25` index rebuilt per query over `resume_text`. Implements experiment E1 (lexical baseline) and is the fallback retriever.
- Retrieval returns `(candidate_id, score)` ordered pairs; rank/score/query provenance is persisted on every `MatchResult`.

### 3.5 Re-ranking (`app/services/reranking/`)

**Cross-Encoder** (`cross_encoder.py`)
- `cross-encoder/ms-marco-MiniLM-L-6-v2` scores `(JD, resume)` pairs jointly — much more precise than bi-encoder cosine but O(K) forward passes. Scores are model-specific logits, not probabilities; we only use their ordering.

**LLM ranker** (`llm.py`)
- Provider-independent interface; Ollama adapter implemented (`/api/generate`, `format: "json"`, temperature 0).
- Prompt contract (SRD §6.3): structured JSON with `relevance_score` (0–100 rubric), `matched_requirements`, `missing_requirements`, `evidence[{requirement, resume_evidence}]`, `reason`. System prompt treats resume text as untrusted, forbids instruction-following inside resumes, forbids inferring absent information.
- **Robustness layers** (learned from real runs with `gemma3:4b`):
  - tolerant JSON extraction (fences, bare objects, concatenated values, trailing prose);
  - candidate-ID resolution tolerating UUID prefixes/truncation;
  - **batch-then-fallback**: one call ranks all candidates; any candidate the model skipped is scored in individual calls (concurrency-limited). Small local models reliably emit only the first object otherwise.
- Every assessment's `reason`/`evidence` is stored in `match_results.explanation` (JSONB) → powers the "matching explanations" feature.

### 3.6 Matching pipeline (`app/services/pipeline.py`)

`run_match(job, top_k, ranking_method)`:
1. Retrieve top-K (`vector` → Qdrant+BGE, `bm25` → lexical; `cross_encoder`/`llm` reuse the same BGE pool).
2. Load candidate rows; dispatch to the chosen ranker.
3. Sort by ranking score desc, ties → retrieval order.
4. Persist `MatchRun` (model versions, top_k, status, latency) + `MatchResult` rows (retrieval rank/score, final rank, ranking score, explanation).

### 3.7 Evaluation (`app/services/evaluation/`)

- `metrics.py`: graded-relevance DCG `Σ(2^rel − 1)/log₂(i+1)`; P/R/MRR use binary threshold **label ≥ 2**; NDCG normalized by ideal DCG. All pure functions — unit-tested against hand-computed cases.
- `engine.py`: for each (job × ranking_method) runs a match, joins `evaluation_labels` (0–3 grades), computes per-query metrics, aggregates means + mean latency, stores an `Experiment` row with full configuration (embedding model, CE model, LLM model + prompt version, top_k, dataset version) for reproducibility.

### 3.8 Persistence

- **PostgreSQL** (SQLAlchemy async + asyncpg): `candidates`, `jobs`, `match_runs`, `match_results`, `evaluation_labels`, `experiments` — exactly the SRD §7 schema. Tables auto-create on startup (`create_all`); Alembic is on the roadmap.
- **Qdrant**: vectors + payload (`candidate_id`, `skills`, `experience_years`, `embedding_model`, `section`).
- **Filesystem**: original resume binaries under `./data/resumes/` (bind-mounted, survives restarts).

### 3.9 Frontend (`frontend/`)

React 18 + TypeScript + Vite + Tailwind, react-router. Five views matching SRD §10:

- **Overview** — counts (resumes/jobs/experiments) + live `/health` (PG, Qdrant, model-load status).
- **Jobs** — create/edit JDs, pick ranking method, run match, inline results table.
- **Candidates** — paginated/searchable list, skill filter, PDF/DOCX upload, resume-text modal.
- **Ranking Comparison** — run all three methods on one job; rank-position matrix per candidate.
- **Evaluation** — select jobs/methods, run benchmark, metrics table (P@5, R@5, NDCG@10, MRR, latency), experiment history.

nginx serves the SPA and proxies `/api/` → `backend:8000`, so no CORS/CORs-config issues in compose.

---

## 4. End-to-End Processes

### Upload → searchable (per resume)

```
POST /resumes/upload (multipart)
  → validate ext/size
  → extract text (PyMuPDF / python-docx)
  → clean + pseudonymize
  → persist file → data/resumes/{uuid}.{ext}
  → Candidate row (status=processing → processed)
  → BGE encode(resume[:8000])          # document-level default
  → Qdrant upsert (collection per model)
```

### Match run

```
POST /jobs/{id}/match {top_k, ranking_method}
  → embed "{title}\n{description}"
  → Qdrant top-K  (or BM25 for method=bm25)
  → ranker: vector | bm25 | cross_encoder | llm
  → MatchRun(status, latency_ms) + MatchResult rows committed
  → GET /match-runs/{id}/results
```

### Evaluation

```
POST /evaluation/run {job_ids, ranking_methods}
  → for each job × method: run_match → join labels → metrics
  → Experiment(configuration, metrics{per-method aggregates + per-job})
  → GET /evaluation, /evaluation/{id}, /evaluation/compare?ids=…
```

---

## 5. Technology Decisions & Trade-offs

| Decision | Choice | Why / trade-off |
|---|---|---|
| Vector DB | **Qdrant** | Purpose-built ANN store, simple Docker image, payload filtering, per-model collections. Weighed against pgvector (fewer moving parts, but weaker for 768-dim ANN at scale) and FAISS (library, no server). |
| Embedding model | **BGE-base-en-v1.5** primary, **MiniLM-L6-v2** baseline | BGE is SOTA-ish for retrieval on MTEB; MiniLM gives the E2-vs-E3 comparison (does a bigger model actually help?). Cost: ~400MB download, CPU-slowest acceptable. |
| Document vs section embeddings | **Document-level default**, section-level implemented behind flag | One vector/resume is simple and avoids dup-candidate complexity; section-level preserves detail but multiplies index size and needs dedup (implemented: fetch 4×K, keep best per candidate). Flagged as experimental per SRD §5.2. |
| Re-rankers | **Cross-Encoder** + **LLM (Ollama)** run independently on same pool | CE is cheap-ish (~ms/pair on CPU) and strong; LLM gives explanations but is slow (~10–60s for 20 candidates on CPU) and less consistent — hence fixed prompt, temp=0, structured output, and recorded prompt version. |
| LLM provider | **Ollama local** (`gemma3:4b`), adapter pattern | No API costs, no PII leaves the machine (SRD privacy §11.3), reproducible. Trade-off: small local models need the batch→per-candidate fallback; an external-API adapter can be added behind the same `rank()` contract. |
| LLM output contract | Structured JSON rubric 0–100 | Comparable numeric scores + human-readable evidence. Trade-off: small models sometimes emit partial JSON → tolerant parser + fallback scoring; failures logged, never crash the run. |
| DB | PostgreSQL for everything except vectors | ACID for runs/labels/experiments; JSONB for skills/explanations avoids schema churn. Qdrant only holds vectors+payload. |
| Tables creation | `create_all` on startup | Fastest path to a runnable MVP. Trade-off: no migration history — Alembic is planned before schema evolves. |
| Retrieval dedup | Fetch 4×K then best-per-candidate | Cheap way to make section-level indexing safe; slight over-fetch cost. |
| PII handling | Pseudonymize before storage | Emails/phones/URLs stripped pre-embedding; UUID candidate IDs; delete endpoint removes vectors too. Trade-off: regex-based — good enough for demo, not a guarantee. |
| Frontend serving | nginx static + `/api` proxy | One container, no dev-server proxy config, works identically in compose and prod-ish deploys. |
| Heavy ML deps | All in one backend image | Simplest compose; ~3.5GB image. Trade-off: slow first build (torch+CUDA wheels ~3GB). CPU-only torch would halve it — noted for later. |
| spaCy | In requirements, lightly used | SRD lists it for preprocessing; current pipeline uses regex/section heuristics which proved sufficient — spaCy pipeline (NER-based skill extraction) is a planned upgrade. |

### Known limitations (honest list)

- **No Alembic migrations** — schema changes require care.
- **No relevance-labeling UI** — labels must be inserted via SQL for now; evaluation needs them to produce meaningful numbers.
- **LLM ranker latency** scales with pool size; top-20 cap is the mitigation.
- **Skill extraction is vocabulary-based** — misses novel terms; resume full text still drives retrieval.
- **Single-node, single-user** — role-based access is architectural only (no auth yet), per SRD MVP allowance.
- **Dataset is 6 sample resumes**, not the 500-resume public corpus — ingestion pipeline is proven; bulk dataset + licensing review is milestone work remaining.
- **E1–E5 benchmark**: methods exist (`bm25`, MiniLM via `retrieval_model`, BGE vector/CE/LLM); a scripted runner producing the full comparison report is partially built into `/evaluation/run`.

---

## 6. Evaluation Framework

- **Labels**: `evaluation_labels` table — `(job_id, candidate_id) → relevance 0–3` (0 irrelevant, 1 weak, 2 good, 3 strong). Binary-relevant threshold = **≥ 2** for P/R/MRR.
- **Metrics** (all in `backend/app/services/evaluation/metrics.py`, unit-tested):
  - `precision@K` = relevant-in-top-K / K
  - `recall@K` = relevant-in-top-K / total relevant for that job
  - `NDCG@K` = DCG/IDCG with graded gains `2^rel − 1` — **primary metric, K=10**
  - `MRR` = mean reciprocal rank of first relevant (label ≥ 2)
- **Benchmark matrix** (SRD §8.7): E1 BM25 · E2 MiniLM · E3 BGE vector · E4 BGE+CE · E5 BGE+LLM — E4/E5 share the BGE top-20 pool.
- **Integrity rules enforced by design**: fixed corpus + fixed test jobs, labels independent of model output, same pool across rerankers, full config + model + prompt versions stored per experiment, failed LLM calls logged and scored 0 rather than silently dropped.

---

## 7. API Reference

Base: `http://localhost:8000/api/v1` (`/health` also at root)

| Method | Path | Purpose |
|---|---|---|
| POST | `/resumes/upload` | multipart PDF/DOCX → ingest+embed+index |
| GET | `/resumes` | list (`page`, `limit`, `search`, `skill`) |
| GET/DELETE | `/resumes/{id}` | detail (incl. resume text) / delete (cascades vectors) |
| POST/GET | `/jobs` | create / list JDs |
| GET/PUT | `/jobs/{id}` | detail / update |
| POST | `/jobs/{id}/match` | `{top_k, ranking_method: vector\|bm25\|cross_encoder\|llm, retrieval_model?}` |
| GET | `/match-runs/{id}` `/match-runs/{id}/results` | run status / ranked results + explanations |
| POST | `/evaluation/run` | `{job_ids, ranking_methods, metrics, top_k, experiment_name}` |
| GET | `/evaluation` `/evaluation/{id}` `/evaluation/compare?ids=` | experiments |
| GET | `/health` | API + PG + Qdrant + model-load status |

Consistent error shape via HTTPException; pagination on list endpoints; structured logging throughout.

---

## 8. Setup & Running

### Prerequisites

- Docker Desktop (running)
- ~6GB free disk for images/models (first build downloads torch+CUDA ~3GB)
- Optional: [Ollama](https://ollama.com) on host for LLM ranking (`gemma3:4b` verified; any instruct model works via `OLLAMA_MODEL`)

### Docker (recommended)

```bash
cp .env.example .env        # edit if needed (DB creds, models, Ollama URL/model)
docker compose up --build   # first build is slow (~30-60 min depending on bandwidth)
```

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| API | http://localhost:8000/api/v1 |
| Qdrant dashboard | http://localhost:6333/dashboard |

Models download lazily on first use into the `models` volume; resumes persist in `./data/`.

### Local dev (no Docker)

```bash
# backend
cd backend && python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# frontend
cd frontend && npm install && npm run dev   # proxies /api → :8000
```

### Tests

```bash
cd backend && pytest          # metrics, preprocessing, DOCX parsing (13 tests)
cd frontend && npm run build  # tsc + vite build is the typecheck
```

### Ollama setup (for `ranking_method: "llm"`)

```powershell
# binary found at D:\Programs\ollama.exe on this machine
D:\Programs\ollama serve      # serves :11434, models in D:\ollama\models
# already pulled: gemma3:4b, qwen3-vl:4b
```

Backend reaches it via `OLLAMA_BASE_URL=http://host.docker.internal:11434`.

---

## 9. Project Structure

```
Elip/
├── backend/
│   ├── app/
│   │   ├── api/            # resumes, jobs, match, evaluation, health routers
│   │   ├── core/           # config (pydantic-settings), db engine, logging
│   │   ├── models/         # SQLAlchemy tables (SRD §7 schema)
│   │   ├── schemas/        # pydantic request/response models
│   │   ├── services/
│   │   │   ├── ingestion/  # parser.py (PDF/DOCX), service.py (pipeline)
│   │   │   ├── preprocessing/ # text.py (clean, PII, skills, sections)
│   │   │   ├── embeddings/ # service.py (lazy ST models, versioning)
│   │   │   ├── retrieval/  # qdrant_store.py, bm25.py
│   │   │   ├── reranking/  # cross_encoder.py, llm.py (Ollama)
│   │   │   ├── evaluation/ # metrics.py, engine.py
│   │   │   └── pipeline.py # match orchestration
│   │   └── main.py
│   ├── tests/              # pytest: metrics, preprocessing, parser
│   ├── requirements.txt / Dockerfile
├── frontend/               # Vite + React + TS + Tailwind
│   └── src/{pages,services,components}/
├── evaluation/             # datasets/, relevance_labels/, experiments/, reports/
├── data/{raw,processed,embeddings}/  # resume storage (bind mount)
├── docker-compose.yml
├── .env.example / .env
└── README.md
```

---

## 10. Current Status & Known Limitations

**Working & verified live:**
- 6 sample resumes ingested end-to-end (parse → PII-strip → BGE embed → Qdrant)
- Match runs for `vector`, `bm25`, `cross_encoder`, `llm` — all persist results
- LLM ranker produces scored, explained rankings via local `gemma3:4b`
- Dashboard renders all five views against the live API

**Not yet done:**
- 500-resume public dataset + 30 JDs + human 0–3 labeling workflow
- Alembic migrations; relevance-labeling UI; benchmark report export
- Section-level embedding experiment (code ready, flag off)
- MiniLM E2 experiment wiring (model supported, needs collection + runner config)
- Auth/RBAC (allowed to be simple for MVP)

---

## 11. Roadmap

1. **Dataset milestone** — source/license a public resume corpus, bulk-ingest, prepare 30 JDs, build labeling UI.
2. **Experiments** — run E1–E5, produce the §8.8 benchmark report with CIs.
3. **Hardening** — Alembic, auth, section-level embeddings A/B, hybrid BM25+dense retrieval.
4. **Scale** — async batch ingestion, CE batching, optional GPU for LLM/CE.
