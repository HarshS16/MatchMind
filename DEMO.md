# Demo Guide — How to Use & Present the App

## 30-second pitch

> "This is a semantic job-matching system. Instead of keyword search, it embeds resumes and job descriptions with a transformer model (BGE), finds the top candidates via vector similarity in Qdrant, then re-ranks them two different ways — a Cross-Encoder and a local LLM — and scores ranking quality with standard IR metrics like NDCG. The interesting part: all rankers see the *same* candidate pool, so we can fairly measure whether re-ranking actually helps."

## The tabs, one by one

### 1. Overview (dashboard home)

- **What it shows:** counts of indexed resumes, jobs, experiments, and live system health (PostgreSQL, Qdrant, which embedding model is loaded).
- **How to explain:** "This is the mission-control view — it proves all the infrastructure (database, vector store, ML models) is alive before we do anything."
- **Demo move:** point at the health badges; refresh after running a match to show `embedding_loaded: true`.

### 2. Jobs

- **What it does:** create job descriptions (title, description, required/preferred skills, experience range), then hit **Run Match** and pick a ranking method (`vector`, `bm25`, `cross_encoder`, `llm`).
- **What's seeded:** 4 jobs — *Machine Learning Engineer*, *Senior Backend Engineer*, *Frontend Engineer*, *DevOps / SRE Engineer*.
- **How to explain:** "The job description gets embedded into the same vector space as the resumes. Retrieval returns the top-10/20 candidates. Then the chosen ranker re-orders them — vector = raw similarity, bm25 = keyword baseline, cross_encoder = neural pairwise scorer, llm = a local LLM that writes a structured assessment with evidence."
- **Demo move:** run `llm` on the ML Engineer job — show the reasoning in the results table; then run `vector` on the same job and compare order.

### 3. Candidates

- **What it does:** lists all indexed resumes with extracted skills and processing status. Search box filters resume text, skill box filters by skill, **Upload Resume** ingests a new PDF/DOCX live. Clicking a row opens the extracted (PII-stripped) resume text.
- **How to explain:** "Every resume goes through parse → clean → pseudonymize → embed → index. Notice emails/phones are stripped — we never store raw PII. The 'processed' badge means its vector is in Qdrant."
- **Demo move:** upload one of the sample .docx files live and watch it appear; open a resume to show the cleaned text.

### 4. Ranking Comparison

- **What it does:** pick a job, run all three ranking methods, and see a side-by-side table of each candidate's final rank under vector / cross_encoder / llm.
- **How to explain:** "This is the core experiment. Same retrieved pool, three different orderings — you can literally see where the LLM promotes/demotes candidates vs. raw similarity. Rank changes = re-ranking effect."
- **Demo move:** use the DevOps job (already has all three runs) — Frank/DevOps-profile candidates should climb under llm vs vector.

### 5. Evaluation

- **What it does:** run a benchmark across selected jobs and ranking methods; shows P@5, R@5, NDCG@10, MRR, mean latency per method, plus experiment history.
- **How to explain:** "This is where we answer the research question. NDCG@10 is the headline metric — it rewards putting the most relevant candidates at the top. The catch: it needs human relevance labels (0–3) in the `evaluation_labels` table, which we haven't populated yet — so right now it's the framework, not the results."
- **Demo move:** be honest — click Run Evaluation, explain it will produce near-zero/meaningless numbers until labels exist; that's the next milestone.

## Key concepts to drop in conversation

- **Two-stage retrieval:** cheap recall first (vector top-K), expensive precision second (rerankers). Standard IR architecture.
- **Same-pool fairness:** rerankers only reorder retrieved candidates — that's what makes the comparison scientifically valid.
- **Bi-encoder vs cross-encoder:** embeddings compare independently (fast, less precise); cross-encoder reads JD+resume together (slow, precise).
- **LLM as ranker:** not just a score — returns matched/missing requirements + resume evidence (explainability).
- **Metrics:** NDCG@10 primary; P@5/R@5/MRR secondary; binary relevance = label ≥ 2.

## If someone asks "does the LLM actually help?"

"That's exactly what the evaluation tab measures — but we need labeled relevance judgments first. Architecturally everything is in place; the labeling workflow is the remaining piece."

## Troubleshooting during demo

- Everything down → `docker compose up -d`
- LLM ranking fails → is Ollama running? `"D:\Programs\ollama serve"` then check http://localhost:11434
- First match is slow → models download once into the `models` volume, then cached
- Frontend blank → hard refresh; API must be up (check Overview tab health)

## Seeded data summary

- **16 resumes** across roles: ML (×3), backend (×2), frontend (×2), data, DevOps/SRE (×2), mobile, QA, security, PM/analytics
- **4 jobs** with matches already run for vector + cross_encoder on all, and llm on DevOps
- Sample resume files live in `data/raw/sample_resumes/` if you want to upload more live
