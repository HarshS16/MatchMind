"""Matching pipeline orchestrator: retrieval -> optional re-ranking -> persistence."""

import logging
import time
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.tables import Candidate, Job, MatchResult, MatchRun
from app.services.embeddings import service as embedding_service
from app.services.retrieval import bm25 as bm25_service
from app.services.retrieval import qdrant_store
from app.services.reranking import cross_encoder, llm

logger = logging.getLogger(__name__)
settings = get_settings()

RANKING_METHODS = {"vector", "cross_encoder", "llm", "bm25"}


async def _get_candidates(db: AsyncSession, ids: list[uuid.UUID]) -> dict[uuid.UUID, Candidate]:
    if not ids:
        return {}
    rows = (await db.execute(select(Candidate).where(Candidate.id.in_(ids)))).scalars().all()
    return {c.id: c for c in rows}


async def retrieve_candidates(
    db: AsyncSession,
    job: Job,
    top_k: int,
    retrieval_model: str | None = None,
) -> tuple[list[tuple[uuid.UUID, float]], str]:
    """Return (candidate_id, retrieval_score) pairs ordered by retrieval rank."""
    if retrieval_model == "bm25":
        rows = (await db.execute(select(Candidate.id, Candidate.resume_text))).all()
        index = bm25_service.BM25Index()
        index.build([r[0] for r in rows], [r[1] or "" for r in rows])
        query = f"{job.title}\n{job.description}"
        return index.search(query, top_k), "bm25"

    query_text = f"{job.title}\n{job.description}"
    vector = embedding_service.encode_single(query_text, model_name=retrieval_model)
    hits = qdrant_store.search(vector, limit=top_k, model_name=retrieval_model)
    return [(h.candidate_id, h.score) for h in hits], embedding_service.collection_name(retrieval_model)


async def run_match(
    db: AsyncSession,
    job: Job,
    top_k: int = 20,
    ranking_method: str = "cross_encoder",
    retrieval_model: str | None = None,
) -> MatchRun:
    if ranking_method not in RANKING_METHODS:
        raise ValueError(f"Unknown ranking method: {ranking_method}")

    start = time.perf_counter()
    run = MatchRun(
        job_id=job.id,
        retrieval_model=retrieval_model or settings.embedding_model,
        ranking_model=ranking_method,
        top_k=top_k,
        status="running",
    )
    db.add(run)
    await db.flush()

    try:
        retrieved, retriever_name = await retrieve_candidates(db, job, top_k, retrieval_model)
        candidates = await _get_candidates(db, [cid for cid, _ in retrieved])

        # Build ordered pairs (id, resume_text) for re-ranking
        pairs = [
            (cid, (candidates[cid].resume_text if cid in candidates else ""))
            for cid, _ in retrieved
            if cid in candidates
        ]

        explanations: dict[uuid.UUID, dict] = {}
        ranking_scores: dict[uuid.UUID, float] = {}

        if ranking_method == "vector" or ranking_method == "bm25":
            for cid, score in retrieved:
                ranking_scores[cid] = score
        elif ranking_method == "cross_encoder":
            job_text = f"Job Description:\n{job.description}"
            texts = [text for _, text in pairs]
            scores = cross_encoder.score_pairs(job_text, texts)
            for (cid, _), score in zip(pairs, scores):
                ranking_scores[cid] = score
        elif ranking_method == "llm":
            ranker = llm.get_ranker()
            assessments = await ranker.rank(
                job_description=f"{job.title}\n{job.description}",
                candidates=pairs,
                required_skills=job.required_skills or [],
                preferred_skills=job.preferred_skills or [],
                experience_min=job.experience_min,
                experience_max=job.experience_max,
            )
            for a in assessments:
                try:
                    cid = uuid.UUID(a["candidate_id"])
                except (ValueError, KeyError):
                    continue
                ranking_scores[cid] = a.get("relevance_score", 0.0)
                explanations[cid] = a
            # Candidates missing from LLM output get score 0 but retain retrieval order at the end
            for cid, score in retrieved:
                ranking_scores.setdefault(cid, 0.0)

        # Final ordering: by ranking score desc; ties keep retrieval order.
        retrieval_rank = {cid: i + 1 for i, (cid, _) in enumerate(retrieved)}
        retrieval_score = {cid: s for cid, s in retrieved}
        ordered = sorted(
            ranking_scores.keys(),
            key=lambda cid: (-ranking_scores[cid], retrieval_rank.get(cid, 10**9)),
        )

        for final_rank, cid in enumerate(ordered, start=1):
            db.add(MatchResult(
                match_run_id=run.id,
                candidate_id=cid,
                retrieval_score=retrieval_score.get(cid),
                ranking_score=ranking_scores.get(cid),
                retrieval_rank=retrieval_rank.get(cid),
                final_rank=final_rank,
                explanation=explanations.get(cid, {}),
            ))

        run.status = "completed"
        run.retrieval_model = retriever_name
    except Exception as exc:
        logger.exception("Match run failed for job %s", job.id)
        run.status = "failed"
        run.latency_ms = int((time.perf_counter() - start) * 1000)
        await db.commit()
        raise

    run.latency_ms = int((time.perf_counter() - start) * 1000)
    await db.commit()
    await db.refresh(run)
    return run
