"""Evaluation engine: run benchmarks across jobs/ranking methods and store experiments."""

import logging
import uuid
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.tables import EvaluationLabel, Experiment, Job, MatchResult, MatchRun
from app.schemas.schemas import EvaluationRunRequest, ExperimentOut
from app.services.evaluation import metrics as metrics_service
from app.services.pipeline import run_match

logger = logging.getLogger(__name__)
settings = get_settings()


def _labels_for_job(labels: list[EvaluationLabel], ranked_candidate_ids: list[uuid.UUID]) -> tuple[list[int], int]:
    """Return (graded relevance labels in ranking order, total relevant count)."""
    label_map = {(l.job_id, l.candidate_id): l.relevance_label for l in labels}
    # labels list may contain all jobs; filter by ranked ids via job handled by caller
    return [], 0


async def run_evaluation(
    db: AsyncSession,
    request: EvaluationRunRequest,
) -> Experiment:
    """Execute match runs for each job/method and compute metrics vs. labels."""
    jobs_q = select(Job)
    if request.job_ids:
        jobs_q = jobs_q.where(Job.id.in_(request.job_ids))
    jobs = (await db.execute(jobs_q)).scalars().all()
    if not jobs:
        raise ValueError("No jobs to evaluate")

    labels = (await db.execute(select(EvaluationLabel))).scalars().all()
    label_map: dict[tuple[uuid.UUID, uuid.UUID], int] = {
        (l.job_id, l.candidate_id): l.relevance_label for l in labels
    }

    per_query: list[dict] = []
    details: dict[str, list[dict]] = defaultdict(list)
    config = {
        "ranking_methods": request.ranking_methods,
        "top_k": request.top_k,
        "metrics": request.metrics or settings.eval_metric_list,
        "embedding_model": settings.embedding_model,
        "cross_encoder_model": settings.cross_encoder_model,
        "llm_model": settings.ollama_model,
        "llm_prompt_version": settings.llm_prompt_version,
    }

    for method in request.ranking_methods:
        method_queries = []
        for job in jobs:
            run = await run_match(db, job, top_k=request.top_k, ranking_method=method)
            results = (
                await db.execute(
                    select(MatchResult)
                    .where(MatchResult.match_run_id == run.id)
                    .order_by(MatchResult.final_rank.asc().nullslast())
                )
            ).scalars().all()

            ranked_ids = [r.candidate_id for r in results]
            graded = [label_map.get((job.id, cid), 0) for cid in ranked_ids]
            total_relevant = sum(1 for l in labels if l.job_id == job.id and l.relevance_label >= metrics_service.BINARY_THRESHOLD)

            ks = tuple(int(m.split("@")[1]) for m in config["metrics"] if m.startswith(("precision@", "recall@"))) or (5,)
            ndcg_k = next((int(m.split("@")[1]) for m in config["metrics"] if m.startswith("ndcg@")), 10)
            q_metrics = metrics_service.compute_metrics(graded, total_relevant, ks=ks, ndcg_k=ndcg_k)
            q_metrics.update({"job_id": str(job.id), "match_run_id": str(run.id), "latency_ms": run.latency_ms})
            method_queries.append(q_metrics)
            details[method].append(q_metrics)

        agg = metrics_service.aggregate([{k: v for k, v in q.items() if isinstance(v, float)} for q in method_queries])
        latencies = [q["latency_ms"] for q in method_queries if q.get("latency_ms")]
        agg["mean_latency_ms"] = sum(latencies) / len(latencies) if latencies else 0.0
        per_query.append({"ranking_method": method, "metrics": agg, "per_job": method_queries})

    experiment = Experiment(
        experiment_name=request.experiment_name or f"eval-{uuid.uuid4().hex[:8]}",
        configuration=config,
        dataset_version="v1",
        metrics={"queries": per_query},
    )
    db.add(experiment)
    await db.commit()
    await db.refresh(experiment)
    return experiment


async def list_experiments(db: AsyncSession) -> list[Experiment]:
    return (await db.execute(select(Experiment).order_by(Experiment.created_at.desc()))).scalars().all()


async def get_experiment(db: AsyncSession, experiment_id: uuid.UUID) -> Experiment | None:
    return await db.get(Experiment, experiment_id)
