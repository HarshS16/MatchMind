import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.tables import Job, MatchResult, MatchRun
from app.schemas.schemas import MatchRequest, MatchResultsResponse, MatchResultOut, MatchRunOut
from app.services.pipeline import RANKING_METHODS, run_match

router = APIRouter(tags=["matching"])


@router.post("/jobs/{job_id}/match", response_model=MatchRunOut, status_code=201)
async def trigger_match(job_id: uuid.UUID, payload: MatchRequest, db: AsyncSession = Depends(get_db)):
    if payload.ranking_method not in RANKING_METHODS:
        raise HTTPException(400, f"ranking_method must be one of {sorted(RANKING_METHODS)}")
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    run = await run_match(
        db,
        job,
        top_k=payload.top_k,
        ranking_method=payload.ranking_method,
        retrieval_model=payload.retrieval_model,
    )
    return run


@router.get("/match-runs/{match_run_id}/results", response_model=MatchResultsResponse)
async def get_match_results(match_run_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    run = await db.get(MatchRun, match_run_id)
    if not run:
        raise HTTPException(404, "Match run not found")

    rows = (
        await db.execute(
            select(MatchResult)
            .where(MatchResult.match_run_id == match_run_id)
            .order_by(MatchResult.final_rank.asc().nullslast())
        )
    ).scalars().all()

    items = []
    for r in rows:
        explanation = r.explanation or {}
        items.append(MatchResultOut(
            candidate_id=r.candidate_id,
            retrieval_rank=r.retrieval_rank,
            final_rank=r.final_rank,
            retrieval_score=r.retrieval_score,
            ranking_score=r.ranking_score,
            matched_skills=explanation.get("matched_requirements", []),
            explanation=explanation,
        ))

    return MatchResultsResponse(
        match_run_id=run.id,
        job_id=run.job_id,
        ranking_method=run.ranking_model,
        total_candidates=len(items),
        results=items,
    )


@router.get("/match-runs/{match_run_id}", response_model=MatchRunOut)
async def get_match_run(match_run_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    run = await db.get(MatchRun, match_run_id)
    if not run:
        raise HTTPException(404, "Match run not found")
    return run
