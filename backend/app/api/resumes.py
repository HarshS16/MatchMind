import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.tables import Candidate
from app.schemas.schemas import (
    CandidateDetail,
    CandidateListResponse,
    CandidateOut,
    UploadResponse,
)
from app.services.ingestion.parser import UnsupportedFileError
from app.services.ingestion.service import ingest_resume

router = APIRouter(prefix="/resumes", tags=["resumes"])


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_resume(file: UploadFile, db: AsyncSession = Depends(get_db)):
    if not file.filename:
        raise HTTPException(400, "Missing filename")
    data = await file.read()
    try:
        candidate = await ingest_resume(db, file.filename, data)
    except UnsupportedFileError as exc:
        raise HTTPException(415, str(exc))
    except ValueError as exc:
        raise HTTPException(413, str(exc))
    return UploadResponse(candidate_id=candidate.id, filename=file.filename, status=candidate.processing_status)


@router.get("", response_model=CandidateListResponse)
async def list_candidates(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=200),
    search: str | None = None,
    skill: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Candidate)
    if search:
        stmt = stmt.where(Candidate.resume_text.ilike(f"%{search}%"))
    if skill:
        stmt = stmt.where(Candidate.skills.cast(str).ilike(f"%{skill}%"))
    total = (await db.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one()
    rows = (
        await db.execute(stmt.order_by(Candidate.created_at.desc()).offset((page - 1) * limit).limit(limit))
    ).scalars().all()
    return CandidateListResponse(total=total, page=page, limit=limit, items=[CandidateOut.model_validate(r) for r in rows])


@router.get("/{candidate_id}", response_model=CandidateDetail)
async def get_candidate(candidate_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    candidate = await db.get(Candidate, candidate_id)
    if not candidate:
        raise HTTPException(404, "Candidate not found")
    return CandidateDetail.model_validate(candidate)


@router.delete("/{candidate_id}", status_code=204)
async def delete_candidate(candidate_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    candidate = await db.get(Candidate, candidate_id)
    if not candidate:
        raise HTTPException(404, "Candidate not found")
    from app.services.retrieval import qdrant_store

    qdrant_store.delete_candidate(candidate_id)
    await db.delete(candidate)
    await db.commit()
