import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.tables import Job
from app.schemas.schemas import JobCreate, JobOut, JobUpdate

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=JobOut, status_code=201)
async def create_job(payload: JobCreate, db: AsyncSession = Depends(get_db)):
    job = Job(**payload.model_dump())
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


@router.get("", response_model=list[JobOut])
async def list_jobs(db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(Job).order_by(Job.created_at.desc()))).scalars().all()
    return rows


@router.get("/{job_id}", response_model=JobOut)
async def get_job(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.put("/{job_id}", response_model=JobOut)
async def update_job(job_id: uuid.UUID, payload: JobUpdate, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(job, field, value)
    await db.commit()
    await db.refresh(job)
    return job
