import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.schemas import EvaluationRunRequest, ExperimentOut
from app.services.evaluation.engine import get_experiment, list_experiments, run_evaluation

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/run", response_model=ExperimentOut, status_code=201)
async def trigger_evaluation(payload: EvaluationRunRequest, db: AsyncSession = Depends(get_db)):
    try:
        experiment = await run_evaluation(db, payload)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return experiment


@router.get("", response_model=list[ExperimentOut])
async def get_experiments(db: AsyncSession = Depends(get_db)):
    return await list_experiments(db)


@router.get("/compare")
async def compare_experiments(ids: str, db: AsyncSession = Depends(get_db)):
    id_list = [i.strip() for i in ids.split(",") if i.strip()]
    experiments = []
    for raw in id_list:
        exp = await get_experiment(db, uuid.UUID(raw))
        if exp:
            experiments.append(ExperimentOut.model_validate(exp))
    return {"experiments": experiments}


@router.get("/{experiment_id}", response_model=ExperimentOut)
async def get_experiment_detail(experiment_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    exp = await get_experiment(db, experiment_id)
    if not exp:
        raise HTTPException(404, "Experiment not found")
    return exp
