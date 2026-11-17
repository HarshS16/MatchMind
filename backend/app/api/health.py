import asyncio

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import text

from app.core.config import get_settings
from app.core.database import engine
from app.schemas.schemas import HealthResponse
from app.services.embeddings import service as embedding_service
from app.services.retrieval import qdrant_store
from app.services.reranking import cross_encoder

router = APIRouter(tags=["health"])
settings = get_settings()


class PingResponse(BaseModel):
    status: str


async def _check_postgres() -> str:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return "ok"
    except Exception:
        return "unreachable"


def _check_qdrant() -> str:
    try:
        qdrant_store.get_client().get_collections()
        return "ok"
    except Exception:
        return "unreachable"


@router.get("/health", response_model=HealthResponse)
async def health():
    pg, qd = await asyncio.gather(_check_postgres(), asyncio.to_thread(_check_qdrant))
    embedding_loaded = settings.embedding_model in embedding_service._models
    ce_loaded = cross_encoder._model is not None
    status = "ok" if pg == "ok" and qd == "ok" else "degraded"
    return HealthResponse(
        status=status,
        postgres=pg,
        qdrant=qd,
        embedding_model=settings.embedding_model,
        embedding_loaded=embedding_loaded,
        cross_encoder_loaded=ce_loaded,
    )
