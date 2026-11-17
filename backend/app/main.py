from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import evaluation, health, jobs, match, resumes
from app.core.config import get_settings
from app.core.logging import setup_logging

setup_logging()
settings = get_settings()

app = FastAPI(title=settings.app_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api/v1"
app.include_router(resumes.router, prefix=API_PREFIX)
app.include_router(jobs.router, prefix=API_PREFIX)
app.include_router(match.router, prefix=API_PREFIX)
app.include_router(evaluation.router, prefix=API_PREFIX)
app.include_router(health.router, prefix=API_PREFIX)
app.include_router(health.router)


@app.on_event("startup")
async def startup() -> None:
    # Create tables on startup (Alembic migrations can replace this later).
    from app.core.database import Base, engine
    import app.models  # noqa: F401  ensure models are registered

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
