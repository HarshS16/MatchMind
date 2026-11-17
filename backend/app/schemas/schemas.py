import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------- Candidates / Resumes ----------

class CandidateOut(BaseModel):
    id: uuid.UUID
    candidate_name: str | None = None
    email: str | None = None
    skills: list[str] = []
    experience: dict[str, Any] = {}
    education: dict[str, Any] = {}
    processing_status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class CandidateDetail(CandidateOut):
    resume_text: str = ""
    original_filename: str | None = None


class CandidateListResponse(BaseModel):
    total: int
    page: int
    limit: int
    items: list[CandidateOut]


class UploadResponse(BaseModel):
    candidate_id: uuid.UUID
    filename: str
    status: str


# ---------- Jobs ----------

class JobCreate(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    description: str = Field(min_length=1)
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    responsibilities: str | None = None
    experience_min: int | None = None
    experience_max: int | None = None


class JobUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    required_skills: list[str] | None = None
    preferred_skills: list[str] | None = None
    responsibilities: str | None = None
    experience_min: int | None = None
    experience_max: int | None = None


class JobOut(BaseModel):
    id: uuid.UUID
    title: str
    description: str
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    responsibilities: str | None = None
    experience_min: int | None = None
    experience_max: int | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------- Matching ----------

class MatchRequest(BaseModel):
    top_k: int = Field(default=20, ge=1, le=100)
    ranking_method: str = "cross_encoder"
    retrieval_model: str | None = None


class MatchRunOut(BaseModel):
    id: uuid.UUID
    job_id: uuid.UUID
    retrieval_model: str
    ranking_model: str
    top_k: int
    status: str
    latency_ms: int | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class MatchResultOut(BaseModel):
    candidate_id: uuid.UUID
    retrieval_rank: int | None = None
    final_rank: int | None = None
    retrieval_score: float | None = None
    ranking_score: float | None = None
    matched_skills: list[str] = []
    explanation: dict[str, Any] = {}

    model_config = {"from_attributes": True}


class MatchResultsResponse(BaseModel):
    match_run_id: uuid.UUID
    job_id: uuid.UUID
    ranking_method: str
    total_candidates: int
    results: list[MatchResultOut]


# ---------- Evaluation ----------

class EvaluationRunRequest(BaseModel):
    job_ids: list[uuid.UUID] = []
    ranking_methods: list[str] = ["vector", "cross_encoder", "llm"]
    metrics: list[str] = []
    top_k: int = 20
    experiment_name: str | None = None


class ExperimentOut(BaseModel):
    id: uuid.UUID
    experiment_name: str
    configuration: dict[str, Any] = {}
    dataset_version: str | None = None
    metrics: dict[str, Any] = {}
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------- Health ----------

class HealthResponse(BaseModel):
    status: str
    postgres: str
    qdrant: str
    embedding_model: str
    embedding_loaded: bool
    cross_encoder_loaded: bool
