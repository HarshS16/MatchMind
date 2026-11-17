import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Candidate(Base):
    __tablename__ = "candidates"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    resume_text: Mapped[str] = mapped_column(Text, default="")
    skills: Mapped[list] = mapped_column(JSONB, default=list)
    experience: Mapped[dict] = mapped_column(JSONB, default=dict)
    education: Mapped[dict] = mapped_column(JSONB, default=dict)
    file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    original_filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    processing_status: Mapped[str] = mapped_column(String(50), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    match_results: Mapped[list["MatchResult"]] = relationship(back_populates="candidate")
    labels: Mapped[list["EvaluationLabel"]] = relationship(back_populates="candidate")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text)
    required_skills: Mapped[list] = mapped_column(JSONB, default=list)
    preferred_skills: Mapped[list] = mapped_column(JSONB, default=list)
    responsibilities: Mapped[str | None] = mapped_column(Text, nullable=True)
    experience_min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    experience_max: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    match_runs: Mapped[list["MatchRun"]] = relationship(back_populates="job")
    labels: Mapped[list["EvaluationLabel"]] = relationship(back_populates="job")


class MatchRun(Base):
    __tablename__ = "match_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("jobs.id"))
    retrieval_model: Mapped[str] = mapped_column(String(255), default="")
    ranking_model: Mapped[str] = mapped_column(String(255), default="vector")
    top_k: Mapped[int] = mapped_column(Integer, default=20)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    job: Mapped[Job] = relationship(back_populates="match_runs")
    results: Mapped[list["MatchResult"]] = relationship(back_populates="match_run")


class MatchResult(Base):
    __tablename__ = "match_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    match_run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("match_runs.id"))
    candidate_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("candidates.id"))
    retrieval_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ranking_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    retrieval_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    final_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    explanation: Mapped[dict] = mapped_column(JSONB, default=dict)

    match_run: Mapped[MatchRun] = relationship(back_populates="results")
    candidate: Mapped[Candidate] = relationship(back_populates="match_results")


class EvaluationLabel(Base):
    __tablename__ = "evaluation_labels"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("jobs.id"))
    candidate_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("candidates.id"))
    relevance_label: Mapped[int] = mapped_column(Integer)
    reviewer_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    job: Mapped[Job] = relationship(back_populates="labels")
    candidate: Mapped[Candidate] = relationship(back_populates="labels")


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_name: Mapped[str] = mapped_column(String(255))
    configuration: Mapped[dict] = mapped_column(JSONB, default=dict)
    dataset_version: Mapped[str | None] = mapped_column(String(255), nullable=True)
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
