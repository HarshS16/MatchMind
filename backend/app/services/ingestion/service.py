"""Resume ingestion: validation, storage, parsing, preprocessing, embedding, indexing."""

import hashlib
import logging
import os
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.tables import Candidate
from app.services.embeddings import service as embedding_service
from app.services.ingestion.parser import UnsupportedFileError, extract_resume_text
from app.services.preprocessing.text import (
    clean_text,
    extract_experience_years,
    extract_name,
    extract_skills,
    pseudonymize,
    split_sections,
)
from app.services.retrieval import qdrant_store

logger = logging.getLogger(__name__)
settings = get_settings()


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in settings.allowed_extension_list


async def ingest_resume(
    db: AsyncSession,
    filename: str,
    data: bytes,
    embedding_model: str | None = None,
    section_level: bool = False,
) -> Candidate:
    """Full ingestion pipeline for one resume file."""
    if not allowed_file(filename):
        raise UnsupportedFileError(f"File type not allowed: {filename}")
    if len(data) > settings.max_upload_size_mb * 1024 * 1024:
        raise ValueError(f"File exceeds {settings.max_upload_size_mb}MB limit")

    candidate = Candidate(
        original_filename=filename,
        processing_status="processing",
    )
    db.add(candidate)
    await db.flush()

    try:
        raw_text = extract_resume_text(filename, data)
        text = clean_text(pseudonymize(raw_text))
        skills = extract_skills(text)

        # Persist original file
        storage_dir = Path(settings.resume_storage_path)
        storage_dir.mkdir(parents=True, exist_ok=True)
        file_path = storage_dir / f"{candidate.id}{Path(filename).suffix.lower()}"
        file_path.write_bytes(data)

        candidate.resume_text = text
        candidate.candidate_name = extract_name(text)
        candidate.skills = skills
        candidate.experience = {"years": extract_experience_years(text)}
        candidate.education = {}
        candidate.file_path = str(file_path)
        candidate.processing_status = "processed"

        # Embedding + indexing
        model = embedding_model or settings.embedding_model
        if section_level:
            chunks = split_sections(text)
            chunk_vectors = embedding_service.encode([c.text for c in chunks], model_name=model)
            sections = [(c.section, vec.tolist()) for c, vec in zip(chunks, chunk_vectors)]
            qdrant_store.index_resume_sections(
                candidate.id,
                sections,
                {"candidate_id": str(candidate.id), "skills": skills, "embedding_model": model},
                model_name=model,
            )
        else:
            vector = embedding_service.encode_single(text[:8000], model_name=model)
            qdrant_store.index_resume(
                candidate.id,
                vector,
                {
                    "candidate_id": str(candidate.id),
                    "skills": skills,
                    "experience_years": candidate.experience.get("years"),
                    "embedding_model": model,
                    "section": "document",
                },
                model_name=model,
            )

        await db.commit()
        await db.refresh(candidate)
        return candidate

    except Exception as exc:
        logger.exception("Ingestion failed for %s", filename)
        candidate.processing_status = "failed"
        await db.commit()
        raise


async def bulk_ingest_directory(
    db: AsyncSession,
    directory: str,
    embedding_model: str | None = None,
) -> tuple[int, int]:
    """Ingest all PDF/DOCX files in a directory. Returns (succeeded, failed)."""
    ok = fail = 0
    for path in sorted(Path(directory).iterdir()):
        if path.suffix.lower() not in settings.allowed_extension_list:
            continue
        try:
            await ingest_resume(db, path.name, path.read_bytes(), embedding_model=embedding_model)
            ok += 1
        except Exception:
            logger.exception("Bulk ingest failed for %s", path)
            fail += 1
    return ok, fail
