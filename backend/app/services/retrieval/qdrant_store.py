"""Qdrant vector store: collection management, upsert, and similarity search."""

import logging
import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.core.config import get_settings
from app.services.embeddings import service as embedding_service

logger = logging.getLogger(__name__)
settings = get_settings()

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    return _client


def ensure_collection(model_name: str | None = None) -> str:
    client = get_client()
    name = embedding_service.collection_name(model_name)
    dim = embedding_service.dimension_for(model_name)
    try:
        client.get_collection(name)
    except Exception:
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        logger.info("Created Qdrant collection %s (dim=%d)", name, dim)
    return name


def delete_collection(model_name: str | None = None) -> None:
    client = get_client()
    name = embedding_service.collection_name(model_name)
    try:
        client.delete_collection(name)
    except Exception:
        pass


@dataclass
class RetrievedCandidate:
    candidate_id: uuid.UUID
    score: float
    section: str = "document"
    payload: dict | None = None


def index_resume(
    candidate_id: uuid.UUID,
    vector: list[float],
    payload: dict,
    model_name: str | None = None,
) -> None:
    client = get_client()
    collection = ensure_collection(model_name)
    client.upsert(
        collection_name=collection,
        points=[qm.PointStruct(id=str(candidate_id), vector=vector, payload=payload)],
        wait=True,
    )


def index_resume_sections(
    candidate_id: uuid.UUID,
    sections: list[tuple[str, list[float]]],
    base_payload: dict,
    model_name: str | None = None,
) -> None:
    """Index section-level embeddings. Point IDs are candidate_id#section."""
    client = get_client()
    collection = ensure_collection(model_name)
    points = []
    for section, vector in sections:
        pid = f"{candidate_id}#{section}"
        payload = dict(base_payload)
        payload["section"] = section
        points.append(qm.PointStruct(id=pid, vector=vector, payload=payload))
    if points:
        client.upsert(collection_name=collection, points=points, wait=True)


def delete_candidate(candidate_id: uuid.UUID, model_name: str | None = None) -> None:
    client = get_client()
    name = embedding_service.collection_name(model_name)
    try:
        client.delete(
            collection_name=name,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="candidate_id", match=qm.MatchValue(value=str(candidate_id)))]
                )
            ),
        )
    except Exception:
        logger.warning("Failed to delete vectors for %s from %s", candidate_id, name)


def search(
    query_vector: list[float],
    limit: int | None = None,
    model_name: str | None = None,
    section: str | None = None,
) -> list[RetrievedCandidate]:
    """Top-K cosine similarity search; deduplicates by candidate_id (best section wins)."""
    client = get_client()
    collection = ensure_collection(model_name)
    k = limit or settings.qdrant_search_limit
    # Fetch extra when section-level embeddings exist so dedup still yields k candidates.
    fetch_k = min(k * 4, 200)

    query_filter = None
    if section:
        query_filter = qm.Filter(
            must=[qm.FieldCondition(key="section", match=qm.MatchValue(value=section))]
        )

    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=fetch_k,
        query_filter=query_filter,
        with_payload=True,
    ).points

    best: dict[str, RetrievedCandidate] = {}
    for point in results:
        payload = point.payload or {}
        cid = payload.get("candidate_id") or str(point.id).split("#")[0]
        try:
            cid_uuid = uuid.UUID(cid)
        except (ValueError, AttributeError):
            continue
        score = float(point.score or 0.0)
        if cid not in best or score > best[cid].score:
            best[cid] = RetrievedCandidate(
                candidate_id=cid_uuid,
                score=score,
                section=payload.get("section", "document"),
                payload=payload,
            )
        if len(best) >= k:
            break
    return sorted(best.values(), key=lambda r: r.score, reverse=True)[:k]
