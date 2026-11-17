"""Sentence-Transformer embedding service with lazy model loading and caching."""

import logging
import threading
from functools import lru_cache

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_lock = threading.Lock()
_models: dict[str, object] = {}


def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name, device=settings.embedding_device)
    model.max_seq_length = 512
    return model


def get_model(model_name: str | None = None):
    name = model_name or settings.embedding_model
    if name not in _models:
        with _lock:
            if name not in _models:
                _models[name] = _load_model(name)
    return _models[name]


def model_version(model_name: str | None = None) -> str:
    name = model_name or settings.embedding_model
    model = get_model(name)
    return getattr(model, "model_card_data", None) and name or name


def encode(texts: list[str], model_name: str | None = None, normalize: bool = True) -> np.ndarray:
    model = get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def encode_single(text: str, model_name: str | None = None) -> list[float]:
    vec = encode([text], model_name=model_name)
    return vec[0].tolist()


def collection_name(model_name: str | None = None) -> str:
    """Versioned Qdrant collection name per embedding model to avoid mixing."""
    name = (model_name or settings.embedding_model).lower().replace("/", "_").replace(".", "-")
    return f"{settings.qdrant_collection_prefix}_{name}"


def dimension_for(model_name: str | None = None) -> int:
    if model_name and "minilm" in model_name.lower():
        return 384
    return settings.embedding_dimension
