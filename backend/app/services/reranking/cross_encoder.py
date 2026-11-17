"""Cross-Encoder re-ranking: pairwise (job description, resume) relevance scoring."""

import logging
import threading

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_lock = threading.Lock()
_model = None


def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import CrossEncoder

                logger.info("Loading cross-encoder: %s", settings.cross_encoder_model)
                _model = CrossEncoder(settings.cross_encoder_model, device=settings.embedding_device)
    return _model


def score_pairs(job_text: str, resume_texts: list[str], batch_size: int = 16) -> list[float]:
    """Return relevance scores for each (job, resume) pair."""
    if not resume_texts:
        return []
    model = get_model()
    pairs = [[job_text, text] for text in resume_texts]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return [float(s) for s in scores]
