"""Embedding service for generating address vectors."""
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import get_settings
from config.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class EmbeddingService:
    """Service for generating text embeddings."""

    _instance = None
    _model: SentenceTransformer | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load_model()

    def _load_model(self) -> None:
        """Load the embedding model."""
        logger.info("loading_embedding_model", model=settings.embedding_model)
        self._model = SentenceTransformer(
            settings.embedding_model,
            device=settings.embedding_device,
        )
        logger.info("embedding_model_loaded", 
                   model=settings.embedding_model,
                   vector_size=self._model.get_sentence_embedding_dimension())

    def encode(self, texts: str | List[str]) -> np.ndarray:
        """Generate embeddings for text(s)."""
        if isinstance(texts, str):
            texts = [texts]

        # Add task prefix for E5 models
        if "e5" in settings.embedding_model.lower():
            texts = [f"query: {t}" for t in texts]

        embeddings = self._model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def encode_single(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        embedding = self.encode([text])[0]
        return embedding.tolist()

    @property
    def vector_size(self) -> int:
        """Get embedding vector size."""
        return self._model.get_sentence_embedding_dimension()


# Global instance
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
