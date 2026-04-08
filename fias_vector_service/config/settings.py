"""Application settings."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "fias_vector"
    postgres_user: str = "fias_user"
    postgres_password: str = "fias_password"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None
    qdrant_collection: str = "fias_addresses"
    qdrant_vector_size: int = 1024  # USER-bge-m3

    # Embedding Model
    embedding_model: str = "deepvk/USER-bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32

    # Similarity thresholds
    similarity_high: float = 0.95
    similarity_medium: float = 0.82
    similarity_low: float = 0.75

    # FIAS
    fias_schema: str = "fias"

    # App
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
