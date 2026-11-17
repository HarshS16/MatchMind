from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Intelligent Job Matching System"
    environment: str = "local"
    secret_key: str = "change-me"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    postgres_user: str = "elip"
    postgres_password: str = "elip"
    postgres_db: str = "elip"
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    database_url: str = "postgresql+asyncpg://elip:elip@postgres:5432/elip"

    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection_prefix: str = "resume_embeddings"
    qdrant_search_limit: int = 20

    embedding_model: str = "BAAI/bge-base-en-v1.5"
    baseline_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 768
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    hf_home: str = "/models/huggingface"

    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    llm_provider: str = "ollama"
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "llama3.1:8b"
    llm_prompt_version: str = "v1"
    llm_temperature: float = 0.0
    llm_timeout_seconds: int = 120

    resume_storage_path: str = "/data/resumes"
    max_upload_size_mb: int = 20
    allowed_extensions: str = ".pdf,.docx"

    evaluation_top_k: int = 20
    eval_metrics: str = "precision@5,recall@5,ndcg@10,mrr"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_extension_list(self) -> list[str]:
        return [e.strip().lower() for e in self.allowed_extensions.split(",")]

    @property
    def eval_metric_list(self) -> list[str]:
        return [m.strip() for m in self.eval_metrics.split(",") if m.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
