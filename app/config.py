"""Application configuration using Pydantic Settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: str

    # Database Configuration
    database_url: str

    # API Authentication
    api_keys: str  # Comma-separated list of valid API keys

    # Model Configuration
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    router_model: str = "gpt-3.5-turbo"

    # Document Configuration
    documents_path: str = "./documents"

    # RAG Configuration
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.7

    @property
    def api_keys_list(self) -> list[str]:
        """Parse comma-separated API keys into a list."""
        return [key.strip() for key in self.api_keys.split(",") if key.strip()]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
