import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    livekit_url: str = Field(..., description="wss:// URL of your LiveKit project")
    livekit_api_key: str
    livekit_api_secret: str
    groq_api_key: str

    github_raw_url: str = (
        "https://raw.githubusercontent.com/tnahddisttud/sample-doc/"
        "refs/heads/main/AtliqAI_HR_Policies.pdf"
    )

    embedding_model: str = "all-MiniLM-L6-v2"
    qdrant_path: str = "/tmp/langchain_qdrant"
    collection_name: str = "docs"
    chunk_size: int = 50

    groq_model: str = "llama-3.3-70b-versatile"


settings = Settings()


os.environ.setdefault("LIVEKIT_URL", settings.livekit_url)
os.environ.setdefault("LIVEKIT_API_KEY", settings.livekit_api_key)
os.environ.setdefault("LIVEKIT_API_SECRET", settings.livekit_api_secret)
os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)