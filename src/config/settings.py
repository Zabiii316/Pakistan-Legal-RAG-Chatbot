from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Pakistan Legal RAG Chatbot"
    app_env: str = "development"
    debug: bool = True
    host: str = "127.0.0.1"
    port: int = 8000

    anthropic_api_key: str = ""
    model_name: str = "claude-3-5-sonnet-latest"

    data_dir: str = "data"
    raw_docs_dir: str = "data/raw"
    index_dir: str = "data/index"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()