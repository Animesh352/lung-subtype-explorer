from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "lung-transcriptomics-api"
    debug: bool = False
    database_url: str = "postgresql://lung_user:lung_password@localhost:5432/lung_transcriptomics"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
