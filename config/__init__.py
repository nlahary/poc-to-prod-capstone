from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    API_VERSION: str = "v1"
    API_HOST: str = "localhost"
    API_PORT: int = 3000

    MODEL_CONFIG_PATH: str
    ARTEFACTS_PATH: str
    DATASET_PATH: str

    HOSTNAME: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file="./.dev.env",
        env_file_encoding="utf-8",
        extra="allow"   # Allow extra fields for the Airflow settings
    )

    def model_post_init(self, __context) -> None:
        """Post-initialization to validate adjust API_HOST wether we are running
        in a container or not. HOSTNAME is an env var only available in a container.
        """
        if self.HOSTNAME is None:
            self.API_HOST = "0.0.0.0"


settings = Settings()
