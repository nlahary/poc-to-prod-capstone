from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os
import logging


class Settings(BaseSettings):
    API_VERSION: str = "v1"
    API_HOST: str = "localhost"
    API_PORT: int = 3000

    MODEL_CONFIG_PATH: str
    ARTEFACTS_PATH: str
    DATASET_PATH: str
    # If no path is provided, download the model from Hugging Face
    BERT_MODEL_PATH: Optional[str] = None

    HOSTNAME: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file="./.dev.env",
        env_file_encoding="utf-8",
    )

    def model_post_init(self, __context) -> None:

        if self.HOSTNAME is not None:
            self.API_HOST = "0.0.0.0"

        if not self.BERT_MODEL_PATH or not os.path.exists(self.BERT_MODEL_PATH):
            logging.warning(
                f"BERT_MODEL_PATH ({self.BERT_MODEL_PATH}) not provided or does not exist . The model will be downloaded from Hugging Face."
            )
            self.BERT_MODEL_PATH = 'bert-base-uncased'
        else:
            snapshots_path = os.path.join(self.BERT_MODEL_PATH, "snapshots")
            if os.path.exists(snapshots_path) and os.path.isdir(snapshots_path):
                snapshots = os.listdir(snapshots_path)
                if snapshots:
                    logging.info(
                        f"Snapshots found in {snapshots_path}. Loading the latest snapshot.")
                    self.BERT_MODEL_PATH = os.path.join(
                        snapshots_path, sorted(snapshots)[-1])
                else:
                    logging.warning(
                        f"No snapshots directory found in {snapshots_path}. Please provide a valid path.")
            else:
                logging.warning(
                    f"Snapshots path {snapshots_path} does not exist.")


settings = Settings()
