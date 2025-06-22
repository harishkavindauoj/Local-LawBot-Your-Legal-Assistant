import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Application configuration using Pydantic settings."""

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # Gemini Configuration
    gemini_api_key: str
    gemini_model: str = "gemini-1.5-pro"

    # HuggingFace Configuration (needed for legal datasets)
    huggingface_token: Optional[str] = None

    # Vector Database Configuration
    vector_db_type: str = "chromadb"
    chromadb_persist_dir: Path = Path("./data/chromadb")
    embedding_model: str = "all-MiniLM-L6-v2"

    # Dataset Configuration - Fixed field names to match LegalDataLoader
    legal_dataset: str
    legal_dataset_config: Optional[str]
    chunk_size: int   # Increased for better legal document coverage
    chunk_overlap: int
    max_documents: int  # More reasonable for legal RAG

    # Multiple dataset configuration support
    use_multiple_configs: bool = True
    multiple_configs: str = "federal_register,cfr,uscode"

    # RAG Configuration
    top_k_retrieval: int = 5
    max_context_length: int = 2048
    temperature: float = 0.3

    # Cache Settings
    enable_caching: bool = True
    cache_directory: Path = Path("./data/processed")

    # Logging
    log_level: str = "INFO"

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "env_file_encoding": "utf-8"
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set HuggingFace token for datasets library if provided
        if self.huggingface_token:
            os.environ["HF_TOKEN"] = self.huggingface_token
            os.environ["HUGGINGFACE_HUB_TOKEN"] = self.huggingface_token

    @property
    def multiple_configs_list(self) -> List[str]:
        """Convert comma-separated configs to list."""
        if not self.multiple_configs:
            return []
        return [config.strip() for config in self.multiple_configs.split(",")]


# Global config instance
config = Config()

# Ensure data directories exist
config.chromadb_persist_dir.mkdir(parents=True, exist_ok=True)
config.cache_directory.mkdir(parents=True, exist_ok=True)

# Set up logging level
import logging

logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))