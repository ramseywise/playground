"""Eval-suite settings (LangFuse, datasets, LangSmith) — env-driven."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EvalSettings(BaseSettings):
    """Configuration for eval CLI, LangFuse, and LangSmith."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_dataset_name: str = "golden_eval"
    eval_dataset_path: str = ""
    langchain_project: str | None = Field(
        default=None, validation_alias="LANGCHAIN_PROJECT"
    )
    langchain_api_key: str | None = Field(
        default=None, validation_alias="LANGCHAIN_API_KEY"
    )
    langchain_endpoint: str | None = Field(
        default=None, validation_alias="LANGCHAIN_ENDPOINT"
    )


settings = EvalSettings()
