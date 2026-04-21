from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    provider: str
    ollama_base_url: str
    openai_base_url: str
    openai_api_key: str | None
    generator_model: str
    judge_model: str
    embedding_model: str
    workspace: Path

    @property
    def cache_dir(self) -> Path:
        return self.workspace / "cache"

    @property
    def output_dir(self) -> Path:
        return self.workspace / "output"

    def ensure_dirs(self) -> None:
        for path in (self.workspace, self.cache_dir, self.output_dir):
            path.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    load_dotenv(override=False)
    workspace = Path(os.getenv("DISTILLERY_WORKSPACE", "./distillery_workspace")).expanduser().resolve()
    provider = os.getenv("DISTILLERY_PROVIDER", "ollama").strip().lower()
    if provider not in {"ollama", "openai"}:
        raise ValueError(f"DISTILLERY_PROVIDER must be 'ollama' or 'openai', got '{provider}'")

    api_key = os.getenv("DISTILLERY_OPENAI_API_KEY") or None
    if api_key is not None and api_key.strip() == "":
        api_key = None

    settings = Settings(
        provider=provider,
        ollama_base_url=os.getenv("DISTILLERY_OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
        openai_base_url=os.getenv("DISTILLERY_OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        openai_api_key=api_key,
        generator_model=os.getenv("DISTILLERY_GENERATOR_MODEL", "llama3.1:8b"),
        judge_model=os.getenv("DISTILLERY_JUDGE_MODEL", "llama3.1:8b"),
        embedding_model=os.getenv("DISTILLERY_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        workspace=workspace,
    )
    settings.ensure_dirs()
    return settings
