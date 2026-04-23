"""SQLite-backed cache for LLM calls.

Wrap any LLMProvider with `CachingProvider(inner, path)`. Keys are a SHA-256
hash of (model, temperature, max_tokens, system, prompt). Regenerating the same
dataset hits the cache and pays nothing.

Thread-safe: uses a single lock plus check_same_thread=False. The cache is
crash-safe: writes are committed immediately (SQLite default isolation_level
is autocommit when isolation_level=None isn't set, but we commit explicitly for
clarity).
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from pathlib import Path

from .llm import LLMProvider


log = logging.getLogger(__name__)


class LLMCache:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                temperature REAL NOT NULL,
                max_tokens INTEGER NOT NULL,
                prompt TEXT NOT NULL,
                system TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _key(
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        prompt: str,
        system: str,
    ) -> str:
        # Normalize float so "0.7" and "0.70" hash the same.
        temp_s = f"{float(temperature):.4f}"
        blob = f"{model}|{temp_s}|{int(max_tokens)}|{system}|{prompt}".encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def get(
        self,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        prompt: str,
        system: str,
    ) -> str | None:
        k = self._key(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt=prompt,
            system=system,
        )
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM llm_cache WHERE key = ?", (k,)
            ).fetchone()
        if row is not None:
            self._hits += 1
            return row[0]
        self._misses += 1
        return None

    def put(
        self,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        prompt: str,
        system: str,
        response: str,
    ) -> None:
        k = self._key(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt=prompt,
            system=system,
        )
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache
                    (key, model, temperature, max_tokens, prompt, system, response, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (k, model, float(temperature), int(max_tokens), prompt, system, response, time.time()),
            )
            self._conn.commit()

    def stats(self) -> dict[str, int]:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
        return {"hits": self._hits, "misses": self._misses, "entries": int(total)}

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class CachingProvider:
    """Decorator around an LLMProvider that memoizes deterministic calls.

    Only memoizes when temperature <= caching_threshold (default 0.0). Sampling
    calls above the threshold are never cached — caching a high-temperature
    response would defeat the point of sampling.
    """

    def __init__(
        self,
        inner: LLMProvider,
        cache: LLMCache,
        *,
        caching_threshold: float = 0.0,
        default_model: str = "unknown",
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.caching_threshold = float(caching_threshold)
        self.default_model = default_model

    def _resolve_model(self, model: str | None) -> str:
        if model:
            return model
        inner_default = getattr(self.inner, "default_model", None)
        return inner_default or self.default_model

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: str = "",
    ) -> str:
        resolved = self._resolve_model(model)
        cache_eligible = float(temperature) <= self.caching_threshold
        if cache_eligible:
            hit = self.cache.get(
                model=resolved,
                temperature=temperature,
                max_tokens=max_tokens,
                prompt=prompt,
                system=system,
            )
            if hit is not None:
                return hit
        response = self.inner.generate(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )
        if cache_eligible:
            self.cache.put(
                model=resolved,
                temperature=temperature,
                max_tokens=max_tokens,
                prompt=prompt,
                system=system,
                response=response,
            )
        return response

    async def agenerate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: str = "",
    ) -> str:
        import asyncio as _asyncio

        resolved = self._resolve_model(model)
        cache_eligible = float(temperature) <= self.caching_threshold
        if cache_eligible:
            hit = self.cache.get(
                model=resolved,
                temperature=temperature,
                max_tokens=max_tokens,
                prompt=prompt,
                system=system,
            )
            if hit is not None:
                return hit
        native = getattr(self.inner, "agenerate", None)
        if native is not None:
            response = await native(
                prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system=system,
            )
        else:
            response = await _asyncio.to_thread(
                self.inner.generate,
                prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system=system,
            )
        if cache_eligible:
            self.cache.put(
                model=resolved,
                temperature=temperature,
                max_tokens=max_tokens,
                prompt=prompt,
                system=system,
                response=response,
            )
        return response
