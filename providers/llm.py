from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Protocol, runtime_checkable

import requests

from ..config import Settings


log = logging.getLogger(__name__)


@runtime_checkable
class LLMProvider(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: str = "",
    ) -> str: ...


async def agenerate(
    provider: LLMProvider,
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    system: str = "",
) -> str:
    """Async wrapper for any LLMProvider.

    If the provider exposes `agenerate`, call it directly; otherwise offload the
    sync call to a worker thread. Network-bound — threads are fine.
    """
    native = getattr(provider, "agenerate", None)
    if native is not None:
        return await native(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )
    return await asyncio.to_thread(
        provider.generate,
        prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system=system,
    )


class OllamaProvider:
    def __init__(self, base_url: str, default_model: str, timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: str = "",
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
                r.raise_for_status()
                return str(r.json().get("response", "")).strip()
            except requests.RequestException as exc:
                last_exc = exc
                log.warning("ollama generate attempt %d failed: %s", attempt + 1, exc)
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"ollama generate failed after retries: {last_exc}")


class OpenAIProvider:
    """OpenAI-compatible chat completions. Works with OpenAI, Groq, Together, vLLM, LM Studio, etc."""

    def __init__(self, base_url: str, api_key: str | None, default_model: str, timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: str = "",
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                r.raise_for_status()
                data = r.json()
                choices = data.get("choices") or []
                if not choices:
                    return ""
                message = choices[0].get("message") or {}
                return str(message.get("content", "")).strip()
            except requests.RequestException as exc:
                last_exc = exc
                log.warning("openai-compat generate attempt %d failed: %s", attempt + 1, exc)
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"openai-compat generate failed after retries: {last_exc}")


def build_provider(settings: Settings) -> LLMProvider:
    if settings.provider == "ollama":
        return OllamaProvider(settings.ollama_base_url, settings.generator_model)
    if settings.provider == "openai":
        return OpenAIProvider(settings.openai_base_url, settings.openai_api_key, settings.generator_model)
    raise ValueError(f"Unknown provider: {settings.provider}")
