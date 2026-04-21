from __future__ import annotations

import logging

from ..providers.llm import LLMProvider, agenerate
from ..types import Chunk
from ..utils import safe_json_loads


log = logging.getLogger(__name__)


SEED_FROM_CHUNK_PROMPT = """You are proposing instruction-tuning seeds grounded in a document chunk.

Chunk:
---
{chunk}
---

Goal: produce {n} diverse user instructions that a specialized assistant should be
able to answer USING ONLY the information in this chunk. Each instruction must be
answerable from the chunk alone — do not invent facts.

Language: {language}.

Return ONLY a JSON array of strings. No markdown, no prose.
"""


SEED_FROM_DESCRIPTION_PROMPT = """You are proposing instruction-tuning seeds for a specialized assistant.

Assistant purpose:
{description}

Generate {n} diverse, realistic user instructions spanning:
- direct questions
- multi-step tasks
- troubleshooting scenarios
- comparisons, summaries, plans

Language: {language}.

Return ONLY a JSON array of strings.
"""


def _parse_seeds(raw: str) -> list[str]:
    parsed = safe_json_loads(raw)
    if not isinstance(parsed, list):
        return []
    seeds: list[str] = []
    for item in parsed:
        if isinstance(item, str):
            cleaned = item.strip()
            if 8 <= len(cleaned) <= 500:
                seeds.append(cleaned)
        elif isinstance(item, dict):
            value = item.get("instruction") or item.get("question") or item.get("prompt")
            if isinstance(value, str):
                cleaned = value.strip()
                if 8 <= len(cleaned) <= 500:
                    seeds.append(cleaned)
    return seeds


def seed_from_chunk(
    provider: LLMProvider,
    chunk: Chunk,
    *,
    n: int = 6,
    language: str = "English",
    model: str | None = None,
) -> list[str]:
    prompt = SEED_FROM_CHUNK_PROMPT.format(chunk=chunk.text, n=n, language=language)
    try:
        raw = provider.generate(
            prompt, model=model, temperature=0.9, max_tokens=768
        )
    except RuntimeError as exc:
        log.warning("seed_from_chunk failed for %s: %s", chunk.id, exc)
        return []
    return _parse_seeds(raw)


def seed_from_description(
    provider: LLMProvider,
    description: str,
    *,
    n: int = 20,
    language: str = "English",
    model: str | None = None,
) -> list[str]:
    prompt = SEED_FROM_DESCRIPTION_PROMPT.format(description=description, n=n, language=language)
    try:
        raw = provider.generate(
            prompt, model=model, temperature=0.95, max_tokens=1200
        )
    except RuntimeError as exc:
        log.warning("seed_from_description failed: %s", exc)
        return []
    return _parse_seeds(raw)


async def aseed_from_chunk(
    provider: LLMProvider,
    chunk: Chunk,
    *,
    n: int = 6,
    language: str = "English",
    model: str | None = None,
) -> list[str]:
    prompt = SEED_FROM_CHUNK_PROMPT.format(chunk=chunk.text, n=n, language=language)
    try:
        raw = await agenerate(provider, prompt, model=model, temperature=0.9, max_tokens=768)
    except RuntimeError as exc:
        log.warning("aseed_from_chunk failed for %s: %s", chunk.id, exc)
        return []
    return _parse_seeds(raw)


async def aseed_from_description(
    provider: LLMProvider,
    description: str,
    *,
    n: int = 20,
    language: str = "English",
    model: str | None = None,
) -> list[str]:
    prompt = SEED_FROM_DESCRIPTION_PROMPT.format(description=description, n=n, language=language)
    try:
        raw = await agenerate(provider, prompt, model=model, temperature=0.95, max_tokens=1200)
    except RuntimeError as exc:
        log.warning("aseed_from_description failed: %s", exc)
        return []
    return _parse_seeds(raw)
