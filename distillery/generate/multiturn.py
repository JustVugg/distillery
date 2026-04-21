"""Multi-turn dialogue generation.

Given a chunk (or just a description), produce a 2-3 turn conversation in which
a user asks a question, the assistant answers grounded in the chunk, the user
follows up, and the assistant answers again. The output is a sequence of
messages in OpenAI/ChatML-style format.

We enforce the JSON shape and limit the number of turns so dialogues stay short
enough to fit context during fine-tuning.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..providers.llm import LLMProvider, agenerate
from ..types import Chunk
from ..utils import safe_json_loads


log = logging.getLogger(__name__)


MULTITURN_GROUNDED_PROMPT = """You are generating a multi-turn dialogue for training a specialized assistant.

Reference material (authoritative):
---
{context}
---

Produce a realistic conversation of {turns} turns total, alternating user and assistant.
The user asks a question, the assistant answers using ONLY the reference. The user
then asks a natural follow-up, the assistant answers it, and so on.

Rules:
- Language: {language}.
- First turn must be the user; the last turn must be the assistant.
- Every assistant message must be grounded in the reference.
- No apologies, no meta commentary, no restatement of the question.

Return ONLY JSON in this shape:
{{"messages": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}
"""


MULTITURN_FREE_PROMPT = """You are generating a multi-turn dialogue for training a specialized assistant.

Assistant purpose:
{description}

Produce a realistic conversation of {turns} turns total, alternating user and assistant.

Rules:
- Language: {language}.
- First turn must be the user; the last turn must be the assistant.
- Stay on-topic for the assistant's stated purpose.
- No apologies, no meta commentary.

Return ONLY JSON in this shape:
{{"messages": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}
"""


@dataclass
class Dialogue:
    messages: list[dict[str, str]]
    source_chunks: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_openai_jsonl(self, system_prompt: str | None = None) -> dict:
        payload: list[dict[str, str]] = []
        if system_prompt:
            payload.append({"role": "system", "content": system_prompt})
        payload.extend(self.messages)
        out: dict[str, Any] = {"messages": payload}
        if self.source_chunks:
            out["source_chunks"] = list(self.source_chunks)
        return out


def _parse_dialogue(raw: str, *, turns: int) -> list[dict[str, str]] | None:
    parsed = safe_json_loads(raw)
    if not isinstance(parsed, dict):
        return None
    messages = parsed.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    cleaned: list[dict[str, str]] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return None
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "")).strip()
        if not content:
            return None
        expected = "user" if i % 2 == 0 else "assistant"
        if role != expected:
            return None
        cleaned.append({"role": role, "content": content})
    if cleaned[-1]["role"] != "assistant":
        return None
    # Clamp to requested turn count if the model overshot.
    return cleaned[:turns]


def generate_dialogue(
    provider: LLMProvider,
    *,
    chunk: Chunk | None,
    description: str,
    turns: int = 4,
    language: str = "English",
    model: str | None = None,
    temperature: float = 0.8,
    max_tokens: int = 1200,
) -> Dialogue | None:
    turns = max(2, min(turns, 8))
    if turns % 2 != 0:
        turns += 1  # force even so it ends on assistant
    prompt = (
        MULTITURN_GROUNDED_PROMPT.format(context=chunk.text, turns=turns, language=language)
        if chunk is not None
        else MULTITURN_FREE_PROMPT.format(description=description, turns=turns, language=language)
    )
    try:
        raw = provider.generate(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
    except RuntimeError as exc:
        log.warning("generate_dialogue failed: %s", exc)
        return None
    messages = _parse_dialogue(raw, turns=turns)
    if not messages:
        return None
    return Dialogue(
        messages=messages,
        source_chunks=[chunk.id] if chunk is not None else [],
    )


async def agenerate_dialogue(
    provider: LLMProvider,
    *,
    chunk: Chunk | None,
    description: str,
    turns: int = 4,
    language: str = "English",
    model: str | None = None,
    temperature: float = 0.8,
    max_tokens: int = 1200,
) -> Dialogue | None:
    turns = max(2, min(turns, 8))
    if turns % 2 != 0:
        turns += 1
    prompt = (
        MULTITURN_GROUNDED_PROMPT.format(context=chunk.text, turns=turns, language=language)
        if chunk is not None
        else MULTITURN_FREE_PROMPT.format(description=description, turns=turns, language=language)
    )
    try:
        raw = await agenerate(
            provider, prompt, model=model, temperature=temperature, max_tokens=max_tokens
        )
    except RuntimeError as exc:
        log.warning("agenerate_dialogue failed: %s", exc)
        return None
    messages = _parse_dialogue(raw, turns=turns)
    if not messages:
        return None
    return Dialogue(
        messages=messages,
        source_chunks=[chunk.id] if chunk is not None else [],
    )
