from __future__ import annotations

import logging

from ..providers.llm import LLMProvider, agenerate
from ..types import Chunk, Example


log = logging.getLogger(__name__)


ANSWER_PROMPT_GROUNDED = """You are a specialized assistant.

Reference material (authoritative — do not contradict it):
---
{context}
---

User instruction:
{instruction}

Answer in {language}. Be specific, actionable, and grounded in the reference.
If the reference does not contain the answer, say so briefly instead of inventing.
Do not restate the question. Reply with the answer only.
"""


ANSWER_PROMPT_FREE = """You are a specialized assistant.

Assistant purpose:
{description}

User instruction:
{instruction}

Answer in {language}. Be specific, actionable, and concise.
Reply with the answer only — do not restate the question.
"""


def _answer_grounded(
    provider: LLMProvider,
    instruction: str,
    context: str,
    *,
    language: str,
    model: str | None,
    temperature: float,
    max_tokens: int,
) -> str:
    prompt = ANSWER_PROMPT_GROUNDED.format(context=context, instruction=instruction, language=language)
    return provider.generate(prompt, model=model, temperature=temperature, max_tokens=max_tokens)


def _answer_free(
    provider: LLMProvider,
    instruction: str,
    description: str,
    *,
    language: str,
    model: str | None,
    temperature: float,
    max_tokens: int,
) -> str:
    prompt = ANSWER_PROMPT_FREE.format(description=description, instruction=instruction, language=language)
    return provider.generate(prompt, model=model, temperature=temperature, max_tokens=max_tokens)


def expand_seeds(
    provider: LLMProvider,
    seeds: list[tuple[str, Chunk | None]],
    *,
    description: str,
    language: str = "English",
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 800,
) -> list[Example]:
    """Turn a list of (instruction, chunk_or_None) pairs into grounded examples."""
    out: list[Example] = []
    for instruction, chunk in seeds:
        try:
            if chunk is not None:
                answer = _answer_grounded(
                    provider,
                    instruction,
                    chunk.text,
                    language=language,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                source_chunks = [chunk.id]
            else:
                answer = _answer_free(
                    provider,
                    instruction,
                    description,
                    language=language,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                source_chunks = []
        except RuntimeError as exc:
            log.warning("answer generation failed for seed: %s", exc)
            continue

        answer = answer.strip()
        if len(answer) < 12:
            continue
        out.append(
            Example(
                instruction=instruction.strip(),
                output=answer,
                source_chunks=source_chunks,
                format="sft",
            )
        )
    return out


async def aexpand_one(
    provider: LLMProvider,
    instruction: str,
    chunk: Chunk | None,
    *,
    description: str,
    language: str = "English",
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 800,
) -> Example | None:
    """Async single-seed expansion. Returns None if the generation fails or is empty."""
    if chunk is not None:
        prompt = ANSWER_PROMPT_GROUNDED.format(
            context=chunk.text, instruction=instruction, language=language
        )
        source_chunks = [chunk.id]
    else:
        prompt = ANSWER_PROMPT_FREE.format(
            description=description, instruction=instruction, language=language
        )
        source_chunks = []
    try:
        answer = await agenerate(
            provider, prompt, model=model, temperature=temperature, max_tokens=max_tokens
        )
    except RuntimeError as exc:
        log.warning("aexpand_one failed: %s", exc)
        return None
    answer = (answer or "").strip()
    if len(answer) < 12:
        return None
    return Example(
        instruction=instruction.strip(),
        output=answer,
        source_chunks=source_chunks,
        format="sft",
    )
