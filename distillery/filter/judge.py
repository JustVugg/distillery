from __future__ import annotations

import logging

from ..providers.llm import LLMProvider, agenerate
from ..types import Example
from ..utils import safe_json_loads


log = logging.getLogger(__name__)


JUDGE_PROMPT = """You are grading an instruction/output pair for a training dataset.

Purpose of the assistant:
{description}

Instruction:
{instruction}

Output:
{output}

Grade on a 1-10 scale considering:
- relevance to purpose
- factual correctness (given any reference context below)
- specificity and helpfulness
- clarity and tone

{reference_block}

Return ONLY JSON: {{"score": <int>, "reason": "short sentence"}}
"""


WEAK_PROMPT = """You are deliberately producing a mediocre answer for pairwise preference data.

Instruction:
{instruction}

Produce an answer that is:
- superficially on-topic
- vague, generic, or slightly incorrect
- shorter than an ideal answer

Return ONLY the answer text — no preamble.
"""


def judge_example(
    provider: LLMProvider,
    example: Example,
    *,
    description: str,
    reference_text: str | None = None,
    model: str | None = None,
) -> tuple[int, str]:
    reference_block = ""
    if reference_text:
        reference_block = f"Reference context (authoritative):\n---\n{reference_text[:2000]}\n---"

    prompt = JUDGE_PROMPT.format(
        description=description,
        instruction=example.instruction,
        output=example.output,
        reference_block=reference_block,
    )
    try:
        raw = provider.generate(prompt, model=model, temperature=0.0, max_tokens=200)
    except RuntimeError as exc:
        log.warning("judge failed: %s", exc)
        return 0, f"judge error: {exc}"

    parsed = safe_json_loads(raw)
    if not isinstance(parsed, dict):
        return 0, "unparseable judge response"
    try:
        score = int(round(float(parsed.get("score", 0))))
    except (TypeError, ValueError):
        score = 0
    reason = str(parsed.get("reason", "")).strip()[:240]
    return max(0, min(10, score)), reason


def weak_answer(
    provider: LLMProvider,
    instruction: str,
    *,
    model: str | None = None,
) -> str:
    try:
        return provider.generate(
            WEAK_PROMPT.format(instruction=instruction),
            model=model,
            temperature=0.9,
            max_tokens=300,
        ).strip()
    except RuntimeError as exc:
        log.warning("weak_answer failed: %s", exc)
        return ""


def _parse_judge(raw: str) -> tuple[int, str]:
    parsed = safe_json_loads(raw)
    if not isinstance(parsed, dict):
        return 0, "unparseable judge response"
    try:
        score = int(round(float(parsed.get("score", 0))))
    except (TypeError, ValueError):
        score = 0
    reason = str(parsed.get("reason", "")).strip()[:240]
    return max(0, min(10, score)), reason


async def ajudge_example(
    provider: LLMProvider,
    example: Example,
    *,
    description: str,
    reference_text: str | None = None,
    model: str | None = None,
) -> tuple[int, str]:
    reference_block = ""
    if reference_text:
        reference_block = f"Reference context (authoritative):\n---\n{reference_text[:2000]}\n---"
    prompt = JUDGE_PROMPT.format(
        description=description,
        instruction=example.instruction,
        output=example.output,
        reference_block=reference_block,
    )
    try:
        raw = await agenerate(provider, prompt, model=model, temperature=0.0, max_tokens=200)
    except RuntimeError as exc:
        log.warning("ajudge failed: %s", exc)
        return 0, f"judge error: {exc}"
    return _parse_judge(raw)


async def aweak_answer(
    provider: LLMProvider,
    instruction: str,
    *,
    model: str | None = None,
) -> str:
    try:
        raw = await agenerate(
            provider,
            WEAK_PROMPT.format(instruction=instruction),
            model=model,
            temperature=0.9,
            max_tokens=300,
        )
        return raw.strip()
    except RuntimeError as exc:
        log.warning("aweak_answer failed: %s", exc)
        return ""
