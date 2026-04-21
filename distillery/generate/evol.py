"""Evol-Instruct: iteratively deepen / constrain / complicate seed instructions.

Based on the recipe from WizardLM (Xu et al. 2023). Given a seed, we sample one
of several mutation operators (deepen, add constraint, concretize, reason step,
broaden) and ask the provider to rewrite the seed under that mutation. The
output is a single instruction string, which the pipeline then answers as usual.

A good dataset usually mixes original seeds with 1-2 evolved variants per seed
to improve difficulty distribution without drifting off-topic.
"""
from __future__ import annotations

import logging
import random

from ..providers.llm import LLMProvider, agenerate


log = logging.getLogger(__name__)


MUTATIONS = {
    "deepen": (
        "Rewrite the instruction so it asks for a DEEPER, more thorough answer. "
        "Require the rewritten task to demand reasoning about underlying mechanisms, "
        "trade-offs, or edge cases. Keep the topic unchanged."
    ),
    "constrain": (
        "Rewrite the instruction by adding ONE realistic constraint (a deadline, a "
        "resource limit, a required format, a forbidden approach, or a user persona). "
        "Keep the topic unchanged."
    ),
    "concretize": (
        "Rewrite the instruction so it refers to a SPECIFIC, plausible scenario with "
        "concrete details (numbers, names, dates, or environment). Keep the topic unchanged."
    ),
    "reason": (
        "Rewrite the instruction so the answer requires a multi-step reasoning plan, "
        "explicitly asking for a step-by-step approach. Keep the topic unchanged."
    ),
    "broaden": (
        "Rewrite the instruction so it asks to compare or contrast with a related "
        "situation, widening the scope slightly. Keep the topic unchanged."
    ),
    "troubleshoot": (
        "Rewrite the instruction as a troubleshooting scenario: something has gone "
        "wrong, and the user needs to diagnose and fix it. Keep the topic unchanged."
    ),
}


EVOL_PROMPT = """You are an instruction rewriter for an AI training dataset.

Original instruction:
{seed}

Mutation to apply:
{mutation}

Rules:
- Produce ONE rewritten instruction, in {language}.
- Keep it plausible and actionable — do not invent fantasy constraints.
- Do NOT answer the instruction.
- Do NOT add any preamble, markdown, or quotes.
- Output ONLY the rewritten instruction, nothing else.
"""


def pick_mutation(rng: random.Random | None = None) -> str:
    rng = rng or random.Random()
    return rng.choice(list(MUTATIONS.keys()))


def evolve_seed(
    provider: LLMProvider,
    seed: str,
    *,
    mutation: str | None = None,
    language: str = "English",
    model: str | None = None,
    rng: random.Random | None = None,
) -> str | None:
    name = mutation or pick_mutation(rng)
    spec = MUTATIONS.get(name)
    if spec is None:
        return None
    prompt = EVOL_PROMPT.format(seed=seed, mutation=spec, language=language)
    try:
        raw = provider.generate(prompt, model=model, temperature=0.8, max_tokens=400)
    except RuntimeError as exc:
        log.warning("evolve_seed failed: %s", exc)
        return None
    return _clean_evolution(raw, seed)


async def aevolve_seed(
    provider: LLMProvider,
    seed: str,
    *,
    mutation: str | None = None,
    language: str = "English",
    model: str | None = None,
    rng: random.Random | None = None,
) -> str | None:
    name = mutation or pick_mutation(rng)
    spec = MUTATIONS.get(name)
    if spec is None:
        return None
    prompt = EVOL_PROMPT.format(seed=seed, mutation=spec, language=language)
    try:
        raw = await agenerate(provider, prompt, model=model, temperature=0.8, max_tokens=400)
    except RuntimeError as exc:
        log.warning("aevolve_seed failed: %s", exc)
        return None
    return _clean_evolution(raw, seed)


def _clean_evolution(raw: str, original: str) -> str | None:
    text = (raw or "").strip()
    # Strip wrapping quotes or "Rewritten:" prefixes a model sometimes adds.
    for prefix in ("Rewritten instruction:", "Rewritten:", "Instruction:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    text = text.strip('"\'`')
    if len(text) < 8 or len(text) > 800:
        return None
    if text.lower() == original.strip().lower():
        return None
    return text
