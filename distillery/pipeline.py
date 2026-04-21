from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .checkpoint import Checkpoint, compute_signature
from .filter.diversity import DiversityFilter
from .filter.hallucination import grounded_ok
from .filter.judge import ajudge_example, aweak_answer, judge_example, weak_answer
from .generate.evol import aevolve_seed, evolve_seed, pick_mutation
from .generate.expand import aexpand_one, expand_seeds
from .generate.multiturn import Dialogue, agenerate_dialogue, generate_dialogue
from .generate.seed import (
    aseed_from_chunk,
    aseed_from_description,
    seed_from_chunk,
    seed_from_description,
)
from .providers.embeddings import EmbeddingBackend
from .providers.llm import LLMProvider
from .types import Chunk, Example, PipelineStats, PreferencePair


log = logging.getLogger(__name__)


ProgressFn = Callable[[float, str], None]


@dataclass
class PipelineConfig:
    description: str
    language: str = "English"
    target_examples: int = 200
    seeds_per_chunk: int = 6
    seeds_from_description: int = 40
    diversity_threshold: float = 0.9
    min_judge_score: int = 7
    min_hallucination_overlap: float = 0.35
    min_semantic_similarity: float | None = None
    generator_model: str | None = None
    judge_model: str | None = None
    generator_temperature: float = 0.7
    generate_dpo_pairs: bool = False
    dpo_target_pairs: int = 50
    answer_max_tokens: int = 800
    seed: int = 1234
    concurrency: int = 1
    evol_factor: float = 0.0             # 0.0 = no evolution, 1.0 = one evolved per seed
    evol_rounds: int = 1                 # how many mutations per seed when evol_factor > 0
    checkpoint_path: Path | None = None  # when set, pipeline resumes on re-run


@dataclass
class PipelineOutput:
    examples: list[Example] = field(default_factory=list)
    rejected: list[Example] = field(default_factory=list)
    dpo_pairs: list[PreferencePair] = field(default_factory=list)
    stats: PipelineStats = field(default_factory=PipelineStats)


@dataclass
class MultiturnConfig:
    description: str
    language: str = "English"
    target_dialogues: int = 100
    turns: int = 4
    model: str | None = None
    temperature: float = 0.8
    max_tokens: int = 1200
    concurrency: int = 1
    seed: int = 1234


@dataclass
class MultiturnOutput:
    dialogues: list[Dialogue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _noop(_f: float, _m: str) -> None:
    return None


def run_pipeline(
    *,
    config: PipelineConfig,
    chunks: Iterable[Chunk],
    provider: LLMProvider,
    embedder: EmbeddingBackend,
    judge_provider: LLMProvider | None = None,
    progress: ProgressFn | None = None,
) -> PipelineOutput:
    """Synchronous public API. Internally dispatches to async when concurrency > 1."""
    if config.concurrency > 1:
        return asyncio.run(
            arun_pipeline(
                config=config,
                chunks=chunks,
                provider=provider,
                embedder=embedder,
                judge_provider=judge_provider,
                progress=progress,
            )
        )
    return _run_pipeline_sync(
        config=config,
        chunks=chunks,
        provider=provider,
        embedder=embedder,
        judge_provider=judge_provider,
        progress=progress,
    )


def _run_pipeline_sync(
    *,
    config: PipelineConfig,
    chunks: Iterable[Chunk],
    provider: LLMProvider,
    embedder: EmbeddingBackend,
    judge_provider: LLMProvider | None = None,
    progress: ProgressFn | None = None,
) -> PipelineOutput:
    progress = progress or _noop
    judge_provider = judge_provider or provider

    stats = PipelineStats()
    started = time.time()
    rng = random.Random(config.seed)

    chunk_list = list(chunks)
    by_id: dict[str, Chunk] = {c.id: c for c in chunk_list}

    checkpoint = _open_checkpoint(config, chunk_list)

    progress(0.02, f"collecting seeds from {len(chunk_list)} chunks")
    seeds: list[tuple[str, Chunk | None]] = []

    for chunk in chunk_list:
        items = seed_from_chunk(
            provider,
            chunk,
            n=config.seeds_per_chunk,
            language=config.language,
            model=config.generator_model,
        )
        for item in items:
            seeds.append((item, chunk))
        stats.seeds += len(items)

    if config.seeds_from_description > 0:
        free_seeds = seed_from_description(
            provider,
            config.description,
            n=config.seeds_from_description,
            language=config.language,
            model=config.generator_model,
        )
        for item in free_seeds:
            seeds.append((item, None))
        stats.seeds += len(free_seeds)

    if config.evol_factor > 0 and seeds:
        seeds = _apply_evol_sync(
            seeds, provider, config, rng
        )
        stats.seeds = len(seeds)

    if not seeds:
        log.warning("no seeds produced — check provider connectivity or model availability")
        stats.elapsed_sec = time.time() - started
        return PipelineOutput(stats=stats)

    rng.shuffle(seeds)
    progress(0.15, f"generating answers for {len(seeds)} seeds")

    diversity = DiversityFilter(embedder, threshold=config.diversity_threshold)
    kept: list[Example] = []
    rejected: list[Example] = []

    if checkpoint is not None:
        kept = list(checkpoint.kept)
        rejected = list(checkpoint.rejected)
        for ex in kept + rejected:
            diversity.accept(ex.instruction)  # reseed the filter with prior keeps

    candidates = expand_seeds(
        provider,
        seeds,
        description=config.description,
        language=config.language,
        model=config.generator_model,
        temperature=config.generator_temperature,
        max_tokens=config.answer_max_tokens,
    )
    stats.generated = len(candidates)

    for idx, example in enumerate(candidates):
        if len(kept) >= config.target_examples:
            break
        if checkpoint is not None and checkpoint.is_processed(idx):
            continue

        decision = _process_one(
            example,
            by_id=by_id,
            diversity=diversity,
            config=config,
            judge_provider=judge_provider,
            embedder=embedder,
            stats=stats,
        )
        if decision == "kept":
            kept.append(example)
        elif decision == "rejected":
            rejected.append(example)
        if checkpoint is not None:
            checkpoint.record_seed(idx, decision, example)

        progress(
            0.15 + 0.7 * min(1.0, len(kept) / max(1, config.target_examples)),
            f"kept {len(kept)}/{config.target_examples}",
        )

    stats.kept = len(kept)

    dpo_pairs: list[PreferencePair] = (
        list(checkpoint.dpo_pairs) if checkpoint is not None else []
    )
    if config.generate_dpo_pairs and kept:
        progress(0.9, "generating DPO preference pairs")
        pool = list(kept)
        rng.shuffle(pool)
        target = min(config.dpo_target_pairs, len(pool))
        for example in pool[:target]:
            key = _dpo_key(example)
            if checkpoint is not None and key in checkpoint.dpo_processed:
                continue
            pair = _build_dpo_pair(
                example, provider, judge_provider, config
            )
            if pair is not None:
                dpo_pairs.append(pair)
                if checkpoint is not None:
                    checkpoint.record_dpo(key, pair)

    stats.dpo_pairs = len(dpo_pairs)
    stats.elapsed_sec = time.time() - started

    progress(1.0, f"done: {stats.kept} kept, {len(rejected)} rejected")
    return PipelineOutput(examples=kept, rejected=rejected, dpo_pairs=dpo_pairs, stats=stats)


async def arun_pipeline(
    *,
    config: PipelineConfig,
    chunks: Iterable[Chunk],
    provider: LLMProvider,
    embedder: EmbeddingBackend,
    judge_provider: LLMProvider | None = None,
    progress: ProgressFn | None = None,
) -> PipelineOutput:
    progress = progress or _noop
    judge_provider = judge_provider or provider

    stats = PipelineStats()
    started = time.time()
    rng = random.Random(config.seed)

    chunk_list = list(chunks)
    by_id: dict[str, Chunk] = {c.id: c for c in chunk_list}
    concurrency = max(1, int(config.concurrency))
    sem = asyncio.Semaphore(concurrency)

    checkpoint = _open_checkpoint(config, chunk_list)

    progress(0.02, f"collecting seeds from {len(chunk_list)} chunks (concurrency={concurrency})")

    async def _seed_chunk(c: Chunk) -> list[tuple[str, Chunk | None]]:
        async with sem:
            items = await aseed_from_chunk(
                provider,
                c,
                n=config.seeds_per_chunk,
                language=config.language,
                model=config.generator_model,
            )
        return [(i, c) for i in items]

    async def _seed_desc() -> list[tuple[str, Chunk | None]]:
        if config.seeds_from_description <= 0:
            return []
        async with sem:
            items = await aseed_from_description(
                provider,
                config.description,
                n=config.seeds_from_description,
                language=config.language,
                model=config.generator_model,
            )
        return [(i, None) for i in items]

    seed_tasks = [_seed_chunk(c) for c in chunk_list]
    seed_tasks.append(_seed_desc())
    seeded_groups = await asyncio.gather(*seed_tasks)

    seeds: list[tuple[str, Chunk | None]] = []
    for group in seeded_groups:
        seeds.extend(group)
    stats.seeds = len(seeds)

    if config.evol_factor > 0 and seeds:
        seeds = await _apply_evol_async(seeds, provider, config, rng, sem)
        stats.seeds = len(seeds)

    if not seeds:
        stats.elapsed_sec = time.time() - started
        return PipelineOutput(stats=stats)

    rng.shuffle(seeds)
    progress(0.15, f"generating answers for {len(seeds)} seeds")

    async def _expand(item: tuple[str, Chunk | None]) -> Example | None:
        instruction, chunk = item
        async with sem:
            return await aexpand_one(
                provider,
                instruction,
                chunk,
                description=config.description,
                language=config.language,
                model=config.generator_model,
                temperature=config.generator_temperature,
                max_tokens=config.answer_max_tokens,
            )

    candidates = [c for c in await asyncio.gather(*[_expand(s) for s in seeds]) if c is not None]
    stats.generated = len(candidates)

    diversity = DiversityFilter(embedder, threshold=config.diversity_threshold)
    kept: list[Example] = []
    rejected: list[Example] = []

    if checkpoint is not None:
        kept = list(checkpoint.kept)
        rejected = list(checkpoint.rejected)
        for ex in kept + rejected:
            diversity.accept(ex.instruction)

    # Process candidates sequentially for diversity/judge — diversity is stateful,
    # and we want to stop at target_examples. But we can parallelize the judge
    # call inside the loop using a pre-warmed queue of accepted-diversity examples.

    pre: list[tuple[int, Example, Chunk | None]] = []
    for idx, example in enumerate(candidates):
        if checkpoint is not None and checkpoint.is_processed(idx):
            continue
        if not diversity.accept(example.instruction):
            stats.diversity_rejected += 1
            rejected.append(example)
            if checkpoint is not None:
                checkpoint.record_seed(idx, "rejected", example)
            continue
        chunk = None
        if example.source_chunks:
            chunk = by_id.get(example.source_chunks[0])
        if chunk is not None:
            passed, token, semantic = grounded_ok(
                example.output,
                chunk.text,
                min_token_overlap=config.min_hallucination_overlap,
                min_semantic_similarity=config.min_semantic_similarity,
                embedder=embedder if config.min_semantic_similarity is not None else None,
            )
            example.hallucination_score = token
            if semantic is not None:
                example.metadata["semantic_similarity"] = semantic
            if not passed:
                stats.hallucination_rejected += 1
                rejected.append(example)
                if checkpoint is not None:
                    checkpoint.record_seed(idx, "rejected", example)
                continue
        pre.append((idx, example, chunk))

    async def _judge(idx: int, example: Example, chunk: Chunk | None) -> tuple[int, Example, int, str]:
        async with sem:
            score, reason = await ajudge_example(
                judge_provider,
                example,
                description=config.description,
                reference_text=chunk.text if chunk is not None else None,
                model=config.judge_model,
            )
        return idx, example, score, reason

    # Judge in parallel, but commit to kept/rejected sequentially so we can
    # stop cleanly at target_examples.
    judge_results = await asyncio.gather(*[_judge(i, e, c) for (i, e, c) in pre])
    for idx, example, score, reason in judge_results:
        example.judge_score = score
        example.judge_reason = reason
        stats.judged += 1
        if len(kept) >= config.target_examples:
            rejected.append(example)
            if checkpoint is not None:
                checkpoint.record_seed(idx, "rejected", example)
            continue
        if score < config.min_judge_score:
            stats.judge_rejected += 1
            rejected.append(example)
            if checkpoint is not None:
                checkpoint.record_seed(idx, "rejected", example)
            continue
        kept.append(example)
        if checkpoint is not None:
            checkpoint.record_seed(idx, "kept", example)
        progress(
            0.15 + 0.7 * min(1.0, len(kept) / max(1, config.target_examples)),
            f"kept {len(kept)}/{config.target_examples}",
        )

    stats.kept = len(kept)

    dpo_pairs: list[PreferencePair] = (
        list(checkpoint.dpo_pairs) if checkpoint is not None else []
    )
    if config.generate_dpo_pairs and kept:
        progress(0.9, "generating DPO preference pairs")
        pool = list(kept)
        rng.shuffle(pool)
        target = min(config.dpo_target_pairs, len(pool))

        async def _build(example: Example):
            key = _dpo_key(example)
            if checkpoint is not None and key in checkpoint.dpo_processed:
                return None
            async with sem:
                weak = await aweak_answer(provider, example.instruction, model=config.generator_model)
            if not weak or weak.strip() == example.output.strip():
                return None
            probe = Example(
                instruction=example.instruction,
                output=weak,
                source_chunks=example.source_chunks,
            )
            async with sem:
                weak_score, _ = await ajudge_example(
                    judge_provider,
                    probe,
                    description=config.description,
                    model=config.judge_model,
                )
            if example.judge_score is None or weak_score >= example.judge_score:
                return None
            pair = PreferencePair(
                instruction=example.instruction,
                chosen=example.output,
                rejected=weak,
                chosen_score=example.judge_score,
                rejected_score=weak_score,
                source_chunks=list(example.source_chunks),
            )
            return key, pair

        built = await asyncio.gather(*[_build(e) for e in pool[:target]])
        for item in built:
            if item is None:
                continue
            key, pair = item
            dpo_pairs.append(pair)
            if checkpoint is not None:
                checkpoint.record_dpo(key, pair)

    stats.dpo_pairs = len(dpo_pairs)
    stats.elapsed_sec = time.time() - started

    progress(1.0, f"done: {stats.kept} kept, {len(rejected)} rejected")
    return PipelineOutput(examples=kept, rejected=rejected, dpo_pairs=dpo_pairs, stats=stats)


def run_multiturn_pipeline(
    *,
    config: MultiturnConfig,
    chunks: Iterable[Chunk],
    provider: LLMProvider,
    progress: ProgressFn | None = None,
) -> MultiturnOutput:
    if config.concurrency > 1:
        return asyncio.run(
            arun_multiturn_pipeline(config=config, chunks=chunks, provider=provider, progress=progress)
        )
    progress = progress or _noop
    rng = random.Random(config.seed)
    chunk_list = list(chunks)
    dialogues: list[Dialogue] = []
    pool: list[Chunk | None] = list(chunk_list) or [None]
    started = time.time()
    while len(dialogues) < config.target_dialogues:
        remaining = config.target_dialogues - len(dialogues)
        chunk = rng.choice(pool)
        d = generate_dialogue(
            provider,
            chunk=chunk,
            description=config.description,
            turns=config.turns,
            language=config.language,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        if d is not None:
            dialogues.append(d)
        progress(
            min(1.0, len(dialogues) / max(1, config.target_dialogues)),
            f"dialogues {len(dialogues)}/{config.target_dialogues}",
        )
        if remaining > 0 and d is None and len(dialogues) == 0:
            # All failures — bail after some attempts.
            if time.time() - started > 120:
                break
    stats = {"generated": len(dialogues), "elapsed_sec": round(time.time() - started, 2)}
    return MultiturnOutput(dialogues=dialogues, stats=stats)


async def arun_multiturn_pipeline(
    *,
    config: MultiturnConfig,
    chunks: Iterable[Chunk],
    provider: LLMProvider,
    progress: ProgressFn | None = None,
) -> MultiturnOutput:
    progress = progress or _noop
    rng = random.Random(config.seed)
    chunk_list = list(chunks)
    pool: list[Chunk | None] = list(chunk_list) or [None]
    sem = asyncio.Semaphore(max(1, int(config.concurrency)))
    started = time.time()
    target = config.target_dialogues

    async def _one() -> Dialogue | None:
        async with sem:
            return await agenerate_dialogue(
                provider,
                chunk=rng.choice(pool),
                description=config.description,
                turns=config.turns,
                language=config.language,
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

    # Over-sample because some will fail to parse.
    batch = int(target * 1.3) + 4
    results = await asyncio.gather(*[_one() for _ in range(batch)])
    dialogues = [d for d in results if d is not None][:target]
    progress(1.0, f"dialogues {len(dialogues)}/{target}")
    stats = {"generated": len(dialogues), "elapsed_sec": round(time.time() - started, 2)}
    return MultiturnOutput(dialogues=dialogues, stats=stats)


# --- helpers -----------------------------------------------------------------


def _open_checkpoint(config: PipelineConfig, chunks: list[Chunk]) -> Checkpoint | None:
    if config.checkpoint_path is None:
        return None
    sig = compute_signature(
        {k: v for k, v in asdict(config).items() if k != "checkpoint_path"},
        [c.id for c in chunks],
    )
    return Checkpoint.load_or_create(Path(config.checkpoint_path), config_signature=sig)


def _apply_evol_sync(
    seeds: list[tuple[str, Chunk | None]],
    provider: LLMProvider,
    config: PipelineConfig,
    rng: random.Random,
) -> list[tuple[str, Chunk | None]]:
    factor = max(0.0, min(1.0, float(config.evol_factor)))
    evolved: list[tuple[str, Chunk | None]] = []
    for instruction, chunk in seeds:
        for _ in range(config.evol_rounds):
            if rng.random() > factor:
                continue
            mutated = evolve_seed(
                provider,
                instruction,
                mutation=pick_mutation(rng),
                language=config.language,
                model=config.generator_model,
                rng=rng,
            )
            if mutated:
                evolved.append((mutated, chunk))
    return seeds + evolved


async def _apply_evol_async(
    seeds: list[tuple[str, Chunk | None]],
    provider: LLMProvider,
    config: PipelineConfig,
    rng: random.Random,
    sem: asyncio.Semaphore,
) -> list[tuple[str, Chunk | None]]:
    factor = max(0.0, min(1.0, float(config.evol_factor)))
    plan: list[tuple[str, Chunk | None]] = []
    for instruction, chunk in seeds:
        for _ in range(config.evol_rounds):
            if rng.random() <= factor:
                plan.append((instruction, chunk))

    async def _one(instruction: str, chunk: Chunk | None) -> tuple[str, Chunk | None] | None:
        async with sem:
            mutated = await aevolve_seed(
                provider,
                instruction,
                mutation=pick_mutation(rng),
                language=config.language,
                model=config.generator_model,
                rng=rng,
            )
        return (mutated, chunk) if mutated else None

    results = await asyncio.gather(*[_one(i, c) for (i, c) in plan])
    evolved = [r for r in results if r is not None]
    return seeds + evolved


def _process_one(
    example: Example,
    *,
    by_id: dict[str, Chunk],
    diversity: DiversityFilter,
    config: PipelineConfig,
    judge_provider: LLMProvider,
    embedder: EmbeddingBackend,
    stats: PipelineStats,
) -> str:
    if not diversity.accept(example.instruction):
        stats.diversity_rejected += 1
        return "rejected"

    chunk = None
    if example.source_chunks:
        chunk = by_id.get(example.source_chunks[0])
    if chunk is not None:
        passed, token, semantic = grounded_ok(
            example.output,
            chunk.text,
            min_token_overlap=config.min_hallucination_overlap,
            min_semantic_similarity=config.min_semantic_similarity,
            embedder=embedder if config.min_semantic_similarity is not None else None,
        )
        example.hallucination_score = token
        if semantic is not None:
            example.metadata["semantic_similarity"] = semantic
        if not passed:
            stats.hallucination_rejected += 1
            return "rejected"

    score, reason = judge_example(
        judge_provider,
        example,
        description=config.description,
        reference_text=chunk.text if chunk is not None else None,
        model=config.judge_model,
    )
    example.judge_score = score
    example.judge_reason = reason
    stats.judged += 1

    if score < config.min_judge_score:
        stats.judge_rejected += 1
        return "rejected"
    return "kept"


def _build_dpo_pair(
    example: Example,
    provider: LLMProvider,
    judge_provider: LLMProvider,
    config: PipelineConfig,
) -> PreferencePair | None:
    weak = weak_answer(provider, example.instruction, model=config.generator_model)
    if not weak or weak.strip() == example.output.strip():
        return None
    probe = Example(
        instruction=example.instruction,
        output=weak,
        source_chunks=example.source_chunks,
    )
    weak_score, _ = judge_example(
        judge_provider,
        probe,
        description=config.description,
        model=config.judge_model,
    )
    if example.judge_score is None or weak_score >= example.judge_score:
        return None
    return PreferencePair(
        instruction=example.instruction,
        chosen=example.output,
        rejected=weak,
        chosen_score=example.judge_score,
        rejected_score=weak_score,
        source_chunks=list(example.source_chunks),
    )


def _dpo_key(example: Example) -> str:
    import hashlib
    h = hashlib.sha1()
    h.update(example.instruction.strip().lower().encode("utf-8"))
    h.update(b"|")
    h.update(example.output.strip().lower().encode("utf-8"))
    return h.hexdigest()
