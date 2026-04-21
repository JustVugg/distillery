"""Hugging Face dataset card generator.

Produces a README.md with the YAML frontmatter the Hub expects, followed by a
human-readable description of what the dataset contains, how it was built,
and the filter thresholds that produced it. Drop it into the dataset folder
and `datasets-cli upload` (or `huggingface-cli upload`) picks it up.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DatasetCardInfo:
    title: str
    description: str
    language: str
    source_description: str
    license: str
    train_count: int
    eval_count: int
    dpo_count: int
    rejected_count: int
    provider: str
    generator_model: str
    judge_model: str
    embedding_model: str
    stats: dict[str, Any]
    config: dict[str, Any]
    tags: list[str]


YAML_TEMPLATE = """---
language:
- {language_code}
license: {license}
task_categories:
- text-generation
- question-answering
pretty_name: "{title}"
size_categories:
- {size_category}
tags:
{tags_block}
configs:
- config_name: default
  data_files:
  - split: train
    path: "*.train.jsonl"
  - split: validation
    path: "*.eval.jsonl"
---
"""


MARKDOWN_TEMPLATE = """# {title}

{description}

## Dataset details

- **Train set**: {train_count} examples
- **Eval set**: {eval_count} examples
- **DPO preference pairs**: {dpo_count}
- **Rejected candidates** (kept for auditing): {rejected_count}
- **Language**: {language}
- **License**: {license}

## How it was built

This dataset was distilled from {source_description} using
[Distillery](https://github.com/JustVugg/distillery). Each example is:

- **grounded** in a specific chunk of the source (or flagged as free-form)
- **judged** by an LLM grader against its reference context
- **deduplicated** by instruction embedding similarity
- **hallucination-checked** against the source chunk

### Pipeline configuration

| Knob | Value |
|------|-------|
| Provider | `{provider}` |
| Generator model | `{generator_model}` |
| Judge model | `{judge_model}` |
| Embedding model | `{embedding_model}` |
| Minimum judge score | {min_judge_score} |
| Diversity threshold | {diversity_threshold} |
| Minimum hallucination overlap | {min_hallucination_overlap} |
| Target examples | {target_examples} |

### Pipeline stats

{stats_table}

## Intended uses

Supervised fine-tuning (SFT) and preference optimization (DPO) of open-weights
language models. The dataset is NOT intended as a factual knowledge source for
end-users; the reference material linked above is authoritative.

## Limitations

- Quality is bounded by the judge model used during construction.
- Free-form examples (without `source_chunks`) are not grounded in source material.
- No personal data was intentionally included, but if the source material contains
  any, verify before public release.
"""


_LANG_CODE = {
    "english": "en",
    "italian": "it",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "portuguese": "pt",
    "dutch": "nl",
    "japanese": "ja",
    "chinese": "zh",
}


def _infer_language_code(language: str) -> str:
    code = _LANG_CODE.get(language.strip().lower())
    return code or "en"


def _size_category(n: int) -> str:
    if n < 1_000:
        return "n<1K"
    if n < 10_000:
        return "1K<n<10K"
    if n < 100_000:
        return "10K<n<100K"
    if n < 1_000_000:
        return "100K<n<1M"
    return "n>1M"


def _format_stats_table(stats: dict[str, Any]) -> str:
    lines = ["| Metric | Value |", "|--------|-------|"]
    for k, v in stats.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def render_dataset_card(info: DatasetCardInfo) -> str:
    tags = list(info.tags) + ["distillery", "synthetic", "instruction-tuning"]
    seen = set()
    deduped = []
    for t in tags:
        lt = t.lower().strip()
        if lt and lt not in seen:
            seen.add(lt)
            deduped.append(lt)
    tags_block = "\n".join(f"- {t}" for t in deduped) or "- distillery"

    yaml = YAML_TEMPLATE.format(
        language_code=_infer_language_code(info.language),
        license=info.license,
        title=info.title,
        size_category=_size_category(info.train_count),
        tags_block=tags_block,
    )

    body = MARKDOWN_TEMPLATE.format(
        title=info.title,
        description=info.description.strip(),
        train_count=info.train_count,
        eval_count=info.eval_count,
        dpo_count=info.dpo_count,
        rejected_count=info.rejected_count,
        language=info.language,
        license=info.license,
        source_description=info.source_description.strip(),
        provider=info.provider,
        generator_model=info.generator_model,
        judge_model=info.judge_model,
        embedding_model=info.embedding_model,
        min_judge_score=info.config.get("min_judge_score", "?"),
        diversity_threshold=info.config.get("diversity_threshold", "?"),
        min_hallucination_overlap=info.config.get("min_hallucination_overlap", "?"),
        target_examples=info.config.get("target_examples", "?"),
        stats_table=_format_stats_table(info.stats),
    )
    return yaml + "\n" + body


def write_dataset_card(path: Path, info: DatasetCardInfo) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dataset_card(info), encoding="utf-8")
    return path
