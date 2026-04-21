from __future__ import annotations

from typing import Iterable

from ..types import Example


def to_hf_dataset(examples: Iterable[Example]):
    """Return a 🤗 Datasets object with columns: instruction, output, source_chunks, judge_score.

    Requires the [hf] extra: `pip install 'distillery[hf]'`.
    """
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets not installed. Install with: pip install 'distillery[hf]'"
        ) from exc

    rows = {
        "instruction": [],
        "output": [],
        "source_chunks": [],
        "judge_score": [],
    }
    for ex in examples:
        rows["instruction"].append(ex.instruction)
        rows["output"].append(ex.output)
        rows["source_chunks"].append(list(ex.source_chunks))
        rows["judge_score"].append(ex.judge_score if ex.judge_score is not None else -1)
    return Dataset.from_dict(rows)
