from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A unit of source text that can be used to ground generation."""

    id: str
    text: str
    source: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Example:
    """A single instruction/output pair with provenance."""

    instruction: str
    output: str
    source_chunks: list[str] = field(default_factory=list)
    judge_score: int | None = None
    judge_reason: str | None = None
    hallucination_score: float | None = None
    format: str = "sft"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "instruction": self.instruction,
            "output": self.output,
            "format": self.format,
        }
        if self.source_chunks:
            d["source_chunks"] = list(self.source_chunks)
        if self.judge_score is not None:
            d["judge_score"] = self.judge_score
        if self.judge_reason is not None:
            d["judge_reason"] = self.judge_reason
        if self.hallucination_score is not None:
            d["hallucination_score"] = self.hallucination_score
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        return d


@dataclass
class PreferencePair:
    """A DPO-style preference pair."""

    instruction: str
    chosen: str
    rejected: str
    chosen_score: int
    rejected_score: int
    source_chunks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "chosen_score": self.chosen_score,
            "rejected_score": self.rejected_score,
            "source_chunks": list(self.source_chunks),
            "format": "dpo",
        }


@dataclass
class PipelineStats:
    seeds: int = 0
    generated: int = 0
    judged: int = 0
    diversity_rejected: int = 0
    judge_rejected: int = 0
    hallucination_rejected: int = 0
    kept: int = 0
    dpo_pairs: int = 0
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": self.seeds,
            "generated": self.generated,
            "judged": self.judged,
            "diversity_rejected": self.diversity_rejected,
            "judge_rejected": self.judge_rejected,
            "hallucination_rejected": self.hallucination_rejected,
            "kept": self.kept,
            "dpo_pairs": self.dpo_pairs,
            "elapsed_sec": round(self.elapsed_sec, 2),
        }
