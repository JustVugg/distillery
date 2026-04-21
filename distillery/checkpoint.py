"""Pipeline checkpoint/resume.

A single JSONL file per run stores:
  - one config line at the top (for sanity checks)
  - one JSON record per processed seed: {"idx": int, "decision": "kept"|"rejected", "payload": {...}}

On restart, the pipeline reads the file, rebuilds kept/rejected lists, and skips
already-processed seed indices. The format is append-only and human-readable.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from .types import Example, PreferencePair


log = logging.getLogger(__name__)


CHECKPOINT_VERSION = 1


class Checkpoint:
    def __init__(self, path: Path, *, config_signature: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.config_signature = config_signature or {}
        self.processed: set[int] = set()
        self.kept: list[Example] = []
        self.rejected: list[Example] = []
        self.dpo_processed: set[str] = set()
        self.dpo_pairs: list[PreferencePair] = []

    @classmethod
    def load_or_create(cls, path: Path, *, config_signature: dict[str, Any]) -> "Checkpoint":
        cp = cls(path, config_signature=config_signature)
        if not cp.path.exists():
            cp._write_header()
            return cp

        lines = cp.path.read_text(encoding="utf-8").splitlines()
        if not lines:
            cp._write_header()
            return cp

        try:
            header = json.loads(lines[0])
        except json.JSONDecodeError:
            log.warning("checkpoint header unreadable, starting fresh: %s", cp.path)
            cp.path.unlink()
            cp._write_header()
            return cp

        if header.get("version") != CHECKPOINT_VERSION:
            log.warning("checkpoint version mismatch, starting fresh")
            cp.path.unlink()
            cp._write_header()
            return cp

        if header.get("config") != config_signature:
            log.warning(
                "checkpoint config mismatch — ignoring existing checkpoint for a clean run"
            )
            cp.path.unlink()
            cp._write_header()
            return cp

        for raw in lines[1:]:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue

            kind = rec.get("kind")
            if kind == "seed":
                idx = rec.get("idx")
                if isinstance(idx, int):
                    cp.processed.add(idx)
                    ex = _example_from_dict(rec.get("example") or {})
                    if rec.get("decision") == "kept":
                        cp.kept.append(ex)
                    elif rec.get("decision") == "rejected":
                        cp.rejected.append(ex)
            elif kind == "dpo":
                key = rec.get("key")
                if isinstance(key, str):
                    cp.dpo_processed.add(key)
                    pair = _pair_from_dict(rec.get("pair") or {})
                    if pair is not None:
                        cp.dpo_pairs.append(pair)

        log.info(
            "checkpoint resumed from %s: %d kept, %d rejected, %d dpo",
            cp.path,
            len(cp.kept),
            len(cp.rejected),
            len(cp.dpo_pairs),
        )
        return cp

    def _write_header(self) -> None:
        header = {"version": CHECKPOINT_VERSION, "config": self.config_signature}
        with self.path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(header, ensure_ascii=False) + "\n")

    def _append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()

    def record_seed(self, idx: int, decision: str, example: Example) -> None:
        if idx in self.processed:
            return
        self.processed.add(idx)
        if decision == "kept":
            self.kept.append(example)
        elif decision == "rejected":
            self.rejected.append(example)
        self._append({
            "kind": "seed",
            "idx": idx,
            "decision": decision,
            "example": example.to_dict(),
        })

    def record_dpo(self, key: str, pair: PreferencePair) -> None:
        if key in self.dpo_processed:
            return
        self.dpo_processed.add(key)
        self.dpo_pairs.append(pair)
        self._append({"kind": "dpo", "key": key, "pair": pair.to_dict()})

    def is_processed(self, idx: int) -> bool:
        return idx in self.processed

    def delete(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _example_from_dict(d: dict[str, Any]) -> Example:
    return Example(
        instruction=str(d.get("instruction", "")),
        output=str(d.get("output", "")),
        source_chunks=list(d.get("source_chunks") or []),
        judge_score=d.get("judge_score"),
        judge_reason=d.get("judge_reason"),
        hallucination_score=d.get("hallucination_score"),
        format=str(d.get("format", "sft")),
        metadata=dict(d.get("metadata") or {}),
    )


def _pair_from_dict(d: dict[str, Any]) -> PreferencePair | None:
    try:
        return PreferencePair(
            instruction=str(d["instruction"]),
            chosen=str(d["chosen"]),
            rejected=str(d["rejected"]),
            chosen_score=int(d["chosen_score"]),
            rejected_score=int(d["rejected_score"]),
            source_chunks=list(d.get("source_chunks") or []),
        )
    except (KeyError, TypeError, ValueError):
        return None


def compute_signature(config: Any, chunk_ids: Iterable[str]) -> dict[str, Any]:
    """Produce a stable dict summarizing the config and data inputs.

    Only fields that would meaningfully change outputs go in. Avoid embedding
    the full chunk text — use IDs + count to keep the header small while still
    catching swapped inputs.
    """
    if is_dataclass(config):
        cfg_dict = asdict(config)
    else:
        cfg_dict = dict(config)
    ids = sorted(chunk_ids)
    return {
        "config": cfg_dict,
        "chunk_count": len(ids),
        "chunk_ids_hash": _hash_ids(ids),
    }


def _hash_ids(ids: list[str]) -> str:
    import hashlib
    h = hashlib.sha1()
    for i in ids:
        h.update(i.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()
