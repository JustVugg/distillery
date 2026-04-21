from __future__ import annotations

from distillery.checkpoint import Checkpoint, compute_signature
from distillery.types import Example, PreferencePair


def _ex(instruction: str, output: str = "out") -> Example:
    return Example(instruction=instruction, output=output)


def test_checkpoint_roundtrip(tmp_path):
    sig = {"config": {"x": 1}, "chunk_count": 2, "chunk_ids_hash": "abc"}
    path = tmp_path / "ckpt.jsonl"

    cp = Checkpoint.load_or_create(path, config_signature=sig)
    cp.record_seed(0, "kept", _ex("seed-0"))
    cp.record_seed(1, "rejected", _ex("seed-1"))
    cp.record_dpo("k1", PreferencePair(
        instruction="i", chosen="a", rejected="b", chosen_score=8, rejected_score=4,
    ))

    resumed = Checkpoint.load_or_create(path, config_signature=sig)
    assert resumed.is_processed(0)
    assert resumed.is_processed(1)
    assert not resumed.is_processed(2)
    assert [e.instruction for e in resumed.kept] == ["seed-0"]
    assert [e.instruction for e in resumed.rejected] == ["seed-1"]
    assert "k1" in resumed.dpo_processed
    assert len(resumed.dpo_pairs) == 1


def test_checkpoint_rejects_mismatched_signature(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    sig_a = {"config": {"x": 1}, "chunk_count": 1, "chunk_ids_hash": "aa"}
    sig_b = {"config": {"x": 2}, "chunk_count": 1, "chunk_ids_hash": "aa"}

    cp = Checkpoint.load_or_create(path, config_signature=sig_a)
    cp.record_seed(0, "kept", _ex("seed-0"))

    resumed = Checkpoint.load_or_create(path, config_signature=sig_b)
    assert not resumed.is_processed(0)
    assert resumed.kept == []
    assert resumed.rejected == []


def test_compute_signature_stable_across_chunk_order():
    a = compute_signature({"k": 1}, ["c2", "c1", "c3"])
    b = compute_signature({"k": 1}, ["c1", "c2", "c3"])
    assert a == b


def test_compute_signature_changes_with_chunks():
    a = compute_signature({"k": 1}, ["c1"])
    b = compute_signature({"k": 1}, ["c1", "c2"])
    assert a != b
