from __future__ import annotations

import numpy as np

from distillery.filter.hallucination import grounded_ok, semantic_similarity
from distillery.providers.embeddings import HashEmbedder


def test_grounded_ok_passes_on_token_overlap():
    ref = "OAuth2 refresh tokens allow clients to obtain new access tokens without re-authenticating."
    out = "OAuth2 refresh tokens let clients obtain new access tokens without re-authenticating."
    passed, token, semantic = grounded_ok(
        out, ref,
        min_token_overlap=0.4,
        min_semantic_similarity=None,
        embedder=None,
    )
    assert passed is True
    assert token >= 0.4
    assert semantic is None


def test_grounded_ok_fails_when_no_overlap_and_no_semantic():
    ref = "Database migration steps require careful planning."
    out = "Elephants roam the savannah at dawn."
    passed, token, semantic = grounded_ok(
        out, ref,
        min_token_overlap=0.35,
        min_semantic_similarity=None,
        embedder=None,
    )
    assert passed is False
    assert token < 0.35
    assert semantic is None


def test_grounded_ok_semantic_rescues_low_token_overlap():
    class _ConstEmbedder:
        dim = 4

        def embed(self, texts):
            # Emit identical vectors so cosine = 1.0 — simulates a paraphrase rescue.
            return np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(texts), 1))

    ref = "alpha beta gamma delta"
    out = "completely different words here"  # no token overlap
    passed, token, semantic = grounded_ok(
        out, ref,
        min_token_overlap=0.9,
        min_semantic_similarity=0.5,
        embedder=_ConstEmbedder(),
    )
    assert passed is True  # passed via semantic path
    assert semantic is not None and semantic >= 0.99


def test_semantic_similarity_handles_empty():
    emb = HashEmbedder(dim=64)
    assert semantic_similarity("", "something", emb) == 0.0
    assert semantic_similarity("something", "", emb) == 0.0


def test_semantic_similarity_identical_text_is_one():
    emb = HashEmbedder(dim=64)
    s = semantic_similarity("the quick brown fox", "the quick brown fox", emb)
    assert s >= 0.99
