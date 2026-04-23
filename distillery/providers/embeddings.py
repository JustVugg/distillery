from __future__ import annotations

import hashlib
import math
from typing import Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Deterministic hashing embedder for environments without sentence-transformers.

    Produces unit-normalized vectors over a fixed feature space. Good enough to detect
    near-duplicates when a heavier model isn't available.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim < 32:
            raise ValueError("dim must be >= 32")
        self.dim = dim

    def _vectorize(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = text.lower().split()
        for token in tokens:
            h = hashlib.sha1(token.encode("utf-8")).digest()
            bucket = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if (h[4] & 1) == 0 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(float(np.dot(vec, vec)))
        if norm > 0:
            vec /= norm
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vectorize(t) for t in texts], axis=0)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers not installed. Install with: pip install 'distillery[embeddings]'"
            ) from exc
        self._model = SentenceTransformer(model_name)
        probe = self._model.encode(["probe"], convert_to_numpy=True, normalize_embeddings=True)
        self.dim = int(probe.shape[1])

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return vectors.astype(np.float32, copy=False)


def build_embedder(model_name: str, prefer_transformers: bool = True) -> EmbeddingBackend:
    """Try sentence-transformers first, fall back to the hash-based embedder."""
    if prefer_transformers:
        try:
            return SentenceTransformerEmbedder(model_name)
        except RuntimeError:
            pass
    return HashEmbedder()
