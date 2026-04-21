from __future__ import annotations

import numpy as np

from ..providers.embeddings import EmbeddingBackend


class DiversityFilter:
    """Reject examples whose instruction is too close to one already accepted.

    Uses cosine similarity on normalized embeddings. Maintains a running matrix of
    kept vectors so the check is O(n·kept) per new example.
    """

    def __init__(self, embedder: EmbeddingBackend, threshold: float = 0.92) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.embedder = embedder
        self.threshold = float(threshold)
        self._matrix = np.zeros((0, embedder.dim), dtype=np.float32)
        self._texts: list[str] = []

    def reset(self) -> None:
        self._matrix = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._texts = []

    def __len__(self) -> int:
        return len(self._texts)

    def _vector(self, text: str) -> np.ndarray:
        vec = self.embedder.embed([text])[0]
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32, copy=False)

    def accept(self, text: str) -> bool:
        """Return True if the text is novel enough; if so, it is added to the index."""
        vec = self._vector(text)
        if self._matrix.shape[0] > 0:
            sims = self._matrix @ vec
            if float(sims.max()) >= self.threshold:
                return False
        self._matrix = np.vstack([self._matrix, vec[None, :]])
        self._texts.append(text)
        return True

    def batch_accept(self, texts: list[str]) -> list[bool]:
        return [self.accept(t) for t in texts]
