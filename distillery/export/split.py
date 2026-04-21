from __future__ import annotations

import random
from typing import Sequence, TypeVar


T = TypeVar("T")


def train_eval_split(
    items: Sequence[T],
    eval_fraction: float = 0.1,
    *,
    seed: int = 42,
    min_eval: int = 1,
    max_eval: int | None = None,
) -> tuple[list[T], list[T]]:
    """Deterministic random split. Returns (train, eval)."""
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in (0, 1)")
    if not items:
        return [], []

    indices = list(range(len(items)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    eval_size = max(min_eval, int(round(len(items) * eval_fraction)))
    if max_eval is not None:
        eval_size = min(eval_size, max_eval)
    eval_size = min(eval_size, max(0, len(items) - 1))

    eval_idx = set(indices[:eval_size])
    train = [items[i] for i in range(len(items)) if i not in eval_idx]
    eval_ = [items[i] for i in sorted(eval_idx)]
    return train, eval_
