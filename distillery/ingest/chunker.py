from __future__ import annotations

import hashlib
import re
from typing import Iterator

from ..types import Chunk


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ÿ0-9])")
_WHITESPACE = re.compile(r"[ \t]+")
_MULTIBLANK = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    lines = [_WHITESPACE.sub(" ", line).rstrip() for line in text.splitlines()]
    joined = "\n".join(lines)
    return _MULTIBLANK.sub("\n\n", joined).strip()


def split_sentences(text: str) -> list[str]:
    if not text.strip():
        return []
    parts = _SENT_SPLIT.split(text.replace("\n", " "))
    return [p.strip() for p in parts if p.strip()]


def _chunk_id(source: str, index: int, body: str) -> str:
    digest = hashlib.sha1(body.encode("utf-8")).hexdigest()[:10]
    return f"{source}:{index:04d}:{digest}"


def chunk_text(
    text: str,
    *,
    source: str,
    target_chars: int = 1200,
    overlap_chars: int = 150,
) -> Iterator[Chunk]:
    """Yield overlapping chunks while respecting sentence boundaries.

    Chunks aim for `target_chars` but may be slightly larger to avoid splitting a
    sentence mid-word. Overlap is applied by keeping the tail of the previous chunk.
    """
    if target_chars <= 0:
        raise ValueError("target_chars must be positive")
    overlap_chars = max(0, min(overlap_chars, target_chars // 2))

    normalized = normalize(text)
    if not normalized:
        return

    sentences = split_sentences(normalized)
    if not sentences:
        return

    buffer: list[str] = []
    buffer_len = 0
    chunk_index = 0

    def flush() -> Chunk | None:
        nonlocal chunk_index
        if not buffer:
            return None
        body = " ".join(buffer).strip()
        if not body:
            return None
        chunk = Chunk(
            id=_chunk_id(source, chunk_index, body),
            text=body,
            source=source,
            index=chunk_index,
            metadata={"char_count": len(body)},
        )
        chunk_index += 1
        return chunk

    for sentence in sentences:
        if buffer_len + len(sentence) + 1 > target_chars and buffer:
            chunk = flush()
            if chunk is not None:
                yield chunk
            if overlap_chars > 0:
                tail: list[str] = []
                tail_len = 0
                for s in reversed(buffer):
                    if tail_len + len(s) + 1 > overlap_chars:
                        break
                    tail.insert(0, s)
                    tail_len += len(s) + 1
                buffer = list(tail)
                buffer_len = sum(len(s) + 1 for s in buffer)
            else:
                buffer = []
                buffer_len = 0

        buffer.append(sentence)
        buffer_len += len(sentence) + 1

    tail_chunk = flush()
    if tail_chunk is not None:
        yield tail_chunk
