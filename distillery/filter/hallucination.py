from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..providers.embeddings import EmbeddingBackend


_TOKEN = re.compile(r"[\wÀ-ÿ]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "to", "of", "in",
    "on", "for", "with", "at", "by", "from", "as", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "can",
    "could", "should", "would", "may", "might", "will", "shall", "this", "that",
    "these", "those", "it", "its", "they", "their", "them", "he", "she", "his",
    "her", "you", "your", "we", "our",
    # italian stopwords (common since codebase is bilingual)
    "il", "lo", "la", "le", "gli", "un", "uno", "una", "di", "da", "del", "della",
    "dei", "delle", "che", "come", "per", "con", "su", "sono", "essere", "avere",
}


def _tokens(text: str) -> set[str]:
    return {
        tok.lower()
        for tok in _TOKEN.findall(text)
        if len(tok) > 2 and tok.lower() not in _STOPWORDS
    }


def hallucination_score(output: str, reference_text: str) -> float:
    """Fraction of content-word tokens in `output` that also appear in `reference_text`.

    1.0 means every content word is grounded; 0.0 means none is. This is a cheap
    sanity check, not a semantic guarantee. Use alongside an LLM judge.
    """
    if not output.strip() or not reference_text.strip():
        return 0.0
    out_tokens = _tokens(output)
    ref_tokens = _tokens(reference_text)
    if not out_tokens:
        return 0.0
    overlap = out_tokens & ref_tokens
    return round(len(overlap) / len(out_tokens), 4)


def semantic_similarity(
    output: str,
    reference_text: str,
    embedder: "EmbeddingBackend",
) -> float:
    """Cosine similarity between embedding of `output` and `reference_text`.

    Complements `hallucination_score`: an answer that paraphrases the source
    will have low token overlap but high semantic similarity. Use a combined
    rule (e.g. pass if either signal is strong) for a forgiving filter.
    """
    if not output.strip() or not reference_text.strip():
        return 0.0
    vecs = embedder.embed([output.strip(), reference_text.strip()])
    if vecs.shape[0] != 2:
        return 0.0
    a, b = vecs[0], vecs[1]
    import math
    denom = math.sqrt(float((a * a).sum())) * math.sqrt(float((b * b).sum()))
    if denom <= 0:
        return 0.0
    return round(float((a * b).sum() / denom), 4)


def grounded_ok(
    output: str,
    reference_text: str,
    *,
    min_token_overlap: float,
    min_semantic_similarity: float | None,
    embedder: "EmbeddingBackend | None",
) -> tuple[bool, float, float | None]:
    """Return (is_grounded, token_overlap, semantic_similarity_or_None).

    Pass if either signal clears its threshold. Semantic-only is skipped when
    no embedder is provided or min_semantic_similarity is None.
    """
    token = hallucination_score(output, reference_text)
    semantic: float | None = None
    if embedder is not None and min_semantic_similarity is not None:
        semantic = semantic_similarity(output, reference_text, embedder)

    passed = token >= min_token_overlap
    if not passed and semantic is not None:
        passed = semantic >= min_semantic_similarity
    return passed, token, semantic
