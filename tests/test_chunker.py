from distillery.ingest.chunker import chunk_text, normalize, split_sentences


def test_normalize_collapses_whitespace():
    raw = "Hello   world.\n\n\n\nNext  paragraph."
    assert normalize(raw) == "Hello world.\n\nNext paragraph."


def test_split_sentences_basic():
    text = "First sentence. Second sentence! Third one?"
    assert split_sentences(text) == [
        "First sentence.",
        "Second sentence!",
        "Third one?",
    ]


def test_chunk_text_respects_target_size():
    text = " ".join([f"Sentence number {i}." for i in range(50)])
    chunks = list(chunk_text(text, source="doc", target_chars=200, overlap_chars=30))
    assert len(chunks) >= 3
    for c in chunks:
        assert len(c.text) <= 320  # target + tolerance for boundary sentence
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_text_handles_empty_input():
    assert list(chunk_text("   \n\n   ", source="empty")) == []


def test_chunk_text_overlap_creates_context_bleed():
    text = ". ".join([f"Unique fact number {i}" for i in range(30)]) + "."
    chunks = list(chunk_text(text, source="doc", target_chars=150, overlap_chars=80))
    assert len(chunks) >= 2
    # overlap means the second chunk shares content with the first
    assert any(word in chunks[1].text for word in chunks[0].text.split()[-3:])
