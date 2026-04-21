from distillery.filter.hallucination import hallucination_score


def test_hallucination_high_overlap():
    ref = "The quick brown fox jumps over the lazy dog near the river."
    out = "The fox jumps over the dog near the river."
    score = hallucination_score(out, ref)
    assert score > 0.7


def test_hallucination_low_overlap():
    ref = "Database migration steps require careful planning and testing."
    out = "The elephant walked across the Sahara desert at dawn."
    assert hallucination_score(out, ref) < 0.2


def test_hallucination_empty_inputs():
    assert hallucination_score("", "something") == 0.0
    assert hallucination_score("something", "") == 0.0
