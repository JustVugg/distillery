from distillery.generate.evol import _clean_evolution


def test_clean_strips_rewritten_prefix():
    out = _clean_evolution("Rewritten: Explain how OAuth2 refresh tokens work in detail.", "orig")
    assert out == "Explain how OAuth2 refresh tokens work in detail."


def test_clean_strips_wrapping_quotes():
    out = _clean_evolution('"Explain how OAuth2 works in a mobile app."', "orig")
    assert out == "Explain how OAuth2 works in a mobile app."


def test_clean_rejects_no_op_rewrite():
    assert _clean_evolution("same text", "  SAME TEXT  ") is None


def test_clean_rejects_too_short():
    assert _clean_evolution("hi", "orig") is None


def test_clean_rejects_too_long():
    assert _clean_evolution("x" * 801, "orig") is None


def test_clean_handles_empty():
    assert _clean_evolution("", "orig") is None
    assert _clean_evolution("   ", "orig") is None
