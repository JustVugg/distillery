from distillery.filter.diversity import DiversityFilter
from distillery.providers.embeddings import HashEmbedder


def test_diversity_accepts_distinct():
    f = DiversityFilter(HashEmbedder(dim=128), threshold=0.95)
    assert f.accept("How do I reset the database?")
    assert f.accept("What is the policy for refunds on annual plans?")
    assert len(f) == 2


def test_diversity_rejects_near_duplicate():
    f = DiversityFilter(HashEmbedder(dim=128), threshold=0.75)
    assert f.accept("How do I reset the database?")
    # exact duplicate must be rejected
    assert not f.accept("How do I reset the database?")
    assert len(f) == 1


def test_diversity_reset_clears_state():
    f = DiversityFilter(HashEmbedder(dim=128), threshold=0.9)
    f.accept("first instruction")
    f.accept("second instruction")
    assert len(f) == 2
    f.reset()
    assert len(f) == 0
    assert f.accept("first instruction")
