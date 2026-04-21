from distillery.export.split import train_eval_split


def test_split_fraction_and_determinism():
    items = list(range(100))
    train_a, eval_a = train_eval_split(items, eval_fraction=0.1, seed=1)
    train_b, eval_b = train_eval_split(items, eval_fraction=0.1, seed=1)
    assert train_a == train_b
    assert eval_a == eval_b
    assert len(eval_a) == 10
    assert len(train_a) == 90
    assert set(train_a + eval_a) == set(items)


def test_split_respects_max_eval():
    items = list(range(100))
    _, eval_ = train_eval_split(items, eval_fraction=0.5, max_eval=5)
    assert len(eval_) == 5


def test_split_small_inputs():
    items = list(range(3))
    train, eval_ = train_eval_split(items, eval_fraction=0.3, min_eval=1)
    assert len(train) + len(eval_) == 3
    assert len(eval_) >= 1
