from distillery.export.datacard import DatasetCardInfo, _infer_language_code, _size_category, render_dataset_card


def test_infer_language_code_known():
    assert _infer_language_code("Italian") == "it"
    assert _infer_language_code(" ENGLISH ") == "en"


def test_infer_language_code_unknown_defaults_to_en():
    assert _infer_language_code("Klingon") == "en"


def test_size_category_buckets():
    assert _size_category(100) == "n<1K"
    assert _size_category(5_000) == "1K<n<10K"
    assert _size_category(50_000) == "10K<n<100K"
    assert _size_category(500_000) == "100K<n<1M"
    assert _size_category(2_000_000) == "n>1M"


def test_render_dataset_card_has_yaml_and_markdown():
    info = DatasetCardInfo(
        title="Test Dataset",
        description="A test dataset for HR assistants.",
        language="Italian",
        source_description="internal handbook PDFs",
        license="mit",
        train_count=1200,
        eval_count=150,
        dpo_count=30,
        rejected_count=80,
        provider="ollama",
        generator_model="llama3.1:8b",
        judge_model="llama3.1:8b",
        embedding_model="hash",
        stats={"seeds": 200, "kept": 180, "elapsed_sec": 12.5},
        config={
            "min_judge_score": 7,
            "diversity_threshold": 0.9,
            "min_hallucination_overlap": 0.35,
            "target_examples": 1500,
        },
        tags=["hr", "italian", "distillery"],  # distillery will be dedup'd with auto-tags
    )
    rendered = render_dataset_card(info)
    assert rendered.startswith("---\n")
    assert "language:\n- it" in rendered
    assert "pretty_name: \"Test Dataset\"" in rendered
    assert "1K<n<10K" in rendered
    assert "# Test Dataset" in rendered
    assert "| seeds | 200 |" in rendered
    assert rendered.count("- distillery") == 1  # deduped
    assert "internal handbook PDFs" in rendered


def test_render_dataset_card_unique_tags_case_insensitive():
    info = DatasetCardInfo(
        title="T", description="d", language="english", source_description="s",
        license="mit", train_count=10, eval_count=1, dpo_count=0, rejected_count=0,
        provider="p", generator_model="g", judge_model="j", embedding_model="e",
        stats={}, config={}, tags=["Foo", "foo", "FOO", "bar"],
    )
    rendered = render_dataset_card(info)
    # Should appear once each.
    assert rendered.count("- foo") == 1
    assert rendered.count("- bar") == 1
