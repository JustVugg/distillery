import json

from distillery.generate.multiturn import Dialogue, _parse_dialogue


def test_parse_valid_alternation():
    raw = json.dumps({
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "More?"},
            {"role": "assistant", "content": "Sure"},
        ]
    })
    msgs = _parse_dialogue(raw, turns=4)
    assert msgs is not None
    assert len(msgs) == 4
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]


def test_parse_rejects_bad_alternation():
    raw = json.dumps({
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "user", "content": "Again"},  # two users in a row
        ]
    })
    assert _parse_dialogue(raw, turns=4) is None


def test_parse_rejects_ending_with_user():
    raw = json.dumps({
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Follow-up"},
        ]
    })
    assert _parse_dialogue(raw, turns=4) is None


def test_parse_rejects_empty_content():
    raw = json.dumps({
        "messages": [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "x"},
        ]
    })
    assert _parse_dialogue(raw, turns=4) is None


def test_parse_rejects_non_json():
    assert _parse_dialogue("not json", turns=4) is None


def test_parse_clamps_to_turns():
    raw = json.dumps({
        "messages": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
            {"role": "user", "content": "e"},
            {"role": "assistant", "content": "f"},
        ]
    })
    msgs = _parse_dialogue(raw, turns=4)
    assert msgs is not None
    assert len(msgs) == 4


def test_dialogue_to_openai_jsonl_with_system():
    d = Dialogue(
        messages=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ],
        source_chunks=["c1"],
    )
    payload = d.to_openai_jsonl(system_prompt="You are helpful.")
    assert payload["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert payload["messages"][1]["role"] == "user"
    assert payload["source_chunks"] == ["c1"]


def test_dialogue_to_openai_jsonl_without_system():
    d = Dialogue(messages=[
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ])
    payload = d.to_openai_jsonl()
    assert payload["messages"][0]["role"] == "user"
    assert "source_chunks" not in payload
