import json
from pathlib import Path

from distillery.utils import extract_json_blob, safe_json_loads, write_jsonl


def test_extract_json_blob_from_fenced():
    raw = "Here you go:\n```json\n[1, 2, 3]\n```"
    assert safe_json_loads(raw) == [1, 2, 3]


def test_extract_json_blob_from_noise():
    raw = "ok: {\"a\": 1, \"b\": [2, 3]} trailing"
    assert safe_json_loads(raw) == {"a": 1, "b": [2, 3]}


def test_extract_json_blob_returns_none_on_garbage():
    assert extract_json_blob("no json here") is None
    assert safe_json_loads("also no json") is None


def test_write_jsonl(tmp_path: Path):
    path = tmp_path / "out.jsonl"
    count = write_jsonl(path, [{"a": 1}, {"b": 2}])
    assert count == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"b": 2}
