import json

from distillery.export.jsonl import export_tool_calling_jsonl
from distillery.types import Example


def test_tool_calling_export_emits_valid_shape(tmp_path):
    ex = Example(
        instruction="What's the weather in Rome?",
        output='{"temperature_c": 22, "condition": "sunny"}',
        metadata={"tool_call": {"name": "get_weather", "arguments": {"city": "Rome"}}},
    )
    path = tmp_path / "tc.jsonl"
    count = export_tool_calling_jsonl(path, [ex])
    assert count == 1
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    msgs = row["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "get_weather"
    # arguments must be serialized to a JSON string per OpenAI spec.
    assert isinstance(msgs[2]["tool_calls"][0]["function"]["arguments"], str)
    assert json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"]) == {"city": "Rome"}
    assert msgs[3]["role"] == "tool"
    assert msgs[3]["tool_call_id"] == msgs[2]["tool_calls"][0]["id"]


def test_tool_calling_export_skips_examples_without_metadata(tmp_path):
    without = Example(instruction="Plain", output="No tool call.")
    with_tool = Example(
        instruction="Fetch",
        output="{}",
        metadata={"tool_call": {"name": "noop", "arguments": {}}},
    )
    path = tmp_path / "tc.jsonl"
    count = export_tool_calling_jsonl(path, [without, with_tool])
    assert count == 1


def test_tool_calling_export_attaches_tool_schema(tmp_path):
    ex = Example(
        instruction="Fetch",
        output="{}",
        metadata={"tool_call": {"name": "noop", "arguments": {}}},
    )
    schema = {"name": "noop", "parameters": {"type": "object"}}
    path = tmp_path / "tc.jsonl"
    export_tool_calling_jsonl(path, [ex], tool_schema=schema)
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["tools"] == [{"type": "function", "function": schema}]
