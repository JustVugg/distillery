from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..generate.multiturn import Dialogue
from ..types import Example, PreferencePair
from ..utils import write_json, write_jsonl


def export_jsonl(path: Path, examples: Iterable[Example]) -> int:
    """Write examples as JSONL with full metadata (recommended default)."""
    return write_jsonl(path, (ex.to_dict() for ex in examples))


def export_openai_messages(
    path: Path,
    examples: Iterable[Example],
    *,
    system_prompt: str = "You are a helpful assistant.",
) -> int:
    """Write examples in OpenAI `messages` JSONL format (compatible with OpenAI
    fine-tuning, Together, Fireworks, Axolotl `sharegpt` loader)."""
    def _iter():
        for ex in examples:
            yield {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": ex.instruction},
                    {"role": "assistant", "content": ex.output},
                ]
            }

    return write_jsonl(path, _iter())


def export_dpo_jsonl(path: Path, pairs: Iterable[PreferencePair]) -> int:
    """Write preference pairs in a DPO-ready JSONL format."""
    return write_jsonl(path, (p.to_dict() for p in pairs))


def export_legacy_instruction_json(path: Path, examples: Iterable[Example]) -> int:
    """Flat JSON array of {instruction, output} — for tools that want a single file."""
    items = [{"instruction": ex.instruction, "output": ex.output} for ex in examples]
    write_json(path, items)
    return len(items)


def export_multiturn_jsonl(
    path: Path,
    dialogues: Iterable[Dialogue],
    *,
    system_prompt: str | None = None,
) -> int:
    """Write multi-turn dialogues as OpenAI-style JSONL (one dialogue per line)."""
    return write_jsonl(path, (d.to_openai_jsonl(system_prompt=system_prompt) for d in dialogues))


def export_tool_calling_jsonl(
    path: Path,
    examples: Iterable[Example],
    *,
    tool_schema: dict | None = None,
    system_prompt: str = "You are a helpful assistant with access to tools.",
) -> int:
    """Export in OpenAI tool-calling format.

    Each example is wrapped into a messages[] with a tool call. If the example's
    metadata includes a `tool_call` dict ({"name": ..., "arguments": {...}}), we
    use it verbatim. Otherwise we skip the example — tool-calling examples must
    be explicit about which function is called, we do not invent them.
    """
    def _iter():
        for ex in examples:
            tool_call = (ex.metadata or {}).get("tool_call")
            if not isinstance(tool_call, dict):
                continue
            name = str(tool_call.get("name", "")).strip()
            arguments = tool_call.get("arguments")
            if not name or not isinstance(arguments, (dict, str)):
                continue
            args_json = arguments if isinstance(arguments, str) else _json_compact(arguments)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ex.instruction},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{abs(hash((ex.instruction, name))) % 10**8:08d}",
                            "type": "function",
                            "function": {"name": name, "arguments": args_json},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": None,
                    "name": name,
                    "content": ex.output,
                },
            ]
            # Link tool response to the call id
            messages[-1]["tool_call_id"] = messages[-2]["tool_calls"][0]["id"]
            out: dict = {"messages": messages}
            if tool_schema is not None:
                out["tools"] = [{"type": "function", "function": tool_schema}]
            yield out

    return write_jsonl(path, _iter())


def _json_compact(obj) -> str:
    import json as _json
    return _json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
