from __future__ import annotations

from ..types import Example


def build_sft_example(example: Example, system_prompt: str = "You are a helpful assistant.") -> dict:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example.instruction},
            {"role": "assistant", "content": example.output},
        ],
        "format": "sft",
        "source_chunks": list(example.source_chunks),
    }


def build_multiturn_example(
    example: Example,
    follow_up_instruction: str,
    follow_up_output: str,
    system_prompt: str = "You are a helpful assistant.",
) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example.instruction},
            {"role": "assistant", "content": example.output},
            {"role": "user", "content": follow_up_instruction.strip()},
            {"role": "assistant", "content": follow_up_output.strip()},
        ],
        "format": "multiturn",
        "source_chunks": list(example.source_chunks),
    }
