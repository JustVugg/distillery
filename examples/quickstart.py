"""End-to-end example: PDF in, dataset out.

Run from the project root after `pip install -e '.[embeddings]'` and ensure an
Ollama server is reachable (or set DISTILLERY_PROVIDER=openai).

    python examples/quickstart.py path/to/doc.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

from distillery.config import load_settings
from distillery.export.jsonl import export_jsonl, export_openai_messages
from distillery.export.split import train_eval_split
from distillery.ingest.chunker import chunk_text
from distillery.ingest.pdf import load_pdf
from distillery.pipeline import PipelineConfig, run_pipeline
from distillery.providers.embeddings import build_embedder
from distillery.providers.llm import build_provider


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python examples/quickstart.py <pdf_path>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    settings = load_settings()

    raw_text = load_pdf(pdf_path)
    chunks = list(chunk_text(raw_text, source=str(pdf_path), target_chars=1200, overlap_chars=150))
    print(f"{len(chunks)} chunks from {pdf_path}")

    provider = build_provider(settings)
    embedder = build_embedder(settings.embedding_model)

    config = PipelineConfig(
        description="Technical assistant for the product described in the PDF.",
        target_examples=80,
        seeds_per_chunk=4,
        seeds_from_description=20,
        min_judge_score=7,
    )

    result = run_pipeline(
        config=config,
        chunks=chunks,
        provider=provider,
        embedder=embedder,
        progress=lambda f, m: print(f"  {f * 100:5.1f}%  {m}"),
    )

    train, eval_ = train_eval_split(result.examples, eval_fraction=0.1)
    out_dir = settings.output_dir / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    export_jsonl(out_dir / "train.jsonl", train)
    export_jsonl(out_dir / "eval.jsonl", eval_)
    export_openai_messages(out_dir / "openai.jsonl", train)

    print(f"\nstats: {result.stats.to_dict()}")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
