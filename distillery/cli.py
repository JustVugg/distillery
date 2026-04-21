from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .config import load_settings
from .export.datacard import DatasetCardInfo, write_dataset_card
from .export.jsonl import (
    export_dpo_jsonl,
    export_jsonl,
    export_legacy_instruction_json,
    export_multiturn_jsonl,
    export_openai_messages,
    export_tool_calling_jsonl,
)
from .export.split import train_eval_split
from .ingest.chunker import chunk_text
from .ingest.pdf import load_pdf
from .ingest.text import load_text
from .ingest.url import load_url
from .pipeline import (
    MultiturnConfig,
    PipelineConfig,
    run_multiturn_pipeline,
    run_pipeline,
)
from .providers.cache import CachingProvider, LLMCache
from .providers.embeddings import build_embedder
from .providers.llm import build_provider
from .types import Chunk
from .utils import write_json


app = typer.Typer(
    name="distillery",
    help="Turn documents into high-quality instruction datasets.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()


def _collect_chunks(
    pdfs: list[Path],
    texts: list[Path],
    urls: list[str],
    target_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    out: list[Chunk] = []
    for pdf in pdfs:
        console.print(f"[cyan]ingesting pdf[/cyan] {pdf}")
        body = load_pdf(pdf)
        out.extend(chunk_text(body, source=str(pdf), target_chars=target_chars, overlap_chars=overlap_chars))
    for path in texts:
        console.print(f"[cyan]ingesting text[/cyan] {path}")
        body = load_text(path)
        out.extend(chunk_text(body, source=str(path), target_chars=target_chars, overlap_chars=overlap_chars))
    for url in urls:
        console.print(f"[cyan]ingesting url[/cyan] {url}")
        body = load_url(url)
        out.extend(chunk_text(body, source=url, target_chars=target_chars, overlap_chars=overlap_chars))
    return out


@app.command()
def ingest(
    output: Path = typer.Option(..., "--output", "-o", help="Where to write the chunk JSON"),
    pdf: list[Path] = typer.Option(default=[], help="PDF files to ingest (repeatable)"),
    text: list[Path] = typer.Option(default=[], help="Text/markdown files (repeatable)"),
    url: list[str] = typer.Option(default=[], help="URLs (repeatable)"),
    target_chars: int = typer.Option(1200, help="Target chunk size in characters"),
    overlap_chars: int = typer.Option(150, help="Overlap between consecutive chunks"),
) -> None:
    """Extract and chunk source material into a reusable chunk file."""
    if not pdf and not text and not url:
        raise typer.BadParameter("Provide at least one --pdf, --text, or --url source.")
    chunks = _collect_chunks(pdf, text, url, target_chars, overlap_chars)
    payload = [
        {
            "id": c.id,
            "text": c.text,
            "source": c.source,
            "index": c.index,
            "metadata": c.metadata,
        }
        for c in chunks
    ]
    write_json(output, payload)
    console.print(f"[green]wrote[/green] {len(payload)} chunks → {output}")


def _load_chunks_from_json(path: Path) -> list[Chunk]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[Chunk] = []
    for item in raw:
        out.append(
            Chunk(
                id=item["id"],
                text=item["text"],
                source=item["source"],
                index=int(item.get("index", 0)),
                metadata=dict(item.get("metadata") or {}),
            )
        )
    return out


@app.command()
def generate(
    description: str = typer.Option(..., "--description", "-d", help="Purpose of the assistant"),
    output_dir: Path = typer.Option(..., "--output-dir", "-o"),
    chunks_file: Optional[Path] = typer.Option(None, "--chunks", help="JSON from `distillery ingest`"),
    pdf: list[Path] = typer.Option(default=[], help="PDFs to ingest inline"),
    text: list[Path] = typer.Option(default=[], help="Text files to ingest inline"),
    url: list[str] = typer.Option(default=[], help="URLs to ingest inline"),
    target_examples: int = typer.Option(200, "--target", "-n"),
    language: str = typer.Option("English", "--language", "-l"),
    min_judge_score: int = typer.Option(7),
    diversity_threshold: float = typer.Option(0.9),
    min_hallucination_overlap: float = typer.Option(0.35),
    seeds_per_chunk: int = typer.Option(6),
    seeds_from_description: int = typer.Option(40),
    generate_dpo: bool = typer.Option(False, "--dpo/--no-dpo"),
    dpo_target_pairs: int = typer.Option(50),
    target_chars: int = typer.Option(1200),
    overlap_chars: int = typer.Option(150),
    eval_fraction: float = typer.Option(0.1),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Parallel LLM calls"),
    cache_path: Optional[Path] = typer.Option(
        None, "--cache", help="SQLite cache path for LLM calls (caches deterministic calls only)"
    ),
    resume: Optional[Path] = typer.Option(
        None, "--resume", help="Checkpoint file — reused across re-runs to resume on crash"
    ),
    evol_factor: float = typer.Option(0.0, "--evol", help="Evol-Instruct fraction (0.0-1.0)"),
    min_semantic_similarity: Optional[float] = typer.Option(
        None, "--min-semantic", help="Require embedding-cosine similarity >= N to pass the grounding check"
    ),
    datacard: bool = typer.Option(True, "--datacard/--no-datacard", help="Write HF-style dataset card"),
    tool_calling: bool = typer.Option(
        False, "--tool-calling/--no-tool-calling", help="Also export tool-calling JSONL (examples must carry metadata.tool_call)"
    ),
    formats: str = typer.Option(
        "jsonl,openai,flat",
        help="Comma-separated subset of: jsonl, openai, flat",
    ),
    verbose: bool = typer.Option(False, "--verbose/--quiet"),
) -> None:
    """Run the full generation pipeline and export to disk."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = load_settings()
    provider = build_provider(settings)
    embedder = build_embedder(settings.embedding_model)

    cache: LLMCache | None = None
    if cache_path is not None:
        cache = LLMCache(cache_path)
        provider = CachingProvider(provider, cache, default_model=settings.generator_model)

    chunks: list[Chunk] = []
    if chunks_file is not None:
        chunks.extend(_load_chunks_from_json(chunks_file))
    if pdf or text or url:
        chunks.extend(_collect_chunks(pdf, text, url, target_chars, overlap_chars))

    console.print(
        f"[cyan]pipeline start[/cyan] provider={settings.provider} "
        f"chunks={len(chunks)} target={target_examples} concurrency={concurrency}"
    )

    cfg = PipelineConfig(
        description=description,
        language=language,
        target_examples=target_examples,
        seeds_per_chunk=seeds_per_chunk,
        seeds_from_description=seeds_from_description,
        diversity_threshold=diversity_threshold,
        min_judge_score=min_judge_score,
        min_hallucination_overlap=min_hallucination_overlap,
        min_semantic_similarity=min_semantic_similarity,
        generator_model=settings.generator_model,
        judge_model=settings.judge_model,
        generate_dpo_pairs=generate_dpo,
        dpo_target_pairs=dpo_target_pairs,
        concurrency=concurrency,
        evol_factor=evol_factor,
        checkpoint_path=resume,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task("starting", total=None)

        def cb(frac: float, msg: str) -> None:
            progress.update(task_id, description=f"{msg} ({frac * 100:.0f}%)")

        output = run_pipeline(
            config=cfg,
            chunks=chunks,
            provider=provider,
            embedder=embedder,
            progress=cb,
        )

    wanted = {name.strip().lower() for name in formats.split(",") if name.strip()}
    stem = f"dataset_{int(time.time())}"

    train, eval_ = train_eval_split(output.examples, eval_fraction=eval_fraction)

    manifest: dict[str, object] = {
        "stats": output.stats.to_dict(),
        "train_size": len(train),
        "eval_size": len(eval_),
        "config": {
            "description": description,
            "language": language,
            "target_examples": target_examples,
            "min_judge_score": min_judge_score,
            "diversity_threshold": diversity_threshold,
            "min_hallucination_overlap": min_hallucination_overlap,
            "provider": settings.provider,
            "generator_model": settings.generator_model,
            "judge_model": settings.judge_model,
            "embedding_model": settings.embedding_model,
        },
        "files": {},
    }

    if "jsonl" in wanted:
        train_path = output_dir / f"{stem}.train.jsonl"
        eval_path = output_dir / f"{stem}.eval.jsonl"
        export_jsonl(train_path, train)
        export_jsonl(eval_path, eval_)
        manifest["files"]["jsonl_train"] = str(train_path)
        manifest["files"]["jsonl_eval"] = str(eval_path)

    if "openai" in wanted:
        path = output_dir / f"{stem}.openai.jsonl"
        export_openai_messages(path, train)
        manifest["files"]["openai_messages"] = str(path)

    if "flat" in wanted:
        path = output_dir / f"{stem}.flat.json"
        export_legacy_instruction_json(path, train)
        manifest["files"]["flat_json"] = str(path)

    if output.dpo_pairs:
        dpo_path = output_dir / f"{stem}.dpo.jsonl"
        export_dpo_jsonl(dpo_path, output.dpo_pairs)
        manifest["files"]["dpo"] = str(dpo_path)

    rejected_path = output_dir / f"{stem}.rejected.jsonl"
    export_jsonl(rejected_path, output.rejected)
    manifest["files"]["rejected"] = str(rejected_path)

    if tool_calling:
        tc_path = output_dir / f"{stem}.tool_calling.jsonl"
        count = export_tool_calling_jsonl(tc_path, train)
        manifest["files"]["tool_calling"] = str(tc_path)
        manifest["tool_calling_count"] = count

    if datacard:
        card_path = output_dir / "README.md"
        write_dataset_card(
            card_path,
            DatasetCardInfo(
                title=f"Distillery dataset — {description[:60]}",
                description=description,
                language=language,
                source_description=_describe_sources(pdf, text, url, chunks_file),
                license="mit",
                train_count=len(train),
                eval_count=len(eval_),
                dpo_count=len(output.dpo_pairs),
                rejected_count=len(output.rejected),
                provider=settings.provider,
                generator_model=settings.generator_model,
                judge_model=settings.judge_model,
                embedding_model=settings.embedding_model,
                stats=output.stats.to_dict(),
                config={
                    "min_judge_score": min_judge_score,
                    "diversity_threshold": diversity_threshold,
                    "min_hallucination_overlap": min_hallucination_overlap,
                    "target_examples": target_examples,
                },
                tags=[language.lower()],
            ),
        )
        manifest["files"]["dataset_card"] = str(card_path)

    if cache is not None:
        manifest["cache"] = cache.stats()

    manifest_path = output_dir / f"{stem}.manifest.json"
    write_json(manifest_path, manifest)

    _print_summary(output.stats, manifest_path, manifest["files"])


def _describe_sources(
    pdfs: list[Path],
    texts: list[Path],
    urls: list[str],
    chunks_file: Path | None,
) -> str:
    parts: list[str] = []
    if chunks_file is not None:
        parts.append(f"pre-ingested chunks from `{chunks_file}`")
    if pdfs:
        parts.append(f"PDFs: {', '.join(str(p) for p in pdfs)}")
    if texts:
        parts.append(f"text files: {', '.join(str(p) for p in texts)}")
    if urls:
        parts.append(f"URLs: {', '.join(urls)}")
    if not parts:
        return "a product description (no source documents provided)"
    return "; ".join(parts)


@app.command()
def multiturn(
    description: str = typer.Option(..., "--description", "-d"),
    output: Path = typer.Option(..., "--output", "-o", help="JSONL output path"),
    chunks_file: Optional[Path] = typer.Option(None, "--chunks"),
    target: int = typer.Option(100, "--target", "-n"),
    turns: int = typer.Option(4, "--turns"),
    language: str = typer.Option("English", "--language"),
    concurrency: int = typer.Option(1, "--concurrency", "-c"),
    system_prompt: Optional[str] = typer.Option(None, "--system"),
) -> None:
    """Generate multi-turn dialogues as OpenAI-messages JSONL."""
    settings = load_settings()
    provider = build_provider(settings)
    chunks: list[Chunk] = []
    if chunks_file is not None:
        chunks.extend(_load_chunks_from_json(chunks_file))

    cfg = MultiturnConfig(
        description=description,
        language=language,
        target_dialogues=target,
        turns=turns,
        model=settings.generator_model,
        concurrency=concurrency,
    )
    with Progress(SpinnerColumn(), TextColumn("[bold cyan]{task.description}"), TimeElapsedColumn()) as p:
        tid = p.add_task("generating", total=None)

        def cb(frac, msg):
            p.update(tid, description=f"{msg} ({frac * 100:.0f}%)")

        result = run_multiturn_pipeline(config=cfg, chunks=chunks, provider=provider, progress=cb)
    count = export_multiturn_jsonl(output, result.dialogues, system_prompt=system_prompt)
    console.print(f"[green]wrote[/green] {count} dialogues → {output}")
    console.print_json(json.dumps(result.stats))


@app.command()
def cache_info(
    cache_path: Path = typer.Option(..., "--cache", "-c"),
) -> None:
    """Inspect an LLM call cache."""
    cache = LLMCache(cache_path)
    console.print_json(json.dumps(cache.stats()))


def _print_summary(stats, manifest_path: Path, files: dict) -> None:
    table = Table(title="Distillery pipeline summary", show_header=True, header_style="bold cyan")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key, value in stats.to_dict().items():
        table.add_row(key, str(value))
    console.print(table)

    if files:
        files_table = Table(title="Artifacts", show_header=True, header_style="bold cyan")
        files_table.add_column("name")
        files_table.add_column("path")
        for key, path in files.items():
            files_table.add_row(key, str(path))
        console.print(files_table)

    console.print(f"[green]manifest:[/green] {manifest_path}")


@app.command()
def version() -> None:
    """Print the installed version."""
    from . import __version__
    console.print(__version__)


if __name__ == "__main__":
    app()
