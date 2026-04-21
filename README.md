# Distillery

**Turn your documents into high-quality instruction datasets — with grounding, quality filtering, and provenance baked in.**

Distillery is a focused Python library and CLI for building training datasets from real sources (PDFs, text files, URLs). It is opinionated about what a *good* dataset looks like:

- every example is **grounded** in a specific chunk of the source material (or flagged as free-form)
- every example is **graded** by an LLM judge against its reference context
- every example is **deduplicated** by instruction embedding, not string match
- every example has **provenance** — you can trace the answer back to the chunk that produced it
- outputs are exported in **every common format** (JSONL, OpenAI `messages`, HF `datasets`, flat `{instruction,output}`, DPO preference pairs)

No platform, no cloud, no account. A CLI, a library, and an optional MCP server.

---

## Why

Most teams fine-tune on garbage datasets because producing a clean one is manual work. Distillery automates the loop:

```
source docs ─► chunking ─► seed proposals ─► grounded answers ─► diversity filter
                                                                      │
                                                                      ▼
                                  LLM-judge score ◄── hallucination check ◄── ...
                                         │
                                         ▼
                   train.jsonl + eval.jsonl + dpo.jsonl + manifest
```

The result is a dataset you'd actually want to fine-tune on, with a `manifest.json` that records exactly what went in, what was rejected, and why.

---

## Quickstart

```bash
pip install -e '.[embeddings]'       # adds sentence-transformers for real diversity
cp .env.example .env
# edit .env: pick provider=ollama or provider=openai

# Option A: ingest → generate in two steps
distillery ingest --pdf docs/handbook.pdf --output chunks.json
distillery generate \
  --chunks chunks.json \
  --description "Internal support assistant for HR policies." \
  --target 300 \
  --output-dir datasets/

# Option B: one-shot
distillery generate \
  --pdf docs/handbook.pdf \
  --description "Internal support assistant for HR policies." \
  --target 300 \
  --output-dir datasets/
```

Output:

```
datasets/
├── dataset_1712345678.train.jsonl        # 270 grounded examples
├── dataset_1712345678.eval.jsonl         # 30 held-out examples
├── dataset_1712345678.openai.jsonl       # same examples in OpenAI messages format
├── dataset_1712345678.flat.json          # flat {instruction,output} for Axolotl/Unsloth
├── dataset_1712345678.rejected.jsonl     # examples that failed judge/diversity/hallucination
└── dataset_1712345678.manifest.json      # stats, config, provenance
```

---

## What makes a Distillery dataset different

| Stage                  | What it does                                                              | Knob |
|------------------------|---------------------------------------------------------------------------|------|
| Chunking               | Sentence-aware overlap, deterministic IDs                                 | `--target-chars`, `--overlap-chars` |
| Seed proposals         | Grounded seeds per chunk + free-form seeds from the description           | `--seeds-per-chunk`, `--seeds-from-description` |
| Answer generation      | Uses the originating chunk as authoritative context                       | `--language`, provider/model in `.env` |
| Diversity filter       | Cosine-similarity embedding dedup, not string hashing                     | `--diversity-threshold` |
| Hallucination check    | Content-word overlap between answer and source chunk                      | `--min-hallucination-overlap` |
| LLM-judge              | Scores each example 1-10 with reason, rejects below threshold             | `--min-judge-score` |
| DPO pair generation    | Optional — weak answer + strong answer scored and kept only if weak loses | `--dpo`, `--dpo-target-pairs` |
| Train/eval split       | Deterministic, reproducible                                               | `--eval-fraction` |

Additional knobs for scale and iteration:

| Flag                   | What it does                                                                 |
|------------------------|------------------------------------------------------------------------------|
| `--concurrency N`      | Run up to N LLM calls in parallel (asyncio, semaphore-bounded)               |
| `--cache path.sqlite`  | Memoize deterministic (temperature ≤ 0) LLM calls in SQLite                  |
| `--resume ckpt.jsonl`  | Append-only checkpoint: resume a crashed run from the last processed seed    |
| `--evol 0.4`           | Evol-Instruct mutation fraction — rewrites seeds with WizardLM-style prompts |
| `--min-semantic 0.55`  | Require embedding-cosine similarity with source to pass the grounding check  |
| `--tool-calling`       | Also export OpenAI tool-calling JSONL (from examples with `metadata.tool_call`) |
| `--datacard/--no-datacard` | Toggle HF-style `README.md` dataset card                                 |

A separate `distillery multiturn` command generates OpenAI-format multi-turn dialogues, and `distillery cache-info --cache ...` inspects the LLM cache.

---

## Providers

Distillery works with anything OpenAI-compatible. Set `DISTILLERY_PROVIDER` in `.env`:

```bash
# Local Ollama
DISTILLERY_PROVIDER=ollama
DISTILLERY_OLLAMA_BASE_URL=http://127.0.0.1:11434
DISTILLERY_GENERATOR_MODEL=llama3.1:8b
DISTILLERY_JUDGE_MODEL=llama3.1:8b

# OpenAI
DISTILLERY_PROVIDER=openai
DISTILLERY_OPENAI_BASE_URL=https://api.openai.com/v1
DISTILLERY_OPENAI_API_KEY=sk-...
DISTILLERY_GENERATOR_MODEL=gpt-4o-mini
DISTILLERY_JUDGE_MODEL=gpt-4o-mini

# Together / Fireworks / Groq / vLLM / LM Studio
DISTILLERY_PROVIDER=openai
DISTILLERY_OPENAI_BASE_URL=https://api.together.xyz/v1
DISTILLERY_OPENAI_API_KEY=...
DISTILLERY_GENERATOR_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
```

Embeddings default to a deterministic hashing embedder so the tool runs with zero ML deps. Install the `embeddings` extra to switch to sentence-transformers:

```bash
pip install -e '.[embeddings]'
```

---

## Library usage

For integration into your own pipeline:

```python
from distillery.config import load_settings
from distillery.ingest.chunker import chunk_text
from distillery.ingest.pdf import load_pdf
from distillery.pipeline import PipelineConfig, run_pipeline
from distillery.providers.embeddings import build_embedder
from distillery.providers.llm import build_provider
from distillery.export.jsonl import export_jsonl
from distillery.export.split import train_eval_split

settings = load_settings()
provider = build_provider(settings)
embedder = build_embedder(settings.embedding_model)

chunks = list(chunk_text(load_pdf("docs/handbook.pdf"), source="handbook.pdf"))

result = run_pipeline(
    config=PipelineConfig(
        description="Internal HR support assistant",
        target_examples=300,
        min_judge_score=7,
    ),
    chunks=chunks,
    provider=provider,
    embedder=embedder,
)

train, eval_ = train_eval_split(result.examples, eval_fraction=0.1)
export_jsonl("train.jsonl", train)
export_jsonl("eval.jsonl", eval_)

print(result.stats.to_dict())
```

See [`examples/quickstart.py`](examples/quickstart.py) for a runnable version.

---

## MCP server (Claude Desktop, Cursor, etc.)

Distillery also ships an MCP server so an LLM agent can drive dataset generation end-to-end. Install with the `mcp` extra and wire the `distillery-mcp` entry point into your MCP client config.

```bash
pip install -e '.[embeddings,mcp]'
```

Claude Desktop example (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "distillery": {
      "command": "distillery-mcp",
      "env": {
        "DISTILLERY_PROVIDER": "ollama",
        "DISTILLERY_GENERATOR_MODEL": "llama3.1:8b",
        "DISTILLERY_JUDGE_MODEL": "llama3.1:8b",
        "DISTILLERY_WORKSPACE": "/absolute/path/to/distillery_workspace",
        "DISTILLERY_CACHE_LLM": "1"
      }
    }
  }
}
```

The server exposes these tools:

| Tool                  | What it does |
|-----------------------|--------------|
| `generate_from_text`  | Start a job from inline text; returns `job_id` immediately |
| `generate_from_pdf`   | Same, but ingests a PDF from disk |
| `generate_from_url`   | Same, but ingests a URL |
| `get_job_status`      | Poll a job's progress, message, and final result manifest |
| `list_jobs`           | List recent jobs, newest first |
| `cancel_job`          | Request cooperative cancellation |
| `wait_for_job`        | Block until the job reaches a terminal state (or timeout) |
| `health`              | Workspace, provider, cache stats |

Jobs run in a thread pool (size via `DISTILLERY_MAX_JOBS`, default 1) and their state is persisted to SQLite inside the workspace. Restarting the server marks previously-running jobs as `failed` so the agent doesn't get stuck waiting on them.

---

## Project layout

```
distillery/
├── cli.py             # typer-based CLI (ingest, generate, multiturn, cache-info, version)
├── mcp_server.py      # Optional MCP server (distillery-mcp)
├── config.py          # Settings from env
├── pipeline.py        # End-to-end orchestration (sync + async parallel)
├── checkpoint.py      # Append-only JSONL checkpoint/resume
├── types.py           # Chunk, Example, PreferencePair, PipelineStats
├── ingest/            # PDF / text / URL ingestion + sentence-aware chunker
├── generate/          # Seed proposals, grounded expansion, evol-instruct, multiturn
├── filter/            # Diversity (embeddings), LLM-judge, hallucination (token+semantic)
├── export/            # JSONL / OpenAI / DPO / flat / tool-calling / HF datacard + split
└── providers/         # LLM + embedding abstractions + SQLite LLM-call cache
```

---

## Development

```bash
pip install -e '.[embeddings,dev]'
pytest
ruff check distillery tests
```

Tests cover chunking, diversity filtering, split determinism, hallucination heuristics, and JSON utilities. They run in under a second with no ML deps.

---

## Limits and non-goals

- **Quality depends on the judge.** A small, weak model will rubber-stamp bad answers. For production datasets use a 70B-class judge via an OpenAI-compatible endpoint, or GPT-4o / Claude Sonnet.
- **No scraping.** URL ingestion expects a single page. For large site crawls, use a dedicated scraper upstream and feed the text in.
- **No reward model training.** DPO pair export is supported; actual DPO/RM training is out of scope — feed the pairs to TRL / Axolotl / your preferred trainer.

---

## License

MIT. See [LICENSE](LICENSE).
