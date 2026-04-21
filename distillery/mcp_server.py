"""Thin MCP server exposing Distillery's dataset-generation pipeline.

This is a light wrapper over the library. Jobs run in a ThreadPoolExecutor and
are persisted to a small SQLite table so the transport never blocks. Callers
interact via:

    generate_from_text / generate_from_pdf / generate_from_url
    get_job_status
    list_jobs
    cancel_job
    wait_for_job

All tool calls return a `job_id` immediately and the agent polls. Results land
in the configured workspace directory.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool
except ImportError as exc:  # pragma: no cover - only hit when extras missing
    raise RuntimeError(
        "MCP dependency missing. Install with: pip install 'distillery[mcp]'"
    ) from exc

from .config import load_settings
from .export.datacard import DatasetCardInfo, write_dataset_card
from .export.jsonl import (
    export_dpo_jsonl,
    export_jsonl,
    export_legacy_instruction_json,
    export_openai_messages,
)
from .export.split import train_eval_split
from .ingest.chunker import chunk_text
from .ingest.pdf import load_pdf
from .ingest.url import load_url
from .pipeline import PipelineConfig, run_pipeline
from .providers.cache import CachingProvider, LLMCache
from .providers.embeddings import build_embedder
from .providers.llm import build_provider
from .types import Chunk
from .utils import write_json


log = logging.getLogger(__name__)


TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "generate_from_text": {
        "description": (
            "Generate a grounded instruction dataset from a block of text. Returns a "
            "job_id immediately; poll with get_job_status or use wait_for_job."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["description", "text"],
            "properties": {
                "description": {"type": "string", "minLength": 10},
                "text": {"type": "string", "minLength": 100},
                "target_examples": {"type": "integer", "minimum": 8, "maximum": 5000, "default": 200},
                "language": {"type": "string", "default": "English"},
                "min_judge_score": {"type": "integer", "minimum": 1, "maximum": 10, "default": 7},
                "diversity_threshold": {"type": "number", "default": 0.9},
                "concurrency": {"type": "integer", "minimum": 1, "maximum": 32, "default": 4},
                "evol_factor": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.0},
                "generate_dpo_pairs": {"type": "boolean", "default": False},
                "name": {"type": "string"},
            },
        },
    },
    "generate_from_pdf": {
        "description": "Like generate_from_text, but ingests a PDF from the filesystem first.",
        "inputSchema": {
            "type": "object",
            "required": ["description", "pdf_path"],
            "properties": {
                "description": {"type": "string", "minLength": 10},
                "pdf_path": {"type": "string"},
                "target_examples": {"type": "integer", "minimum": 8, "maximum": 5000, "default": 200},
                "language": {"type": "string", "default": "English"},
                "min_judge_score": {"type": "integer", "minimum": 1, "maximum": 10, "default": 7},
                "diversity_threshold": {"type": "number", "default": 0.9},
                "concurrency": {"type": "integer", "minimum": 1, "maximum": 32, "default": 4},
                "evol_factor": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.0},
                "generate_dpo_pairs": {"type": "boolean", "default": False},
                "name": {"type": "string"},
            },
        },
    },
    "generate_from_url": {
        "description": "Like generate_from_text, but ingests a single URL first.",
        "inputSchema": {
            "type": "object",
            "required": ["description", "url"],
            "properties": {
                "description": {"type": "string", "minLength": 10},
                "url": {"type": "string"},
                "target_examples": {"type": "integer", "minimum": 8, "maximum": 5000, "default": 200},
                "language": {"type": "string", "default": "English"},
                "min_judge_score": {"type": "integer", "minimum": 1, "maximum": 10, "default": 7},
                "diversity_threshold": {"type": "number", "default": 0.9},
                "concurrency": {"type": "integer", "minimum": 1, "maximum": 32, "default": 4},
                "evol_factor": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.0},
                "generate_dpo_pairs": {"type": "boolean", "default": False},
                "name": {"type": "string"},
            },
        },
    },
    "get_job_status": {
        "description": "Fetch current status, progress, and result for a job.",
        "inputSchema": {
            "type": "object",
            "required": ["job_id"],
            "properties": {"job_id": {"type": "string"}},
        },
    },
    "list_jobs": {
        "description": "List recent dataset jobs (newest first).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["queued", "running", "succeeded", "failed", "cancelled"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20},
            },
        },
    },
    "cancel_job": {
        "description": "Request cooperative cancellation of a running job.",
        "inputSchema": {
            "type": "object",
            "required": ["job_id"],
            "properties": {"job_id": {"type": "string"}},
        },
    },
    "wait_for_job": {
        "description": "Block until a job is terminal or timeout expires, then return its final record.",
        "inputSchema": {
            "type": "object",
            "required": ["job_id"],
            "properties": {
                "job_id": {"type": "string"},
                "timeout_sec": {"type": "number", "minimum": 1, "maximum": 86400, "default": 600},
                "poll_interval_sec": {"type": "number", "minimum": 0.5, "maximum": 30, "default": 2.0},
            },
        },
    },
    "health": {
        "description": "Distillery server health: workspace, provider, cache stats.",
        "inputSchema": {"type": "object", "properties": {}},
    },
}


# ---------------------------------------------------------------------------
# Job store — minimal SQLite
# ---------------------------------------------------------------------------


@dataclass
class Job:
    id: str
    kind: str
    params: dict
    status: str = "queued"
    progress: float = 0.0
    message: str = ""
    result: dict = field(default_factory=dict)
    error: str | None = None
    created_at: float = 0.0
    finished_at: float | None = None
    cancel_requested: bool = False

    def to_public(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "params": self.params,
            "status": self.status,
            "progress": round(self.progress, 3),
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "cancel_requested": self.cancel_requested,
        }


class MiniStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                params TEXT NOT NULL,
                status TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0.0,
                message TEXT NOT NULL DEFAULT '',
                result TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                created_at REAL NOT NULL,
                finished_at REAL,
                cancel_requested INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.commit()

    def create(self, kind: str, params: dict) -> Job:
        job = Job(id=str(uuid.uuid4()), kind=kind, params=params, created_at=time.time())
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, kind, params, status, progress, message, result, error, created_at) "
                "VALUES (?, ?, ?, ?, 0, '', '{}', NULL, ?)",
                (job.id, kind, json.dumps(params), job.status, job.created_at),
            )
            self._conn.commit()
        return job

    def update(self, job_id: str, **fields) -> None:
        if not fields:
            return
        cols = []
        vals = []
        for k, v in fields.items():
            if k in ("params", "result"):
                v = json.dumps(v, ensure_ascii=False)
            if k == "cancel_requested":
                v = 1 if v else 0
            cols.append(f"{k} = ?")
            vals.append(v)
        vals.append(job_id)
        with self._lock:
            self._conn.execute(f"UPDATE jobs SET {', '.join(cols)} WHERE id = ?", vals)
            self._conn.commit()

    def request_cancel(self, job_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET cancel_requested = 1 WHERE id = ? AND status IN ('queued', 'running')",
                (job_id,),
            )
            self._conn.commit()
            return (cur.rowcount or 0) > 0

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cur.fetchone()
            cols = [d[0] for d in cur.description]
        if row is None:
            return None
        return self._row_to_job(dict(zip(cols, row)))

    def list(self, *, status: str | None = None, limit: int = 20) -> list[Job]:
        q = "SELECT * FROM jobs"
        vals: list[Any] = []
        if status:
            q += " WHERE status = ?"
            vals.append(status)
        q += " ORDER BY created_at DESC LIMIT ?"
        vals.append(max(1, min(limit, 200)))
        with self._lock:
            cur = self._conn.execute(q, vals)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
        return [self._row_to_job(dict(zip(cols, r))) for r in rows]

    def mark_orphans_failed(self) -> int:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status = 'failed', error = COALESCE(error, 'process restart'), finished_at = ? "
                "WHERE status IN ('queued', 'running')",
                (time.time(),),
            )
            self._conn.commit()
            return cur.rowcount or 0

    def cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return bool(row and row[0])

    @staticmethod
    def _row_to_job(row: dict) -> Job:
        return Job(
            id=row["id"],
            kind=row["kind"],
            params=json.loads(row["params"]) if row["params"] else {},
            status=row["status"],
            progress=row["progress"] or 0.0,
            message=row["message"] or "",
            result=json.loads(row["result"]) if row["result"] else {},
            error=row["error"],
            created_at=row["created_at"],
            finished_at=row["finished_at"],
            cancel_requested=bool(row.get("cancel_requested")),
        )


# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------


class PipelineCancelled(Exception):
    pass


def _make_progress_cb(store: MiniStore, job_id: str):
    def cb(frac: float, msg: str) -> None:
        store.update(job_id, progress=float(frac), message=str(msg))
        if store.cancel_requested(job_id):
            raise PipelineCancelled(f"job {job_id} cancelled")
    return cb


def _run_dataset_job(
    store: MiniStore,
    job_id: str,
    params: dict,
    *,
    workspace: Path,
    cache_path: Path | None,
) -> None:
    store.update(job_id, status="running", message="starting")
    try:
        chunks = _collect_chunks_for_params(params)
        settings = load_settings()
        provider = build_provider(settings)
        embedder = build_embedder(settings.embedding_model)
        if cache_path is not None:
            cache = LLMCache(cache_path)
            provider = CachingProvider(provider, cache, default_model=settings.generator_model)

        cfg = PipelineConfig(
            description=params["description"],
            language=params.get("language", "English"),
            target_examples=int(params.get("target_examples", 200)),
            min_judge_score=int(params.get("min_judge_score", 7)),
            diversity_threshold=float(params.get("diversity_threshold", 0.9)),
            generator_model=settings.generator_model,
            judge_model=settings.judge_model,
            generate_dpo_pairs=bool(params.get("generate_dpo_pairs", False)),
            concurrency=int(params.get("concurrency", 4)),
            evol_factor=float(params.get("evol_factor", 0.0)),
        )

        name = params.get("name") or params["description"][:40]
        slug = "".join(c if c.isalnum() else "-" for c in name).strip("-").lower()[:48] or "run"
        stem = f"{slug}_{int(time.time())}"
        out_dir = workspace / "datasets" / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        cb = _make_progress_cb(store, job_id)
        result = run_pipeline(
            config=cfg,
            chunks=chunks,
            provider=provider,
            embedder=embedder,
            progress=cb,
        )

        train, eval_ = train_eval_split(result.examples, eval_fraction=0.1)
        files: dict[str, str] = {}

        p = out_dir / f"{stem}.train.jsonl"
        export_jsonl(p, train)
        files["train"] = str(p)

        p = out_dir / f"{stem}.eval.jsonl"
        export_jsonl(p, eval_)
        files["eval"] = str(p)

        p = out_dir / f"{stem}.openai.jsonl"
        export_openai_messages(p, train)
        files["openai"] = str(p)

        p = out_dir / f"{stem}.flat.json"
        export_legacy_instruction_json(p, train)
        files["flat"] = str(p)

        p = out_dir / f"{stem}.rejected.jsonl"
        export_jsonl(p, result.rejected)
        files["rejected"] = str(p)

        if result.dpo_pairs:
            p = out_dir / f"{stem}.dpo.jsonl"
            export_dpo_jsonl(p, result.dpo_pairs)
            files["dpo"] = str(p)

        write_dataset_card(
            out_dir / "README.md",
            DatasetCardInfo(
                title=f"Distillery — {name}"[:96],
                description=params["description"],
                language=cfg.language,
                source_description=params.get("_source_description", "provided text"),
                license="mit",
                train_count=len(train),
                eval_count=len(eval_),
                dpo_count=len(result.dpo_pairs),
                rejected_count=len(result.rejected),
                provider=settings.provider,
                generator_model=settings.generator_model,
                judge_model=settings.judge_model,
                embedding_model=settings.embedding_model,
                stats=result.stats.to_dict(),
                config={
                    "min_judge_score": cfg.min_judge_score,
                    "diversity_threshold": cfg.diversity_threshold,
                    "min_hallucination_overlap": cfg.min_hallucination_overlap,
                    "target_examples": cfg.target_examples,
                },
                tags=[cfg.language.lower()],
            ),
        )

        manifest = {
            "stats": result.stats.to_dict(),
            "files": files,
            "output_dir": str(out_dir),
        }
        write_json(out_dir / f"{stem}.manifest.json", manifest)

        store.update(
            job_id,
            status="succeeded",
            progress=1.0,
            message="completed",
            result=manifest,
            finished_at=time.time(),
        )
    except PipelineCancelled as exc:
        store.update(
            job_id,
            status="cancelled",
            message=str(exc),
            finished_at=time.time(),
        )
    except Exception as exc:  # noqa: BLE001
        trace = traceback.format_exc()
        log.error("job %s failed: %s", job_id, trace)
        store.update(
            job_id,
            status="failed",
            error=f"{exc}\n{trace}",
            message=f"failed: {exc}",
            finished_at=time.time(),
        )


def _collect_chunks_for_params(params: dict) -> list[Chunk]:
    if "text" in params:
        body = params["text"]
        source = params.get("_source_tag", "inline-text")
    elif "pdf_path" in params:
        path = Path(params["pdf_path"])
        body = load_pdf(path)
        source = str(path)
    elif "url" in params:
        body = load_url(params["url"])
        source = params["url"]
    else:
        raise ValueError("need one of: text, pdf_path, url")
    return list(chunk_text(body, source=source))


# ---------------------------------------------------------------------------
# Server wiring
# ---------------------------------------------------------------------------


def _text(payload: Any) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))]


def _workspace_root() -> Path:
    return Path(os.getenv("DISTILLERY_WORKSPACE", "./distillery_workspace")).expanduser().resolve()


def build_server() -> Server:
    app: Server = Server("distillery")
    workspace = _workspace_root()
    workspace.mkdir(parents=True, exist_ok=True)

    store = MiniStore(workspace / "jobs.sqlite")
    store.mark_orphans_failed()

    cache_path: Path | None = None
    if os.getenv("DISTILLERY_CACHE_LLM", "").lower() in ("1", "true", "yes"):
        cache_path = workspace / "llm_cache.sqlite"

    max_workers = max(1, int(os.getenv("DISTILLERY_MAX_JOBS", "1")))
    executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="distillery-job")

    @app.list_tools()
    async def _list() -> list[Tool]:
        return [
            Tool(name=n, description=s["description"], inputSchema=s["inputSchema"])
            for n, s in TOOL_SCHEMAS.items()
        ]

    def _submit(kind: str, params: dict) -> Job:
        job = store.create(kind, params)
        executor.submit(_run_dataset_job, store, job.id, params, workspace=workspace, cache_path=cache_path)
        return job

    @app.call_tool()
    async def _call(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        arguments = arguments or {}
        try:
            if name == "generate_from_text":
                arguments.setdefault("_source_description", "provided inline text")
                job = _submit("generate_from_text", arguments)
                return _text({"job_id": job.id, "status": job.status})

            if name == "generate_from_pdf":
                arguments.setdefault("_source_description", f"PDF at {arguments.get('pdf_path')}")
                job = _submit("generate_from_pdf", arguments)
                return _text({"job_id": job.id, "status": job.status})

            if name == "generate_from_url":
                arguments.setdefault("_source_description", f"URL {arguments.get('url')}")
                job = _submit("generate_from_url", arguments)
                return _text({"job_id": job.id, "status": job.status})

            if name == "get_job_status":
                job = store.get(str(arguments["job_id"]))
                if job is None:
                    return _text({"error": "not found"})
                return _text(job.to_public())

            if name == "list_jobs":
                jobs = store.list(
                    status=arguments.get("status"),
                    limit=int(arguments.get("limit", 20)),
                )
                return _text({"jobs": [j.to_public() for j in jobs]})

            if name == "cancel_job":
                jid = str(arguments["job_id"])
                ok = store.request_cancel(jid)
                job = store.get(jid)
                return _text({
                    "ok": ok,
                    "job": job.to_public() if job else None,
                })

            if name == "wait_for_job":
                jid = str(arguments["job_id"])
                timeout = float(arguments.get("timeout_sec", 600))
                interval = max(0.5, float(arguments.get("poll_interval_sec", 2.0)))
                deadline = time.time() + timeout
                job = store.get(jid)
                if job is None:
                    return _text({"error": "not found"})
                while time.time() < deadline and job.status in ("queued", "running"):
                    await asyncio.sleep(interval)
                    job = store.get(jid)
                    if job is None:
                        return _text({"error": "vanished"})
                return _text({
                    "timed_out": job.status in ("queued", "running"),
                    "job": job.to_public(),
                })

            if name == "health":
                cache_stats = None
                if cache_path is not None:
                    cache_stats = LLMCache(cache_path).stats()
                settings = load_settings()
                return _text({
                    "workspace": str(workspace),
                    "provider": settings.provider,
                    "generator_model": settings.generator_model,
                    "judge_model": settings.judge_model,
                    "cache_enabled": cache_path is not None,
                    "cache": cache_stats,
                    "max_workers": max_workers,
                })

            return _text({"error": f"unknown tool: {name}"})
        except Exception as exc:  # noqa: BLE001
            log.exception("tool %s failed", name)
            return _text({"error": f"{type(exc).__name__}: {exc}"})

    return app


async def _run_async() -> None:
    app = build_server()
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


def main() -> None:
    logging.basicConfig(
        level=os.getenv("DISTILLERY_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_run_async())


if __name__ == "__main__":
    main()
