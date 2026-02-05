#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_CMD = os.getenv("MLXSMITH_CLI_CODEX_CMD") or (
    "codex exec --full-auto --disable shell_tool --disable shell_snapshot "
    "-c 'features.collab=false' "
    "-c 'model_reasoning_effort=\"medium\"' "
    "-c 'model_reasoning_summaries=\"never\"' "
    "-c 'mcp_servers.github.enabled=false' "
    "-c 'mcp_servers.aleph.enabled=false' "
    "-c 'mcp_servers.hegelion.enabled=false' "
    "-c 'hooks.agent-turn-complete=[]' "
    "--color never "
    "--model gpt-5.2"
)
DEFAULT_TIMEOUT = float(os.getenv("MLXSMITH_CLI_TIMEOUT", "1800"))


DOC_FILES = [
    "README.md",
    "docs/getting-started.md",
    "docs/COMPATIBILITY.md",
    "docs/ENVIRONMENTS.md",
    "docs/PROJECT_FORMAT.md",
    "docs/VERIFIERS.md",
    "docs/FAQ.md",
    "docs/troubleshooting.md",
    "docs/cli/README.md",
    "docs/cli/configuration.md",
    "docs/cli/model-management.md",
    "docs/cli/project-setup.md",
    "docs/cli/data.md",
    "docs/cli/sft.md",
    "docs/cli/preference-training.md",
    "docs/cli/online-dpo.md",
    "docs/cli/reinforcement-training.md",
    "docs/cli/rlm.md",
    "docs/cli/self-verify.md",
    "docs/cli/judge.md",
    "docs/cli/distillation.md",
    "docs/cli/synthetic-data.md",
    "docs/cli/eval-and-bench.md",
    "docs/cli/serving.md",
    "src/mlxsmith/cli.py",
    "src/mlxsmith/config_models.py",
    "src/mlxsmith/models.py",
]


STOPWORDS = {
    "the",
    "and",
    "or",
    "to",
    "a",
    "an",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "be",
    "as",
    "how",
    "what",
    "why",
    "when",
    "where",
    "show",
    "give",
    "provide",
    "explain",
    "does",
    "do",
    "can",
    "i",
    "we",
    "you",
    "it",
    "this",
    "that",
    "these",
    "those",
    "from",
    "into",
    "via",
    "using",
}


@dataclass(frozen=True)
class Doc:
    path: str
    text: str
    text_lower: str


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    if not text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in cleaned.lower():
        cleaned = re.split(r"</think>", cleaned, flags=re.IGNORECASE)[-1]
    return cleaned.strip()


def _read_jsonl_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = str(row.get("prompt", "")).strip()
            if prompt:
                prompts.append(prompt)
    return prompts


def _load_existing(out: Path) -> set[str]:
    existing: set[str] = set()
    if not out.exists():
        return existing
    with out.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = str(row.get("prompt", "")).strip()
            if prompt:
                existing.add(prompt)
    return existing


def _tokenize_query(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9_./-]{3,}", text.lower())
    out: list[str] = []
    for w in words:
        if w in STOPWORDS:
            continue
        if len(w) < 3:
            continue
        out.append(w)
    # De-dupe but keep order
    seen = set()
    final: list[str] = []
    for w in out:
        if w in seen:
            continue
        seen.add(w)
        final.append(w)
    return final[:12]


def _extract_window(text: str, idx: int, *, before: int, after: int) -> str:
    start = max(0, idx - before)
    end = min(len(text), idx + after)
    snippet = text[start:end].strip("\n")
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet.strip()


def select_context(prompt: str, docs: list[Doc], *, max_chars: int, max_files: int = 2) -> str:
    keys = _tokenize_query(prompt)
    if not keys:
        keys = ["mlxsmith"]

    scored: list[tuple[int, int, Doc]] = []
    for doc in docs:
        score = 0
        first_idx = None
        for k in keys:
            idx = doc.text_lower.find(k)
            if idx != -1:
                score += 1
                if first_idx is None or idx < first_idx:
                    first_idx = idx
        if score > 0:
            scored.append((score, first_idx or 0, doc))

    scored.sort(key=lambda x: (-x[0], x[1], x[2].path))
    chosen = [d for _score, _idx, d in scored[:max_files]]
    if not chosen:
        # fallback to README and CLI
        chosen = [d for d in docs if d.path in ("README.md", "src/mlxsmith/cli.py")][:2]

    parts: list[str] = []
    remaining = max(400, int(max_chars))
    per_file = max(200, remaining // max(1, len(chosen)))

    for doc in chosen:
        idx = None
        for k in keys:
            pos = doc.text_lower.find(k)
            if pos != -1:
                idx = pos if idx is None else min(idx, pos)
        if idx is None:
            idx = 0
        snippet = _extract_window(doc.text, idx, before=450, after=1100)
        snippet = snippet[:per_file]
        parts.append(f"# {doc.path}\n{snippet}")

    ctx = "\n\n".join(parts).strip()
    return ctx[:max_chars].strip()


def build_batch_prompt(items: list[tuple[int, str, str]]) -> str:
    header = (
        "You are a senior engineer answering questions about the MLXSmith repository.\n"
        "Use ONLY the provided context for each question. If the context is insufficient, say so briefly and suggest the most relevant file to read.\n\n"
        "Formatting rules:\n"
        "- Be concise and command-focused.\n"
        "- Put commands and file paths in code fences or backticks.\n"
        "- Do NOT output `<think>` blocks, hidden reasoning, or internal notes.\n\n"
        "Output rules:\n"
        "- Output EXACTLY one `<answer id=\"...\">...</answer>` block per item.\n"
        "- Each answer block must use the matching numeric id.\n"
        "- You may include newlines inside an answer block.\n"
        "- Do not include any other XML tags besides `<answer>`.\n"
        "- Do not include any other text before, between, or after the answer blocks.\n"
        "\n"
        "Example (format only):\n"
        "<answer id=\"1\">...your answer...</answer>\n"
    )
    body_lines: list[str] = []
    for item_id, prompt, context in items:
        body_lines.append(
            f"\n<item id=\"{item_id}\">\n"
            f"<prompt>\n{prompt}\n</prompt>\n"
            f"<context>\n{context}\n</context>\n"
            f"</item>\n"
        )
    return header + "\n".join(body_lines).strip() + "\n"


_ANSWER_RE = re.compile(r"<answer\s+id\s*=\s*\"?(\d+)\"?\s*>(.*?)</answer>", flags=re.DOTALL | re.IGNORECASE)


def parse_answers(text: str) -> dict[int, str]:
    answers: dict[int, str] = {}
    for mid, content in _ANSWER_RE.findall(text or ""):
        try:
            i = int(mid)
        except ValueError:
            continue
        value = content.strip()
        if value:
            answers[i] = strip_think(value)
    return answers


def run_batch(cmd: str, prompt: str, timeout_s: float) -> str:
    proc = subprocess.run(
        shlex.split(cmd),
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"codex exec failed (code {proc.returncode}): {detail}")
    return proc.stdout or ""


def load_docs(root: Path, rel_paths: Iterable[str]) -> list[Doc]:
    docs: list[Doc] = []
    for rel in rel_paths:
        path = (root / rel).resolve()
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        docs.append(Doc(path=str(Path(rel)), text=text, text_lower=text.lower()))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate repo-grounded SFT pairs via codex exec (batched).")
    parser.add_argument("--prompts", required=True, help="Input JSONL with {prompt}")
    parser.add_argument("--out", required=True, help="Output JSONL with {prompt, response}")
    parser.add_argument("--num", type=int, default=300, help="Total pairs to generate")
    parser.add_argument("--batch-size", type=int, default=6, help="Prompts per Codex call")
    parser.add_argument("--max-context-chars", type=int, default=2800, help="Max context characters per prompt")
    parser.add_argument("--cmd", default=DEFAULT_CMD, help="Codex exec command")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Codex timeout seconds")
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds to sleep between batches")
    parser.add_argument("--max-errors", type=int, default=10, help="Abort after this many errors")
    parser.add_argument("--retries-per-batch", type=int, default=3, help="Retries for a single batch before counting an error")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for prompt ordering")
    parser.add_argument("--root", default=".", help="Repo root for context retrieval")
    parser.add_argument("--docs", nargs="*", default=None, help="Override context files list")
    parser.add_argument("--debug-dir", help="If set, write raw codex prompts/outputs here")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    prompts_path = Path(args.prompts)
    if not prompts_path.is_absolute():
        prompts_path = (root / prompts_path).resolve()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc_paths = args.docs if args.docs else DOC_FILES
    docs = load_docs(root, doc_paths)
    if not docs:
        raise RuntimeError("No docs loaded for context retrieval.")

    prompts = _read_jsonl_prompts(prompts_path)
    if not prompts:
        raise RuntimeError("No prompts found.")

    rng = random.Random(args.seed)
    rng.shuffle(prompts)

    existing = _load_existing(out_path)
    pending = [p for p in prompts if p not in existing]
    target = min(int(args.num), len(pending))
    pending = pending[:target]

    errors = 0
    debug_dir = Path(args.debug_dir).expanduser().resolve() if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    def log(message: str, *, error: bool = False) -> None:
        stream = sys.stderr if error else sys.stdout
        print(message, file=stream, flush=True)

    with out_path.open("a", encoding="utf-8") as f:
        idx = 0
        while idx < len(pending):
            batch_prompts = pending[idx : idx + int(args.batch_size)]
            items: list[tuple[int, str, str]] = []
            for j, p in enumerate(batch_prompts, start=1):
                ctx = select_context(p, docs, max_chars=int(args.max_context_chars))
                items.append((j, p, ctx))

            batch_prompt = build_batch_prompt(items)
            answers: dict[int, str] | None = None
            last_out_text: str = ""
            ok = False

            for attempt in range(1, max(1, int(args.retries_per_batch)) + 1):
                try:
                    log(f"Starting batch: {idx}/{len(pending)} (+{len(batch_prompts)}) attempt={attempt}")
                    last_out_text = run_batch(args.cmd, batch_prompt, args.timeout)
                    answers = parse_answers(last_out_text)
                    if len(answers) < len(items):
                        missing = sorted({item_id for item_id, _p, _c in items} - set(answers))
                        raise RuntimeError(f"Missing answers for ids: {missing}")
                    ok = True
                    break
                except Exception as exc:
                    log(f"[warn] {exc}", error=True)
                    if debug_dir:
                        ts = int(time.time() * 1000)
                        (debug_dir / f"batch_{idx:05d}_{ts}_prompt.txt").write_text(batch_prompt, encoding="utf-8")
                        (debug_dir / f"batch_{idx:05d}_{ts}_output.txt").write_text(last_out_text or "", encoding="utf-8")
                    time.sleep(float(args.sleep))

            if not ok:
                errors += 1
                log(f"[error] Batch failed after {args.retries_per_batch} attempts", error=True)
                if errors >= int(args.max_errors):
                    raise RuntimeError(f"Aborting after {errors} batch errors")
                # Skip this batch and continue.
                idx += len(batch_prompts)
                continue

            wrote = 0
            for item_id, prompt, _ctx in items:
                response = answers.get(item_id)
                if not response:
                    continue
                if prompt in existing:
                    continue
                f.write(json.dumps({"prompt": prompt, "response": response}) + "\n")
                existing.add(prompt)
                wrote += 1
            f.flush()

            log(f"Wrote {wrote}/{len(items)} (total {len(existing)}) -> {out_path}")
            idx += len(batch_prompts)
            time.sleep(float(args.sleep))

    log(f"Done. Wrote {len(existing)} total -> {out_path}")


if __name__ == "__main__":
    main()
