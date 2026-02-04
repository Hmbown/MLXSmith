#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_CMD = os.getenv("MLXSMITH_CLI_CODEX_CMD") or (
    "codex exec --full-auto --disable shell_tool --disable shell_snapshot "
    "-c 'features.collab=false' "
    "-c 'mcp_servers.github.enabled=false' "
    "-c 'mcp_servers.aleph.enabled=false' "
    "-c 'mcp_servers.hegelion.enabled=false' "
    "-c 'mcp_servers.gxmcp.enabled=false' "
    "-c 'mcp_servers.acz.enabled=false' "
    "-c 'hooks.agent-turn-complete=[]' "
    "--model gpt-5.2"
)
DEFAULT_TIMEOUT = float(os.getenv("MLXSMITH_CLI_TIMEOUT", "1800"))


def _read_text_value(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    if value.startswith("@"):
        path = Path(value[1:])
        if path.exists():
            return path.read_text(encoding="utf-8")
    path = Path(value)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return value


def _read_prompts(path: Path) -> list[str]:
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


def _strip_code_fences(text: str) -> str:
    if "```" not in text:
        return text
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```") and lines[-1].startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return text


def _parse_prompts(text: str) -> list[str]:
    text = _strip_code_fences(text.strip())
    if not text:
        return []
    prompts: list[str] = []

    # Try JSON array or object with "prompts"
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict) and "prompt" in item:
                    p = str(item["prompt"]).strip()
                    if p:
                        prompts.append(p)
            if prompts:
                return prompts
            for item in obj:
                if isinstance(item, str) and item.strip():
                    prompts.append(item.strip())
            if prompts:
                return prompts
        if isinstance(obj, dict) and "prompts" in obj:
            for item in obj["prompts"]:
                if isinstance(item, dict) and "prompt" in item:
                    p = str(item["prompt"]).strip()
                    if p:
                        prompts.append(p)
                elif isinstance(item, str) and item.strip():
                    prompts.append(item.strip())
            if prompts:
                return prompts
    except json.JSONDecodeError:
        pass

    # Fallback: parse JSONL lines
    for line in text.splitlines():
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
    if prompts:
        return prompts

    # Last resort: extract JSON objects from mixed text
    for line in text.splitlines():
        if "{" not in line or "}" not in line:
            continue
        chunk = line[line.find("{") : line.rfind("}") + 1]
        try:
            row = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        prompt = str(row.get("prompt", "")).strip()
        if prompt:
            prompts.append(prompt)
    return prompts


def build_batch_prompt(
    system_prompt: str,
    *,
    examples: Iterable[str],
    batch_size: int,
) -> str:
    example_block = "\n".join(f"- {e}" for e in examples)
    return (
        f"{system_prompt}\n\n"
        "You will generate multiple prompts.\n"
        "Output JSONL only (one JSON object per line) with the shape {\"prompt\": \"...\"}.\n"
        f"Generate exactly {batch_size} prompts.\n\n"
        "Examples:\n"
        f"{example_block}\n\n"
        "Generate the prompts now."
    )


def run_batch(cmd: str, prompt: str, timeout_s: float) -> list[str]:
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
    return _parse_prompts(proc.stdout or "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MLXSmith prompts in Codex batches.")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--num", type=int, default=200, help="Total prompts to generate")
    parser.add_argument("--batch-size", type=int, default=20, help="Prompts per CLI call")
    parser.add_argument("--seed-prompts", required=True, help="Seed prompts JSONL for few-shot")
    parser.add_argument("--system-prompt", required=True, help="System prompt text or @file path")
    parser.add_argument("--cmd", default=DEFAULT_CMD, help="CLI command for codex exec")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="CLI timeout seconds")
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds to sleep between batches")
    parser.add_argument("--max-errors", type=int, default=3, help="Abort after this many errors")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for example selection")
    parser.add_argument("--log", help="Optional log file to append progress")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log).expanduser().resolve() if args.log else None
    log_handle = log_file.open("a", encoding="utf-8") if log_file else None

    def log(message: str, *, error: bool = False) -> None:
        stream = sys.stderr if error else sys.stdout
        print(message, file=stream, flush=True)
        if log_handle:
            log_handle.write(message + "\n")
            log_handle.flush()

    system_prompt = _read_text_value(args.system_prompt) or ""
    if not system_prompt.strip():
        raise RuntimeError("system prompt is empty")

    random.seed(args.seed)
    seed_prompts = _read_prompts(Path(args.seed_prompts))
    if not seed_prompts:
        raise RuntimeError("seed prompts are empty")

    existing = _load_existing(out)
    total_target = args.num
    errors = 0

    with out.open("a", encoding="utf-8") as f:
        while len(existing) < total_target:
            batch_n = min(args.batch_size, total_target - len(existing))
            examples = random.sample(seed_prompts, min(5, len(seed_prompts)))
            batch_prompt = build_batch_prompt(system_prompt, examples=examples, batch_size=batch_n)
            try:
                log(f"Starting batch: target {batch_n} (total {len(existing)}/{total_target})")
                prompts = run_batch(args.cmd, batch_prompt, args.timeout)
            except Exception as exc:
                errors += 1
                log(f"[error] {exc}", error=True)
                if errors >= args.max_errors:
                    raise
                time.sleep(args.sleep)
                continue

            wrote = 0
            for prompt in prompts:
                if prompt in existing:
                    continue
                f.write(json.dumps({"prompt": prompt}) + "\n")
                existing.add(prompt)
                wrote += 1
            f.flush()
            log(f"Wrote {wrote} prompts (total {len(existing)}/{total_target})")
            time.sleep(args.sleep)

    log(f"Done. Wrote {len(existing)} prompts -> {out}")
    if log_handle:
        log_handle.close()


if __name__ == "__main__":
    main()
