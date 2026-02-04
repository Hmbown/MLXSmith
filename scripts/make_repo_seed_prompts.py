#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DOC_FILES = [
    "README.md",
    "docs/README.md",
    "docs/getting-started.md",
    "docs/COMPATIBILITY.md",
    "docs/ENVIRONMENTS.md",
    "docs/PROJECT_FORMAT.md",
    "docs/VERIFIERS.md",
    "docs/rlm-ctl.md",
    "docs/orchestrator.md",
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
]


CURATED_PROMPTS = [
    "You are an MLXSmith assistant. Show the exact command sequence to initialize a new project, run doctor, pull a model, prepare data, run SFT, and serve the adapter.",
    "Explain the differences between `mlxsmith sft`, `mlxsmith pref`, and `mlxsmith online-dpo`, and when you would choose each.",
    "Give a minimal example of the JSONL format required for SFT (`{prompt, response}`) and preference training (`{prompt, chosen, rejected}`).",
    "How do you run preference training with ORPO or SIMPO? Show the command and the key flag to set.",
    "Show how to use `mlxsmith data` commands to pull a preset dataset, split it into train/valid/test, and validate it.",
    "Explain how MLXSmith uses `mlxsmith config` and environment variables for configuration overrides. Provide one example with `MLXSMITH__MODEL__ID`.",
    "How do you generate synthetic prompts, then generate synthetic SFT pairs, then train on them? Provide the command sequence.",
    "What does `mlxsmith judge` do and what dataset format does it expect? Provide a minimal example.",
    "How do you run online DPO with a judge model? Provide a command template and mention required args.",
    "Explain how `mlxsmith rft` (GRPO) uses environments and verifiers. Provide an example command.",
    "Show how to run the RLM loop in orchestrated mode and how to resume a run.",
    "What does `mlxsmith pipeline` do? Give a minimal example of running it end-to-end.",
    "How do you serve a fine-tuned adapter with the OpenAI-compatible server? Provide a sample command.",
    "How do you evaluate a model with `mlxsmith eval` and benchmark it with `mlxsmith bench`?",
    "Troubleshoot: `mlxsmith doctor` shows `metal: False` — list the likely steps to fix it.",
    "Troubleshoot: `mlxsmith pull` fails during conversion — give two concrete steps to try.",
    "Show how to merge multiple adapters with `mlxsmith adapters merge` and explain why you might do this.",
    "Explain the role of verifiers in MLXSmith and the required interface for a verifier.",
    "What is the difference between `mlxsmith serve` with a base model vs. an adapter? Provide example commands.",
    "Explain how to set `train.iters`, `save_every`, and `eval_every` for faster iteration in a config file.",
    "Give an example of using `mlxsmith pull` with quantization options.",
    "How do you create and run an MLXSmith environment package using `mlxsmith env` commands?",
    "Explain how `mlxsmith self-verify` generates rewards and how it differs from LLM-judge based training.",
    "Show a minimal config file snippet that sets the model id, backend, and training hyperparameters.",
    "How do you validate a config file and inspect the merged config in MLXSmith?",
    "Where are run artifacts stored, and how do you load an adapter from a previous run?",
    "Provide a concise overview of all major MLXSmith training methods and their dataset requirements.",
    "Explain how `mlxsmith distill` works and what inputs it expects.",
    "How do you use `mlxsmith data import` for ShareGPT format and then split it?",
    "Show how to run RFT using a packaged environment (`mlxsmith env run`).",
    "What does `mlxsmith losses` list and when would you use it?",
    "Explain the role of `mlxsmith accel status` and when to check it.",
    "How do you switch models for a single run without editing the config file?",
    "Explain the purpose of `mlxsmith quantize` and why `mlxsmith pull --quantize` is preferred.",
    "Show a minimal end-to-end workflow for preference tuning: SFT -> Pref -> Serve.",
    "Explain how `mlxsmith rlm history` and `mlxsmith rlm status` are used during long runs.",
    "Explain what an MLXSmith project directory contains (data/, runs/, eval/, verifiers/, envs/).",
    "Give a command sequence to run online-dpo, then serve the resulting adapter.",
    "How do you set up and use a judge rubric for filtering synthetic SFT or DPO data?",
]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_commands(text: str) -> set[str]:
    cmds = set()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("mlxsmith "):
            cmds.add(line)
    # inline code
    for m in re.findall(r"`(mlxsmith[^`]+)`", text):
        if m.strip():
            cmds.add(m.strip())
    return cmds


def extract_headings(text: str) -> list[str]:
    headings = []
    for line in text.splitlines():
        if line.startswith("#"):
            head = line.lstrip("#").strip()
            if 4 <= len(head) <= 60:
                headings.append(head)
    return headings


def extract_env_vars(text: str) -> set[str]:
    return set(re.findall(r"MLXSMITH__[_A-Z0-9]+", text))


def extract_formats(text: str) -> set[str]:
    formats = set()
    for m in re.findall(r"\{[^}]{6,80}\}", text):
        if "prompt" in m or "response" in m or "chosen" in m or "rejected" in m:
            formats.add(m.strip())
    return formats


def build_prompts(root: Path, max_prompts: int | None = None) -> list[str]:
    commands: set[str] = set()
    headings: list[str] = []
    env_vars: set[str] = set()
    formats: set[str] = set()

    for rel in DOC_FILES:
        path = root / rel
        if not path.exists():
            continue
        text = _read_text(path)
        commands |= extract_commands(text)
        headings.extend(extract_headings(text))
        env_vars |= extract_env_vars(text)
        formats |= extract_formats(text)

    prompts = list(CURATED_PROMPTS)

    for cmd in sorted(commands):
        prompts.append(f"What does this MLXSmith command do, and when should it be used? Command: `{cmd}`")
        prompts.append(f"Provide a short explanation plus any required inputs for: `{cmd}`")

    for head in headings:
        if head.lower() in {"overview", "notes", "examples", "example", "quickstart"}:
            continue
        prompts.append(f"Explain MLXSmith: {head}. Include a minimal example command.")

    for env in sorted(env_vars):
        prompts.append(f"How does `{env}` affect MLXSmith configuration? Give a one-line example.")

    for fmt in sorted(formats):
        prompts.append(f"In MLXSmith, explain this dataset format and give a minimal JSONL example: {fmt}")

    # De-duplicate, keep order
    seen = set()
    final = []
    for p in prompts:
        key = p.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        final.append(key)

    if max_prompts is not None:
        final = final[: max_prompts]
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate seed prompts for MLXSmith repo training.")
    parser.add_argument("--root", default=".", help="Repo root")
    parser.add_argument("--out", default="data/mlxsmith_seed_prompts.jsonl", help="Output JSONL path")
    parser.add_argument("--max-prompts", type=int, default=250, help="Maximum prompts to write")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = build_prompts(root, max_prompts=args.max_prompts)
    with out_path.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p}) + "\n")

    print(f"Wrote {len(prompts)} prompts -> {out_path}")


if __name__ == "__main__":
    main()
