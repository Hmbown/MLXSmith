"""Synthetic data generation: prompts, SFT pairs, and DPO preference pairs."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable, Optional

from rich.console import Console

from .config import ProjectConfig
from .llm.backend import BackendNotAvailable
from .llm.registry import get_llm_backend
from .models import resolve_model_spec
from .verifiers.llm_judge import verify as judge_verify

console = Console()

_DEFAULT_PROMPT_SYSTEM = (
    "You are a helpful assistant that generates diverse, high-quality task prompts. "
    "Each prompt should be a self-contained instruction or question. "
    "Output exactly one prompt per response, with no additional commentary."
)


def _load_llm(model: str, cfg: ProjectConfig, project_root: Path):
    """Load an LLM backend and model, returning (backend, base_model)."""
    llm = get_llm_backend(cfg.model.backend)
    base_model, adapter_path, _meta = resolve_model_spec(project_root, model, cfg)
    try:
        llm.load(
            base_model,
            max_seq_len=cfg.model.max_seq_len,
            dtype=cfg.model.dtype,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        if adapter_path:
            llm.apply_adapter(str(adapter_path))
    except BackendNotAvailable as exc:
        raise RuntimeError(f"MLX backend unavailable: {exc}") from exc
    return llm, base_model


def _iter_prompts(path: Path, *, limit: Optional[int] = None) -> Iterable[str]:
    """Yield prompts from a JSONL file (supports prompt/instruction/input/question/messages)."""
    bad = 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            prompt = (
                row.get("prompt")
                or row.get("instruction")
                or row.get("input")
                or row.get("question")
                or ""
            )
            if not prompt and "messages" in row:
                msgs = row.get("messages") or []
                if msgs:
                    prompt = "\n".join([m.get("content", "") for m in msgs])
            if prompt:
                yield str(prompt)
                count += 1
                if limit is not None and count >= limit:
                    break
    if bad:
        console.print(f"[yellow]Skipped[/yellow] {bad} malformed rows in {path}")


def _read_prompts(path: Path, *, limit: Optional[int] = None) -> list[str]:
    """Read prompts from a JSONL file (expects {"prompt": "..."} rows)."""
    return list(_iter_prompts(path, limit=limit))


def _strip_leading_marker(text: str) -> str:
    text = text.strip()
    while True:
        if text.startswith(("-", "*", "•")):
            text = text[1:].strip()
            continue
        if len(text) > 2 and text[0].isdigit() and text[1] in (".", ")"):
            text = text[2:].strip()
            continue
        break
    return text


def _extract_generated_prompt(text: str, prefix: str, markers: Iterable[str]) -> str:
    cleaned = text
    if cleaned.startswith(prefix):
        cleaned = cleaned[len(prefix) :]
    for marker in markers:
        if marker in cleaned:
            cleaned = cleaned.split(marker)[-1]
    return _strip_leading_marker(cleaned.strip())


_EVOLVE_SYSTEM = (
    "You are a prompt engineer. Transform the base prompt into a new, higher-quality instruction. "
    "Return only the evolved prompt text, with no commentary."
)

_EVOLVE_MODES: dict[str, str] = {
    "deepen": "Add multi-step reasoning, edge cases, and require justification.",
    "broaden": "Broaden the scope while keeping the task clear and specific.",
    "complexify": "Add constraints, formatting requirements, and a higher difficulty level.",
    "constraints": "Introduce precise constraints or evaluation criteria for the answer.",
    "multi_turn": "Turn it into a multi-turn conversation task with at least 3 turns.",
}


def generate_prompts(
    model: str,
    cfg: ProjectConfig,
    out: Path,
    *,
    num: int = 50,
    seed_prompts: Optional[Path] = None,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    seed: int = 42,
    project_root: Optional[Path] = None,
) -> int:
    """Generate synthetic task prompts and write them as JSONL.

    Returns the number of prompts written.
    """
    root = project_root or Path.cwd()
    llm, _base = _load_llm(model, cfg, root)
    rng = random.Random(seed)

    seeds: list[str] = []
    if seed_prompts is not None:
        if not seed_prompts.exists():
            raise RuntimeError(f"Seed prompts file not found: {seed_prompts}")
        seeds = _read_prompts(seed_prompts)

    sys_prompt = system_prompt or _DEFAULT_PROMPT_SYSTEM
    written = 0
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for i in range(num):
            # Build a few-shot prefix from seed prompts when available
            if seeds:
                examples = rng.sample(seeds, min(3, len(seeds)))
                prefix = (
                    sys_prompt
                    + "\n\nExamples:\n"
                    + "\n".join(f"- {e}" for e in examples)
                    + "\n\nGenerate a new prompt:"
                )
            else:
                prefix = sys_prompt + "\n\nGenerate a new prompt:"

            gen = llm.generate(
                prefix,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=rng.randint(0, 2**31 - 1),
            )
            text = _extract_generated_prompt(gen.text, prefix, markers=["Generate a new prompt:", "Prompt:"])
            if text:
                f.write(json.dumps({"prompt": text}) + "\n")
                written += 1

    console.print(f"[green]Wrote[/green] {written} prompts -> {out}")
    return written


def generate_evolved_prompts(
    model: str,
    cfg: ProjectConfig,
    seed_prompts: Path,
    out: Path,
    *,
    num: int = 50,
    mode: str = "mix",
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    seed: int = 42,
    project_root: Optional[Path] = None,
) -> int:
    """Generate evolved prompts from a seed set (Evol-Instruct style)."""
    if not seed_prompts.exists():
        raise RuntimeError(f"Seed prompts file not found: {seed_prompts}")
    seeds = _read_prompts(seed_prompts)
    if not seeds:
        raise RuntimeError(f"No prompts found in {seed_prompts}")

    root = project_root or Path.cwd()
    llm, _base = _load_llm(model, cfg, root)
    rng = random.Random(seed)
    written = 0
    out.parent.mkdir(parents=True, exist_ok=True)

    sys_prompt = system_prompt or _EVOLVE_SYSTEM
    modes = list(_EVOLVE_MODES)
    if mode != "mix" and mode not in _EVOLVE_MODES:
        raise RuntimeError(f"Unknown evolve mode: {mode}. Choose from {', '.join(['mix'] + modes)}")

    with out.open("w", encoding="utf-8") as f:
        for _i in range(num):
            base = rng.choice(seeds)
            picked = rng.choice(modes) if mode == "mix" else mode
            instruction = _EVOLVE_MODES[picked]
            prefix = (
                f"{sys_prompt}\n\n"
                f"Base prompt:\n{base}\n\n"
                f"Transformation:\n{instruction}\n\n"
                "Return only the evolved prompt:"
            )
            gen = llm.generate(
                prefix,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=rng.randint(0, 2**31 - 1),
            )
            text = _extract_generated_prompt(
                gen.text,
                prefix,
                markers=["Return only the evolved prompt:", "Evolved prompt:", "Prompt:"],
            )
            if text:
                f.write(json.dumps({"prompt": text, "mode": picked, "seed_prompt": base}) + "\n")
                written += 1

    console.print(f"[green]Wrote[/green] {written} evolved prompts -> {out}")
    return written


def generate_sft(
    model: str,
    cfg: ProjectConfig,
    prompts_path: Path,
    out: Path,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    seed: int = 42,
    candidates_per_prompt: int = 1,
    judge_model: Optional[str] = None,
    judge_backend: str = "mlx-lm",
    rubric: Optional[str] = None,
    min_score: Optional[float] = None,
    max_prompts: Optional[int] = None,
    project_root: Optional[Path] = None,
) -> int:
    """Generate SFT pairs from prompts and write as JSONL.

    Reads {"prompt": "..."} rows, generates a response for each.
    Returns the number of pairs written.
    """
    root = project_root or Path.cwd()
    llm, _base = _load_llm(model, cfg, root)
    rng = random.Random(seed)
    if candidates_per_prompt < 1:
        raise RuntimeError("candidates_per_prompt must be >= 1")
    needs_judge = candidates_per_prompt > 1 or min_score is not None or rubric is not None
    if needs_judge and judge_model is None and not os.environ.get("MLXSMITH_JUDGE_MODEL"):
        raise RuntimeError("Judge model required for rejection sampling or filtering.")

    prompt_iter = _iter_prompts(prompts_path, limit=max_prompts)

    written = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    judge_dir = out.parent / "judge_artifacts"
    if needs_judge:
        judge_dir.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        any_prompt = False
        for prompt in prompt_iter:
            any_prompt = True
            best_response = None
            best_score = None
            for _k in range(candidates_per_prompt):
                gen = llm.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    seed=rng.randint(0, 2**31 - 1),
                )
                response = gen.text[len(prompt):] if gen.text.startswith(prompt) else gen.text
                response = response.strip()
                if not response:
                    continue
                if needs_judge:
                    res = judge_verify(
                        prompt,
                        response,
                        str(judge_dir),
                        model=judge_model,
                        backend=judge_backend,
                        rubric=rubric,
                        reward_mode="score",
                    )
                    score = float(getattr(res, "reward", 0.0))
                    if best_score is None or score > best_score:
                        best_score = score
                        best_response = response
                else:
                    best_response = response
                    break

            if best_response is None:
                continue
            if min_score is not None and best_score is not None and best_score < min_score:
                continue

            row = {"prompt": prompt, "response": best_response}
            if best_score is not None:
                row["score"] = best_score
            f.write(json.dumps(row) + "\n")
            written += 1

        if not any_prompt:
            raise RuntimeError(f"No prompts found in {prompts_path}")

    console.print(f"[green]Wrote[/green] {written} SFT pairs -> {out}")
    return written


def generate_dpo(
    model: str,
    cfg: ProjectConfig,
    prompts_path: Path,
    out: Path,
    *,
    candidates_per_prompt: int = 4,
    judge_model: Optional[str] = None,
    judge_backend: str = "mlx-lm",
    rubric: Optional[str] = None,
    min_margin: Optional[float] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    seed: int = 42,
    max_prompts: Optional[int] = None,
    project_root: Optional[Path] = None,
) -> int:
    """Generate DPO preference pairs from prompts.

    For each prompt, generates N candidates, scores them with llm_judge,
    and picks the best/worst as chosen/rejected.
    Returns the number of pairs written.
    """
    root = project_root or Path.cwd()
    llm, _base = _load_llm(model, cfg, root)
    rng = random.Random(seed)
    if candidates_per_prompt < 2:
        raise RuntimeError("candidates_per_prompt must be >= 2")
    if judge_model is None and not os.environ.get("MLXSMITH_JUDGE_MODEL"):
        raise RuntimeError("Judge model required for DPO pair generation.")
    prompt_iter = _iter_prompts(prompts_path, limit=max_prompts)

    written = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    judge_dir = out.parent / "judge_artifacts"
    judge_dir.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        any_prompt = False
        for prompt in prompt_iter:
            any_prompt = True
            candidates: list[tuple[str, float]] = []
            for _k in range(candidates_per_prompt):
                gen = llm.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    seed=rng.randint(0, 2**31 - 1),
                )
                completion = gen.text[len(prompt):] if gen.text.startswith(prompt) else gen.text
                completion = completion.strip()
                if not completion:
                    continue
                res = judge_verify(
                    prompt,
                    completion,
                    str(judge_dir),
                    model=judge_model,
                    backend=judge_backend,
                    rubric=rubric,
                    reward_mode="score",
                )
                reward = float(getattr(res, "reward", 0.0))
                candidates.append((completion, reward))

            if len(candidates) < 2:
                continue
            candidates.sort(key=lambda x: x[1], reverse=True)
            chosen, chosen_r = candidates[0]
            rejected = None
            rejected_r = None
            for comp, score in reversed(candidates):
                if comp != chosen:
                    rejected = comp
                    rejected_r = score
                    break
            if rejected is None:
                continue
            if chosen_r == rejected_r:
                continue
            if min_margin is not None and (chosen_r - float(rejected_r)) < min_margin:
                continue

            row = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            if rejected_r is not None:
                row["chosen_score"] = chosen_r
                row["rejected_score"] = rejected_r
            f.write(json.dumps(row) + "\n")
            written += 1

        if not any_prompt:
            raise RuntimeError(f"No prompts found in {prompts_path}")

    console.print(f"[green]Wrote[/green] {written} DPO pairs -> {out}")
    return written
