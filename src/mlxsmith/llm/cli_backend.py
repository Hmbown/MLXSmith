from __future__ import annotations

import os
import shlex
import subprocess
from typing import Sequence, Optional, Any, Dict, List

from .backend import Generation, BackendNotAvailable


def _env(name: str) -> Optional[str]:
    value = os.getenv(name)
    return value if value else None


class CliBackend:
    """Backend that shells out to a CLI for generation (judge/data only)."""

    name = "cli"

    def __init__(self):
        self.model_id: Optional[str] = None
        self.command: Optional[str] = None
        self.timeout_s: float = 120.0

    def load(
        self,
        model_id_or_path: str,
        *,
        max_seq_len: int | None = None,
        dtype: str | None = None,
        tokenizer_config: dict | None = None,
        model_config: dict | None = None,
        adapter_path: str | None = None,
        trust_remote_code: bool | None = None,
    ) -> None:
        self.model_id = model_id_or_path
        self.command = self._resolve_command(model_id_or_path)
        try:
            self.timeout_s = float(_env("MLXSMITH_CLI_TIMEOUT") or "900")
        except ValueError:
            self.timeout_s = 900.0

    def _resolve_command(self, model_id: str) -> str:
        override = _env("MLXSMITH_CLI_COMMAND")
        if override:
            return override
        presets = {
            "codex": _env("MLXSMITH_CLI_CODEX_CMD") or "codex --auto-exec",
            "claude": _env("MLXSMITH_CLI_CLAUDE_CMD") or "claude",
            "gemini": _env("MLXSMITH_CLI_GEMINI_CMD") or "gemini",
        }
        if model_id in presets:
            return presets[model_id]
        return model_id

    def _build_command(self, prompt: str) -> tuple[list[str], Optional[str]]:
        template = _env("MLXSMITH_CLI_TEMPLATE")
        prompt_flag = _env("MLXSMITH_CLI_PROMPT_FLAG")
        if not self.command:
            raise RuntimeError("Backend not loaded")

        if template:
            cmd_str = template.replace("{prompt}", prompt)
            return shlex.split(cmd_str), None

        cmd = shlex.split(self.command)
        if prompt_flag:
            return cmd + [prompt_flag, prompt], None
        return cmd, prompt

    def apply_adapter(self, adapter_path: str) -> None:
        raise RuntimeError("CLI backend does not support adapters")

    def apply_lora_from_config(self, cfg) -> dict:
        raise RuntimeError("CLI backend does not support LoRA")

    def encode(self, text: str) -> list[int]:
        raise RuntimeError("CLI backend does not support tokenization")

    def decode(self, ids: Sequence[int]) -> str:
        raise RuntimeError("CLI backend does not support tokenization")

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 1.0,
        top_k: int | None = None,
        seed: int | None = None,
    ) -> Generation:
        if not self.command:
            raise RuntimeError("Backend not loaded")

        cmd, stdin_text = self._build_command(prompt)
        try:
            proc = subprocess.run(
                cmd,
                input=stdin_text,
                text=True,
                capture_output=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"CLI backend timed out after {self.timeout_s}s") from exc
        except FileNotFoundError as exc:
            raise BackendNotAvailable(f"CLI command not found: {cmd[0]}") from exc

        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            stdout = proc.stdout.strip()
            detail = stderr or stdout or "Unknown error"
            raise RuntimeError(f"CLI backend error (code {proc.returncode}): {detail}")

        text = proc.stdout or ""
        return Generation(text=text, token_ids=[], prompt_len=0)

    def generate_with_logprobs(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 1.0,
        top_k_sampling: int | None = None,
        seed: int | None = None,
        logprobs: int = 0,
    ) -> Generation:
        raise RuntimeError("CLI backend does not support logprobs or training calls")

    def sft_loss(self, token_ids: Sequence[int], *, train_on_prompt: bool, prompt_len: int) -> Any:
        raise RuntimeError("CLI backend does not support training")

    def rl_loss(self, token_ids: Sequence[int], *, prompt_len: int, advantage: float) -> Any:
        raise RuntimeError("CLI backend does not support training")

    def sequence_logprob(self, token_ids: Sequence[int], *, prompt_len: int) -> Any:
        raise RuntimeError("CLI backend does not support training")

    def token_logprobs(
        self,
        token_ids: Sequence[int],
        *,
        prompt_len: int,
        top_k: int = 0,
        include_prompt: bool = False,
    ) -> tuple[list[float], List[Dict[str, float]] | None]:
        raise RuntimeError("CLI backend does not support token logprobs")

    def value_and_grad(self, loss_fn):
        raise RuntimeError("CLI backend does not support training")

    def optimizer_and_params(
        self,
        *,
        lr: float,
        weight_decay: float = 0.0,
        optimizer: str | None = None,
        optimizer_kwargs: dict | None = None,
    ) -> tuple[Any, Any]:
        raise RuntimeError("CLI backend does not support training")

    def apply_grads(self, optimizer: Any, grads: Any) -> None:
        raise RuntimeError("CLI backend does not support training")

    def save_adapter(self, out_dir: str, *, metadata: dict | None = None) -> None:
        raise RuntimeError("CLI backend does not support training")
