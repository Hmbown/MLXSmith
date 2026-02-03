from __future__ import annotations

import os
from typing import Sequence, Optional, Any, Dict, List

from .backend import Generation, BackendNotAvailable


class OpenAIBackend:
    """OpenAI-compatible backend for remote generation (chat/completions).

    This backend is intended for judging + data generation, not training.
    Tokenization and gradient-based methods are not supported.
    """

    name = "openai"

    def __init__(self):
        self.model_id: Optional[str] = None
        self.base_url: str = "https://api.openai.com/v1"
        self.api_key: Optional[str] = None
        self.org: Optional[str] = None
        self.project: Optional[str] = None
        self.timeout_s: float = 60.0
        self._client = None

    def _require(self):
        try:
            import httpx  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise BackendNotAvailable(
                "httpx is not installed. Try: pip install -e '.[serve]'"
            ) from exc
        return httpx

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
        # Store config for subsequent calls; most fields are not used.
        self.model_id = model_id_or_path
        self.base_url = (
            os.getenv("MLXSMITH_API_BASE")
            or os.getenv("MLXSMITH_OPENAI_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or self.base_url
        ).rstrip("/")
        self.api_key = (
            os.getenv("MLXSMITH_API_KEY")
            or os.getenv("MLXSMITH_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.org = os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")
        self.project = os.getenv("OPENAI_PROJECT_ID")
        try:
            self.timeout_s = float(os.getenv("MLXSMITH_API_TIMEOUT", "60"))
        except ValueError:
            self.timeout_s = 60.0
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        httpx = self._require()
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.org:
            headers["OpenAI-Organization"] = self.org
        if self.project:
            headers["OpenAI-Project"] = self.project
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout_s,
        )
        return self._client

    def _ensure_loaded(self):
        if not self.model_id:
            raise RuntimeError("Backend not loaded")

    def apply_adapter(self, adapter_path: str) -> None:
        raise RuntimeError("OpenAI backend does not support adapters")

    def apply_lora_from_config(self, cfg) -> dict:
        raise RuntimeError("OpenAI backend does not support LoRA")

    def encode(self, text: str) -> list[int]:
        raise RuntimeError("OpenAI backend does not support tokenization")

    def decode(self, ids: Sequence[int]) -> str:
        raise RuntimeError("OpenAI backend does not support tokenization")

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
        self._ensure_loaded()
        client = self._get_client()

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_new_tokens),
            "stream": False,
        }
        if seed is not None:
            payload["seed"] = int(seed)
        if top_k is not None:
            payload["top_k"] = int(top_k)

        resp = client.post("/chat/completions", json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text}")

        data = resp.json()
        text = ""
        if isinstance(data, dict):
            choices = data.get("choices") or []
            if choices:
                choice = choices[0]
                if isinstance(choice, dict):
                    msg = choice.get("message") or {}
                    if isinstance(msg, dict) and msg.get("content") is not None:
                        text = str(msg.get("content"))
                    elif choice.get("text") is not None:
                        text = str(choice.get("text"))

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
        raise RuntimeError(
            "OpenAI backend does not support token-level logprobs; "
            "use mlx-lm for training backends that require generate_with_logprobs."
        )

    def sft_loss(self, token_ids: Sequence[int], *, train_on_prompt: bool, prompt_len: int) -> Any:
        raise RuntimeError("OpenAI backend does not support training")

    def rl_loss(self, token_ids: Sequence[int], *, prompt_len: int, advantage: float) -> Any:
        raise RuntimeError("OpenAI backend does not support training")

    def sequence_logprob(self, token_ids: Sequence[int], *, prompt_len: int) -> Any:
        raise RuntimeError("OpenAI backend does not support training")

    def token_logprobs(
        self,
        token_ids: Sequence[int],
        *,
        prompt_len: int,
        top_k: int = 0,
        include_prompt: bool = False,
    ) -> tuple[list[float], List[Dict[str, float]] | None]:
        raise RuntimeError("OpenAI backend does not support token logprobs")

    def value_and_grad(self, loss_fn):
        raise RuntimeError("OpenAI backend does not support training")

    def optimizer_and_params(
        self,
        *,
        lr: float,
        weight_decay: float = 0.0,
        optimizer: str | None = None,
        optimizer_kwargs: dict | None = None,
    ) -> tuple[Any, Any]:
        raise RuntimeError("OpenAI backend does not support training")

    def apply_grads(self, optimizer: Any, grads: Any) -> None:
        raise RuntimeError("OpenAI backend does not support training")

    def save_adapter(self, out_dir: str, *, metadata: dict | None = None) -> None:
        raise RuntimeError("OpenAI backend does not support training")
