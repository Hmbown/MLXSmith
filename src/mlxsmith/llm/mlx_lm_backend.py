from __future__ import annotations

import inspect
from typing import Sequence, Any, List, Dict, Optional

from .backend import Generation, BackendNotAvailable
from ..train.lora import apply_adapter, apply_lora, LoRAConfig, save_adapter


class MlxLMBackend:
    """Backend built on MLX + mlx-lm."""

    name = "mlx-lm"

    def __init__(self, *, lora_config: dict | None = None):
        self.lora_config = lora_config or {}
        self.model = None
        self.tokenizer = None
        self.nn = None
        self.mx = None
        self.optim = None
        self._lora_applied = False
        self._adapter_config: dict | None = None

    def _require(self):
        try:
            import mlx.core as mx  # type: ignore
            import mlx.nn as nn  # type: ignore
            import mlx.optimizers as optim  # type: ignore
        except Exception as e:  # pragma: no cover
            raise BackendNotAvailable(
                "MLX is not installed. Try: pip install -e '.[mlx,llm]'"
            ) from e
        self.mx = mx
        self.nn = nn
        self.optim = optim

        try:
            import mlx_lm  # type: ignore
        except Exception as e:  # pragma: no cover
            raise BackendNotAvailable(
                "mlx-lm is not installed. Try: pip install -e '.[llm]'"
            ) from e
        return mlx_lm

    def _call_with_supported_kwargs(self, fn, *args, **kwargs):
        try:
            sig = inspect.signature(fn)
            supported = {}
            for k, v in kwargs.items():
                if k in sig.parameters:
                    supported[k] = v
            return fn(*args, **supported)
        except Exception:
            return fn(*args, **kwargs)

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
        mlx_lm = self._require()

        if tokenizer_config is None:
            tokenizer_config = {}
        if trust_remote_code is not None:
            tokenizer_config = dict(tokenizer_config)
            tokenizer_config.setdefault("trust_remote_code", trust_remote_code)

        if model_config is None:
            model_config = {}
        # Pass dtype/max_seq_len as hints when supported in model config.
        if dtype is not None:
            model_config = dict(model_config)
            model_config.setdefault("dtype", dtype)
        if max_seq_len is not None:
            model_config = dict(model_config)
            model_config.setdefault("max_seq_len", max_seq_len)

        load_fn = getattr(mlx_lm, "load", None)
        if callable(load_fn):
            model, tokenizer = self._call_with_supported_kwargs(
                load_fn,
                model_id_or_path,
                tokenizer_config=tokenizer_config,
                model_config=model_config,
                adapter_path=adapter_path,
            )
        else:  # pragma: no cover
            utils = getattr(mlx_lm, "utils", None)
            if utils is None or not callable(getattr(utils, "load", None)):
                raise BackendNotAvailable("Could not find mlx_lm.load(...) API")
            model, tokenizer = self._call_with_supported_kwargs(
                utils.load,
                model_id_or_path,
                tokenizer_config=tokenizer_config,
                model_config=model_config,
                adapter_path=adapter_path,
            )

        self.model = model
        self.tokenizer = tokenizer
        self._lora_applied = False
        self._adapter_config = None

    def apply_adapter(self, adapter_path: str) -> None:
        if self.model is None:
            raise RuntimeError("Backend not loaded")
        self._adapter_config = apply_adapter(self.model, adapter_path)
        self._lora_applied = True

    def apply_lora_from_config(self, cfg: LoRAConfig) -> dict:
        if self.model is None:
            raise RuntimeError("Backend not loaded")
        adapter_cfg = apply_lora(self.model, cfg)
        self._lora_applied = True
        self._adapter_config = adapter_cfg
        return adapter_cfg

    def encode(self, text: str) -> list[int]:
        if self.tokenizer is None:
            raise RuntimeError("Backend not loaded")
        tok = self.tokenizer
        if hasattr(tok, "encode"):
            out = tok.encode(text)
            if isinstance(out, dict) and "input_ids" in out:
                return list(out["input_ids"])
            if isinstance(out, (list, tuple)):
                return list(out)
        if hasattr(tok, "__call__"):
            out = tok(text)
            if isinstance(out, dict) and "input_ids" in out:
                return list(out["input_ids"])
        raise RuntimeError("Tokenizer does not support encode")

    def decode(self, ids: Sequence[int]) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Backend not loaded")
        tok = self.tokenizer
        if hasattr(tok, "decode"):
            return tok.decode(list(ids))
        raise RuntimeError("Tokenizer does not support decode")

    def _forward_logits(self, ids: Sequence[int]):
        assert self.mx is not None
        if self.model is None:
            raise RuntimeError("Backend not loaded")
        mx = self.mx
        x = mx.array([list(ids)], dtype=mx.int32)
        return self.model(x)

    def _response_logprobs(self, ids: Sequence[int], *, prompt_len: int) -> list[float]:
        assert self.mx is not None
        mx = self.mx
        if not ids:
            return []
        logits = self._forward_logits(ids)
        logits = logits[:, :-1, :]
        labels = mx.array([list(ids)[1:]], dtype=mx.int32)
        lse = mx.logsumexp(logits, axis=-1)
        chosen = mx.take_along_axis(logits, labels[..., None], axis=-1).squeeze(-1)
        logp = chosen - lse
        start = max(0, prompt_len - 1)
        if start >= int(getattr(logp, "size", len(ids) - 1)):
            return []
        values = logp[:, start:]
        try:
            flat = values.flatten().tolist()
        except Exception:
            try:
                flat = [float(v) for v in values.reshape(-1)]
            except Exception:
                flat = [float(v) for v in values]
        return [float(v) for v in flat]

    def _extract_top_k_logprobs(
        self, 
        logits: Any, 
        k: int,
        sampled_ids: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, float]]:
        """Extract top-k logprobs from logits.
        
        Args:
            logits: Logits array [batch, seq_len, vocab_size]
            k: Number of top logprobs to extract
            sampled_ids: Optional token IDs that were actually sampled
            
        Returns:
            List of dicts mapping token string to logprob for each position
        """
        assert self.mx is not None
        mx = self.mx
        
        # Get log softmax
        log_probs = mx.log(mx.softmax(logits, axis=-1))
        
        # Get top-k indices and values
        # MLX doesn't have topk directly, so we use argsort
        sorted_indices = mx.argsort(-log_probs, axis=-1)
        
        results = []
        batch_size, seq_len, vocab_size = log_probs.shape
        
        # Limit k to vocab size
        k = min(k, vocab_size)
        
        for b in range(batch_size):
            for t in range(seq_len):
                # Get top-k for this position
                top_k_indices = sorted_indices[b, t, :k]
                top_k_logprobs = log_probs[b, t, top_k_indices]
                
                # Build dict
                token_logprobs = {}
                for idx, logprob in zip(top_k_indices.tolist(), top_k_logprobs.tolist()):
                    # Try to decode token
                    try:
                        token_str = self.decode([idx])
                        # Escape special characters for JSON compatibility
                        token_str = token_str.replace('\n', '\\n').replace('\t', '\\t')
                    except Exception:
                        token_str = f"<token_{idx}>"
                    token_logprobs[token_str] = float(logprob)
                
                results.append(token_logprobs)
        
        return results

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
        assert self.mx is not None
        mx = self.mx
        if seed is not None:
            mx.random.seed(seed)

        prompt_ids = self.encode(prompt)
        ids = list(prompt_ids)
        prompt_len = len(prompt_ids)

        # Prefer mlx_lm sampler if available
        sampler = None
        try:
            from mlx_lm.sample_utils import make_sampler  # type: ignore

            sampler = make_sampler(
                temp=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k or 0),
            )
        except Exception:
            sampler = None

        for _ in range(max_new_tokens):
            logits = self._forward_logits(ids)
            last = logits[:, -1, :]
            if temperature <= 0:
                next_id = int(mx.argmax(last, axis=-1).item())
            elif sampler is not None:
                next_id = int(sampler(last).item())
            else:
                probs = mx.softmax(last / float(temperature), axis=-1)
                next_id = int(mx.random.categorical(mx.log(probs)).item())
            ids.append(next_id)
        text = self.decode(ids)
        return Generation(text=text, token_ids=ids, prompt_len=prompt_len)

    def generate_with_logprobs(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 1.0,
        top_k_sampling: int | None = None,
        seed: int | None = None,
        logprobs: int = 0,  # Number of top logprobs to return per token
    ) -> Generation:
        """Generate with logprobs support including top-k logprobs.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k_sampling: Top-k sampling parameter (named to avoid conflict with logprobs)
            seed: Random seed
            logprobs: Number of top logprobs to return per token (0 = only sampled token)
            
        Returns:
            Generation with logprobs and optionally top_k_logprobs
        """
        assert self.mx is not None
        mx = self.mx
        if seed is not None:
            mx.random.seed(seed)

        prompt_ids = self.encode(prompt)
        ids = list(prompt_ids)
        prompt_len = len(prompt_ids)
        
        # Storage for per-token info
        per_token_logprobs: list[float] = []
        per_token_top_k: list[dict[str, float]] = []

        # Prefer mlx_lm sampler if available
        sampler = None
        try:
            from mlx_lm.sample_utils import make_sampler  # type: ignore

            sampler = make_sampler(
                temp=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k_sampling or 0),
            )
        except Exception:
            sampler = None

        for _ in range(max_new_tokens):
            logits = self._forward_logits(ids)
            last = logits[:, -1, :]  # [batch=1, vocab_size]
            
            # Get log probabilities for this position
            log_probs = mx.log(mx.softmax(last, axis=-1))
            
            if temperature <= 0:
                next_id = int(mx.argmax(last, axis=-1).item())
            elif sampler is not None:
                next_id = int(sampler(last).item())
            else:
                probs = mx.softmax(last / float(temperature), axis=-1)
                next_id = int(mx.random.categorical(mx.log(probs)).item())
            
            # Get logprob of sampled token
            sampled_logprob = float(log_probs[0, next_id].item())
            per_token_logprobs.append(sampled_logprob)
            
            # Get top-k logprobs if requested
            if logprobs > 0:
                top_k_logprobs = self._extract_top_k_logprobs(
                    last, 
                    k=logprobs,
                    sampled_ids=[next_id]
                )
                if top_k_logprobs:
                    per_token_top_k.append(top_k_logprobs[0])
            
            ids.append(next_id)
        
        text = self.decode(ids)
        
        return Generation(
            text=text, 
            token_ids=ids, 
            prompt_len=prompt_len,
            logprobs=per_token_logprobs,
            top_k_logprobs=per_token_top_k if per_token_top_k else None,
        )

    def sft_loss(self, token_ids: Sequence[int], *, train_on_prompt: bool, prompt_len: int) -> Any:
        assert self.mx is not None
        mx = self.mx
        ids = list(token_ids)
        logits = self._forward_logits(ids)
        logits = logits[:, :-1, :]
        labels = mx.array([ids[1:]], dtype=mx.int32)

        if not train_on_prompt:
            mask = [0] * max(0, prompt_len - 1) + [1] * (len(ids) - prompt_len)
            mask = mx.array([mask], dtype=mx.float32)
        else:
            mask = mx.ones(labels.shape, dtype=mx.float32)

        lse = mx.logsumexp(logits, axis=-1)
        chosen = mx.take_along_axis(logits, labels[..., None], axis=-1).squeeze(-1)
        nll = (lse - chosen) * mask
        denom = mx.maximum(mask.sum(), mx.array(1.0))
        return nll.sum() / denom

    def rl_loss(self, token_ids: Sequence[int], *, prompt_len: int, advantage: float) -> Any:
        assert self.mx is not None
        mx = self.mx
        ids = list(token_ids)
        logits = self._forward_logits(ids)
        logits = logits[:, :-1, :]
        labels = mx.array([ids[1:]], dtype=mx.int32)

        lse = mx.logsumexp(logits, axis=-1)
        chosen = mx.take_along_axis(logits, labels[..., None], axis=-1).squeeze(-1)
        logp = chosen - lse

        start = max(0, prompt_len - 1)
        logp_resp = logp[:, start:]
        return -float(advantage) * logp_resp.sum() / mx.maximum(mx.array(1.0), mx.array(logp_resp.size))

    def sequence_logprob(self, token_ids: Sequence[int], *, prompt_len: int) -> Any:
        assert self.mx is not None
        mx = self.mx
        ids = list(token_ids)
        logits = self._forward_logits(ids)
        logits = logits[:, :-1, :]
        labels = mx.array([ids[1:]], dtype=mx.int32)
        lse = mx.logsumexp(logits, axis=-1)
        chosen = mx.take_along_axis(logits, labels[..., None], axis=-1).squeeze(-1)
        logp = chosen - lse
        start = max(0, prompt_len - 1)
        return logp[:, start:].sum()

    def token_logprobs(
        self,
        token_ids: Sequence[int],
        *,
        prompt_len: int,
        top_k: int = 0,
        include_prompt: bool = False,
    ) -> tuple[list[float], list[dict[str, float]] | None]:
        assert self.mx is not None
        mx = self.mx
        ids = list(token_ids)
        if len(ids) < 2:
            return [], [] if top_k > 0 else None

        logits = self._forward_logits(ids)
        logits = logits[:, :-1, :]
        labels = mx.array([ids[1:]], dtype=mx.int32)
        lse = mx.logsumexp(logits, axis=-1)
        chosen = mx.take_along_axis(logits, labels[..., None], axis=-1).squeeze(-1)
        logp = chosen - lse

        start = 0 if include_prompt else max(0, prompt_len - 1)
        values = logp[:, start:]
        try:
            flat = values.flatten().tolist()
        except Exception:
            try:
                flat = [float(v) for v in values.reshape(-1)]
            except Exception:
                flat = [float(v) for v in values]
        logprobs = [float(v) for v in flat]

        if top_k <= 0:
            return logprobs, None

        top_k_all = self._extract_top_k_logprobs(logits, k=int(top_k))
        top_k_list = top_k_all[start:] if top_k_all else []
        return logprobs, top_k_list

    def value_and_grad(self, loss_fn):
        if self.nn is None or self.model is None:
            return loss_fn(self.model), None
        vag = getattr(self.nn, "value_and_grad", None)
        if callable(vag):
            return vag(self.model, loss_fn)(self.model)
        return loss_fn(self.model), None

    def optimizer_and_params(self, *, lr: float, weight_decay: float = 0.0) -> tuple[Any, Any]:
        assert self.optim is not None
        if self.model is None:
            raise RuntimeError("Backend not loaded")

        params = None
        if hasattr(self.model, "trainable_parameters"):
            params = self.model.trainable_parameters()
        if params is None or not params:
            # fallback: train LoRA params if injected
            from ..train.lora import lora_parameters

            params = lora_parameters(self.model)
        if not params:
            params = getattr(self.model, "parameters", lambda: self.model)()

        opt = self.optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
        opt.init(params)
        return opt, params

    def apply_grads(self, optimizer: Any, grads: Any) -> None:
        assert self.mx is not None
        mx = self.mx
        if self.model is None:
            raise RuntimeError("Backend not loaded")
        optimizer.update(self.model, grads)
        try:
            mx.eval(self.model.parameters(), optimizer.state)
        except Exception:  # pragma: no cover
            pass

    def save_adapter(self, out_dir: str, *, metadata: dict | None = None) -> None:
        if self.model is None:
            raise RuntimeError("Backend not loaded")
        adapter_cfg = self._adapter_config or {
            "fine_tune_type": "lora",
            "num_layers": 0,
            "lora_parameters": {},
        }
        save_adapter(self.model, out_dir, adapter_config=adapter_cfg, metadata=metadata)
