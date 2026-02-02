"""LoRA adapter utilities.

This module prefers MLX-LM's LoRA utilities and adapter format when available.
Fallback implementations are provided for environments without MLX/MLX-LM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _require_mlx():
    import mlx.core as mx  # type: ignore
    import mlx.nn as nn  # type: ignore
    from mlx.utils import tree_flatten  # type: ignore

    return mx, nn, tree_flatten


def _try_mlx_lm_utils():
    try:
        from mlx_lm.tuner import utils as tuner_utils  # type: ignore
        from mlx_lm.utils import save_config as mlx_save_config  # type: ignore
    except Exception:
        return None, None
    return tuner_utils, mlx_save_config


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: list[str] | None = None
    num_layers: int = 0  # 0 => all layers
    scale: float | None = None
    fine_tune_type: str = "lora"  # lora | dora | full


class LoRALinear:
    """Minimal fallback LoRA wrapper for mlx.nn.Linear."""

    def __init__(self, base_linear: Any, *, r: int, alpha: int, dropout: float = 0.0):
        mx, nn, _ = _require_mlx()
        self.nn = nn
        self.mx = mx

        self.base = base_linear
        self.r = int(r)
        self.alpha = float(alpha)
        self.scale = float(alpha) / float(r) if r > 0 else 0.0
        self.dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else None

        w = getattr(base_linear, "weight")
        out_dim, in_dim = int(w.shape[0]), int(w.shape[1])

        self.A = mx.random.normal((self.r, in_dim)) * 0.01
        self.B = mx.zeros((out_dim, self.r))

    def __call__(self, x):
        mx = self.mx
        base_w = mx.stop_gradient(self.base.weight)
        y = x @ base_w.T
        if getattr(self.base, "bias", None) is not None:
            y = y + mx.stop_gradient(self.base.bias)
        z = x
        if self.dropout is not None:
            z = self.dropout(z)
        z = (z @ self.A.T) @ self.B.T
        return y + z * self.scale


def _iter_named_modules(root: Any, prefix: str = ""):
    for name in dir(root):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(root, name)
        except Exception:
            continue
        if callable(getattr(obj, "__call__", None)) and hasattr(obj, "__dict__"):
            full = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
            yield full, root, name, obj


def inject_lora(
    model: Any,
    *,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
) -> int:
    _mx, _nn, _ = _require_mlx()
    targets = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
    wrapped = 0

    for full, parent, attr, obj in list(_iter_named_modules(model)):
        if not hasattr(obj, "weight"):
            continue
        if not any(attr.endswith(t) or full.endswith(t) for t in targets):
            continue
        cls_name = obj.__class__.__name__.lower()
        if "linear" not in cls_name:
            continue
        try:
            setattr(parent, attr, LoRALinear(obj, r=r, alpha=alpha, dropout=dropout))
            wrapped += 1
        except Exception:
            continue

    return wrapped


def lora_parameters(model: Any) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for full, _parent, _attr, obj in list(_iter_named_modules(model)):
        if obj.__class__.__name__ == "LoRALinear" or (hasattr(obj, "A") and hasattr(obj, "B")):
            try:
                params[f"{full}.A"] = obj.A
                params[f"{full}.B"] = obj.B
            except Exception:
                pass
    return params


def _scale_for_config(cfg: LoRAConfig) -> float:
    if cfg.scale is not None:
        return float(cfg.scale)
    if cfg.r == 0:
        return float(cfg.alpha)
    return float(cfg.alpha) / float(cfg.r)


def _keys_for_target_modules(model: Any, target_modules: list[str]) -> set[str]:
    keys: set[str] = set()
    # Prefer model.named_modules if present (MLX-LM models support this)
    if hasattr(model, "named_modules"):
        for name, _mod in model.named_modules():
            if any(name.endswith(t) for t in target_modules):
                keys.add(name)
        return keys

    # Fallback: walk attributes
    for full, _parent, _attr, obj in list(_iter_named_modules(model)):
        if any(full.endswith(t) for t in target_modules):
            keys.add(full)
    return keys


def apply_lora(model: Any, cfg: LoRAConfig) -> dict:
    """Apply LoRA layers (prefer MLX-LM utilities) and return adapter config."""
    tuner_utils, _save_cfg = _try_mlx_lm_utils()
    scale = _scale_for_config(cfg)
    keys = None
    if cfg.target_modules:
        keys = sorted(_keys_for_target_modules(model, cfg.target_modules))

    if tuner_utils is not None and hasattr(tuner_utils, "linear_to_lora_layers"):
        # MLX-LM format
        config = {
            "rank": int(cfg.r),
            "scale": float(scale),
            "dropout": float(cfg.dropout),
        }
        if keys:
            config["keys"] = keys
        num_layers = int(cfg.num_layers)
        tuner_utils.linear_to_lora_layers(
            model,
            num_layers,
            config,
            use_dora=(cfg.fine_tune_type == "dora"),
        )
        return {
            "fine_tune_type": cfg.fine_tune_type,
            "num_layers": num_layers,
            "lora_parameters": config,
        }

    # Fallback to local LoRA injection
    inject_lora(
        model,
        r=cfg.r,
        alpha=cfg.alpha,
        dropout=cfg.dropout,
        target_modules=cfg.target_modules,
    )
    return {
        "fine_tune_type": "lora",
        "num_layers": 0,
        "lora_parameters": {
            "rank": int(cfg.r),
            "scale": float(scale),
            "dropout": float(cfg.dropout),
        },
    }


def save_adapter(model: Any, out_dir: str | Path, *, adapter_config: dict, metadata: dict | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tuner_utils, mlx_save_config = _try_mlx_lm_utils()
    try:
        mx, _nn, tree_flatten = _require_mlx()
    except Exception:
        mx = None
        tree_flatten = None

    if mx is not None and hasattr(model, "trainable_parameters") and tree_flatten is not None:
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        try:
            mx.save_safetensors(str(out / "adapters.safetensors"), adapter_weights)
        except Exception:
            # fallback to numpy
            import numpy as np

            arrays = {k: mx.array(v).to_numpy() for k, v in adapter_weights.items()}
            np.savez(out / "lora.npz", **arrays)
    else:
        # fallback to local LoRA params
        params = lora_parameters(model)
        if params:
            import numpy as np

            mx_local, _nn_local, _ = _require_mlx()
            arrays = {k: mx_local.array(v).to_numpy() for k, v in params.items()}
            np.savez(out / "lora.npz", **arrays)

    config_path = out / "adapter_config.json"
    if mlx_save_config is not None:
        cfg = dict(adapter_config)
        mlx_save_config(cfg, config_path)
    else:
        config_path.write_text(json.dumps(adapter_config, indent=2), encoding="utf-8")

    if metadata is not None:
        (out / "adapter_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )


def load_adapter_config(adapter_dir: str | Path) -> dict | None:
    path = Path(adapter_dir) / "adapter_config.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def apply_adapter(model: Any, adapter_dir: str | Path) -> dict | None:
    adapter_dir = Path(adapter_dir)
    adapter_cfg = load_adapter_config(adapter_dir)
    tuner_utils, _ = _try_mlx_lm_utils()
    if adapter_cfg is not None and tuner_utils is not None and hasattr(tuner_utils, "load_adapters"):
        tuner_utils.load_adapters(model, str(adapter_dir))
        return adapter_cfg

    # Fallback: load local lora.npz into LoRALinear wrappers
    lora_file = adapter_dir / "lora.npz"
    if not lora_file.exists():
        return adapter_cfg

    import numpy as np

    weights = dict(np.load(lora_file))
    mx, _nn, _ = _require_mlx()
    # apply
    for full, _parent, _attr, obj in list(_iter_named_modules(model)):
        if hasattr(obj, "A") and hasattr(obj, "B"):
            key_a = f"{full}.A"
            key_b = f"{full}.B"
            if key_a in weights:
                obj.A = mx.array(weights[key_a])
            if key_b in weights:
                obj.B = mx.array(weights[key_b])
    return adapter_cfg
