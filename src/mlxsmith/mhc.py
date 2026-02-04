"""Experimental Manifold-Constrained Hyper-Connections (mHC) adapters.

This module implements a *block-local* variant of mHC that can be applied to
existing mlx-lm Transformer blocks **without changing model shapes**:

- Block input/output stays ``(..., d_model)`` (so pretrained weights remain valid).
- Inside each Transformer block, we temporarily expand the residual stream into
  ``n`` parallel streams and apply:
    - non-negative read/write maps (H_pre, H_post)
    - a doubly-stochastic residual mixing map (H_res) via Sinkhorn-Knopp

Note: This is not the full paper architecture (which carries an ``n*d_model``
residual stream across all layers). The full design would require end-to-end
reparameterization and fused kernels for efficiency.
"""

from __future__ import annotations

from typing import Any


# Broad attribute matching for mlx-lm Transformer blocks (same heuristic as ZMLX).
_ATTN_NORM_NAMES = (
    "input_layernorm",
    "norm1",
    "operator_norm",
    "attention_norm",
    "attention_layernorm",
)
_POST_NORM_NAMES = (
    "post_attention_layernorm",
    "norm2",
    "ffn_norm",
    "pre_ff_layernorm",
    "feedforward_layernorm",
)
_ATTN_NAMES = ("self_attn", "attention", "conv")
_MLP_NAMES = ("mlp", "feed_forward", "ffn", "block_sparse_moe")


def _require_mlx():
    try:
        import mlx.core as mx  # type: ignore
        import mlx.nn as nn  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "mHC requires MLX. Install with: pip install -e '.[mlx,llm]'"
        ) from e
    return mx, nn


def _first_match(module: Any, names: tuple[str, ...]) -> tuple[str | None, Any]:
    for name in names:
        attr = getattr(module, name, None)
        if attr is not None:
            return name, attr
    return None, None


def sinkhorn_knopp(logits: Any, *, tmax: int = 20, eps: float = 1e-6) -> Any:
    """Entropic projection onto the Birkhoff polytope via Sinkhorn-Knopp.

    Args:
        logits: ``(..., n, n)`` array (unconstrained).
        tmax: Number of alternating row/col normalizations.
        eps: Small constant to avoid division-by-zero.

    Returns:
        Approx. doubly-stochastic matrix with the same shape as ``logits``.
    """
    if tmax < 1:
        raise ValueError("sinkhorn_knopp: tmax must be >= 1")

    mx, _nn = _require_mlx()

    x = logits.astype(mx.float32)
    # Stabilize exp for large logits.
    x = x - mx.max(x)
    m = mx.exp(x)
    for _ in range(int(tmax)):
        m = m / (mx.sum(m, axis=-1, keepdims=True) + eps)  # row normalize
        m = m / (mx.sum(m, axis=-2, keepdims=True) + eps)  # col normalize
    return m


def _call_attn(attn_mod: Any, x_in: Any, mask: Any | None, cache: Any | None) -> Any:
    if mask is None and cache is None:
        return attn_mod(x_in)
    try:
        return attn_mod(x_in, mask=mask, cache=cache)
    except TypeError:
        if cache is not None:
            try:
                return attn_mod(x_in, mask, cache)
            except TypeError:
                return attn_mod(x_in, mask)
        if mask is not None:
            try:
                return attn_mod(x_in, mask)
            except TypeError:
                return attn_mod(x_in)
        return attn_mod(x_in)


def _init_streams(x: Any, n: int) -> Any:
    mx, _nn = _require_mlx()
    if n == 1:
        return mx.expand_dims(x, axis=-2)
    return mx.stack([x] * n, axis=-2)


def _matches_transformer_block(mod: Any) -> bool:
    _mx, nn = _require_mlx()
    if not isinstance(mod, nn.Module):
        return False
    _, attn_norm = _first_match(mod, _ATTN_NORM_NAMES)
    _, post_norm = _first_match(mod, _POST_NORM_NAMES)
    _, attn = _first_match(mod, _ATTN_NAMES)
    _, mlp = _first_match(mod, _MLP_NAMES)
    return attn_norm is not None and post_norm is not None and attn is not None and mlp is not None


def _walk_children(module: Any) -> list[Any]:
    _mx, nn = _require_mlx()
    children: dict[str, Any] = {}
    if hasattr(module, "children") and callable(module.children):
        children = dict(module.children())
    out: list[Any] = []
    for child in children.values():
        if isinstance(child, list):
            for item in child:
                if isinstance(item, nn.Module):
                    out.append(item)
        elif isinstance(child, nn.Module):
            out.append(child)
    return out


def _make_connections(*, n: int, tmax: int):
    mx, nn = _require_mlx()

    class MHCConnections(nn.Module):
        """Static (token-independent) mHC coefficient parameterization for one residual layer."""

        def __init__(self):
            super().__init__()
            self.n = int(n)
            self.tmax = int(tmax)

            self.pre_logits = mx.zeros((self.n,), dtype=mx.float32)
            self.post_logits = mx.zeros((self.n,), dtype=mx.float32)
            self.res_logits = mx.zeros((self.n, self.n), dtype=mx.float32)

        def h_pre(self) -> Any:
            w = mx.sigmoid(self.pre_logits)
            return w / (mx.sum(w) + 1e-6)

        def h_post(self) -> Any:
            return 2.0 * mx.sigmoid(self.post_logits)

        def h_res(self) -> Any:
            return sinkhorn_knopp(self.res_logits, tmax=self.tmax)

        def read(self, streams: Any) -> Any:
            """Apply H_pre: (..., n, d) -> (..., d)."""
            w = self.h_pre().astype(mx.float32)
            w = w.reshape((1,) * (streams.ndim - 2) + (self.n, 1))
            x = mx.sum(streams.astype(mx.float32) * w, axis=-2)
            return x.astype(streams.dtype)

        def mix_and_write(self, streams: Any, layer_out: Any) -> Any:
            """Apply H_res and H_post^T update: (..., n, d) + (..., d) -> (..., n, d)."""
            hres = self.h_res().astype(mx.float32)
            flat = streams.astype(mx.float32).reshape((-1, self.n, streams.shape[-1]))
            mixed = mx.matmul(hres, flat).reshape(streams.shape).astype(streams.dtype)

            hpost = self.h_post().astype(mx.float32)
            hpost = hpost.reshape((1,) * (layer_out.ndim - 1) + (self.n, 1))
            post = mx.expand_dims(layer_out.astype(mx.float32), axis=-2) * hpost
            return mixed + post.astype(streams.dtype)

    return MHCConnections


def _patch_block(mod: Any, *, n: int, tmax: int) -> bool:
    _mx, nn = _require_mlx()
    if getattr(mod, "_mlxsmith_mhc_patched", False):
        return False
    if not isinstance(mod, nn.Module):
        return False

    attn_norm_name, _ = _first_match(mod, _ATTN_NORM_NAMES)
    post_norm_name, _ = _first_match(mod, _POST_NORM_NAMES)
    attn_name, _ = _first_match(mod, _ATTN_NAMES)
    mlp_name, _ = _first_match(mod, _MLP_NAMES)
    if attn_norm_name is None or post_norm_name is None or attn_name is None or mlp_name is None:
        return False

    original_call = mod.__call__.__func__ if hasattr(mod.__call__, "__func__") else None

    MHCConnections = _make_connections(n=int(n), tmax=int(tmax))
    mod._mlxsmith_mhc_attn = MHCConnections()  # type: ignore[attr-defined]
    mod._mlxsmith_mhc_mlp = MHCConnections()  # type: ignore[attr-defined]
    mod._mlxsmith_mhc_original_call = original_call  # type: ignore[attr-defined]

    def patched_call(self_mod: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
        mask = kwargs.get("mask")
        cache = kwargs.get("cache")
        if len(args) > 0:
            mask = args[0]
        if len(args) > 1:
            cache = args[1]

        attn_norm = getattr(self_mod, attn_norm_name)
        post_norm = getattr(self_mod, post_norm_name)
        attn_mod = getattr(self_mod, attn_name)
        mlp_mod = getattr(self_mod, mlp_name)

        streams = _init_streams(x, int(self_mod._mlxsmith_mhc_attn.n))  # type: ignore[attr-defined]

        # Attention "layer"
        x_attn_in = self_mod._mlxsmith_mhc_attn.read(streams)  # type: ignore[attr-defined]
        attn_out = _call_attn(attn_mod, attn_norm(x_attn_in), mask, cache)
        streams = self_mod._mlxsmith_mhc_attn.mix_and_write(streams, attn_out)  # type: ignore[attr-defined]

        # MLP "layer"
        x_mlp_in = self_mod._mlxsmith_mhc_mlp.read(streams)  # type: ignore[attr-defined]
        mlp_out = mlp_mod(post_norm(x_mlp_in))
        streams = self_mod._mlxsmith_mhc_mlp.mix_and_write(streams, mlp_out)  # type: ignore[attr-defined]

        # Readout: take stream 0 (all streams start identical at init).
        return streams[..., 0, :]

    mod._mlxsmith_mhc_patched = True  # type: ignore[attr-defined]
    mod.__class__ = type(
        mod.__class__.__name__,
        (mod.__class__,),
        {"__call__": patched_call},
    )
    return True


def apply_mhc(
    model: Any,
    *,
    n: int = 4,
    tmax: int = 20,
    verbose: bool = False,
) -> int:
    """Patch Transformer blocks in-place to use block-local mHC adapters.

    Returns:
        The number of patched blocks.
    """
    _mx, nn = _require_mlx()
    if n < 1:
        raise ValueError("apply_mhc: n must be >= 1")
    if tmax < 1:
        raise ValueError("apply_mhc: tmax must be >= 1")

    if not isinstance(model, nn.Module):
        return 0

    patched = 0
    stack: list[Any] = [model]
    while stack:
        mod = stack.pop()
        if _matches_transformer_block(mod):
            if _patch_block(mod, n=int(n), tmax=int(tmax)):
                patched += 1
        stack.extend(_walk_children(mod))

    if verbose:
        print(f"[mlxsmith.mhc] Patched {patched} transformer blocks (n={n}, tmax={tmax}).")
    return patched

