"""Tests for mlxsmith.mhc (experimental mHC adapters)."""

from __future__ import annotations

import subprocess
import sys

import pytest

def _mlx_import_works() -> bool:
    """Return True if `import mlx` works without aborting the interpreter.

    Some environments (e.g. CI runners without Metal) may have MLX installed but
    crash on import. Probe in a subprocess so pytest can skip safely.
    """
    code = "import mlx.core, mlx.nn; print('ok')"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return False
    return proc.returncode == 0


if not _mlx_import_works():
    pytest.skip("mlx not available or failed to initialize", allow_module_level=True)

import mlx.core as mx  # type: ignore
import mlx.nn as nn  # type: ignore


class _DummyAttn(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)

    def __call__(self, x, mask=None, cache=None):  # noqa: ANN001
        return self.proj(x)


class _DummyMLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc = nn.Linear(d, d, bias=False)

    def __call__(self, x):  # noqa: ANN001
        return self.fc(x)


class _DummyBlock(nn.Module):
    """Pre-norm Transformer-like block compatible with mlxsmith.mhc patching."""

    def __init__(self, d: int):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(d)
        self.post_attention_layernorm = nn.RMSNorm(d)
        self.self_attn = _DummyAttn(d)
        self.mlp = _DummyMLP(d)

    def __call__(self, x, mask=None, cache=None):  # noqa: ANN001
        h = x + self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        return h + self.mlp(self.post_attention_layernorm(h))


def test_sinkhorn_knopp_doubly_stochastic():
    from mlxsmith.mhc import sinkhorn_knopp

    logits = mx.zeros((4, 4))
    m = sinkhorn_knopp(logits, tmax=5)
    mx.eval(m)

    row = mx.sum(m, axis=-1)
    col = mx.sum(m, axis=-2)
    mx.eval(row, col)

    assert mx.allclose(row, mx.ones_like(row), atol=1e-4), f"row_sums: {row.tolist()}"
    assert mx.allclose(col, mx.ones_like(col), atol=1e-4), f"col_sums: {col.tolist()}"


def test_apply_mhc_preserves_block_behavior_at_init():
    from mlxsmith.mhc import apply_mhc

    d = 16
    block = _DummyBlock(d)

    x = mx.random.normal((2, 3, d))
    y_ref = block(x)
    mx.eval(y_ref)

    patched = apply_mhc(block, n=4, tmax=5)
    assert patched == 1

    y = block(x)
    mx.eval(y)

    assert y.shape == y_ref.shape
    assert mx.allclose(y, y_ref, atol=1e-4), (
        f"patched != baseline\npatched: {y.tolist()}\nbaseline: {y_ref.tolist()}"
    )
