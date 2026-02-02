from __future__ import annotations

from typing import Any, Optional

from ..util import tree_map


class Muon:
    """Muon optimizer wrapper (AdamW with Newton-Schulz orthogonalization for 2D grads)."""

    def __init__(
        self,
        learning_rate: Optional[float] = None,
        lr: Optional[float] = None,
        weight_decay: float = 0.0,
        clip: Optional[float] = None,
        ns_iters: int = 5,
        a: float = 3.4445,
        b: float = -4.7750,
        c: float = 2.0315,
    ) -> None:
        import mlx.optimizers as optim  # type: ignore

        self.learning_rate = float(lr if lr is not None else learning_rate if learning_rate is not None else 1e-4)
        self.weight_decay = float(weight_decay)
        self.clip = clip
        self.ns_iters = int(ns_iters)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self._base = optim.AdamW(learning_rate=self.learning_rate, weight_decay=self.weight_decay)
        self.state: dict[str, Any] = {}

    def init(self, params: Any) -> None:
        self._base.init(params)
        self.state = getattr(self._base, "state", {})

    def _clip_grad(self, g):
        if self.clip is None or self.clip <= 0:
            return g
        import mlx.core as mx  # type: ignore

        norm = mx.sqrt((g * g).sum())
        scale = mx.minimum(mx.array(1.0), mx.array(float(self.clip)) / mx.maximum(norm, mx.array(1e-8)))
        return g * scale

    def _orthogonalize(self, g):
        import mlx.core as mx  # type: ignore

        g = self._clip_grad(g)
        if getattr(g, "ndim", None) != 2:
            return g
        in_dim = int(g.shape[1])
        eye = mx.eye(in_dim, dtype=g.dtype)
        out = g
        for _ in range(self.ns_iters):
            gtg = mx.matmul(out.T, out)
            gtg2 = mx.matmul(gtg, gtg)
            out = mx.matmul(out, self.a * eye + self.b * gtg + self.c * gtg2)
        return out

    def update(self, model: Any, grads: Any) -> None:
        if grads is None:
            return
        transformed = tree_map(self._orthogonalize, grads)
        self._base.update(model, transformed)
        self.state = getattr(self._base, "state", self.state)


class MuonClip(Muon):
    """Muon with explicit gradient clipping before orthogonalization."""

    def __init__(
        self,
        learning_rate: Optional[float] = None,
        lr: Optional[float] = None,
        weight_decay: float = 0.0,
        clip: Optional[float] = 1.0,
        ns_iters: int = 5,
        a: float = 3.4445,
        b: float = -4.7750,
        c: float = 2.0315,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            lr=lr,
            weight_decay=weight_decay,
            clip=clip,
            ns_iters=ns_iters,
            a=a,
            b=b,
            c=c,
        )
