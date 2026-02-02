from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..util import ensure_dir


@dataclass
class GatingState:
    best_score: Optional[float] = None
    best_adapter: Optional[str] = None
    ema_score: Optional[float] = None
    last_iteration: int = 0
    current_adapter: Optional[str] = None


def load_state(path: Path) -> GatingState:
    if not path.exists():
        return GatingState()
    data = json.loads(path.read_text(encoding="utf-8"))
    return GatingState(
        best_score=data.get("best_score"),
        best_adapter=data.get("best_adapter"),
        ema_score=data.get("ema_score"),
        last_iteration=int(data.get("last_iteration", 0)),
        current_adapter=data.get("current_adapter"),
    )


def save_state(path: Path, state: GatingState) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(
            {
                "best_score": state.best_score,
                "best_adapter": state.best_adapter,
                "ema_score": state.ema_score,
                "last_iteration": state.last_iteration,
                "current_adapter": state.current_adapter,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def should_accept(
    score: float,
    state: GatingState,
    *,
    mode: str,
    threshold: float = 0.0,
    ema_alpha: float = 0.2,
) -> bool:
    if state.best_score is None:
        return True
    mode = (mode or "strict").lower()
    if mode == "threshold":
        return score >= float(state.best_score) + float(threshold)
    if mode == "ema":
        ema = state.ema_score if state.ema_score is not None else state.best_score
        return score >= float(ema)
    # strict
    return score > float(state.best_score)


def update_state(
    state: GatingState,
    *,
    iteration: int,
    score: float,
    adapter_path: str,
    accepted: bool,
    ema_alpha: float = 0.2,
) -> GatingState:
    state.last_iteration = iteration
    if state.ema_score is None:
        state.ema_score = score
    else:
        state.ema_score = float(ema_alpha) * score + (1.0 - float(ema_alpha)) * float(state.ema_score)

    if accepted:
        state.current_adapter = adapter_path
        if state.best_score is None or score > float(state.best_score):
            state.best_score = score
            state.best_adapter = adapter_path
    return state
