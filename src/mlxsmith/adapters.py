from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from .util import ensure_dir, copytree, now_ts


def merge_adapters(
    base_model: str,
    adapters: Iterable[Path],
    out_dir: Path,
    *,
    weights: Optional[list[float]] = None,
) -> Path:
    adapter_list = [Path(a) for a in adapters]
    if not adapter_list:
        raise RuntimeError("No adapters provided")
    ensure_dir(out_dir)

    try:
        from mlx_lm.tuner import utils as tuner_utils  # type: ignore

        if hasattr(tuner_utils, "merge_adapters"):
            tuner_utils.merge_adapters(
                base_model,
                [str(p) for p in adapter_list],
                str(out_dir),
                weights=weights,
            )
            return out_dir
    except Exception:
        pass

    # Fallback: copy the first adapter and record metadata.
    copytree(adapter_list[0], out_dir)
    meta = {
        "base_model": base_model,
        "merged_from": [str(p) for p in adapter_list],
        "weights": weights,
        "merged_at": now_ts(),
        "note": "merge_adapters fallback: copied first adapter (mlx_lm merge unavailable)",
    }
    (out_dir / "adapter_merge.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir
