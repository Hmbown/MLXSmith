from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from ..util import ensure_dir


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def load_corpus(path: Path, *, max_size: int | None = None) -> List[dict]:
    rows = _read_jsonl(path)
    if max_size is not None and len(rows) > max_size:
        return rows[-max_size:]
    return rows


def append_corpus(path: Path, rows: Iterable[dict], *, max_size: int) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if max_size <= 0:
        return

    all_rows = _read_jsonl(path)
    if len(all_rows) > max_size:
        trimmed = all_rows[-max_size:]
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in trimmed) + "\n", encoding="utf-8")


def sample_corpus(rows: List[dict], *, n: int, hard_ratio: float = 0.0) -> List[dict]:
    if n <= 0 or not rows:
        return []
    if n >= len(rows):
        return list(rows)

    hard_n = int(round(n * max(0.0, min(1.0, hard_ratio))))
    easy_n = n - hard_n

    # Hard samples = longest prompts (proxy for difficulty).
    sorted_rows = sorted(rows, key=lambda r: len((r.get("prompt") or "")), reverse=True)
    hard = sorted_rows[:hard_n] if hard_n > 0 else []

    # Easy samples = shortest prompts.
    easy = sorted_rows[-easy_n:] if easy_n > 0 else []

    # Deduplicate while preserving order.
    seen = set()
    out = []
    for row in hard + easy:
        key = row.get("id") or row.get("hash") or (row.get("prompt"), row.get("response"))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= n:
            break
    return out
