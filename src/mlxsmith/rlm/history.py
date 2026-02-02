from __future__ import annotations

import json
from pathlib import Path

from ..util import ensure_dir


def append_history(path: Path, record: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
