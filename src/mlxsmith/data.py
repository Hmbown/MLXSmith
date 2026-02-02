from __future__ import annotations

import json
from pathlib import Path

from .util import ensure_dir

def import_sharegpt(in_path: Path, out_path: Path):
    """Convert ShareGPT-ish JSONL into {prompt, response} JSONL.

    Expected input lines: {"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":"..."}], ...}
    """
    ensure_dir(out_path.parent)
    n = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            conv = obj.get("conversations") or []
            # naive: take first human then first assistant after it
            prompt = None
            response = None
            for turn in conv:
                frm = (turn.get("from") or "").lower()
                if prompt is None and frm in ("human", "user"):
                    prompt = turn.get("value") or ""
                elif prompt is not None and response is None and frm in ("gpt", "assistant"):
                    response = turn.get("value") or ""
                    break
            if prompt is None or response is None:
                continue
            fout.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")
            n += 1
    return n

def split_jsonl(in_path: Path, out_dir: Path, valid_frac: float, test_frac: float, seed: int = 1337):
    import random
    random.seed(seed)
    rows = [json.loads(line) for line in in_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    random.shuffle(rows)
    n = len(rows)
    n_test = int(n * test_frac)
    n_valid = int(n * valid_frac)
    test = rows[:n_test]
    valid = rows[n_test:n_test+n_valid]
    train = rows[n_test+n_valid:]
    ensure_dir(out_dir)
    for name, part in [("train.jsonl", train), ("valid.jsonl", valid), ("test.jsonl", test)]:
        (out_dir / name).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in part) + ("\n" if part else ""), encoding="utf-8")
    return {"n": n, "train": len(train), "valid": len(valid), "test": len(test)}
