#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_PATTERN = r"(mlxsmith|data/|runs/|eval/|envs/|verifiers/|docs/)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter prompts to repo-specific ones.")
    parser.add_argument("--in", dest="inp", required=True, help="Input JSONL with {prompt}")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="Regex to keep prompts")
    args = parser.parse_args()

    pattern = re.compile(args.pattern, flags=re.IGNORECASE)
    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0
    with inp.open("r", encoding="utf-8") as f_in, out.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                dropped += 1
                continue
            if not pattern.search(prompt):
                dropped += 1
                continue
            f_out.write(json.dumps({"prompt": prompt}) + "\n")
            kept += 1

    print(f"Kept {kept}, dropped {dropped} -> {out}")


if __name__ == "__main__":
    main()
