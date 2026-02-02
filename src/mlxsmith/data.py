from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .util import ensure_dir


PRESET_DATASETS: dict[str, dict] = {
    "alpaca": {
        "dataset": "tatsu-lab/alpaca",
        "kind": "sft",
        "prompt_field": "instruction",
        "response_field": "output",
        "license": "cc-by-nc-4.0",
    },
    "hh-rlhf": {
        "dataset": "Anthropic/hh-rlhf",
        "kind": "prefs",
        "chosen_field": "chosen",
        "rejected_field": "rejected",
        "license": "mit",
    },
    "ultrachat-200k": {
        "dataset": "HuggingFaceH4/ultrachat_200k",
        "kind": "sft",
        "split": "train_sft",
        "config": "default",
        "license": "mit",
    },
    "ultrafeedback-binarized-prefs": {
        "dataset": "HuggingFaceH4/ultrafeedback_binarized",
        "kind": "prefs",
        "split": "train_prefs",
        "config": "default",
        "prompt_field": "prompt",
        "chosen_field": "chosen",
        "rejected_field": "rejected",
        "license": "mit",
    },
    "ultrafeedback-binarized-sft": {
        "dataset": "HuggingFaceH4/ultrafeedback_binarized",
        "kind": "sft",
        "split": "train_sft",
        "config": "default",
        "license": "mit",
    },
}


def list_presets() -> dict[str, dict]:
    return {k: dict(v) for k, v in PRESET_DATASETS.items()}


def resolve_preset(name: str) -> dict:
    key = name.strip()
    if key not in PRESET_DATASETS:
        options = ", ".join(sorted(PRESET_DATASETS.keys()))
        raise ValueError(f"Unknown preset: {name}. Available: {options}")
    return dict(PRESET_DATASETS[key])


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _select_field_value(row: dict, keys: list[str | None]):
    for key in keys:
        if not key:
            continue
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _messages_to_text(msgs: list) -> str:
    parts: list[str] = []
    for msg in msgs:
        if msg is None:
            continue
        if isinstance(msg, dict):
            text = msg.get("content") or msg.get("value") or msg.get("text") or ""
        else:
            text = str(msg)
        if text:
            parts.append(str(text))
    return "\n".join(parts)


def _coerce_text(val) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, (int, float, bool)):
        return str(val)
    if isinstance(val, list):
        if not val:
            return ""
        if all(isinstance(v, dict) for v in val):
            return _messages_to_text(val)
        return "\n".join(str(v) for v in val if v not in (None, ""))
    if isinstance(val, dict):
        for key in ("text", "content", "value"):
            if key in val and val[key] not in (None, ""):
                return str(val[key])
        if "messages" in val and isinstance(val["messages"], list):
            return _messages_to_text(val["messages"])
    return str(val)


def _messages_from_value(val) -> list | None:
    if isinstance(val, dict) and isinstance(val.get("messages"), list):
        return val.get("messages")
    if isinstance(val, list) and val and all(isinstance(m, dict) for m in val):
        return val
    return None


def _select_field(row: dict, keys: list[str]) -> str:
    val = _select_field_value(row, keys)
    return _coerce_text(val) if val is not None else ""


def _extract_prompt_response_from_messages(val) -> tuple[str, str]:
    msgs = _messages_from_value(val)
    if not msgs:
        return "", ""
    if len(msgs) == 1:
        return "", _coerce_text(msgs[0])
    prompt = _messages_to_text(msgs[:-1])
    response = _coerce_text(msgs[-1])
    return prompt, response


def _row_to_prompt_response(row: dict, prompt_field: str | None = None, response_field: str | None = None) -> tuple[str, str]:
    prompt = _select_field(
        row,
        [
            prompt_field,
            "prompt",
            "instruction",
            "input",
            "question",
        ],
    )
    response = _select_field(
        row,
        [
            response_field,
            "response",
            "output",
            "answer",
            "completion",
        ],
    )
    if "messages" in row and isinstance(row.get("messages"), list):
        msg_prompt, msg_response = _extract_prompt_response_from_messages(row.get("messages"))
        if msg_prompt and (not prompt or len(msg_prompt) > len(prompt)):
            prompt = msg_prompt
        if not response and msg_response:
            response = msg_response
    return prompt, response


def _row_to_pref(
    row: dict,
    prompt_field: str | None = None,
    chosen_field: str | None = None,
    rejected_field: str | None = None,
) -> tuple[str, str, str]:
    prompt = _select_field(
        row,
        [
            prompt_field,
            "prompt",
            "instruction",
            "input",
            "question",
            "query",
            "context",
            "history",
        ],
    )

    chosen_val = _select_field_value(
        row,
        [
            chosen_field,
            "chosen",
            "accepted",
            "preferred",
            "chosen_response",
            "response",
            "output",
            "answer",
        ],
    )
    rejected_val = _select_field_value(
        row,
        [
            rejected_field,
            "rejected",
            "rejected_response",
            "rejected_output",
            "rejected_answer",
            "dispreferred",
        ],
    )

    chosen = _coerce_text(chosen_val)
    rejected = _coerce_text(rejected_val)

    if chosen_val is not None:
        msg_prompt, msg_response = _extract_prompt_response_from_messages(chosen_val)
        if msg_prompt and (not prompt or len(msg_prompt) > len(prompt)):
            prompt = msg_prompt
        if msg_response:
            chosen = msg_response

    if rejected_val is not None:
        msg_prompt, msg_response = _extract_prompt_response_from_messages(rejected_val)
        if msg_prompt and (not prompt or len(msg_prompt) > len(prompt)):
            prompt = msg_prompt
        if msg_response:
            rejected = msg_response

    if (not prompt or not chosen) and chosen_val is not None:
        msg_prompt, msg_response = _extract_prompt_response_from_messages(chosen_val)
        if not prompt and msg_prompt:
            prompt = msg_prompt
        if not chosen and msg_response:
            chosen = msg_response

    if (not prompt or not rejected) and rejected_val is not None:
        msg_prompt, msg_response = _extract_prompt_response_from_messages(rejected_val)
        if not prompt and msg_prompt:
            prompt = msg_prompt
        if not rejected and msg_response:
            rejected = msg_response

    if not prompt and "messages" in row and isinstance(row.get("messages"), list):
        msg_prompt, _msg_response = _extract_prompt_response_from_messages(row.get("messages"))
        if msg_prompt:
            prompt = msg_prompt

    return prompt, chosen, rejected


def _write_provenance(out_dir: Path, metadata: dict) -> Path:
    ensure_dir(out_dir)
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta_path

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


def pull_hf_dataset(
    dataset: str,
    out_dir: Path,
    split: str = "train",
    limit: int | None = None,
    prompt_field: str | None = None,
    response_field: str | None = None,
    chosen_field: str | None = None,
    rejected_field: str | None = None,
    config: str | None = None,
    revision: str | None = None,
    kind: str = "sft",
    license: str | None = None,
    notes: str | None = None,
    preset: str | None = None,
    write_metadata: bool = True,
) -> dict:
    """Download a HF dataset split and write prompt/response JSONL.

    Args:
        dataset: HF dataset name (e.g. "tatsu-lab/alpaca")
        out_dir: Output directory (data/sft or data/prefs)
        split: Dataset split to pull
        limit: Optional max rows to write
        prompt_field: Optional override for prompt field name
        response_field: Optional override for response field name
        config: Optional dataset config/subset name
        revision: Optional dataset revision

    Returns:
        Stats dict with counts.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"datasets not available: {e}")

    ensure_dir(out_dir)
    ds = load_dataset(dataset, name=config, split=split, revision=revision)
    total = len(ds)
    n = 0
    skipped = 0
    kind_norm = kind.strip().lower()
    if kind_norm in ("pref", "prefs", "preference", "preferences"):
        kind_norm = "prefs"
    elif kind_norm in ("sft", "supervised"):
        kind_norm = "sft"
    else:
        raise ValueError(f"Unsupported kind: {kind}")
    out_path = out_dir / "train.jsonl"
    with out_path.open("w", encoding="utf-8") as fout:
        for row in ds:
            if kind_norm == "prefs":
                prompt, chosen, rejected = _row_to_pref(row, prompt_field, chosen_field, rejected_field)
                if not (prompt and chosen and rejected):
                    skipped += 1
                    continue
                fout.write(
                    json.dumps(
                        {"prompt": prompt, "chosen": chosen, "rejected": rejected},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            else:
                prompt, response = _row_to_prompt_response(row, prompt_field, response_field)
                if not (prompt and response):
                    skipped += 1
                    continue
                fout.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")
            n += 1
            if limit and n >= limit:
                break
    metadata = {
        "dataset": dataset,
        "preset": preset,
        "config": config,
        "split": split,
        "revision": revision,
        "license": license,
        "notes": notes,
        "kind": kind_norm,
        "total": total,
        "written": n,
        "skipped": skipped,
        "limit": limit,
        "prompt_field": prompt_field,
        "response_field": response_field if kind_norm == "sft" else None,
        "chosen_field": chosen_field if kind_norm == "prefs" else None,
        "rejected_field": rejected_field if kind_norm == "prefs" else None,
        "out": str(out_path),
        "generated_at": _utc_now_iso(),
    }
    if write_metadata:
        meta_path = _write_provenance(out_dir, metadata)
        metadata["metadata"] = str(meta_path)
    return metadata


def _normalize_kind(kind: str | None) -> str | None:
    if not kind:
        return None
    kind_norm = kind.strip().lower()
    if kind_norm in ("pref", "prefs", "preference", "preferences"):
        return "prefs"
    if kind_norm in ("sft", "supervised"):
        return "sft"
    return kind_norm


def _infer_kind_from_row(row: dict) -> str:
    if "chosen" in row or "rejected" in row:
        return "prefs"
    if "response" in row:
        return "sft"
    if "messages" in row:
        return "sft"
    return "unknown"


def analyze_jsonl(path: Path, kind: str | None = None, limit: int | None = None) -> dict:
    stats = {
        "rows": 0,
        "empty_lines": 0,
        "bad_json": 0,
        "missing_prompt": 0,
        "missing_response": 0,
        "missing_chosen": 0,
        "missing_rejected": 0,
        "prompt_chars": 0,
        "response_chars": 0,
        "chosen_chars": 0,
        "rejected_chars": 0,
        "prompt_count": 0,
        "response_count": 0,
        "chosen_count": 0,
        "rejected_count": 0,
        "kind": None,
    }
    kind_norm = _normalize_kind(kind)
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if limit and stats["rows"] >= limit:
                break
            if not line.strip():
                stats["empty_lines"] += 1
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json"] += 1
                continue
            stats["rows"] += 1
            if not kind_norm:
                kind_norm = _infer_kind_from_row(row)
            prompt = _coerce_text(row.get("prompt"))
            if not prompt:
                stats["missing_prompt"] += 1
            else:
                stats["prompt_chars"] += len(prompt)
                stats["prompt_count"] += 1
            if kind_norm == "prefs":
                chosen = _coerce_text(row.get("chosen"))
                rejected = _coerce_text(row.get("rejected"))
                if not chosen:
                    stats["missing_chosen"] += 1
                else:
                    stats["chosen_chars"] += len(chosen)
                    stats["chosen_count"] += 1
                if not rejected:
                    stats["missing_rejected"] += 1
                else:
                    stats["rejected_chars"] += len(rejected)
                    stats["rejected_count"] += 1
            elif kind_norm == "sft":
                response = _coerce_text(row.get("response"))
                if not response:
                    stats["missing_response"] += 1
                else:
                    stats["response_chars"] += len(response)
                    stats["response_count"] += 1
            else:
                response = _coerce_text(row.get("response"))
                if response:
                    stats["response_chars"] += len(response)
                    stats["response_count"] += 1
                else:
                    stats["missing_response"] += 1
    stats["kind"] = kind_norm or "unknown"
    return stats
