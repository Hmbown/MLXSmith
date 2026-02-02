from pathlib import Path

from mlxsmith.data import analyze_jsonl


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_analyze_jsonl_sft(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "sft.jsonl",
        "\n".join(
            [
                '{"prompt": "hi", "response": "ok"}',
                '{"prompt": "", "response": "x"}',
                "not-json",
                "",
            ]
        )
        + "\n",
    )
    stats = analyze_jsonl(path, kind="sft")
    assert stats["rows"] == 2
    assert stats["bad_json"] == 1
    assert stats["empty_lines"] == 1
    assert stats["missing_prompt"] == 1
    assert stats["missing_response"] == 0


def test_analyze_jsonl_prefs(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "prefs.jsonl",
        "\n".join(
            [
                '{"prompt": "p", "chosen": "c", "rejected": "r"}',
                '{"prompt": "p", "chosen": "", "rejected": "r"}',
            ]
        )
        + "\n",
    )
    stats = analyze_jsonl(path, kind="prefs")
    assert stats["rows"] == 2
    assert stats["missing_chosen"] == 1
    assert stats["missing_rejected"] == 0
