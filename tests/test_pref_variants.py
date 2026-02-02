from pathlib import Path

from mlxsmith.config import ProjectConfig
from mlxsmith.train.pref import run_pref
from mlxsmith.train.sft import run_sft


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(__import__("json").dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_pref_loss_variants(tmp_path: Path):
    cfg = ProjectConfig()
    cfg.model.backend = "mock"
    cfg.train.iters = 2
    cfg.train.save_every = 1
    cfg.train.log_every = 1

    _write_jsonl(tmp_path / "data" / "sft" / "train.jsonl", [
        {"prompt": "Hello", "completion": " world"},
    ])
    run = run_sft(tmp_path, cfg, tmp_path / "data" / "sft", "dummy/model", "none")

    _write_jsonl(tmp_path / "data" / "prefs" / "train.jsonl", [
        {"prompt": "Say hi", "chosen": " hi", "rejected": " bye"},
    ])

    for loss_type in ["dpo", "cpo", "orpo", "ipo", "hinge", "simpo", "tdpo"]:
        cfg.pref.loss_type = loss_type
        run_pref(tmp_path, cfg, tmp_path / "data" / "prefs", run.adapter_dir, "none")
