from pathlib import Path

from mlxsmith.config import ProjectConfig
from mlxsmith.train.kto import run_kto


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(__import__("json").dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_kto_mock(tmp_path: Path):
    cfg = ProjectConfig()
    cfg.model.backend = "mock"
    cfg.train.iters = 2
    cfg.train.save_every = 1
    cfg.train.log_every = 1

    data_path = tmp_path / "data" / "kto.jsonl"
    _write_jsonl(
        data_path,
        [
            {"prompt": "Say hi", "response": " hello", "label": True},
            {"prompt": "Say bye", "response": " goodbye", "label": False},
        ],
    )

    run = run_kto(tmp_path, cfg, data_path, "dummy/model", "none")
    assert run.adapter_dir.exists()
