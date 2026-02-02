from pathlib import Path

from mlxsmith.runs import new_run, snapshot_config


def test_run_dirs(tmp_path: Path):
    run = new_run(tmp_path, "sft")
    assert run.run_dir.exists()
    assert run.adapter_dir.exists()
    assert run.logs_dir.exists()


def test_snapshot_config(tmp_path: Path):
    run = new_run(tmp_path, "sft")
    snapshot_config({"model": {"id": "test"}}, run.config_snapshot_path)
    assert run.config_snapshot_path.exists()
    text = run.config_snapshot_path.read_text(encoding="utf-8")
    assert "model" in text
