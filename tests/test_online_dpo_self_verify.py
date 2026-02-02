from pathlib import Path

from mlxsmith.config import ProjectConfig
from mlxsmith.train.online_dpo import run_online_dpo
from mlxsmith.train.self_verify import run_self_verify


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(__import__("json").dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_online_dpo_mock(tmp_path: Path):
    cfg = ProjectConfig()
    cfg.model.backend = "mock"
    cfg.train.iters = 2
    cfg.train.save_every = 1
    cfg.train.log_every = 1
    cfg.rft.rollouts = 2

    data_path = tmp_path / "data" / "prompts.jsonl"
    _write_jsonl(data_path, [{"prompt": "Say hi"}])

    run = run_online_dpo(
        tmp_path,
        cfg,
        data_path,
        "dummy/model",
        "none",
        judge_model="dummy",
        judge_backend="mock",
        group_size=2,
        judge_mock_response=[
            '{"passed": true, "score": 0.9}',
            '{"passed": true, "score": 0.1}',
        ],
    )
    assert run.adapter_dir.exists()


def test_self_verify_mock(tmp_path: Path):
    cfg = ProjectConfig()
    cfg.model.backend = "mock"
    cfg.train.iters = 2
    cfg.train.save_every = 1
    cfg.train.log_every = 1

    data_path = tmp_path / "data" / "prompts.jsonl"
    _write_jsonl(data_path, [{"prompt": "Check"}])

    run = run_self_verify(
        tmp_path,
        cfg,
        data_path,
        "dummy/model",
        "none",
        verifier_model="dummy",
        verifier_backend="mock",
        judge_mock_response='{"passed": true, "score": 0.8}',
    )
    assert run.adapter_dir.exists()
