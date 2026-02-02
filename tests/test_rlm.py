from pathlib import Path

from mlxsmith.config import ProjectConfig
import socket

from mlxsmith.rlm.loop import run_rlm, run_rlm_orchestrated


def test_rlm_loop_smoke(tmp_path: Path):
    cfg = ProjectConfig()
    cfg.model.backend = "mock"
    cfg.model.use_chat_template = False
    cfg.train.iters = 1
    cfg.rlm.iterations = 1
    cfg.rlm.tasks_per_iter = 1
    cfg.rlm.rollouts_per_task = 1
    cfg.rlm.mix_old_ratio = 0.0
    cfg.rlm.benchmark_suite = ""
    cfg.rlm.holdout_suite = None

    run_rlm(tmp_path, cfg, model_spec="dummy/model", iterations=1, resume=False)

    state_path = tmp_path / "runs" / "rlm_state.json"
    history_path = tmp_path / "runs" / "rlm_history.jsonl"
    assert state_path.exists()
    assert history_path.exists()


def test_rlm_orchestrated_smoke(tmp_path: Path):
    cfg = ProjectConfig()
    cfg.model.backend = "mock"
    cfg.model.use_chat_template = False
    cfg.train.iters = 1
    cfg.rlm.iterations = 1
    cfg.rlm.tasks_per_iter = 1
    cfg.rlm.rollouts_per_task = 1
    cfg.rlm.mix_old_ratio = 0.0
    cfg.rlm.benchmark_suite = ""
    cfg.rlm.holdout_suite = None
    cfg.rlm.verifier_timeout_s = 5
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    cfg.serve.port = sock.getsockname()[1]
    sock.close()

    run_rlm_orchestrated(tmp_path, cfg, model_spec="dummy/model", iterations=1, resume=False)

    state_path = tmp_path / "runs" / "rlm_state.json"
    history_path = tmp_path / "runs" / "rlm_history.jsonl"
    assert state_path.exists()
    assert history_path.exists()
