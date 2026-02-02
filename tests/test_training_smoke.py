from pathlib import Path

from mlxsmith.config import ProjectConfig
from mlxsmith.train.sft import run_sft
from mlxsmith.train.pref import run_pref
from mlxsmith.train.rft import run_rft


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(__import__("json").dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_training_smoke(tmp_path: Path):
    cfg = ProjectConfig()
    cfg.model.backend = "mock"
    cfg.train.iters = 2
    cfg.train.save_every = 1
    cfg.train.log_every = 1
    cfg.rft.rollouts = 2
    cfg.rft.max_new_tokens = 4

    # SFT data
    _write_jsonl(tmp_path / "data" / "sft" / "train.jsonl", [
        {"prompt": "Hello", "completion": " world"},
    ])

    run = run_sft(tmp_path, cfg, tmp_path / "data" / "sft", "dummy/model", "none")
    assert run.adapter_dir.exists()

    # Pref data
    _write_jsonl(tmp_path / "data" / "prefs" / "train.jsonl", [
        {"prompt": "Say hi", "chosen": " hi", "rejected": " bye"},
    ])

    run_pref(tmp_path, cfg, tmp_path / "data" / "prefs", run.adapter_dir, "none")

    # RFT env + verifier
    env = tmp_path / "envs" / "coding.yaml"
    env.parent.mkdir(parents=True, exist_ok=True)
    env.write_text(
        """
name: test

tasks:
  - id: t1
    prompt: "Say hello"
    verifier_kwargs:
      pattern: "hello"
""",
        encoding="utf-8",
    )

    verifier = tmp_path / "verifiers" / "regex.py"
    verifier.parent.mkdir(parents=True, exist_ok=True)
    verifier.write_text(
        """
from mlxsmith.verifiers.regex import verify as _verify

def verify(prompt, completion, workdir, **kwargs):
    return _verify(prompt, completion, workdir, **kwargs)
""",
        encoding="utf-8",
    )

    run_rft(tmp_path, cfg, env, verifier, run.adapter_dir, "none")

    # RFT variants
    cfg.rft.loss_type = "dr_grpo"
    run_rft(tmp_path, cfg, env, verifier, run.adapter_dir, "none")
    cfg.rft.loss_type = "dapo"
    run_rft(tmp_path, cfg, env, verifier, run.adapter_dir, "none")

    # Token-level env (custom) smoke
    token_env_py = tmp_path / "envs" / "token_env.py"
    token_env_py.write_text(
        """
from mlxsmith.envs.token_env import TokenEnvStep


class Env:
    def __init__(self, max_steps=2, **kwargs):
        self.max_steps = max_steps
        self.steps = 0
        self.obs = [1, 2]

    def initial_observation(self):
        self.steps = 0
        self.obs = [1, 2]
        return list(self.obs)

    def step(self, action: int):
        self.steps += 1
        self.obs.append(int(action))
        done = self.steps >= self.max_steps
        reward = 1.0 if done else 0.0
        return TokenEnvStep(observation=list(self.obs), reward=reward, done=done, info={\"passed\": done})
""",
        encoding="utf-8",
    )
    token_env_yaml = tmp_path / "envs" / "token_env.yaml"
    token_env_yaml.write_text(
        """
name: token-env
token_env:
  path: envs/token_env.py
  class: Env
""",
        encoding="utf-8",
    )

    run_rft(tmp_path, cfg, token_env_yaml, verifier, run.adapter_dir, "none")
