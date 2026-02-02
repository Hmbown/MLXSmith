import sys

from mlxsmith.integrations.mlx_lm_lora import (
    build_train_command,
    build_synthetic_command,
    build_judge_command,
    build_reward_functions_command,
    run_command,
)


def test_build_train_command():
    cmd = build_train_command(
        config="config.yaml",
        model="some/model",
        data="data/prefs",
        train_mode="dpo",
        train_type="lora",
        extra_args=["--beta", "0.1"],
    )
    assert cmd[0] == sys.executable
    assert "mlx_lm_lora.train" in cmd
    assert "--config" in cmd and "config.yaml" in cmd
    assert "--model" in cmd and "some/model" in cmd
    assert "--data" in cmd and "data/prefs" in cmd
    assert "--train-mode" in cmd and "dpo" in cmd
    assert "--train-type" in cmd and "lora" in cmd
    assert "--beta" in cmd and "0.1" in cmd


def test_build_synthetic_command():
    cmd = build_synthetic_command("prompts", extra_args=["--num-samples", "10"])
    assert cmd[0] == sys.executable
    assert "mlx_lm_lora.synthetic_prompts" in cmd
    assert "--num-samples" in cmd


def test_build_judge_and_reward_commands():
    cmd = build_judge_command(extra_args=["--iters", "5"])
    assert "mlx_lm_lora.train_judge" in cmd
    cmd2 = build_reward_functions_command()
    assert "--list-reward-functions" in cmd2


def test_run_command_dry_run():
    cmd = [sys.executable, "-m", "mlx_lm_lora.train", "--help"]
    assert run_command(cmd, dry_run=True) == 0
