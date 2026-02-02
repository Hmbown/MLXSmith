from __future__ import annotations

import os
import subprocess
from pathlib import Path
from .types import VerifyResult

def _sandbox_env(base_env: dict | None = None, *, workdir: Path) -> dict:
    env = dict(os.environ)
    if base_env:
        env.update(base_env)
    env.setdefault("PYTHONHASHSEED", "0")
    env["HOME"] = str(workdir)
    env["TMPDIR"] = str(workdir)
    env["PYTHONPATH"] = str(workdir)
    env.pop("PYTHONSTARTUP", None)
    env.pop("VIRTUAL_ENV", None)
    return env


def verify(
    prompt: str,
    completion: str,
    workdir: str,
    *,
    tests_subdir: str = "tests",
    timeout_s: int = 30,
    reward_pass: float = 1.0,
    reward_fail: float = 0.0,
) -> VerifyResult:
    """Run pytest in a sandbox directory.

    Expected usage: environment builder writes files into `workdir` and this verifier runs tests there.
    For v0, we simply run `pytest -q` and treat exit code 0 as pass.

    IMPORTANT: This runs locally. For stronger isolation, replace with a sandbox runner.
    """
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)

    # If user is doing code tasks, they can write completion into a file convention, e.g., main.py
    # Here we create a default file if none exist:
    if not any(wd.glob("*.py")):
        (wd / "main.py").write_text(completion, encoding="utf-8")

    # Ensure tests folder exists; if not, fail deterministically.
    tests_path = wd / tests_subdir
    if not tests_path.exists():
        return VerifyResult(
            reward=reward_fail,
            passed=False,
            info={"error": f"Missing tests folder: {tests_subdir}"},
            artifacts_dir=str(wd),
        )

    try:
        proc = subprocess.run(
            ["pytest", "-q"],
            cwd=str(wd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=_sandbox_env(workdir=wd),
        )
        passed = proc.returncode == 0
        return VerifyResult(
            reward=reward_pass if passed else reward_fail,
            passed=passed,
            info={
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
            },
            artifacts_dir=str(wd),
        )
    except subprocess.TimeoutExpired:
        return VerifyResult(
            reward=reward_fail,
            passed=False,
            info={"error": "timeout", "timeout_s": timeout_s},
            artifacts_dir=str(wd),
        )
