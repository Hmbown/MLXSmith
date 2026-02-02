from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .types import VerifyResult
from .pytest_verifier import verify as local_verify


def verify(
    prompt: str,
    completion: str,
    workdir: str,
    *,
    tests_subdir: str = "tests",
    timeout_s: int = 30,
    reward_pass: float = 1.0,
    reward_fail: float = 0.0,
    image: str = "python:3.11-slim",
    memory_mb: int = 512,
    cpus: float = 1.0,
    pids: int = 128,
    use_local_fallback: bool = True,
) -> VerifyResult:
    """Run pytest inside a locked-down Docker container.

    Falls back to local pytest verifier if Docker is unavailable and use_local_fallback is True.
    """
    if shutil.which("docker") is None:
        if use_local_fallback:
            return local_verify(
                prompt,
                completion,
                workdir,
                tests_subdir=tests_subdir,
                timeout_s=timeout_s,
                reward_pass=reward_pass,
                reward_fail=reward_fail,
            )
        return VerifyResult(
            reward=reward_fail,
            passed=False,
            info={"error": "docker_not_found"},
            artifacts_dir=workdir,
        )

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    if not any(wd.glob("*.py")):
        (wd / "main.py").write_text(completion, encoding="utf-8")

    tests_path = wd / tests_subdir
    if not tests_path.exists():
        return VerifyResult(
            reward=reward_fail,
            passed=False,
            info={"error": f"Missing tests folder: {tests_subdir}"},
            artifacts_dir=str(wd),
        )

    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--pids-limit",
        str(int(pids)),
        "--memory",
        f"{int(memory_mb)}m",
        "--cpus",
        str(float(cpus)),
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev",
        "-v",
        f"{wd}:/workspace:rw",
        "-w",
        "/workspace",
        image,
        "pytest",
        "-q",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        passed = proc.returncode == 0
        return VerifyResult(
            reward=reward_pass if passed else reward_fail,
            passed=passed,
            info={
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
                "docker_image": image,
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
