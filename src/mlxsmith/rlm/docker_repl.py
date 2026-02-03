"""Docker-isolated REPL for RLM inference.

Provides a sandboxed Python REPL that runs in a Docker container
for safer execution of model-generated code.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .repl import REPLResult


@dataclass
class DockerREPLConfig:
    """Configuration for Docker REPL."""
    image: str = "python:3.11-slim"
    memory_mb: int = 512
    cpus: float = 1.0
    pids: int = 128
    max_iterations: int = 50
    timeout_s: float = 30.0
    network: bool = False
    max_output_chars: int = 4000


class DockerRLMEnvironment:
    """Docker-isolated REPL environment for RLM inference.

    Like RLMEnvironment but executes code in a Docker container
    for full isolation. Sub-calls (llm_query) are still handled
    by the host process.
    """

    def __init__(
        self,
        context: str,
        llm_query_fn: Callable[[str], str],
        config: Optional[DockerREPLConfig] = None,
    ):
        self.config = config or DockerREPLConfig()
        self._llm_query_fn = llm_query_fn
        self._context = context
        self._final_answer: Optional[str] = None
        self._execution_count = 0
        self._trajectory: List[Dict[str, Any]] = []

        # Persistent state across executions (serialized to JSON)
        self._state: Dict[str, Any] = {"context": context}

        # Check Docker availability
        self._docker_available = self._check_docker()

    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5.0,
            )
            return result.returncode == 0
        except Exception:
            return False

    def execute(self, code: str) -> REPLResult:
        """Execute code in Docker container."""
        self._execution_count += 1
        if self._execution_count > self.config.max_iterations:
            return REPLResult(
                stdout="",
                stderr=f"Maximum iterations ({self.config.max_iterations}) exceeded",
                success=False,
                exception="MaxIterationsExceeded",
            )

        if not self._docker_available:
            return REPLResult(
                stdout="",
                stderr="Docker not available. Install Docker to use isolated execution.",
                success=False,
                exception="DockerNotAvailable",
            )

        # Create temp directory for code and state
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Write state for the container
            state_file = tmppath / "state.json"
            state_file.write_text(json.dumps(self._state), encoding="utf-8")

            # Write the code to execute
            code_file = tmppath / "code.py"
            wrapper_code = self._build_wrapper(code)
            code_file.write_text(wrapper_code, encoding="utf-8")

            # Write llm_query requests file (container writes, host reads)
            requests_file = tmppath / "requests.json"
            requests_file.write_text("[]", encoding="utf-8")

            # Write responses file (host writes, container reads)
            responses_file = tmppath / "responses.json"
            responses_file.write_text("[]", encoding="utf-8")

            # Build Docker command
            cmd = self._build_docker_cmd(tmppath)

            try:
                # Run with iterative llm_query handling
                result = self._run_with_callbacks(cmd, tmppath)
                return result

            except subprocess.TimeoutExpired:
                return REPLResult(
                    stdout="",
                    stderr=f"Execution timed out after {self.config.timeout_s}s",
                    success=False,
                    exception="TimeoutExpired",
                )
            except Exception as e:
                return REPLResult(
                    stdout="",
                    stderr=str(e),
                    success=False,
                    exception=type(e).__name__,
                )

    def _build_wrapper(self, user_code: str) -> str:
        """Build wrapper code that handles state and llm_query."""
        return f'''
import json
import sys
import traceback
from pathlib import Path

# Load state
state = json.loads(Path("/workspace/state.json").read_text())
context = state.get("context", "")
for k, v in list(state.items()):
    if k != "context":
        globals()[k] = v

# Tracking for llm_query
_llm_requests = []
_llm_responses = json.loads(Path("/workspace/responses.json").read_text())
_llm_idx = 0
_final_result = None

_STATE_EXCLUDE = {{
    "json", "sys", "traceback", "Path", "state", "context",
    "_llm_requests", "_llm_responses", "_llm_idx", "_final_result",
    "_STATE_EXCLUDE", "_dump_state",
    "llm_query", "llm_batch", "FINAL", "FINAL_VAR",
}}

def _dump_state():
    state_out = {{"context": context}}
    for k, v in list(globals().items()):
        if k.startswith("_") or k in _STATE_EXCLUDE:
            continue
        try:
            json.dumps(v)
            state_out[k] = v
        except Exception:
            continue
    payload = json.dumps(state_out)
    Path("/workspace/state_out.json").write_text(payload)
    Path("/workspace/state.json").write_text(payload)

def llm_query(prompt):
    global _llm_idx, _llm_requests
    # Check if we have a cached response
    if _llm_idx < len(_llm_responses):
        resp = _llm_responses[_llm_idx]
        _llm_idx += 1
        return resp
    # Otherwise, request it (will be handled by host)
    _llm_requests.append(prompt)
    Path("/workspace/requests.json").write_text(json.dumps(_llm_requests))
    _dump_state()
    sys.exit(42)  # Special exit code

def llm_batch(prompts):
    return [llm_query(p) for p in prompts]

def FINAL(answer):
    global _final_result
    _final_result = str(answer)
    raise SystemExit(0)

def FINAL_VAR(varname):
    global _final_result
    if varname not in globals():
        raise NameError(f"Variable '{{varname}}' not found")
    _final_result = str(globals()[varname])
    raise SystemExit(0)

# Execute user code
try:
{self._indent_code(user_code)}
except SystemExit:
    pass
except Exception as e:
    print(traceback.format_exc(), file=sys.stderr)
    sys.exit(1)

# Save updated state
_dump_state()

if _final_result is not None:
    Path("/workspace/final.txt").write_text(_final_result)
'''

    def _indent_code(self, code: str) -> str:
        """Indent code for inclusion in wrapper."""
        lines = code.split("\n")
        return "\n".join("    " + line for line in lines)

    def _build_docker_cmd(self, workdir: Path) -> List[str]:
        """Build Docker run command."""
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{workdir}:/workspace:rw",
            "--workdir", "/workspace",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "--memory", f"{self.config.memory_mb}m",
            "--cpus", str(self.config.cpus),
            "--pids-limit", str(self.config.pids),
            "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev",
        ]

        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])

        if not self.config.network:
            cmd.extend(["--network", "none"])

        cmd.extend([
            self.config.image,
            "python", "/workspace/code.py",
        ])

        return cmd

    def _run_with_callbacks(self, cmd: List[str], workdir: Path) -> REPLResult:
        """Run Docker with iterative llm_query handling."""
        max_iterations = 50
        all_stdout = []
        all_stderr = []

        for _ in range(max_iterations):
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s,
            )

            all_stdout.append(result.stdout)
            all_stderr.append(result.stderr)

            # Check for llm_query request (exit code 42)
            if result.returncode == 42:
                # Read pending requests
                requests_file = workdir / "requests.json"
                requests = json.loads(requests_file.read_text())

                # Process the last request
                if not requests:
                    break

                prompt = requests[-1]
                response = self._llm_query_fn(prompt)

                self._trajectory.append({
                    "type": "llm_query",
                    "prompt": prompt[:500],
                    "result": response[:500],
                })

                # Write response
                responses_file = workdir / "responses.json"
                existing = json.loads(responses_file.read_text())
                existing.append(response)
                responses_file.write_text(json.dumps(existing))

                # Continue execution
                continue

            # Check for final answer
            final_file = workdir / "final.txt"
            if final_file.exists():
                self._final_answer = final_file.read_text()

                # Update state
                state_out_file = workdir / "state_out.json"
                if state_out_file.exists():
                    self._state = json.loads(state_out_file.read_text())

                stdout = "\n".join(all_stdout)
                stderr = "\n".join(all_stderr)
                if len(stdout) > self.config.max_output_chars:
                    stdout = stdout[: self.config.max_output_chars] + "\n...(truncated)"
                if len(stderr) > self.config.max_output_chars:
                    stderr = stderr[: self.config.max_output_chars] + "\n...(truncated)"
                return REPLResult(
                    stdout=stdout,
                    stderr=stderr,
                    success=True,
                    final_answer=self._final_answer,
                )

            # Normal completion
            if result.returncode == 0:
                # Update state
                state_out_file = workdir / "state_out.json"
                if state_out_file.exists():
                    self._state = json.loads(state_out_file.read_text())

                stdout = "\n".join(all_stdout)
                stderr = "\n".join(all_stderr)

                # Truncate if needed
                if len(stdout) > self.config.max_output_chars:
                    stdout = stdout[:self.config.max_output_chars] + "\n...(truncated)"
                if len(stderr) > self.config.max_output_chars:
                    stderr = stderr[:self.config.max_output_chars] + "\n...(truncated)"

                return REPLResult(stdout=stdout, stderr=stderr, success=True)

            # Error
            return REPLResult(
                stdout="\n".join(all_stdout),
                stderr="\n".join(all_stderr),
                success=False,
                exception=f"Exit code {result.returncode}",
            )

        # Max iterations
        return REPLResult(
            stdout="\n".join(all_stdout)[: self.config.max_output_chars],
            stderr="Max llm_query iterations reached",
            success=False,
            exception="MaxIterations",
        )

    @property
    def is_complete(self) -> bool:
        return self._final_answer is not None

    @property
    def answer(self) -> Optional[str]:
        return self._final_answer

    @property
    def trajectory(self) -> List[Dict[str, Any]]:
        return self._trajectory

    def get_variable(self, name: str) -> Any:
        return self._state.get(name)

    def set_variable(self, name: str, value: Any) -> None:
        self._state[name] = value
