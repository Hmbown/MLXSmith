"""REPL Environment for Recursive Language Model inference.

This module provides the Python REPL environment that the model interacts with
during RLM inference. The model can execute code, observe output, make recursive
sub-calls via llm_query(), and signal completion via FINAL().

Based on the RLM paradigm from Zhang et al. (arXiv:2512.24601).
"""

from __future__ import annotations

import builtins
import io
import re
import signal
import threading
import traceback
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class REPLResult:
    """Result from executing code in the REPL."""
    stdout: str
    stderr: str
    success: bool
    exception: Optional[str] = None
    final_answer: Optional[str] = None
    final_var: Optional[str] = None


@dataclass
class REPLConfig:
    """Configuration for REPL environment."""
    max_output_chars: int = 4000
    max_iterations: int = 50
    timeout_per_exec_s: float = 30.0
    allowed_imports: Optional[List[str]] = None  # None = allow all (not recommended)
    blocked_imports: List[str] = field(
        default_factory=lambda: [
            # system/process
            "os",
            "sys",
            "subprocess",
            "signal",
            "ctypes",
            # filesystem helpers (can bypass blocked open in local mode)
            "builtins",
            "importlib",
            # networking
            "socket",
            "ssl",
            "http",
            "urllib",
            "requests",
            "ftplib",
            "smtplib",
            "imaplib",
            "poplib",
            "telnetlib",
        ]
    )


class RLMEnvironment:
    """Python REPL environment for RLM inference.

    Provides:
    - `context` variable: The input text stored as a string
    - `llm_query(prompt)`: Recursive sub-calls to language models
    - `llm_batch(prompts)`: Parallel sub-calls for efficiency
    - `FINAL(answer)`: Signal completion with direct answer
    - `FINAL_VAR(varname)`: Signal completion, answer is in variable
    - Standard Python execution with output capture
    """

    def __init__(
        self,
        context: str,
        llm_query_fn: Callable[[str], str],
        config: Optional[REPLConfig] = None,
    ):
        self.config = config or REPLConfig()
        self._namespace: Dict[str, Any] = {}
        self._llm_query_fn = llm_query_fn
        self._final_answer: Optional[str] = None
        self._final_var: Optional[str] = None
        self._execution_count = 0
        self._trajectory: List[Dict[str, Any]] = []

        # Best-effort safety: override builtins import/open in the local sandbox.
        # For strong isolation, prefer the Docker sandbox.
        self._namespace["__builtins__"] = self._build_builtins()

        # Initialize namespace with context and tools
        self._namespace["context"] = context
        self._namespace["llm_query"] = self._llm_query
        self._namespace["llm_batch"] = self._llm_batch
        self._namespace["FINAL"] = self._final
        self._namespace["FINAL_VAR"] = self._final_var_fn

    def _build_builtins(self) -> Dict[str, Any]:
        """Return a builtins dict with guarded import and blocked open()."""
        builtins_dict: Dict[str, Any] = dict(builtins.__dict__)
        original_import = builtins_dict.get("__import__")
        if callable(original_import):
            builtins_dict["__import__"] = self._guarded_import(original_import)
        builtins_dict["open"] = self._blocked_open
        builtins_dict["input"] = self._blocked_input
        return builtins_dict

    @staticmethod
    def _module_prefix_matches(module: str, prefix: str) -> bool:
        return module == prefix or module.startswith(prefix + ".")

    def _guarded_import(self, original_import):
        def _import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001
            if level and level > 0:
                raise ImportError("Relative imports are not allowed in the REPL sandbox")

            module = str(name)
            root = module.split(".", 1)[0]

            allowed = self.config.allowed_imports
            blocked = self.config.blocked_imports or []

            if allowed is not None and not any(
                self._module_prefix_matches(module, a) or self._module_prefix_matches(root, a) for a in allowed
            ):
                raise ImportError(f"Import not allowed: {module}")

            if any(
                self._module_prefix_matches(module, b) or self._module_prefix_matches(root, b) for b in blocked
            ):
                raise ImportError(f"Blocked import: {module}")

            return original_import(name, globals, locals, fromlist, level)

        return _import

    @staticmethod
    def _blocked_open(*_args, **_kwargs):  # noqa: ANN001
        raise PermissionError("open() is disabled in the local REPL sandbox; use --sandbox docker for isolation.")

    @staticmethod
    def _blocked_input(*_args, **_kwargs):  # noqa: ANN001
        raise RuntimeError("input() is disabled in the REPL sandbox.")

    def _llm_query(self, prompt: str) -> str:
        """Make a recursive sub-call to the language model."""
        if not isinstance(prompt, str):
            prompt = str(prompt)
        result = self._llm_query_fn(prompt)
        self._trajectory.append({
            "type": "llm_query",
            "prompt": prompt[:500],  # Truncate for logging
            "result": result[:500] if result else "",
        })
        return result

    def _llm_batch(self, prompts: List[str]) -> List[str]:
        """Make parallel sub-calls to the language model.

        Note: Currently sequential, but interface supports future parallelization.
        """
        results = []
        for prompt in prompts:
            results.append(self._llm_query(prompt))
        return results

    def _final(self, answer: Any) -> None:
        """Signal completion with a direct answer."""
        self._final_answer = str(answer) if answer is not None else ""
        raise _FinalSignal(self._final_answer)

    def _final_var_fn(self, varname: str) -> None:
        """Signal completion, answer is stored in the named variable."""
        if varname not in self._namespace:
            raise NameError(f"Variable '{varname}' not found in namespace")
        self._final_var = varname
        self._final_answer = str(self._namespace[varname])
        raise _FinalSignal(self._final_answer)

    @contextmanager
    def _execution_timeout(self, seconds: float):
        """Best-effort per-exec timeout for the local sandbox.

        Uses signals (Unix) and only works in the main thread.
        """
        if seconds <= 0:
            yield
            return
        if threading.current_thread() is not threading.main_thread():
            yield
            return
        if not hasattr(signal, "SIGALRM") or not hasattr(signal, "ITIMER_REAL"):
            yield
            return

        def _raise_timeout(_signum, _frame):  # noqa: ANN001
            raise _ExecTimeout(f"Execution timed out after {seconds:.1f}s")

        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])
            signal.signal(signal.SIGALRM, previous_handler)

    def execute(self, code: str) -> REPLResult:
        """Execute code in the REPL environment."""
        self._execution_count += 1

        if self._execution_count > self.config.max_iterations:
            return REPLResult(
                stdout="",
                stderr=f"Maximum iterations ({self.config.max_iterations}) exceeded",
                success=False,
                exception="MaxIterationsExceeded",
            )

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                with self._execution_timeout(self.config.timeout_per_exec_s):
                    exec(code, self._namespace)

            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()

            # Truncate output if too long
            if len(stdout) > self.config.max_output_chars:
                stdout = stdout[:self.config.max_output_chars] + "\n... (truncated)"
            if len(stderr) > self.config.max_output_chars:
                stderr = stderr[:self.config.max_output_chars] + "\n... (truncated)"

            self._trajectory.append({
                "type": "execute",
                "code": code[:1000],
                "stdout": stdout[:500],
                "stderr": stderr[:500] if stderr else None,
            })

            return REPLResult(
                stdout=stdout,
                stderr=stderr,
                success=True,
            )

        except _ExecTimeout as e:
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()

            self._trajectory.append(
                {
                    "type": "timeout",
                    "code": code[:500],
                    "exception": str(e)[:200],
                }
            )

            return REPLResult(
                stdout=stdout,
                stderr=(stderr + "\n" if stderr else "") + str(e),
                success=False,
                exception="Timeout",
            )

        except _FinalSignal:
            # FINAL() or FINAL_VAR() was called
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()

            self._trajectory.append({
                "type": "final",
                "answer": self._final_answer[:500] if self._final_answer else None,
                "var": self._final_var,
            })

            return REPLResult(
                stdout=stdout,
                stderr=stderr,
                success=True,
                final_answer=self._final_answer,
                final_var=self._final_var,
            )

        except Exception as e:
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            exc_text = traceback.format_exc()

            # Truncate exception if too long
            if len(exc_text) > self.config.max_output_chars:
                exc_text = exc_text[:self.config.max_output_chars] + "\n... (truncated)"

            self._trajectory.append({
                "type": "error",
                "code": code[:500],
                "exception": str(e)[:200],
            })

            return REPLResult(
                stdout=stdout,
                stderr=stderr + "\n" + exc_text if stderr else exc_text,
                success=False,
                exception=str(e),
            )

    @property
    def is_complete(self) -> bool:
        """Check if FINAL() or FINAL_VAR() has been called."""
        return self._final_answer is not None

    @property
    def answer(self) -> Optional[str]:
        """Get the final answer if available."""
        return self._final_answer

    @property
    def trajectory(self) -> List[Dict[str, Any]]:
        """Get the execution trajectory for training."""
        return self._trajectory

    @property
    def namespace(self) -> Dict[str, Any]:
        """Get the current namespace (for inspection)."""
        return self._namespace

    def get_variable(self, name: str) -> Any:
        """Get a variable from the namespace."""
        return self._namespace.get(name)

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the namespace."""
        self._namespace[name] = value


class _FinalSignal(Exception):
    """Internal signal for FINAL() completion."""
    def __init__(self, answer: str):
        self.answer = answer
        super().__init__(answer)


class _ExecTimeout(Exception):
    """Internal signal for per-exec timeout."""


def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """Extract code blocks from model output.

    Returns list of (language, code) tuples.
    Supports ```repl, ```python, and plain ``` blocks.
    """
    blocks = []

    # Pattern for fenced code blocks with optional language
    pattern = r"```(\w*)\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    for lang, code in matches:
        lang = lang.lower() if lang else "python"
        # Treat 'repl', 'python', 'py', or empty as executable
        if lang in ("repl", "python", "py", ""):
            blocks.append((lang or "python", code.strip()))

    return blocks


def format_repl_output(result: REPLResult) -> str:
    """Format REPL result for inclusion in model context."""
    parts = []

    if result.stdout:
        parts.append(f"Output:\n{result.stdout}")

    if result.stderr and result.success:
        parts.append(f"Stderr:\n{result.stderr}")

    if not result.success:
        if result.exception:
            parts.append(f"Error: {result.exception}")
        if result.stderr:
            parts.append(f"Details:\n{result.stderr}")

    if result.final_answer is not None:
        parts.append(f"FINAL: {result.final_answer[:200]}...")

    return "\n".join(parts) if parts else "(no output)"
