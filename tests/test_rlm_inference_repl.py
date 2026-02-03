from __future__ import annotations

from dataclasses import dataclass

import pytest
import signal

from mlxsmith.llm.backend import Generation
from mlxsmith.rlm.repl import RLMEnvironment, REPLConfig, extract_code_blocks
from mlxsmith.rlm.rlm_inference import run_rlm_inference, RLMInferenceConfig


@dataclass
class _ScriptedBackend:
    outputs: list[str]

    def generate(self, prompt: str, **_kwargs) -> Generation:  # noqa: ANN001
        if not self.outputs:
            raise RuntimeError("No scripted outputs left")
        out = self.outputs.pop(0)
        # Simulate prompt echo so run_rlm_inference strips it.
        return Generation(text=prompt + out, token_ids=[0], prompt_len=len(prompt))


def test_extract_code_blocks_accepts_repl_and_python():
    text = "hi\n```repl\nprint(1)\n```\n```python\nprint(2)\n```"
    blocks = extract_code_blocks(text)
    assert [lang for lang, _code in blocks] == ["repl", "python"]


def test_repl_blocks_blocked_imports():
    env = RLMEnvironment("ctx", llm_query_fn=lambda _p: "")
    res = env.execute("import os\nprint('nope')")
    assert res.success is False
    assert res.exception is not None
    assert "Blocked import" in (res.stderr or res.exception)


def test_repl_enforces_timeout_best_effort():
    if not hasattr(signal, "SIGALRM"):
        pytest.skip("SIGALRM not available on this platform")
    env = RLMEnvironment("ctx", llm_query_fn=lambda _p: "", config=REPLConfig(timeout_per_exec_s=0.05))
    res = env.execute("import time\ntime.sleep(1)")
    assert res.success is False
    assert res.exception == "Timeout"


def test_rlm_inference_smoke_calls_final():
    llm = _ScriptedBackend(
        outputs=[
            "```repl\nFINAL('ok')\n```",
        ]
    )
    traj = run_rlm_inference(
        llm,
        "some long context",
        config=RLMInferenceConfig(max_turns=3, sandbox="local"),
    )
    assert traj.success is True
    assert traj.final_answer == "ok"
    assert len(traj.turns) == 1


@pytest.mark.parametrize("sandbox", ["wat", "dockerx"])
def test_rlm_inference_rejects_unknown_sandbox(sandbox: str):
    llm = _ScriptedBackend(outputs=["```repl\nFINAL('ok')\n```"])
    with pytest.raises(ValueError, match="Unknown sandbox"):
        run_rlm_inference(llm, "ctx", config=RLMInferenceConfig(sandbox=sandbox))
