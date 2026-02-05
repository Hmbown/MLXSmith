from __future__ import annotations

import json
from pathlib import Path

import yaml

from mlxsmith.eval import run_eval
from mlxsmith.llm.backend import Generation


class _FixedCodeBackend:
    name = "fixed"

    def load(self, model_id_or_path: str, *, max_seq_len: int | None = None, dtype: str | None = None, **kwargs) -> None:
        return None

    def apply_adapter(self, adapter_path: str) -> None:
        return None

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 1.0,
        top_k: int | None = None,
        seed: int | None = None,
    ) -> Generation:
        code = "def add(a, b):\n    return a + b\n"
        return Generation(text=prompt + code, token_ids=[1, 2, 3], prompt_len=1)


def test_eval_suite_with_embedded_tests(tmp_path: Path, monkeypatch) -> None:
    # Config: use mock backend name, but monkeypatch the factory to return our fixed backend.
    (tmp_path / "mlxsmith.yaml").write_text(
        yaml.safe_dump({"model": {"backend": "mock", "id": "dummy/model"}}),
        encoding="utf-8",
    )

    suite = {
        "name": "unit-eval",
        "tasks": [
            {
                "id": "add",
                "prompt": "Write add(a,b). Return only Python code.",
                "k": 1,
                "tests": "from main import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
            }
        ],
    }
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(suite), encoding="utf-8")

    from mlxsmith import eval as eval_mod

    monkeypatch.setattr(eval_mod, "get_llm_backend", lambda _name: _FixedCodeBackend())

    out_path = run_eval(tmp_path, suite_path, Path("dummy/model"))
    results = json.loads(out_path.read_text(encoding="utf-8"))
    assert results["suite"] == "unit-eval"
    assert results["summary"][0]["task_id"] == "add"
    assert results["summary"][0]["pass@k"] == 1.0
