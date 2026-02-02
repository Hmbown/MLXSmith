from pathlib import Path

from mlxsmith.verifiers.regex import verify as regex_verify
from mlxsmith.verifiers.jsonschema import verify as json_verify
from mlxsmith.verifiers.pytest_verifier import verify as pytest_verify
from mlxsmith.verifiers.llm_judge import verify as llm_verify
from mlxsmith.verifiers.prime import verify as prime_verify


def test_regex_verifier():
    res = regex_verify("", "def add(a,b): return a+b", ".", pattern=r"def\s+add")
    assert res.passed


def test_jsonschema_verifier(tmp_path: Path):
    schema = {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}
    res = json_verify("", "{\"x\": 1}", str(tmp_path), schema=schema)
    assert res.passed


def test_pytest_verifier(tmp_path: Path):
    workdir = tmp_path / "sandbox"
    tests = workdir / "tests"
    tests.mkdir(parents=True)
    (tests / "test_basic.py").write_text("def test_ok():\n    assert 2+2==4\n", encoding="utf-8")
    res = pytest_verify("", "", str(workdir))
    assert res.passed


def test_llm_judge_mock(tmp_path: Path):
    mock_response = '{"passed": true, "score": 0.85, "reason": "ok"}'
    res = llm_verify(
        "prompt",
        "completion",
        str(tmp_path),
        model="dummy",
        backend="mock",
        mock_response=mock_response,
    )
    assert res.passed
    assert abs(res.reward - 0.85) < 1e-6


def test_llm_judge_thinkprm_steps(tmp_path: Path):
    mock_response = '{"passed": true, "steps": [{"text": "a", "score": 0.9}, {"text": "b", "score": 0.5}]}'
    res = llm_verify(
        "prompt",
        "completion",
        str(tmp_path),
        model="dummy",
        backend="mock",
        mock_response=mock_response,
        mode="thinkprm",
        process_agg="mean",
        reward_mode="score",
    )
    assert res.passed
    assert abs(res.reward - 0.7) < 1e-6


def test_prime_verifier(tmp_path: Path):
    completion = "1. Step one\n2. Step two"
    res = prime_verify(
        "prompt",
        completion,
        str(tmp_path),
        verifier=str(Path("src/mlxsmith/verifiers/regex.py")),
        verifier_kwargs={"pattern": "Step"},
        agg="mean",
        reward_mode="process",
    )
    assert res.passed
    assert res.reward > 0.0
