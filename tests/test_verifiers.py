from pathlib import Path

from mlxsmith.verifiers.regex import verify as regex_verify
from mlxsmith.verifiers.jsonschema import verify as json_verify
from mlxsmith.verifiers.pytest_verifier import verify as pytest_verify


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
