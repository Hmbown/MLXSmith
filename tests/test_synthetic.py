import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlxsmith.synthetic import generate_prompts, generate_sft, generate_dpo, generate_evolved_prompts


@dataclass
class _FakeGeneration:
    text: str
    token_ids: list = None
    prompt_len: int = 0

    def __post_init__(self):
        if self.token_ids is None:
            self.token_ids = []


@dataclass
class _FakeVerifyResult:
    reward: float
    passed: bool
    info: dict = None

    def __post_init__(self):
        if self.info is None:
            self.info = {}


def _make_mock_llm(responses):
    """Create a mock LLM backend that yields canned responses."""
    llm = MagicMock()
    idx = {"i": 0}

    def _gen(prompt, **kwargs):
        text = responses[idx["i"] % len(responses)]
        idx["i"] += 1
        return _FakeGeneration(text=text)

    llm.generate.side_effect = _gen
    return llm


def _make_cfg():
    """Minimal mock ProjectConfig."""
    cfg = MagicMock()
    cfg.model.backend = "mock"
    cfg.model.max_seq_len = 2048
    cfg.model.dtype = None
    cfg.model.trust_remote_code = False
    return cfg


@patch("mlxsmith.synthetic._load_llm")
def test_generate_prompts(mock_load):
    system_prompt = "SYS"
    prefix = system_prompt + "\n\nGenerate a new prompt:"
    responses = [
        prefix + " Write a function to sort a list",
        "Explain the difference between a stack and a queue",
        "Implement binary search in Python",
    ]
    mock_load.return_value = (_make_mock_llm(responses), "mock-model")

    cfg = _make_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "prompts.jsonl"
        n = generate_prompts("mock-model", cfg, out, num=3, system_prompt=system_prompt)

        assert n == 3
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            row = json.loads(line)
            assert "prompt" in row
            assert isinstance(row["prompt"], str)
            assert len(row["prompt"]) > 0
        first = json.loads(lines[0])["prompt"]
        assert "SYS" not in first


@patch("mlxsmith.synthetic._load_llm")
def test_generate_evolved_prompts(mock_load):
    responses = [
        "Create a more complex sorting task",
        "Add constraints and require proofs",
    ]
    mock_load.return_value = (_make_mock_llm(responses), "mock-model")

    cfg = _make_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        seeds = Path(tmp) / "seed.jsonl"
        seeds.write_text(
            '{"prompt": "Write a sort function"}\n'
            '{"prompt": "Explain stack vs queue"}\n',
            encoding="utf-8",
        )
        out = Path(tmp) / "evolved.jsonl"
        n = generate_evolved_prompts("mock-model", cfg, seeds, out, num=2, mode="mix")
        assert n == 2
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        row = json.loads(lines[0])
        assert "prompt" in row
        assert "mode" in row
        assert "seed_prompt" in row


@patch("mlxsmith.synthetic._load_llm")
def test_generate_sft(mock_load):
    responses = [
        "Write a sort functiondef sort_list(lst): return sorted(lst)",
        "Explain stack vs queueA stack uses LIFO; a queue uses FIFO.",
    ]
    mock_load.return_value = (_make_mock_llm(responses), "mock-model")

    cfg = _make_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        prompts_path = Path(tmp) / "prompts.jsonl"
        prompts_path.write_text(
            '{"instruction": "Write a sort function"}\n'
            '{"input": "Explain stack vs queue"}\n',
            encoding="utf-8",
        )
        out = Path(tmp) / "sft.jsonl"
        n = generate_sft("mock-model", cfg, prompts_path, out)

        assert n == 2
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            row = json.loads(line)
            assert "prompt" in row
            assert "response" in row
            assert len(row["response"]) > 0


@patch("mlxsmith.synthetic.judge_verify")
@patch("mlxsmith.synthetic._load_llm")
def test_generate_sft_rejection_sampling(mock_load, mock_judge):
    responses = [
        "Answer one",
        "Answer two",
    ]
    mock_load.return_value = (_make_mock_llm(responses), "mock-model")

    scores = [0.2, 0.9]
    idx = {"i": 0}

    def _judge(*args, **kwargs):
        s = scores[idx["i"] % len(scores)]
        idx["i"] += 1
        return _FakeVerifyResult(reward=s, passed=s >= 0.5)

    mock_judge.side_effect = _judge

    cfg = _make_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        prompts_path = Path(tmp) / "prompts.jsonl"
        prompts_path.write_text('{"prompt": "Write a function"}\n', encoding="utf-8")
        out = Path(tmp) / "sft.jsonl"
        n = generate_sft(
            "mock-model",
            cfg,
            prompts_path,
            out,
            candidates_per_prompt=2,
            judge_model="mock-judge",
            min_score=0.5,
        )
        assert n == 1
        row = json.loads(out.read_text(encoding="utf-8").strip())
        assert row["response"] == "Answer two"


@patch("mlxsmith.synthetic._load_llm")
def test_generate_sft_empty_prompts_raises(mock_load):
    mock_load.return_value = (_make_mock_llm(["irrelevant"]), "mock-model")
    cfg = _make_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        prompts_path = Path(tmp) / "prompts.jsonl"
        prompts_path.write_text(" \n", encoding="utf-8")
        out = Path(tmp) / "sft.jsonl"
        with pytest.raises(RuntimeError):
            generate_sft("mock-model", cfg, prompts_path, out)


@patch("mlxsmith.synthetic.judge_verify")
@patch("mlxsmith.synthetic._load_llm")
def test_generate_dpo(mock_load, mock_judge):
    # 4 candidates per prompt, 2 prompts = 8 calls total
    responses = [
        "good answer 1",
        "mediocre answer",
        "bad answer",
        "good answer 2",
        "another good one",
        "another mediocre",
        "another bad",
        "another ok",
    ]
    mock_load.return_value = (_make_mock_llm(responses), "mock-model")

    # Alternate high/low rewards so chosen != rejected
    rewards = [0.9, 0.5, 0.1, 0.8, 0.9, 0.4, 0.2, 0.7]
    reward_idx = {"i": 0}

    def _judge(*args, **kwargs):
        r = rewards[reward_idx["i"] % len(rewards)]
        reward_idx["i"] += 1
        return _FakeVerifyResult(reward=r, passed=r >= 0.5)

    mock_judge.side_effect = _judge

    cfg = _make_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        prompts_path = Path(tmp) / "prompts.jsonl"
        prompts_path.write_text(
            '{"prompt": "Write a sort function"}\n'
            '{"prompt": "Explain recursion"}\n',
            encoding="utf-8",
        )
        out = Path(tmp) / "dpo.jsonl"
        n = generate_dpo(
            "mock-model",
            cfg,
            prompts_path,
            out,
            candidates_per_prompt=4,
            judge_model="mock-judge",
        )

        assert n == 2
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            row = json.loads(line)
            assert "prompt" in row
            assert "chosen" in row
            assert "rejected" in row
            assert row["chosen"] != row["rejected"]


@patch("mlxsmith.synthetic.judge_verify")
@patch("mlxsmith.synthetic._load_llm")
def test_generate_dpo_tie_skips(mock_load, mock_judge):
    responses = ["a", "b"]
    mock_load.return_value = (_make_mock_llm(responses), "mock-model")

    def _judge(*args, **kwargs):
        return _FakeVerifyResult(reward=0.5, passed=True)

    mock_judge.side_effect = _judge
    cfg = _make_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        prompts_path = Path(tmp) / "prompts.jsonl"
        prompts_path.write_text('{"prompt": "Say hi"}\n', encoding="utf-8")
        out = Path(tmp) / "dpo.jsonl"
        n = generate_dpo(
            "mock-model",
            cfg,
            prompts_path,
            out,
            candidates_per_prompt=2,
            judge_model="mock-judge",
        )
        assert n == 0
