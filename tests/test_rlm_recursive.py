from __future__ import annotations

from mlxsmith.llm.mock_backend import MockBackend
from mlxsmith.rlm.recursive import recursive_compact


def test_recursive_compact_noop_when_short():
    llm = MockBackend()
    prompt = "short prompt"
    compacted, stats = recursive_compact(llm, prompt, max_seq_len=128)
    assert compacted == prompt
    assert stats.depth == 0
    assert stats.chunks == 0
    assert stats.truncated is False


def test_recursive_compact_truncates_when_max_depth_zero():
    llm = MockBackend()
    prompt = "A" * 200
    compacted, stats = recursive_compact(llm, prompt, max_seq_len=32, max_depth=0)
    assert stats.truncated is True
    assert compacted != prompt
    assert len(llm.encode(compacted)) <= 32


def test_recursive_compact_compacts_long_prompt():
    llm = MockBackend()
    prompt = "B" * 200
    compacted, stats = recursive_compact(
        llm,
        prompt,
        max_seq_len=64,
        max_depth=2,
        chunk_tokens=32,
        overlap_tokens=0,
        keep_last_tokens=16,
        summary_tokens=8,
    )
    assert compacted != prompt
    assert stats.depth >= 1
    assert stats.chunks >= 1
    assert len(llm.encode(compacted)) <= 64
