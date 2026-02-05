from mlxsmith.api import handlers


def test_strip_think_strips_special_markers_without_think() -> None:
    assert handlers._strip_think("hello<|im_end|>") == "hello"


def test_strip_think_strips_think_blocks_and_markers() -> None:
    text = "A<think>secret reasoning</think> B <|im_end|>"
    out = handlers._strip_think(text)
    assert "<think" not in out.lower()
    assert "<|" not in out
    assert "secret" not in out.lower()


def test_strip_think_strips_unclosed_think() -> None:
    text = "<think>secret\n\nAnswer"
    assert handlers._strip_think(text) == "Answer"
