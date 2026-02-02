from mlxsmith.llm.mock_backend import MockBackend
from mlxsmith.sdk.losses import get_loss


def _make_ids():
    backend = MockBackend()
    prompt = "Hello"
    chosen = " world"
    rejected = " there"
    prompt_ids = backend.encode(prompt)
    chosen_ids = backend.encode(prompt + chosen)
    rejected_ids = backend.encode(prompt + rejected)
    return backend, prompt_ids, chosen_ids, rejected_ids


def test_simpo_loss_runs():
    backend, prompt_ids, chosen_ids, rejected_ids = _make_ids()
    loss_fn = get_loss("simpo")
    loss = loss_fn(
        backend,
        chosen_ids,
        rejected_ids,
        prompt_len_chosen=len(prompt_ids),
        prompt_len_rejected=len(prompt_ids),
    )
    assert loss is not None


def test_tdpo_loss_runs():
    backend, prompt_ids, chosen_ids, rejected_ids = _make_ids()
    loss_fn = get_loss("tdpo")
    loss = loss_fn(
        backend,
        chosen_ids,
        rejected_ids,
        prompt_len_chosen=len(prompt_ids),
        prompt_len_rejected=len(prompt_ids),
    )
    assert loss is not None


def test_kto_loss_runs():
    backend, prompt_ids, chosen_ids, _rejected_ids = _make_ids()
    loss_fn = get_loss("kto")
    loss = loss_fn(
        backend,
        chosen_ids,
        prompt_len=len(prompt_ids),
        desired=True,
    )
    assert loss is not None
