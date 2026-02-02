import inspect
from mlxsmith.verifiers.pytest_verifier import verify as _verify

_SUPPORTED = set(inspect.signature(_verify).parameters.keys())


def verify(prompt, completion, workdir, **kwargs):
    filtered = {k: v for k, v in kwargs.items() if k in _SUPPORTED}
    return _verify(prompt, completion, workdir, **filtered)
