from mlxsmith.verifiers.regex import verify as _verify


def verify(prompt, completion, workdir, **kwargs):
    return _verify(prompt, completion, workdir, **kwargs)
