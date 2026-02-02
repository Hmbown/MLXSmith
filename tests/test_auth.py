from pathlib import Path

from mlxsmith.auth import get_status, login, logout


def test_auth_login_status_logout(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    token = "hf_testtoken_1234567890"
    status = login(token, validate=False)
    assert status.token_present is True
    assert status.token_hint is not None

    status = get_status(validate=False)
    assert status.token_present is True
    assert status.token_hint is not None

    assert logout() is True
    status = get_status(validate=False)
    assert status.token_present is False
