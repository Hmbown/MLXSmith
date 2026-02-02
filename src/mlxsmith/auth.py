from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import os
from huggingface_hub import HfApi, get_token as hf_get_token, logout as hf_logout
from huggingface_hub import constants as hf_constants


@dataclass
class AuthStatus:
    token_present: bool
    token_hint: Optional[str] = None
    user: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


def _mask_token(token: str) -> str:
    if not token:
        return ""
    if len(token) <= 8:
        return "***"
    return f"{token[:4]}...{token[-4:]}"


def _token_path() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "token"
    return Path(getattr(hf_constants, "HF_TOKEN_PATH", Path(hf_constants.HF_HOME) / "token"))


def load_token() -> Optional[str]:
    try:
        token = hf_get_token()
        if token:
            return token
    except Exception:
        pass
    path = _token_path()
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return None


def save_token(token: str) -> None:
    path = _token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token, encoding="utf-8")


def delete_token() -> bool:
    removed = False
    try:
        hf_logout()
        removed = True
    except Exception:
        pass
    path = _token_path()
    if path.exists():
        path.unlink()
        removed = True
    return removed


def get_status(validate: bool = False) -> AuthStatus:
    token = load_token()
    if not token:
        return AuthStatus(token_present=False)
    status = AuthStatus(token_present=True, token_hint=_mask_token(token))
    if validate:
        try:
            info = HfApi().whoami(token=token)
            status.user = info.get("name") or info.get("fullname") or info.get("email")
        except Exception as exc:
            status.warnings.append(f"Token validation failed: {exc}")
    return status


def login(token: str, validate: bool = True) -> AuthStatus:
    save_token(token)
    return get_status(validate=validate)


def logout() -> bool:
    return delete_token()
