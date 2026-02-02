from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from ..util import ensure_dir, now_ts, copytree


@dataclass
class EnvManifest:
    name: str
    version: str
    description: Optional[str] = None
    verifier: Optional[str] = None
    tasks: Optional[list] = None


def _envs_root(project_root: Path) -> Path:
    return project_root / "envs"


def _registry_path(project_root: Path) -> Path:
    return _envs_root(project_root) / "registry.json"


def load_manifest(env_path: Path) -> EnvManifest:
    data = yaml.safe_load(env_path.read_text(encoding="utf-8")) or {}
    return EnvManifest(
        name=str(data.get("name") or env_path.parent.name),
        version=str(data.get("version") or "0.1.0"),
        description=data.get("description"),
        verifier=data.get("verifier"),
        tasks=data.get("tasks"),
    )


def resolve_env_path(project_root: Path, env_ref: str) -> Path:
    ref = Path(env_ref)
    if ref.exists():
        if ref.is_dir():
            candidate = ref / "env.yaml"
            return candidate if candidate.exists() else ref
        return ref
    candidate = _envs_root(project_root) / env_ref / "env.yaml"
    return candidate


def init_env(project_root: Path, name: str) -> Path:
    env_root = _envs_root(project_root) / name
    ensure_dir(env_root)
    manifest = {
        "name": name,
        "version": "0.1.0",
        "description": "Sample environment",
        "verifier": "verifiers/regex.py",
        "tasks": [
            {
                "id": "add",
                "prompt": "Write a Python function add(a, b) that returns the sum.",
                "tests": "from main import add\\n\\n\\n"
                "def test_add():\\n"
                "    assert add(2, 3) == 5\\n",
            }
        ],
    }
    (env_root / "env.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return env_root


def _load_registry(project_root: Path) -> dict:
    path = _registry_path(project_root)
    if not path.exists():
        return {"packages": [], "updated_at": None}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_registry(project_root: Path, data: dict) -> None:
    data["updated_at"] = now_ts()
    path = _registry_path(project_root)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def install_env(project_root: Path, source: str) -> Path:
    src = Path(source)
    ensure_dir(_envs_root(project_root))
    if src.exists():
        if src.is_dir():
            manifest_path = src / "env.yaml"
            if not manifest_path.exists():
                raise RuntimeError(f"Missing env.yaml in {src}")
            manifest = load_manifest(manifest_path)
            dest = _envs_root(project_root) / manifest.name
            copytree(src, dest)
            return dest
        if src.suffixes[-2:] == [".tar", ".gz"]:
            with tarfile.open(src, "r:gz") as tf:
                members = tf.getmembers()
                top = members[0].name.split("/")[0] if members else ""
                tf.extractall(_envs_root(project_root))
                env_path = _envs_root(project_root) / top / "env.yaml"
                if env_path.exists():
                    return env_path.parent
            raise RuntimeError(f"Invalid package: {src}")
        raise RuntimeError(f"Unsupported env source: {src}")

    # treat as registry name
    registry = _load_registry(project_root)
    matches = [p for p in registry.get("packages", []) if p.get("name") == source]
    if not matches:
        raise RuntimeError(f"Env not found in registry: {source}")
    pkg = sorted(matches, key=lambda p: p.get("version", ""))[-1]
    pkg_path = Path(pkg["path"])
    return install_env(project_root, str(pkg_path))


def package_env(project_root: Path, env_name: str, out_path: Optional[str] = None) -> Path:
    env_dir = _envs_root(project_root) / env_name
    manifest_path = env_dir / "env.yaml"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing env.yaml in {env_dir}")
    manifest = load_manifest(manifest_path)

    out_dir = Path(out_path) if out_path else _envs_root(project_root) / "packages"
    ensure_dir(out_dir)
    tar_path = out_dir / f"{manifest.name}-{manifest.version}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(env_dir, arcname=env_dir.name)
    return tar_path


def publish_env(project_root: Path, package_path: str) -> Path:
    pkg = Path(package_path)
    if not pkg.exists():
        raise RuntimeError(f"Missing package: {pkg}")

    # Extract manifest for metadata
    manifest = None
    with tarfile.open(pkg, "r:gz") as tf:
        members = tf.getmembers()
        for m in members:
            if m.name.endswith("env.yaml"):
                f = tf.extractfile(m)
                if f is not None:
                    data = yaml.safe_load(f.read().decode("utf-8")) or {}
                    manifest = EnvManifest(
                        name=str(data.get("name") or Path(m.name).parent.name),
                        version=str(data.get("version") or "0.1.0"),
                        description=data.get("description"),
                        verifier=data.get("verifier"),
                    )
                break

    if manifest is None:
        raise RuntimeError("Package missing env.yaml")

    registry_dir = _envs_root(project_root) / "registry"
    ensure_dir(registry_dir)
    dest = registry_dir / f"{manifest.name}-{manifest.version}.tar.gz"
    dest.write_bytes(pkg.read_bytes())

    registry = _load_registry(project_root)
    registry.setdefault("packages", [])
    registry["packages"] = [
        p
        for p in registry["packages"]
        if not (p.get("name") == manifest.name and p.get("version") == manifest.version)
    ]
    registry["packages"].append(
        {
            "name": manifest.name,
            "version": manifest.version,
            "description": manifest.description,
            "verifier": manifest.verifier,
            "path": str(dest),
        }
    )
    _save_registry(project_root, registry)
    return dest
