from __future__ import annotations

import json
import re
import shutil
import tarfile
import tempfile
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


@dataclass
class EnvRef:
    name: str
    version: Optional[str] = None


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


def _normalize_package_module(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", name.replace("-", "_"))
    cleaned = cleaned.strip("_") or "env"
    if cleaned[0].isdigit():
        cleaned = f"env_{cleaned}"
    return cleaned.lower()


def _env_scaffold_pyproject(name: str, version: str, description: str) -> str:
    return f"""[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{name}"
version = "{version}"
description = "{description}"
readme = "README.md"
requires-python = ">=3.10"

[tool.setuptools]
package-dir = {{"" = "src"}}

[tool.setuptools.packages.find]
where = ["src"]
"""


def _env_scaffold_readme(name: str) -> str:
    return f"""# {name}

Local MLXSmith environment package.

## Files
- `env.yaml`: task manifest consumed by mlxsmith.
- `pyproject.toml`: Python package metadata for Hub publishing.

## Usage
```bash
mlxsmith env package {name}
mlxsmith env publish envs/packages/{name}-0.1.0.tar.gz
```
"""


def _env_scaffold_module() -> str:
    return """from pathlib import Path

ENV_MANIFEST = Path(__file__).resolve().parents[2] / "env.yaml"


def load_environment() -> Path:
    return ENV_MANIFEST
"""


def _env_scaffold_init() -> str:
    return """from .environment import ENV_MANIFEST, load_environment

__all__ = ["ENV_MANIFEST", "load_environment"]
"""


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

    pkg_module = _normalize_package_module(name)
    (env_root / "pyproject.toml").write_text(
        _env_scaffold_pyproject(name, manifest["version"], manifest["description"]),
        encoding="utf-8",
    )
    (env_root / "README.md").write_text(_env_scaffold_readme(name), encoding="utf-8")
    pkg_dir = ensure_dir(env_root / "src" / pkg_module)
    (pkg_dir / "environment.py").write_text(_env_scaffold_module(), encoding="utf-8")
    (pkg_dir / "__init__.py").write_text(_env_scaffold_init(), encoding="utf-8")
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


def _parse_env_ref(env_ref: str, version: Optional[str] = None) -> EnvRef:
    name = env_ref.strip()
    parsed_version = None
    if "==" in name:
        name, parsed_version = name.split("==", 1)
    elif "@" in name:
        name, parsed_version = name.rsplit("@", 1)
    name = name.strip()
    if parsed_version is not None:
        parsed_version = parsed_version.strip()
    if parsed_version in {"", "latest"}:
        parsed_version = None
    if version and parsed_version and version != parsed_version:
        raise RuntimeError(f"Conflicting versions: {parsed_version} vs {version}")
    return EnvRef(name=name, version=version or parsed_version)


def _version_key(version: str) -> tuple:
    if not version:
        return tuple()
    base, _, _build = version.partition("+")
    main, _, pre = base.partition("-")
    parts = []
    for part in main.split("."):
        if part.startswith("v") and part[1:].isdigit():
            part = part[1:]
        if part.isdigit():
            parts.append((0, int(part)))
        else:
            parts.append((1, part))
    if pre:
        parts.append((1, pre))
    else:
        parts.append((2, ""))
    return tuple(parts)


def _select_registry_package(packages: list[dict], name: str, version: Optional[str]) -> dict:
    matches = [p for p in packages if p.get("name") == name]
    if not matches:
        raise RuntimeError(f"Env not found in registry: {name}")
    if version:
        exact = [p for p in matches if p.get("version") == version]
        if not exact:
            available = sorted({p.get("version", "") for p in matches})
            raise RuntimeError(f"Env {name} has no version {version}. Available: {', '.join(available)}")
        return exact[0]
    return sorted(matches, key=lambda p: _version_key(p.get("version", "")))[-1]


def _load_manifest_from_package(package_path: Path) -> EnvManifest:
    with tarfile.open(package_path, "r:gz") as tf:
        for m in tf.getmembers():
            if m.name.endswith("env.yaml"):
                f = tf.extractfile(m)
                if f is None:
                    break
                data = yaml.safe_load(f.read().decode("utf-8")) or {}
                return EnvManifest(
                    name=str(data.get("name") or Path(m.name).parent.name),
                    version=str(data.get("version") or "0.1.0"),
                    description=data.get("description"),
                    verifier=data.get("verifier"),
                    tasks=data.get("tasks"),
                )
    raise RuntimeError("Package missing env.yaml")


def list_registry_packages(project_root: Path, name: Optional[str] = None, all_versions: bool = False) -> list[dict]:
    registry = _load_registry(project_root)
    packages = registry.get("packages", [])
    if name:
        packages = [p for p in packages if p.get("name") == name]
    if all_versions or not packages:
        return sorted(packages, key=lambda p: (p.get("name", ""), _version_key(p.get("version", ""))))

    latest = {}
    for pkg in packages:
        pkg_name = pkg.get("name")
        if not pkg_name:
            continue
        prev = latest.get(pkg_name)
        if prev is None or _version_key(pkg.get("version", "")) > _version_key(prev.get("version", "")):
            latest[pkg_name] = pkg
    return sorted(latest.values(), key=lambda p: p.get("name", ""))


def registry_info(project_root: Path, env_ref: str, version: Optional[str] = None) -> tuple[dict, EnvManifest]:
    registry = _load_registry(project_root)
    ref = _parse_env_ref(env_ref, version=version)
    pkg = _select_registry_package(registry.get("packages", []), ref.name, ref.version)
    pkg_path = Path(pkg["path"])
    if not pkg_path.exists():
        raise RuntimeError(f"Registry package missing: {pkg_path}")
    manifest = _load_manifest_from_package(pkg_path)
    return pkg, manifest


def install_env(project_root: Path, source: str, version: Optional[str] = None) -> Path:
    src = Path(source)
    ensure_dir(_envs_root(project_root))
    if src.exists():
        if version:
            raise RuntimeError("Version pinning only applies to registry installs.")
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
    ref = _parse_env_ref(source, version=version)
    pkg = _select_registry_package(registry.get("packages", []), ref.name, ref.version)
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

    manifest = _load_manifest_from_package(pkg)

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


def pull_env(
    project_root: Path,
    env_ref: str,
    out_dir: Optional[str] = None,
    version: Optional[str] = None,
    force: bool = False,
) -> Path:
    registry = _load_registry(project_root)
    ref = _parse_env_ref(env_ref, version=version)
    pkg = _select_registry_package(registry.get("packages", []), ref.name, ref.version)
    pkg_path = Path(pkg["path"])
    if not pkg_path.exists():
        raise RuntimeError(f"Registry package missing: {pkg_path}")

    dest = Path(out_dir) if out_dir else Path.cwd() / ref.name
    if dest.exists():
        if not force:
            raise RuntimeError(f"Destination exists: {dest}")
        shutil.rmtree(dest)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        with tarfile.open(pkg_path, "r:gz") as tf:
            members = tf.getmembers()
            if not members:
                raise RuntimeError(f"Empty package: {pkg_path}")
            top = members[0].name.split("/")[0]
            tf.extractall(tmp_root)
        src = tmp_root / top
        if not src.exists():
            raise RuntimeError(f"Invalid package layout: {pkg_path}")
        shutil.copytree(src, dest)
    return dest
