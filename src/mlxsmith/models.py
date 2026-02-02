from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import subprocess
import sys
import json

from rich.console import Console

console = Console()


def _sanitize_repo_id(model_id: str) -> str:
    return model_id.replace("/", "__")


def is_adapter_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "adapter_config.json").exists():
        return True
    if (path / "adapters.safetensors").exists():
        return True
    if (path / "lora.npz").exists():
        return True
    return False


def read_adapter_metadata(path: Path) -> dict | None:
    meta = path / "adapter_metadata.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))
    cfg = path / "adapter_config.json"
    if cfg.exists():
        return json.loads(cfg.read_text(encoding="utf-8"))
    return None


def resolve_model_spec(project_root: Path, model_arg: str, cfg) -> Tuple[str, Optional[Path], Optional[dict]]:
    """Resolve a model argument into (base_model, adapter_path, adapter_meta)."""
    p = Path(model_arg)
    if not p.is_absolute():
        candidate = project_root / p
    else:
        candidate = p

    if candidate.exists() and is_adapter_dir(candidate):
        meta = read_adapter_metadata(candidate) or {}
        base_model = meta.get("base_model") or cfg.model.id
        return str(base_model), candidate, meta

    if candidate.exists():
        return str(candidate), None, None

    # assume HF repo id
    return model_arg, None, None


def hf_pull(
    model_id: str,
    cache_dir: Path,
    *,
    convert: bool = True,
    mlx_path: Optional[Path] = None,
    quantize: bool = False,
    q_bits: Optional[int] = None,
    q_group_size: Optional[int] = None,
    q_mode: Optional[str] = None,
    quant_predicate: Optional[str] = None,
    trust_remote_code: bool = False,
    hf_token: Optional[str] = None,
    hf_xet: bool = False,
) -> Path:
    """Download a HF repo snapshot and optionally convert to MLX format."""
    import os
    from huggingface_hub import snapshot_download

    if hf_xet:
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    target = cache_dir / "hf" / _sanitize_repo_id(model_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Downloading[/bold] {model_id} -> {target}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=hf_token,
    )

    if not convert:
        return target

    if mlx_path is None:
        mlx_path = cache_dir / "mlx" / _sanitize_repo_id(model_id)
    mlx_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import mlx_lm  # type: ignore

        console.print("[bold]Converting[/bold] HF -> MLX via mlx_lm.convert")
        mlx_lm.convert(
            hf_path=str(target),
            mlx_path=str(mlx_path),
            quantize=quantize,
            q_group_size=q_group_size or 64,
            q_bits=q_bits or 4,
            q_mode=q_mode or "affine",
            quant_predicate=quant_predicate,
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        cmd = [
            sys.executable,
            "-m",
            "mlx_lm.convert",
            "--hf-path",
            str(target),
            "--mlx-path",
            str(mlx_path),
        ]
        if quantize:
            cmd.append("--quantize")
        if q_bits is not None:
            cmd.extend(["--q-bits", str(q_bits)])
        if q_group_size is not None:
            cmd.extend(["--q-group-size", str(q_group_size)])
        if q_mode is not None:
            cmd.extend(["--q-mode", str(q_mode)])
        if quant_predicate is not None:
            cmd.extend(["--quant-predicate", str(quant_predicate)])
        if trust_remote_code:
            cmd.append("--trust-remote-code")

        console.print(f"[bold]Converting[/bold] HF -> MLX: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            console.print(proc.stdout)
            console.print(proc.stderr)
            raise RuntimeError("mlx_lm.convert failed")

    return mlx_path


def hf_convert(
    hf_path: str,
    cache_dir: Path,
    *,
    mlx_path: Optional[Path] = None,
    download: bool = True,
    quantize: bool = False,
    q_bits: Optional[int] = None,
    q_group_size: Optional[int] = None,
    q_mode: Optional[str] = None,
    quant_predicate: Optional[str] = None,
    trust_remote_code: bool = False,
    upload_repo: Optional[str] = None,
    hf_token: Optional[str] = None,
    hf_xet: bool = False,
) -> Path:
    """Convert a HF model (repo id or local path) to MLX format."""
    import os
    from huggingface_hub import snapshot_download, HfApi

    if hf_xet:
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)

    src = Path(hf_path)
    if src.exists():
        hf_src = str(src)
        default_name = src.name
    elif download:
        target = cache_dir / "hf" / _sanitize_repo_id(hf_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold]Downloading[/bold] {hf_path} -> {target}")
        snapshot_download(
            repo_id=hf_path,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            resume_download=True,
            token=hf_token,
        )
        hf_src = str(target)
        default_name = _sanitize_repo_id(hf_path)
    else:
        hf_src = hf_path
        default_name = _sanitize_repo_id(hf_path)

    if mlx_path is None:
        mlx_path = cache_dir / "mlx" / default_name
    mlx_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import mlx_lm  # type: ignore

        console.print("[bold]Converting[/bold] HF -> MLX via mlx_lm.convert")
        mlx_lm.convert(
            hf_path=hf_src,
            mlx_path=str(mlx_path),
            quantize=quantize,
            q_group_size=q_group_size or 64,
            q_bits=q_bits or 4,
            q_mode=q_mode or "affine",
            quant_predicate=quant_predicate,
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        cmd = [
            sys.executable,
            "-m",
            "mlx_lm.convert",
            "--hf-path",
            str(hf_src),
            "--mlx-path",
            str(mlx_path),
        ]
        if quantize:
            cmd.append("--quantize")
        if q_bits is not None:
            cmd.extend(["--q-bits", str(q_bits)])
        if q_group_size is not None:
            cmd.extend(["--q-group-size", str(q_group_size)])
        if q_mode is not None:
            cmd.extend(["--q-mode", str(q_mode)])
        if quant_predicate is not None:
            cmd.extend(["--quant-predicate", str(quant_predicate)])
        if trust_remote_code:
            cmd.append("--trust-remote-code")

        console.print(f"[bold]Converting[/bold] HF -> MLX: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            console.print(proc.stdout)
            console.print(proc.stderr)
            raise RuntimeError("mlx_lm.convert failed")

    if upload_repo:
        console.print(f"[bold]Uploading[/bold] {mlx_path} -> {upload_repo}")
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=upload_repo, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=upload_repo, repo_type="model", folder_path=str(mlx_path))

    return mlx_path


def quantize_stub(model_path: Path, out_path: Path, to: str):
    """Deprecated: quantization now handled via mlx_lm.convert --quantize."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (out_path / "QUANTIZATION.txt").write_text(
        f"Quantization requested: {to}\nSource: {model_path}\n",
        encoding="utf-8",
    )
    return out_path
