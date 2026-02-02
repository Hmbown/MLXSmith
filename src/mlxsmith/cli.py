from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .auth import get_status as get_auth_status, login as hf_login, logout as hf_logout
from .config import (
    ProjectConfig,
    dump_config,
    get_config,
    load_config,
    resolve_config_path,
    show_merged_config,
    write_default_config,
)
from .data import import_sharegpt, split_jsonl
from .models import hf_pull, quantize_stub
from .util import detect_system, ensure_dir
from .accel import get_backend
from .train.sft import run_sft
from .train.pref import run_pref
from .train.rft import run_rft
from .train.distill import run_distill
from .eval import run_eval
from .bench import run_bench
from .rlm import run_rlm, run_rlm_orchestrated
from .rlm.gating import load_state as _load_rlm_state
from .adapters import merge_adapters
from .envs import (
    init_env as init_env_plugin,
    install_env as install_env_plugin,
    package_env as package_env_plugin,
    publish_env as publish_env_plugin,
    resolve_env_path as resolve_env_path_plugin,
    load_manifest as load_env_manifest,
)

app = typer.Typer(add_completion=False, help="mlxsmith — PrimeRL-style training CLI for MLX")
console = Console()


def project_root_from_cwd() -> Path:
    return Path.cwd()


@app.command()
def init(path: str = typer.Argument(..., help="Project directory to create")):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    for d in ["data/sft", "data/prefs", "models", "envs", "verifiers", "eval/suites", "runs", "cache", "bench"]:
        (p / d).mkdir(parents=True, exist_ok=True)
    cfg_path = p / "mlxsmith.yaml"
    if not cfg_path.exists():
        write_default_config(cfg_path)
    (p / "envs" / "coding.yaml").write_text(_sample_env_yaml(), encoding="utf-8")
    (p / "verifiers" / "regex.py").write_text(_sample_verifier_regex(), encoding="utf-8")
    (p / "verifiers" / "pytest.py").write_text(_sample_verifier_pytest(), encoding="utf-8")
    (p / "verifiers" / "jsonschema.py").write_text(_sample_verifier_jsonschema(), encoding="utf-8")
    (p / "eval" / "suites" / "coding.yaml").write_text(_sample_eval_suite(), encoding="utf-8")
    console.print(f"[green]Initialized[/green] {p.resolve()}")


@app.command()
def doctor():
    info = detect_system()
    table = Table(title="mlxsmith doctor")
    table.add_column("item")
    table.add_column("value")
    table.add_row("python", info.python)
    table.add_row("python_arch", info.python_arch)
    table.add_row("platform", info.platform)
    table.add_row("macos_version", info.macos_version or "n/a")
    table.add_row("machine", info.machine)
    table.add_row("cpu_count", str(info.cpu_count))
    table.add_row("metal", str(info.has_metal))
    table.add_row("mlx", f"{info.has_mlx} {info.mlx_version or ''}".strip())
    table.add_row("zmlx", str(info.has_zmlx))
    console.print(table)


@app.command()
def pull(
    model: str = typer.Argument(..., help="Hugging Face model id"),
    out: Optional[str] = typer.Option(None, "--out", help="MLX output path (defaults to cache/mlx/<model>)"),
    no_convert: bool = typer.Option(False, "--no-convert", help="Only download HF snapshot, skip MLX conversion"),
    quantize: bool = typer.Option(False, "--quantize", help="Quantize during conversion"),
    q_bits: Optional[int] = typer.Option(None, "--q-bits"),
    q_group_size: Optional[int] = typer.Option(None, "--q-group-size"),
    q_mode: Optional[str] = typer.Option(None, "--q-mode"),
    quant_predicate: Optional[str] = typer.Option(None, "--quant-predicate"),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code"),
):
    root = project_root_from_cwd()
    cache_dir = ensure_dir(root / "cache")
    out_path = Path(out) if out else None
    dst = hf_pull(
        model,
        cache_dir,
        convert=not no_convert,
        mlx_path=out_path,
        quantize=quantize,
        q_bits=q_bits,
        q_group_size=q_group_size,
        q_mode=q_mode,
        quant_predicate=quant_predicate,
        trust_remote_code=trust_remote_code,
    )
    console.print(f"[green]Pulled[/green] {model} -> {dst}")


@app.command()
def quantize(
    model_path: str = typer.Argument(...),
    to: str = typer.Option("q4"),
    out: str = typer.Option("models/quantized"),
):
    root = project_root_from_cwd()
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = root / out_path
    result = quantize_stub(Path(model_path), out_path, to)
    console.print(f"[green]Quant stub wrote[/green] {result}")


data_app = typer.Typer(help="Dataset utilities")
app.add_typer(data_app, name="data")


@data_app.command("import")
def data_import(in_path: str = typer.Option(..., "--in"), fmt: str = typer.Option("sharegpt", "--format"), out_path: str = typer.Option(..., "--out")):
    root = project_root_from_cwd()
    inp = Path(in_path)
    outp = Path(out_path)
    if not outp.is_absolute():
        outp = root / outp
    if fmt.lower() == "sharegpt":
        n = import_sharegpt(inp, outp)
        console.print(f"[green]Wrote[/green] {n} rows -> {outp}")
    else:
        raise typer.BadParameter(f"Unsupported format: {fmt}")


@data_app.command("split")
def data_split(
    in_path: str = typer.Option(..., "--in"),
    out_dir: str = typer.Option("data/sft", "--out-dir"),
    valid: float = typer.Option(0.02),
    test: float = typer.Option(0.02),
    seed: int = typer.Option(1337),
):
    root = project_root_from_cwd()
    inp = Path(in_path)
    outd = Path(out_dir)
    if not outd.is_absolute():
        outd = root / outd
    stats = split_jsonl(inp, outd, valid_frac=valid, test_frac=test, seed=seed)
    console.print(f"[green]Split[/green] -> {outd}  {stats}")


@app.command()
def sft(
    config: str = typer.Option("mlxsmith.yaml", "-c", "--config", help="Config file path"),
    data: str = typer.Option("data/sft", "--data"),
    model: Optional[str] = typer.Option(None, "--model", help="Override model.id"),
    accel: Optional[str] = typer.Option(None, "--accel", help="Override accel.backend"),
    lr: Optional[float] = typer.Option(None, "--lr", help="Override train.lr (learning rate)"),
    iters: Optional[int] = typer.Option(None, "--iters", help="Override train.iters"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Override train.batch_size"),
):
    root = project_root_from_cwd()
    cfg = get_config(
        config_path=config,
        root=root,
        model_id=model,
        accel_backend=accel,
        lr=lr,
        iters=iters,
        batch_size=batch_size,
    )
    data_dir = root / data
    run = run_sft(root, cfg, data_dir, cfg.model.id, cfg.accel.backend)
    console.print(f"[bold]Run:[/bold] {run.run_dir}")


@app.command()
def pref(
    config: str = typer.Option("mlxsmith.yaml", "-c", "--config", help="Config file path"),
    data: str = typer.Option("data/prefs", "--data"),
    model: str = typer.Option(..., "--model", help="Base adapter or model path (e.g., runs/sft_0001/adapter)"),
    accel: Optional[str] = typer.Option(None, "--accel", help="Override accel.backend"),
    algo: Optional[str] = typer.Option(None, "--algo", help="Override pref.algo (dpo|orpo|grpo)"),
):
    root = project_root_from_cwd()
    cfg = get_config(
        config_path=config,
        root=root,
        accel_backend=accel,
        algo=algo,
    )
    data_dir = root / data
    run = run_pref(root, cfg, data_dir, Path(model), cfg.accel.backend)
    console.print(f"[bold]Run:[/bold] {run.run_dir}")


@app.command()
def rft(
    config: str = typer.Option("mlxsmith.yaml", "-c", "--config", help="Config file path"),
    env: str = typer.Option("envs/coding.yaml", "--env"),
    verifier: str = typer.Option("verifiers/regex.py", "--verifier"),
    model: str = typer.Option(..., "--model"),
    accel: Optional[str] = typer.Option(None, "--accel", help="Override accel.backend"),
    rollouts: Optional[int] = typer.Option(None, "--rollouts", help="Override rft.rollouts"),
):
    root = project_root_from_cwd()
    cfg = get_config(
        config_path=config,
        root=root,
        accel_backend=accel,
        rollouts=rollouts,
    )
    run = run_rft(root, cfg, root / env, root / verifier, Path(model), cfg.accel.backend)
    console.print(f"[bold]Run:[/bold] {run.run_dir}")


@app.command()
def distill(
    data: str = typer.Option(..., "--data", help="JSONL with prompts"),
    teacher: str = typer.Option(..., "--teacher"),
    student: str = typer.Option(..., "--student"),
    mode: str = typer.Option("offline", "--mode", help="offline | opd"),
    config: str = typer.Option("mlxsmith.yaml", "-c", "--config", help="Config file path"),
    accel: Optional[str] = typer.Option(None, "--accel", help="Override accel.backend"),
    max_new_tokens: int = typer.Option(256, "--max-new-tokens"),
    temperature: float = typer.Option(0.7, "--temperature"),
):
    root = project_root_from_cwd()
    cfg = get_config(config_path=config, root=root, accel_backend=accel)
    run = run_distill(
        root,
        cfg,
        Path(data),
        teacher_model=teacher,
        student_model=student,
        accel=cfg.accel.backend,
        mode=mode,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    console.print(f"[bold]Run:[/bold] {run.run_dir}")


@app.command()
def eval(
    suite: str = typer.Option("eval/suites/coding.yaml", "--suite"),
    model: str = typer.Option(..., "--model"),
):
    root = project_root_from_cwd()
    out = run_eval(root, root / suite, Path(model))
    console.print(f"[bold]Eval:[/bold] {out}")


@app.command()
def serve(
    config: str = typer.Option("mlxsmith.yaml", "-c", "--config", help="Config file path"),
    model: str = typer.Option(..., "--model"),
    host: Optional[str] = typer.Option(None, "--host", help="Override serve.host"),
    port: Optional[int] = typer.Option(None, "--port", help="Override serve.port"),
    ui: Optional[bool] = typer.Option(None, "--ui/--no-ui", help="Override serve.ui"),
):
    root = project_root_from_cwd()
    cfg = get_config(
        config_path=config,
        root=root,
        host=host,
        port=port,
        ui=ui,
    )
    h = cfg.serve.host
    p = cfg.serve.port

    import uvicorn
    from .server import create_app

    server_app = create_app(model, cfg)
    console.print(f"[green]Serving[/green] model={model} on http://{h}:{p}")
    uvicorn.run(server_app, host=h, port=p)


@app.command()
def bench(
    config: str = typer.Option("mlxsmith.yaml", "-c", "--config", help="Config file path"),
    model: Optional[str] = typer.Option(None, "--model", help="Override model.id"),
    accel: Optional[str] = typer.Option(None, "--accel", help="Override accel.backend"),
    prompt: str = typer.Option("Hello", "--prompt"),
    max_tokens: int = typer.Option(128, "--max-tokens"),
    reps: int = typer.Option(3, "--reps"),
    mode: str = typer.Option("inference", "--mode", help="inference | trainer | end_to_end"),
    steps: int = typer.Option(5, "--steps", help="trainer mode steps per rep"),
):
    root = project_root_from_cwd()
    cfg = get_config(
        config_path=config,
        root=root,
        model_id=model,
        accel_backend=accel,
    )
    out = run_bench(
        root,
        cfg,
        cfg.model.id,
        cfg.accel.backend,
        prompt=prompt,
        max_tokens=max_tokens,
        reps=reps,
        mode=mode,
        steps=steps,
    )
    console.print(f"[bold]Bench:[/bold] {out}")


# Config subcommand
config_app = typer.Typer(help="Configuration management")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show(
    config: str = typer.Option("mlxsmith.yaml", "-c", "--config", help="Config file path"),
    format: str = typer.Option("yaml", "-f", "--format", help="Output format: yaml, json, toml"),
    sources: bool = typer.Option(False, "--sources", help="Show value sources (default, env, file, cli)"),
    root: Optional[str] = typer.Option(None, "--root", help="Project root directory"),
):
    """Show merged configuration with all overrides applied."""
    project_root = Path(root) if root else project_root_from_cwd()
    cfg_path = resolve_config_path(config, root=project_root)
    
    if sources:
        from .config import get_config_sources
        cfg, srcs = get_config_sources(cfg_path if cfg_path.exists() else None)
        output = show_merged_config(cfg, show_sources=True, sources=srcs)
        console.print(output)
    else:
        cfg = load_config(cfg_path if cfg_path.exists() else None)
        try:
            output = dump_config(cfg, format=format)
            console.print(output)
        except ValueError as e:
            raise typer.BadParameter(str(e))


@config_app.command("init")
def config_init(
    path: str = typer.Argument("mlxsmith.yaml", help="Output config file path"),
    format: str = typer.Option("yaml", "-f", "--format", help="Output format: yaml, json, toml"),
):
    """Initialize a new configuration file with defaults."""
    out_path = Path(path)
    if out_path.exists():
        overwrite = typer.confirm(f"File {path} already exists. Overwrite?")
        if not overwrite:
            raise typer.Exit()
    
    write_default_config(out_path, format=format)
    console.print(f"[green]Created config file:[/green] {out_path.resolve()}")


@config_app.command("validate")
def config_validate(
    config: str = typer.Argument(..., help="Config file path to validate"),
    root: Optional[str] = typer.Option(None, "--root", help="Project root directory"),
):
    """Validate a configuration file."""
    project_root = Path(root) if root else project_root_from_cwd()
    cfg_path = resolve_config_path(config, root=project_root)
    
    try:
        cfg = load_config(cfg_path, require=True)
        console.print(f"[green]✓ Configuration is valid[/green]")
        
        # Show summary
        table = Table(title="Configuration Summary")
        table.add_column("Section")
        table.add_column("Key Settings")
        
        data = cfg.model_dump()
        for section, values in data.items():
            if isinstance(values, dict):
                summary = ", ".join(f"{k}={v}" for k, v in list(values.items())[:3])
                if len(values) > 3:
                    summary += f" ... ({len(values) - 3} more)"
                table.add_row(section, summary)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]✗ Configuration validation failed:[/red] {e}")
        raise typer.Exit(code=1)


@config_app.command("env")
def config_env(
    prefix: str = typer.Option("MLXSMITH__", "--prefix", help="Environment variable prefix"),
):
    """Show available environment variables."""
    cfg = ProjectConfig()
    
    console.print(f"\n[bold]Environment Variable Configuration[/bold]")
    console.print(f"Prefix: [cyan]{prefix}[/cyan]")
    console.print(f"Nested delimiter: [cyan]__[/cyan] (double underscore)\n")
    
    table = Table(title=f"Available {prefix}* Environment Variables")
    table.add_column("Environment Variable")
    table.add_column("Default Value")
    table.add_column("Description")
    
    data = cfg.model_dump()
    for section_name, section_data in data.items():
        if not isinstance(section_data, dict):
            continue
        for key, value in section_data.items():
            env_var = f"{prefix}{section_name.upper()}__{key.upper()}"
            value_str = str(value) if value is not None else "None"
            if len(value_str) > 40:
                value_str = value_str[:37] + "..."
            table.add_row(env_var, value_str, f"{section_name}.{key}")
    
    console.print(table)
    console.print("\n[dim]Example: MLXSMITH__MODEL__ID=custom/model mlxsmith sft[/dim]")


auth_app = typer.Typer(help="Hugging Face authentication")
app.add_typer(auth_app, name="auth")


@auth_app.command("login")
def auth_login(
    token: Optional[str] = typer.Option(None, "--token", envvar="HF_TOKEN"),
    validate: bool = typer.Option(True, "--validate/--no-validate"),
):
    if not token:
        token = typer.prompt("Hugging Face token", hide_input=True)
    status = hf_login(token, validate=validate)
    if status.user:
        console.print(f"[green]Logged in[/green] as {status.user}")
    else:
        hint = f" ({status.token_hint})" if status.token_hint else ""
        console.print(f"[green]Token saved[/green]{hint}")
    for warning in status.warnings:
        console.print(f"[yellow]{warning}[/yellow]")


@auth_app.command("status")
def auth_status(validate: bool = typer.Option(False, "--validate/--no-validate")):
    status = get_auth_status(validate=validate)
    if not status.token_present:
        console.print("[yellow]No token found[/yellow]")
        return
    if status.user:
        console.print(f"[green]Logged in[/green] as {status.user}")
    else:
        hint = f" ({status.token_hint})" if status.token_hint else ""
        console.print(f"[green]Token present[/green]{hint}")
    for warning in status.warnings:
        console.print(f"[yellow]{warning}[/yellow]")


@auth_app.command("logout")
def auth_logout():
    if hf_logout():
        console.print("[green]Logged out[/green]")
    else:
        console.print("[yellow]No token found[/yellow]")


accel_app = typer.Typer(help="Acceleration utilities")
app.add_typer(accel_app, name="accel")

rlm_app = typer.Typer(help="Recursive Language Model (RLM) loop")
app.add_typer(rlm_app, name="rlm")

env_app = typer.Typer(help="Environment plugins")
app.add_typer(env_app, name="env")

adapter_app = typer.Typer(help="Adapter utilities")
app.add_typer(adapter_app, name="adapters")


@rlm_app.callback(invoke_without_command=True)
def rlm_callback(
    ctx: typer.Context,
    config: str = typer.Option("mlxsmith.yaml", "-c", "--config", help="Config file path"),
    model: Optional[str] = typer.Option(None, "--model", help="Override model.id"),
    iterations: Optional[int] = typer.Option(None, "--iterations", help="Override rlm.iterations"),
    resume: bool = typer.Option(False, "--resume"),
    orchestrated: bool = typer.Option(False, "--orchestrated", help="Use multi-process orchestrator mode"),
):
    if ctx.invoked_subcommand is not None:
        return
    root = project_root_from_cwd()
    cfg = get_config(
        config_path=config,
        root=root,
        model_id=model,
        iterations=iterations,
    )
    if orchestrated:
        run_rlm_orchestrated(root, cfg, model_spec=model, iterations=iterations, resume=resume)
    else:
        run_rlm(root, cfg, model_spec=model, iterations=iterations, resume=resume)


@rlm_app.command("status")
def rlm_status():
    root = project_root_from_cwd()
    state_path = root / "runs" / "rlm_state.json"
    state = _load_rlm_state(state_path)
    table = Table(title="mlxsmith rlm status")
    table.add_column("item")
    table.add_column("value")
    table.add_row("last_iteration", str(state.last_iteration))
    table.add_row("current_adapter", state.current_adapter or "n/a")
    table.add_row("best_adapter", state.best_adapter or "n/a")
    table.add_row("best_score", str(state.best_score) if state.best_score is not None else "n/a")
    table.add_row("ema_score", str(state.ema_score) if state.ema_score is not None else "n/a")
    console.print(table)


@rlm_app.command("history")
def rlm_history(limit: int = typer.Option(10, "--limit")):
    root = project_root_from_cwd()
    history_path = root / "runs" / "rlm_history.jsonl"
    if not history_path.exists():
        console.print("[yellow]No history found[/yellow]")
        return
    lines = history_path.read_text(encoding="utf-8").splitlines()
    tail = lines[-limit:] if limit > 0 else lines
    for line in tail:
        console.print(line)


@accel_app.command("status")
def accel_status():
    backends = ["none", "zmlx"]
    table = Table(title="mlxsmith accel status")
    table.add_column("backend")
    table.add_column("available")
    table.add_column("notes")
    for name in backends:
        b = get_backend(name)
        stats = b.stats()
        table.add_row(stats.backend, "yes" if not stats.notes or "error" not in (stats.notes or {}) else "no", json.dumps(stats.notes or {}))
    console.print(table)


@env_app.command("init")
def env_init(name: str = typer.Argument(..., help="Environment name")):
    root = project_root_from_cwd()
    env_dir = init_env_plugin(root, name)
    console.print(f"[green]Initialized env[/green] {env_dir}")


@env_app.command("install")
def env_install(source: str = typer.Argument(..., help="Env dir, package path, or registry name")):
    root = project_root_from_cwd()
    env_dir = install_env_plugin(root, source)
    console.print(f"[green]Installed env[/green] {env_dir}")


@env_app.command("package")
def env_package(
    name: str = typer.Argument(..., help="Env name (directory under envs/)"),
    out: str = typer.Option(None, "--out", help="Output directory for package"),
):
    root = project_root_from_cwd()
    out_path = package_env_plugin(root, name, out_path=out)
    console.print(f"[green]Packaged env[/green] {out_path}")


@env_app.command("publish")
def env_publish(package: str = typer.Argument(..., help="Path to .tar.gz package")):
    root = project_root_from_cwd()
    dest = publish_env_plugin(root, package)
    console.print(f"[green]Published env[/green] {dest}")


@env_app.command("run")
def env_run(
    env: str = typer.Argument(..., help="Env name or path to env.yaml"),
    model: str = typer.Option(..., "--model"),
    accel: Optional[str] = typer.Option(None, "--accel"),
    verifier: Optional[str] = typer.Option(None, "--verifier"),
    config: str = typer.Option("mlxsmith.yaml", "-c", "--config", help="Config file path"),
):
    root = project_root_from_cwd()
    cfg = get_config(config_path=config, root=root, accel_backend=accel)
    env_path = resolve_env_path_plugin(root, env)
    if env_path.is_dir():
        env_path = env_path / "env.yaml"
    if not env_path.exists():
        raise typer.BadParameter(f"Env not found: {env}")
    manifest = load_env_manifest(env_path)
    verifier_path = verifier or manifest.verifier or "verifiers/regex.py"
    vpath = Path(verifier_path)
    if not vpath.is_absolute():
        vpath = root / vpath
    run = run_rft(root, cfg, env_path, vpath, Path(model), cfg.accel.backend)
    console.print(f"[bold]Run:[/bold] {run.run_dir}")


@env_app.command("registry")
def env_registry():
    root = project_root_from_cwd()
    registry_path = root / "envs" / "registry.json"
    if not registry_path.exists():
        console.print("[yellow]No registry found[/yellow]")
        return
    console.print(registry_path.read_text(encoding="utf-8"))


@adapter_app.command("merge")
def adapters_merge(
    base: str = typer.Option(..., "--base", help="Base model id or path"),
    adapters: str = typer.Option(..., "--adapters", help="Comma-separated adapter paths"),
    out: str = typer.Option("models/merged_adapter", "--out"),
    weights: Optional[str] = typer.Option(None, "--weights", help="Comma-separated weights"),
):
    root = project_root_from_cwd()
    adapter_paths = [Path(p.strip()) for p in adapters.split(",") if p.strip()]
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = root / out_path
    w = None
    if weights:
        w = [float(x) for x in weights.split(",") if x.strip()]
    result = merge_adapters(base, adapter_paths, out_path, weights=w)
    console.print(f"[green]Merged adapters[/green] {result}")


def _sample_env_yaml() -> str:
    return """name: coding-sample
tasks:
  - id: hello
    prompt: |
      Write a Python function `add(a, b)` that returns the sum.
    gold: |
      def add(a, b):
          return a + b
    verifier_kwargs:
      pattern: "def\\s+add\\("
  - id: pytest_task
    prompt: |
      Implement `mul(a,b)` in main.py. Tests provided.
    tests: |
      from main import mul
      def test_mul():
          assert mul(2,3) == 6
          assert mul(-1,5) == -5
    gold: |
      def mul(a,b):
          return a*b
"""


def _sample_verifier_regex() -> str:
    return """from mlxsmith.verifiers.regex import verify as _verify

def verify(prompt: str, completion: str, workdir: str, **kwargs):
    return _verify(prompt, completion, workdir, **kwargs)
"""


def _sample_verifier_pytest() -> str:
    return """from mlxsmith.verifiers.pytest_verifier import verify as _verify

def verify(prompt: str, completion: str, workdir: str, **kwargs):
    return _verify(prompt, completion, workdir, **kwargs)
"""


def _sample_verifier_jsonschema() -> str:
    return """from mlxsmith.verifiers.jsonschema import verify as _verify

def verify(prompt: str, completion: str, workdir: str, **kwargs):
    return _verify(prompt, completion, workdir, **kwargs)
"""


def _sample_eval_suite() -> str:
    return """name: coding-eval-sample
notes: |
  Minimal eval suite for smoke testing.
tasks:
  - id: add
    prompt: |
      Write a Python function `add(a, b)` that returns the sum.
    k: 2
    max_new_tokens: 128
    verifier: verifiers/regex.py
    verifier_kwargs:
      pattern: "def\\s+add\\("
"""
