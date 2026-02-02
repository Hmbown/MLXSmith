"""MLXSmith configuration management with pydantic-settings.

Config precedence (highest to lowest):
1. CLI arguments
2. Config file (TOML/YAML/JSON)
3. Environment variables (MLXSMITH__*)
4. Default values

Environment variables use double underscore as nested delimiter:
  MLXSMITH__MODEL__ID=custom/model
  MLXSMITH__TRAIN__LR=0.001
  MLXSMITH__SERVE__PORT=9000

Config files support @path syntax:
  mlxsmith sft --config @production.toml
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10 fallback
    import tomli as tomllib

from .config_models import (
    AccelConfig,
    InferConfig,
    LoggingConfig,
    LoraConfig,
    ModelConfig,
    PrefConfig,
    ProjectConfig,
    RftConfig,
    RlmConfig,
    ServeConfig,
    TrainConfig,
)

__all__ = [
    # Models
    "ProjectConfig",
    "ModelConfig",
    "TrainConfig",
    "LoraConfig",
    "PrefConfig",
    "RftConfig",
    "InferConfig",
    "ServeConfig",
    "RlmConfig",
    "AccelConfig",
    "LoggingConfig",
    # Functions
    "load_config",
    "get_config",
    "resolve_config_path",
    "write_default_config",
    "dump_config",
    "show_merged_config",
]


class ProjectSettings(BaseSettings):
    """Pydantic-settings model for environment variable loading.
    
    This mirrors ProjectConfig but is used specifically for env var parsing.
    """
    
    model: ModelConfig = Field(default_factory=ModelConfig)
    accel: AccelConfig = Field(default_factory=AccelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    pref: PrefConfig = Field(default_factory=PrefConfig)
    rft: RftConfig = Field(default_factory=RftConfig)
    infer: InferConfig = Field(default_factory=InferConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)
    rlm: RlmConfig = Field(default_factory=RlmConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = SettingsConfigDict(
        env_prefix="MLXSMITH__",
        env_nested_delimiter="__",
        env_parse_enums=True,
        extra="ignore",  # Ignore unknown env vars
    )


# Import CLI aliases from models
from .config_models import CLI_ALIASES as _CLI_ALIASES  # noqa: E402


def resolve_config_path(config: Union[str, Path], root: Optional[Path] = None) -> Path:
    """Resolve config path, handling @prefix syntax.
    
    Args:
        config: Path string, optionally starting with @
        root: Optional root directory for relative paths
        
    Returns:
        Resolved Path object
        
    Example:
        >>> resolve_config_path("@production.toml")
        Path("production.toml")
        >>> resolve_config_path("config.yaml", root=Path("/project"))
        Path("/project/config.yaml")
    """
    if isinstance(config, str) and config.startswith("@"):
        config = config[1:]
    path = Path(config)
    if root and not path.is_absolute():
        path = root / path
    return path


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence.
    
    Args:
        base: Base dictionary
        override: Override dictionary (values take precedence)
        
    Returns:
        Merged dictionary
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "__") -> Dict[str, Any]:
    """Flatten a nested dictionary for env var style keys.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
        
    Example:
        >>> _flatten_dict({"model": {"id": "test"}})
        {"model__id": "test"}
    """
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _unflatten_dict(d: Dict[str, Any], sep: str = "__") -> Dict[str, Any]:
    """Unflatten a dictionary with separator-delimited keys.
    
    Args:
        d: Flat dictionary with separator in keys
        sep: Separator used in keys
        
    Returns:
        Nested dictionary
        
    Example:
        >>> _unflatten_dict({"model__id": "test"})
        {"model": {"id": "test"}}
    """
    result: Dict[str, Any] = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def _read_config_file(path: Path) -> Dict[str, Any]:
    """Read and parse a config file (TOML, YAML, or JSON).
    
    Args:
        path: Path to config file
        
    Returns:
        Parsed configuration dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
        
    Supported formats:
        - .toml, .tml: TOML format
        - .yaml, .yml: YAML format
        - .json: JSON format
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    
    try:
        if suffix in (".toml", ".tml"):
            data = tomllib.loads(raw)
        elif suffix in (".yaml", ".yml"):
            data = yaml.safe_load(raw)
        elif suffix == ".json":
            data = json.loads(raw)
        else:
            # Try YAML as fallback (it's a superset of JSON)
            data = yaml.safe_load(raw)
    except Exception as e:
        raise ValueError(f"Failed to parse config file {path}: {e}") from e
    
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping, got {type(data).__name__}")
    
    return data


def _apply_cli_overrides(
    config: ProjectConfig, 
    overrides: Dict[str, Any]
) -> ProjectConfig:
    """Apply CLI argument overrides to configuration.
    
    Args:
        config: Base configuration
        overrides: Dictionary of CLI overrides
        
    Returns:
        Updated configuration
    """
    if not overrides:
        return config
    
    # Convert config to dict
    data = config.model_dump()
    
    # Apply overrides with nested key support
    for key, value in overrides.items():
        if value is None:
            continue
            
        # Check for aliases (e.g., "lr" -> ("train", "lr"))
        if key in _CLI_ALIASES:
            section, field = _CLI_ALIASES[key]
            if section in data:
                data[section][field] = value
            continue
        
        # Handle nested keys like "model.id" or "train.lr"
        if "." in key:
            parts = key.split(".")
            current = data
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            # Find which section this key belongs to
            found = False
            for section_name, section_data in data.items():
                if isinstance(section_data, dict) and key in section_data:
                    data[section_name][key] = value
                    found = True
                    break
            
            # If not found in any section, check if it's an alias
            if not found and key in _CLI_ALIASES:
                section, field = _CLI_ALIASES[key]
                if section in data:
                    data[section][field] = value
    
    return ProjectConfig.model_validate(data)


def load_config(
    path: Optional[Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
    require: bool = False,
) -> ProjectConfig:
    """Load configuration with proper precedence.
    
    Precedence (highest to lowest):
    1. CLI arguments (cli_overrides)
    2. Config file (if path provided)
    3. Environment variables (MLXSMITH__*)
    4. Default values
    
    Args:
        path: Path to config file (optional)
        cli_overrides: Dictionary of CLI argument overrides
        require: If True, raise FileNotFoundError if config file missing
        
    Returns:
        Merged ProjectConfig
        
    Raises:
        FileNotFoundError: If require=True and config file not found
        ValidationError: If configuration is invalid
        
    Example:
        >>> cfg = load_config(Path("config.yaml"), cli_overrides={"model.id": "custom"})
    """
    # Start with defaults (lowest priority)
    defaults = ProjectConfig()
    
    # Layer 1: Environment variables
    try:
        env_settings = ProjectSettings()
        env_data = env_settings.model_dump()
    except ValidationError as e:
        # Log warning but continue with empty env data
        import warnings
        warnings.warn(f"Failed to parse environment variables: {e}")
        env_data = {}
    
    # Merge env vars into defaults
    merged = _deep_merge(defaults.model_dump(), env_data)
    
    # Layer 2: Config file (higher priority than env)
    if path is not None:
        if path.exists():
            try:
                file_data = _read_config_file(path)
                merged = _deep_merge(merged, file_data)
            except (ValueError, FileNotFoundError) as e:
                if require:
                    raise
                import warnings
                warnings.warn(f"Failed to load config file: {e}")
        elif require:
            raise FileNotFoundError(f"Config file not found: {path}")
    
    # Create config from merged data
    config = ProjectConfig.model_validate(merged)
    
    # Layer 3: CLI overrides (highest priority)
    if cli_overrides:
        config = _apply_cli_overrides(config, cli_overrides)
    
    return config


def get_config(
    config_path: Optional[Union[str, Path]] = None,
    root: Optional[Path] = None,
    **cli_kwargs: Any,
) -> ProjectConfig:
    """Convenience function to get configuration with CLI overrides.
    
    This is the recommended way to load configuration in CLI commands.
    
    Args:
        config_path: Path to config file (supports @prefix syntax)
        root: Project root for resolving relative paths
        **cli_kwargs: CLI argument overrides
        
    Returns:
        ProjectConfig with all overrides applied
        
    Example:
        >>> cfg = get_config("@production.toml", model_id="custom/model")
        >>> cfg = get_config("config.yaml", root=Path("/project"), train__lr=0.001)
    """
    path = None
    if config_path:
        path = resolve_config_path(config_path, root=root)
    
    # Filter out None values
    overrides = {k: v for k, v in cli_kwargs.items() if v is not None}
    
    return load_config(path, cli_overrides=overrides)


def dump_config(cfg: ProjectConfig, format: str = "yaml") -> str:
    """Dump configuration to string.
    
    Args:
        cfg: Configuration to dump
        format: Output format ("yaml", "json", "toml")
        
    Returns:
        Configuration string
        
    Raises:
        ValueError: If format is not supported
    """
    format = format.lower()
    
    if format == "yaml":
        return yaml.safe_dump(cfg.model_dump(), sort_keys=False)
    elif format == "json":
        return cfg.model_dump_json(indent=2)
    elif format in ("toml", "tml"):
        try:
            import tomli_w
            return tomli_w.dumps(cfg.model_dump())
        except ImportError:
            raise ValueError(
                "tomli_w is required for TOML output. "
                "Install with: pip install tomli_w"
            )
    else:
        raise ValueError(f"Unsupported format: {format}")


def write_default_config(path: Path, format: str = "yaml") -> None:
    """Write default configuration to file.
    
    Args:
        path: Output file path
        format: Output format (inferred from path if not specified)
    """
    cfg = ProjectConfig()
    
    # Infer format from path if not specified
    if format == "yaml" and path.suffix.lower() in (".json", ".toml", ".tml"):
        format = path.suffix.lower().lstrip(".")
    
    path.write_text(dump_config(cfg, format=format), encoding="utf-8")


def show_merged_config(
    config: ProjectConfig,
    show_sources: bool = False,
    sources: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a human-readable display of merged configuration.
    
    Args:
        config: Configuration to display
        show_sources: Whether to show which source each value came from
        sources: Dictionary mapping keys to their sources
        
    Returns:
        Formatted string representation
    """
    lines = ["# MLXSmith Configuration", ""]
    
    data = config.model_dump()
    
    for section_name, section_data in data.items():
        if not isinstance(section_data, dict):
            continue
            
        lines.append(f"[{section_name}]")
        
        for key, value in section_data.items():
            if show_sources and sources:
                source = sources.get(f"{section_name}.{key}", "default")
                lines.append(f"  {key} = {value!r}  # from: {source}")
            else:
                lines.append(f"  {key} = {value!r}")
        
        lines.append("")
    
    return "\n".join(lines)


def get_config_sources(
    config_path: Optional[Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[ProjectConfig, Dict[str, str]]:
    """Get configuration and track the source of each value.
    
    Args:
        config_path: Path to config file
        cli_overrides: CLI argument overrides
        
    Returns:
        Tuple of (config, sources_dict) where sources_dict maps
        "section.key" -> "default|env|file|cli"
    """
    sources: Dict[str, str] = {}
    
    # Start with defaults
    defaults = ProjectConfig()
    
    # Track default sources
    for section_name, section_data in defaults.model_dump().items():
        if isinstance(section_data, dict):
            for key in section_data.keys():
                sources[f"{section_name}.{key}"] = "default"
    
    # Apply env vars and track
    try:
        env_settings = ProjectSettings()
        env_data = env_settings.model_dump()
        for section_name, section_data in env_data.items():
            if isinstance(section_data, dict):
                for key in section_data.keys():
                    sources[f"{section_name}.{key}"] = "env"
    except ValidationError:
        env_data = {}
    
    merged = _deep_merge(defaults.model_dump(), env_data)
    
    # Apply file and track
    if config_path and config_path.exists():
        file_data = _read_config_file(config_path)
        for section_name, section_data in file_data.items():
            if isinstance(section_data, dict):
                for key in section_data.keys():
                    sources[f"{section_name}.{key}"] = "file"
        merged = _deep_merge(merged, file_data)
    
    config = ProjectConfig.model_validate(merged)
    
    # Apply CLI and track
    if cli_overrides:
        for key in cli_overrides.keys():
            if "." in key:
                sources[key] = "cli"
            else:
                # Find which section this key belongs to
                for section_name in config.model_dump().keys():
                    if key in config.model_dump()[section_name]:
                        sources[f"{section_name}.{key}"] = "cli"
                        break
        config = _apply_cli_overrides(config, cli_overrides)
    
    return config, sources
