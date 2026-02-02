"""Pydantic models for MLXSmith configuration sections."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Any

from pydantic import BaseModel, Field, field_validator

AccelBackendName = Literal["none"]


class ModelConfig(BaseModel):
    """Model configuration for MLXSmith."""
    
    id: str = Field(
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        description="HF model id or local path",
    )
    backend: str = Field(default="mlx-lm", description="LLM backend implementation")
    dtype: str = Field(default="bf16")
    quantization: str = Field(default="none", description="none|q4|q6|q8")
    max_seq_len: int = Field(default=8192)
    trust_remote_code: bool = Field(default=False)
    use_chat_template: bool = Field(default=True)
    
    @field_validator("quantization")
    @classmethod
    def validate_quantization(cls, v: str) -> str:
        allowed = ["none", "q4", "q6", "q8"]
        if v not in allowed:
            raise ValueError(f"quantization must be one of {allowed}, got {v}")
        return v


class AccelConfig(BaseModel):
    """Acceleration backend configuration."""
    
    backend: AccelBackendName = Field(default="none")
    compile_cache: str = Field(default="cache/compiled_kernels")


class TrainConfig(BaseModel):
    """Training configuration for SFT and other training modes."""
    
    seed: int = 1337
    batch_size: int = 1
    grad_accum: int = 8
    lr: float = 2e-4
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    optimizer_kwargs: Dict[str, Any] = Field(default_factory=dict)
    iters: int = 1000
    save_every: int = 100
    eval_every: int = 100
    log_every: int = 10
    train_on_prompt: bool = False
    max_grad_norm: float = 1.0

    @field_validator("lr", "weight_decay")
    @classmethod
    def validate_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("value must be non-negative")
        return v

    @field_validator("optimizer")
    @classmethod
    def normalize_optimizer(cls, v: str) -> str:
        return v.strip().lower()


class LoraConfig(BaseModel):
    """LoRA/DoRA adapter configuration."""
    
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj", "o_proj"])
    num_layers: int = Field(default=0, description="0 = all layers (MLX-LM LoRA)")
    scale: Optional[float] = Field(default=None, description="Optional LoRA scale (overrides alpha/r)")
    fine_tune_type: Literal["lora", "dora", "full"] = "lora"
    
    @field_validator("r", "alpha")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("value must be positive")
        return v
    
    @field_validator("dropout")
    @classmethod
    def validate_dropout(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("dropout must be between 0 and 1")
        return v


class PrefConfig(BaseModel):
    """Preference tuning configuration (DPO variants)."""
    
    algo: Literal["dpo", "orpo", "grpo"] = "dpo"
    loss_type: Literal["dpo", "cpo", "orpo", "ipo", "hinge", "simpo", "tdpo"] = "dpo"
    beta: float = 0.1
    kl_coeff: float = 0.0
    delta: float = 0.0
    reference_model: Optional[str] = None


class KtoConfig(BaseModel):
    """Kahneman-Tversky Optimization configuration (binary feedback)."""

    beta: float = 0.1
    gain_power: float = 1.0
    loss_power: float = 1.0
    loss_aversion: float = 1.0
    reference_point: float = 0.0
    reference_model: Optional[str] = None


class RftConfig(BaseModel):
    """Reinforcement fine-tuning configuration."""
    
    algo: Literal["grpo"] = "grpo"
    loss_type: Literal["grpo", "dr_grpo", "dapo"] = "grpo"
    rollouts: int = 8
    kl_coeff: float = 0.02
    max_steps_per_task: int = 1
    temperature: float = 0.8
    max_new_tokens: int = 256
    normalize_advantage: bool = True
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
    token_level_loss: bool = False
    reference_model: Optional[str] = None


class ServeConfig(BaseModel):
    """Server configuration for inference serving."""
    
    api: Literal["openai", "simple"] = "openai"
    host: str = "0.0.0.0"
    port: int = 8080
    ui: bool = False
    stream: bool = True
    
    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("port must be between 1 and 65535")
        return v


class InferConfig(BaseModel):
    """Inference configuration for generation."""
    
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: Optional[int] = None
    
    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if v < 0:
            raise ValueError("temperature must be non-negative")
        return v
    
    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("top_p must be between 0 and 1")
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    file: Optional[str] = None
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# CLI argument aliases for field mapping
CLI_ALIASES: dict[str, tuple[str, ...]] = {
    "learning_rate": ("train", "lr"),
    "lr": ("train", "lr"),
    "batch_size": ("train", "batch_size"),
    "iters": ("train", "iters"),
    "optimizer": ("train", "optimizer"),
    "model_id": ("model", "id"),
    "accel_backend": ("accel", "backend"),
    "host": ("serve", "host"),
    "port": ("serve", "port"),
    "ui": ("serve", "ui"),
    "iterations": ("rlm", "iterations"),
    "rollouts": ("rft", "rollouts"),
    "algo": ("pref", "algo"),
}


class RlmConfig(BaseModel):
    """Recursive Language Model (RLM) loop configuration."""
    
    iterations: int = 50  # 0 = infinite
    sleep_between: int = 0  # seconds
    tasks_per_iter: int = 80
    rollouts_per_task: int = 8
    attempts_per_task: int = 3
    corpus_max: int = 8000
    mix_old_ratio: float = 0.4
    hard_ratio: float = 0.6
    mutations_per_task: int = 1
    use_task_mutation: bool = True
    gating: Literal["strict", "threshold", "ema"] = "strict"
    gating_threshold: float = 0.0
    gating_ema_alpha: float = 0.2
    infer_staleness: int = 0  # iterations of staleness allowed between trainer and inference weights
    require_recursion: bool = False
    task_domains: List[str] = Field(default_factory=lambda: ["strings", "arrays", "math", "dp", "graphs"])
    benchmark_suite: str = "eval/suites/rlm_bench.yaml"
    holdout_suite: Optional[str] = "eval/suites/rlm_holdout.yaml"
    verifier_timeout_s: int = 30
    verifier_backend: Literal["pytest", "docker"] = "pytest"
    docker_image: str = "python:3.11-slim"
    docker_memory_mb: int = 512
    docker_cpus: float = 1.0
    docker_pids: int = 128
    task_gen_max_new_tokens: int = 1024
    similarity_threshold: float = 0.85
    min_task_desc_len: int = 10
    min_task_asserts: int = 2
    max_task_prompt_len: int = 2000
    min_task_tests_len: int = 20
    max_task_tests_len: int = 8000
    blocked_task_patterns: List[str] = Field(
        default_factory=lambda: [
            r"\bsubprocess\b",
            r"\bos\.system\b",
            r"\bshutil\.rmtree\b",
            r"\brm\s+-rf\b",
            r"\brequests\b",
            r"\burllib\b",
            r"\bsocket\b",
            r"\bhttp[s]?://",
            r"\bpip\s+install\b",
            r"\bapt-get\b",
            r"\bbrew\s+install\b",
        ]
    )
    
    @field_validator("mix_old_ratio", "hard_ratio", "gating_ema_alpha", "similarity_threshold")
    @classmethod
    def validate_ratio(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("ratio must be between 0 and 1")
        return v


class ProjectConfig(BaseModel):
    """Root configuration model containing all sections."""
    
    model: ModelConfig = Field(default_factory=ModelConfig)
    accel: AccelConfig = Field(default_factory=AccelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    pref: PrefConfig = Field(default_factory=PrefConfig)
    kto: KtoConfig = Field(default_factory=KtoConfig)
    rft: RftConfig = Field(default_factory=RftConfig)
    infer: InferConfig = Field(default_factory=InferConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)
    rlm: RlmConfig = Field(default_factory=RlmConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
    
    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        import yaml
        return yaml.safe_dump(self.model_dump(), sort_keys=False)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON string."""
        return self.model_dump_json(indent=indent)
