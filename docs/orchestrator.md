# MLXSmith Multi-Process Orchestrator

The orchestrator provides PrimeIntellect-style multi-process coordination for RLM training. It splits the RLM loop into separate communicating processes to enable:

- **Non-blocking inference**: The inference server remains responsive during training
- **Hot weight reloading**: Update inference weights without restart
- **Asynchronous training**: Trainer updates weights in parallel with inference
- **Process isolation**: Better fault tolerance and resource management
- **Foundation for distribution**: Architecture supports future distributed training

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Orchestrator Daemon                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Job Queue   │  │ Weight Ptr  │  │ Process Monitor         │  │
│  │ Manager     │  │ Store       │  │ & Health Checks         │  │
│  └──────┬──────┘  └──────┬──────┘  └─────────────────────────┘  │
└─────────┼────────────────┼──────────────────────────────────────┘
          │                │
          │   Queues       │   Weight Pointers
          │   (IPC)        │   (Atomic Files)
          │                │
    ┌─────▼─────┐    ┌─────▼─────┐
    │ Inference │    │  Trainer  │
    │  Worker   │    │  Worker   │
    │  Process  │    │  Process  │
    └─────┬─────┘    └─────┬─────┘
          │                │
          │   HTTP API     │   Checkpoints
          │   /v1/chat     │   (LoRA Adapters)
          │   /internal/*  │
          │                │
    ┌─────▼────────────▼─────┐
    │    Client/Evaluator    │
    └────────────────────────┘
```

## Components

### 1. Orchestrator Daemon (`orchestrator/daemon.py`)

The central coordinator that:
- Manages message queues for IPC
- Spawns and monitors inference and trainer processes
- Handles process lifecycle (start, restart on failure, graceful shutdown)
- Forwards weight updates from trainer to inference
- Performs health checks

```python
from mlxsmith.orchestrator import OrchestratorDaemon, DaemonConfig

daemon = OrchestratorDaemon(config, project_cfg)
daemon.run()
```

### 2. Inference Worker (`orchestrator/inference_worker.py`)

Runs as a separate process with:
- OpenAI-compatible `/v1/chat/completions` endpoint
- Internal `/internal/rollout` endpoint for RLM
- `/internal/adapter/reload` for explicit weight reloading
- Hot-reload via weight pointer watching

```python
from mlxsmith.orchestrator import InferenceWorker, InferenceConfig

config = InferenceConfig(
    model_spec="mlx-community/Llama-3.2-3B-Instruct-4bit",
    host="0.0.0.0",
    port=8080,
    weights_dir=Path("runs/rlm_weights"),
    hot_reload=True,
)
worker = InferenceWorker(config)
worker.run()
```

### 3. Trainer Worker (`orchestrator/trainer_worker.py`)

Consumes training batches and:
- Runs forward/backward passes with PPO/GRPO
- Publishes adapter checkpoints
- Updates weight pointers for hot-reload
- Signals completion to orchestrator

```python
from mlxsmith.orchestrator import TrainerWorker, TrainerConfig

config = TrainerConfig(
    model_spec="mlx-community/Llama-3.2-3B-Instruct-4bit",
    base_model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    weights_dir=Path("runs/rlm_weights"),
    checkpoint_dir=Path("runs/rlm_checkpoints"),
)
worker = TrainerWorker(config)
worker.run()
```

### 4. Message Queue (`orchestrator/queue.py`)

Multi-process communication via:
- `ROLLOUT_REQUEST` / `ROLLOUT_RESPONSE`: Task generation
- `TRAIN_BATCH` / `TRAIN_COMPLETE`: Training coordination
- `WEIGHT_UPDATE` / `WEIGHT_ACK`: Weight synchronization
- `CHECKPOINT`: Checkpoint notifications
- `HEALTH_CHECK` / `SHUTDOWN`: Control messages

```python
from mlxsmith.orchestrator import MessageQueue, MessageType

queue = MessageQueue(maxsize=10000)
queue.start()

# Send message
queue.send("train_batches", MessageType.TRAIN_BATCH, {
    "rollouts": [...],
    "iteration": 5,
})

# Receive message
msg = queue.receive("train_batches", timeout=1.0)
```

### 5. Weight Pointer System (`rlm/weights.py`)

Atomic weight pointer updates for IPC:

```python
from mlxsmith.rlm.weights import WeightPointerStore, WeightPointerIPC

store = WeightPointerStore(Path("runs/rlm_weights"))

# Trainer updates weights
pointer = WeightPointerIPC(
    base_model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    adapter_path="runs/rlm_checkpoints/iter_0005",
    iteration=5,
    updated_at=now_ts(),
    version=5,
    name="trainer",
)
store.save(pointer)

# Inference worker watches for changes
pointer = store.load("inference", base_model)
if pointer.version > current_version:
    apply_adapter(pointer.adapter_path)
```

## Usage

### Command Line

Run orchestrated RLM:

```bash
# Single-process mode (legacy)
mlxsmith rlm --iterations 50

# Multi-process orchestrated mode
mlxsmith rlm --orchestrated --iterations 50

# With specific model
mlxsmith rlm --orchestrated --model mlx-community/Llama-3.2-3B-Instruct-4bit --iterations 100
```

### Programmatic

```python
from mlxsmith.rlm import run_rlm_orchestrated
from mlxsmith.config import ProjectConfig, load_config
from pathlib import Path

cfg = load_config(Path("mlxsmith.yaml"))

run_rlm_orchestrated(
    project_root=Path("."),
    cfg=cfg,
    model_spec="mlx-community/Llama-3.2-3B-Instruct-4bit",
    iterations=50,
    resume=False,
)
```

## Communication Flow

### RLM Iteration

1. **Task Generation** (Main Process)
   ```
   Main → LLM.generate_tasks() → tasks
   ```

2. **Rollout Collection** (Inference Worker via API)
   ```
   Main → HTTP POST /internal/rollout → Inference Worker
   Inference → generate_with_logprobs() → response
   Main ← response ← Inference Worker
   ```

3. **Training** (Trainer Worker via API in current impl)
   ```
   Main → train_on_rollouts() → adapter saved
   ```

4. **Weight Update** (Weight Pointer)
   ```
   Main → WeightPointerStore.save() → disk
   Inference → WeightPointerStore.load() (poll) → hot reload
   ```

5. **Evaluation** (via API)
   ```
   Main → run_eval() → score
   ```

### Hot Reload Mechanism

1. Trainer saves new adapter checkpoint
2. Trainer updates `trainer` weight pointer with new version
3. Main process optionally updates `inference` weight pointer
4. Inference worker polls weight pointer store
5. When version changes, inference reloads adapter via `/internal/adapter/reload`

## Message Types

| Type | Direction | Description |
|------|-----------|-------------|
| `ROLLOUT_REQUEST` | Daemon → Inference | Request rollout generation |
| `ROLLOUT_RESPONSE` | Inference → Daemon | Completed rollout with tokens/logprobs |
| `TRAIN_BATCH` | Daemon → Trainer | Batch of rollouts for training |
| `TRAIN_COMPLETE` | Trainer → Daemon | Training completion with metrics |
| `WEIGHT_UPDATE` | Trainer → Daemon → Inference | New adapter path available |
| `WEIGHT_ACK` | Inference → Daemon | Weight update acknowledged |
| `CHECKPOINT` | Trainer → Daemon | Checkpoint saved notification |
| `HEALTH_CHECK` | Daemon → Workers | Health status request |
| `SHUTDOWN` | Daemon → Workers | Graceful shutdown signal |

## Process Monitoring

The orchestrator monitors worker processes and restarts them on failure:

```python
# Configuration
daemon_config = DaemonConfig(
    max_restarts=3,        # Max restart attempts per worker
    restart_delay=5.0,     # Seconds between restarts
    health_check_interval=10.0,  # Health check frequency
)
```

## Future Enhancements

- **Async Training Queue**: Trainer consumes batches asynchronously
- **Distributed Queue**: Redis backend for multi-node training
- **Gradient Aggregation**: Multi-GPU support via gradient synchronization
- **Model Parallelism**: Split model across multiple workers
- **Dynamic Batching**: Adaptive batch sizing based on throughput

## Comparison with PrimeIntellect

| Feature | MLXSmith Orchestrator | PrimeIntellect |
|---------|----------------------|----------------|
| Process Model | Multi-process | Multi-process |
| Communication | Queues + HTTP API | Custom protocol |
| Weight Updates | Atomic file pointers | Distributed store |
| Hot Reload | Yes | Yes |
| Fault Tolerance | Process restart | Process restart |
| Scalability | Single-node | Multi-node |
