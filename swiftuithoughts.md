# MLXSmith SwiftUI App - Design Thoughts

## Vision
A native macOS/iOS app for MLX model training and inference, inspired by Prime Intellect's clean interface but focused specifically on the Apple Silicon + MLX ecosystem.

## Core Philosophy
- **Native-first**: SwiftUI, not Electron wrapper
- **MLX-native**: Deep integration with mlx-lm ecosystem
- **HuggingFace connected**: OAuth + seamless model discovery
- **Local-first**: Everything runs on-device, no cloud required

## UI Structure (Prime Intellect-inspired)

### Sidebar Navigation
```
┌─────────────────────────────────────┐
│  MLXSmith                           │
│  ─────────                          │
│  🔍 Discover    (HF model browser)  │
│  📥 Downloaded  (Local cache)       │
│  🏋️ Train       (Fine-tuning)       │
│  💬 Chat        (Inference UI)      │
│  ⚙️ Serve       (API server)        │
│  ─────────                          │
│  📊 Dashboard   (Metrics/stats)     │
│  🔧 Settings    (Config)            │
└─────────────────────────────────────┘
```

### 1. Discover Tab (HuggingFace Integration)
- OAuth login with HuggingFace
- Browse MLX-compatible models
- Filter by: architecture, size, quantization, task
- One-click "Pull to MLX" (uses `mlxsmith pull` under hood)
- Preview model card, tags, downloads

### 2. Downloaded Models
- Local cache management
- Visual cards showing:
  - Model name + architecture icon
  - Size (GB)
  - Quantization level (4-bit, 8-bit, etc.)
  - Last used
  - Quick actions: Chat, Train, Serve, Delete

### 3. Train Tab (MLXSmith Core)
**Three-stage pipeline UI:**

**Stage 1: SFT (Supervised Fine-Tuning)**
- Dataset upload (JSONL, CSV)
- Preview samples
- LoRA config: rank, alpha, dropout, target_modules
- Training params: lr, batch_size, steps
- Real-time loss chart
- Save adapters automatically

**Stage 2: Preference Tuning**
- DPO or ORPO selection
- Preference dataset format helper
- KL penalty config
- Continue from SFT adapter

**Stage 3: Verifier-Driven RL (GRPO)**
- Environment selector (coding, math, custom)
- Verifier plugin UI (regex, code execution, LLM-as-judge)
- Rollout visualization (show generated responses)
- Reward trajectory chart
- Sandboxed execution status

### 4. Chat Tab (Inference)
- Model selector dropdown
- Chat interface (similar to ChatGPT/Claude apps)
- System prompt editor
- Parameter sidebar (temperature, top_p, max_tokens)
- Token speed display (tok/s)
- Export conversation

### 5. Serve Tab (API Server)
- Toggle server on/off
- Port configuration
- OpenAI-compatible endpoint display
- Request/response logs
- Connection status indicator

## Technical Architecture

### Swift ↔ Python Bridge
```
SwiftUI App
    ↓
Process wrapper (run mlxsmith commands)
    ↓
MLXCTL Python backend
    ↓
MLX + Metal Performance Shaders
```

Options:
1. **Process wrapper** (easiest): Shell out to `mlxsmith` CLI
2. **PythonKit** (native-ish): Embed Python runtime
3. **XPC service** (robust): Separate process, Swift ↔ Python via XPC

Recommendation: Start with #1 (process wrapper), migrate to #3 for production.

### HuggingFace OAuth Flow
1. App opens browser to HF OAuth authorization
2. User approves, redirect to custom URL scheme (`mlxsmith://auth`)
3. App receives token, stores in Keychain
4. Use HF API for:
   - Model search/filter
   - Model card metadata
   - Download URLs for `mlxsmith pull`

### Data Flow
```
[HF Browse] → Select model → [mlxsmith pull] → [Local Cache]
                                          ↓
[Training Data] → [mlxsmith sft] → [Adapter] → [mlxsmith serve/chat]
```

## SwiftUI Specifics

### macOS Target (Primary)
- Use `.navigationSplitView()` for sidebar + detail
- Toolbar with primary actions
- Draggable window, standard macOS chrome
- Menu bar extra for quick server toggle?

### iPadOS (Secondary)
- Same codebase, adaptive layout
- Touch-optimized controls
- Files app integration for datasets

### Key SwiftUI Components Needed
- `Chart` (for training loss curves)
- `ProgressView` (for downloads, training steps)
- `TextEditor` (for dataset preview, chat)
- `Form` (for training configs)
- `LazyVGrid` (for model cards)

## MLXSmith Integration Points

### Commands to Wrap
```bash
# Discovery
mlxsmith search <query>  # Need to add this command

# Download
mlxsmith pull <model_id> [--quantize --q-bits 4]

# Train
mlxsmith sft --model <path> --data <path> [--adapter-out <path>]
mlxsmith pref --model <path> --data <path>
mlxsmith rft --model <path> --env <path> --verifier <path>

# Serve
mlxsmith serve --model <path> --port 8080

# Chat (CLI mode)
mlxsmith chat --model <path> [--adapter <path>]
```

### Real-time Output
Training commands stream progress. SwiftUI needs:
- Parse stdout for progress updates
- Live loss values for Charts
- Step/total step counter
- ETA calculation

### Async/Await Pattern
```swift
func trainModel(config: TrainingConfig) async throws -> Adapter {
    let process = Process()
    process.executableURL = mlxsmithPath
    process.arguments = ["sft", "--model", config.modelPath, ...]
    
    // Stream output, update UI
    for try await line in process.output {
        updateProgress(line)
    }
    
    return Adapter(path: config.outputPath)
}
```

## Design Details (Prime Intellect Inspiration)

### Visual Language
- Clean, minimal interface
- Monospace for code/model names
- Subtle gradients (Apple-style)
- Dark mode first, light mode supported
- SF Symbols for icons

### Animations
- Smooth transitions between tabs
- Progress rings for downloads/training
- Pulse animation for active server
- Typing indicator in chat

### Empty States
- "No models downloaded yet" → CTA to Discover
- "No training data" → Helper to format datasets
- "Server offline" → Big toggle to start

## MVP Features (Week 1-2)
1. Sidebar navigation shell
2. HuggingFace OAuth + model browser
3. `mlxsmith pull` integration with progress
4. Downloaded models grid view
5. Basic chat interface

## V1 Features (Month 1)
1. SFT training UI with live charts
2. Adapter management
3. API server toggle + logs
4. Settings/preferences

## V2 Features (Month 2-3)
1. Preference tuning (DPO/ORPO)
2. Verifier-driven RL interface
3. iPadOS support
4. Cloud sync for adapters (optional)

## Open Questions

1. **Model Discovery**: Do we need a backend service for MLX-compatible model index, or scrape HF tags?
2. **Dataset Handling**: How to validate/upload training data? Drag-drop JSONL?
3. **Verifier Sandbox**: How to show verifier results in UI? Rich text? Code blocks?
4. **ZMLX Integration**: Should ZMLX kernels auto-apply in SwiftUI app? Preference toggle?
5. **Distribution**: Mac App Store (sandboxed) or direct download (full filesystem access)?

## Resources

### Prime Intellect References
- Clean sidebar nav
- Model cards with key stats
- Training progress visualization
- Chat interface simplicity

### MLX-Specific
- Check mlx-swift for native Swift bindings (may simplify bridge)
- Metal Performance Shaders debugging tools

### HuggingFace
- OAuth docs: https://huggingface.co/docs/hub/oauth
- API for model search: https://huggingface.co/docs/hub/api

---

## Next Steps
1. Create SwiftUI project scaffold
2. Implement HF OAuth flow
3. Build model browser (mock data first)
4. Integrate `mlxsmith pull` with progress bar
5. Chat interface with local model

*Last updated: 2026-02-01*
