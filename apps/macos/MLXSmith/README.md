# MLXSmith macOS App

A native SwiftUI macOS interface for MLXSmith - providing seamless model management, training, and inference capabilities.

## Features

### Sidebar Navigation
- **Discover** - Browse and pull models from HuggingFace
- **Downloaded** - Manage local MLX models
- **Chat** - Interactive inference with streaming responses
- **Train** - Monitor RLM training with live charts
- **Serve** - Control API server with request logs
- **Settings** - Configure HF token, paths, and preferences

## Requirements

- macOS 14.0+
- Xcode 15.0+
- MLXSmith Python backend running

## Architecture

### MVVM Pattern
```
Views/          - SwiftUI view components
Models/         - Data models (HFModel, LocalModel, etc.)
Services/       - API clients (MLXSmithAPI, HuggingFaceAPI)
Utils/          - Keychain, formatters, utilities
```

### Key Components

1. **MLXSmithAPI** - Async/await client for REST API
   - `listLocalModels()` - Get cached models
   - `sendChatMessage()` - Streaming chat completions
   - `getRLMState()` - Training state
   - `pullModel()` - Download from HF

2. **KeychainManager** - Secure token storage
   - HF token persistence
   - API key management

3. **ServerProcessManager** - Local server control
   - Start/stop MLXSmith server
   - Log streaming

## Building

1. Open `MLXSmith.xcodeproj` in Xcode
2. Select your team in Signing & Capabilities
3. Build and run (⌘R)

## Configuration

Default settings (change in Settings tab):
- Server host: `localhost`
- Server port: `8080`
- Project root: `~`
- Cache: `~/.mlxsmith/cache`

## API Endpoints Used

- `GET /health` - Health check
- `GET /internal/models/list` - Local models
- `POST /internal/models/pull` - Download model
- `POST /v1/chat/completions` - Chat (streaming)
- `GET /internal/rlm/state` - Training state
- `GET /internal/rlm/history` - Training metrics

## Design Philosophy

- Native-first: Pure SwiftUI, not Electron
- MLX-native: Deep Apple Silicon integration
- Local-first: Everything on-device
- Clean UI: Following Apple's HIG
