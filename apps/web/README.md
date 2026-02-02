# MLXSmith Web App

A Next.js 14+ web interface for MLXSmith - Train, serve, and chat with MLX models on Apple Silicon.

## Features

- **Models Management**: Browse, pull, and manage local MLX models
- **Chat Interface**: Interactive chat with streaming support
- **Training Dashboard**: RLM training monitor with real-time charts
- **Server Control**: Start/stop the API server with configuration
- **Settings**: HF token management, theme toggle, and preferences

## Tech Stack

- Next.js 14+ with App Router
- TypeScript
- Tailwind CSS + shadcn/ui components
- React Query for API calls
- Recharts for data visualization
- Zustand for state management

## Getting Started

### Prerequisites

- Node.js 18+
- MLXSmith API server running

### Installation

```bash
# Install dependencies
npm install

# Copy environment variables
cp .env.example .env.local

# Update .env.local with your MLXSmith API URL
# NEXT_PUBLIC_API_URL=http://localhost:8080

# Run development server
npm run dev
```

### Build for Production

```bash
npm run build
npm start
```

## Project Structure

```
app/                    # Next.js App Router
  models/              # Models page
  chat/                # Chat interface page
  train/               # Training dashboard page
  serve/               # Server control page
  settings/            # Settings page
components/            # React components
  ui/                  # shadcn/ui components
  layout/              # Layout components
  chat/                # Chat-specific components
  models/              # Model-related components
  train/               # Training-related components
hooks/                 # Custom React Query hooks
lib/                   # Utility functions and API client
stores/                # Zustand state stores
types/                 # TypeScript type definitions
```

## API Integration

The web app communicates with the MLXSmith API server. Ensure the server is running at the configured URL (default: `http://localhost:8080`).

### Key Endpoints

- `GET /internal/models/list` - List cached models
- `POST /internal/models/pull` - Pull new models
- `POST /v1/chat/completions` - Chat completions (streaming supported)
- `GET /internal/rlm/state` - RLM training state
- `GET /internal/rlm/history` - RLM training history

## License

Same as MLXSmith project
