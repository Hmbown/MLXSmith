"""MLXSmith FastAPI server with OpenAI-compatible chat completions.

This module provides:
- OpenAI-compatible chat completions with streaming support
- Internal rollout endpoint (tokens + logprobs)
- Adapter hot-reload
- RLM state and history endpoints
- Web UI and RLM monitor (when enabled)
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .config import ProjectConfig
from .models import resolve_model_spec
from .llm.registry import get_llm_backend

# Import new API handlers
from .api.handlers import create_router, InternalAuthMiddleware
from .api.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    RolloutRequest,
    RolloutResponse,
    AdapterReloadRequest,
    AdapterReloadResponse,
)

# Re-export schemas for backward compatibility
__all__ = [
    "create_app",
    "ChatMessage",
    "ChatRequest", 
    "ChatResponse",
    "RolloutRequest",
    "RolloutResponse",
    "AdapterReloadRequest",
    "AdapterReloadResponse",
]


def _ui_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>mlxsmith serve</title>
  <style>
    :root {
      --bg: #f7f2ea;
      --ink: #231f20;
      --accent: #0b6e4f;
      --accent-2: #b44a1d;
      --card: #fff8ee;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 10% 10%, rgba(11,110,79,0.12), transparent 35%),
        radial-gradient(circle at 90% 20%, rgba(180,74,29,0.12), transparent 40%),
        var(--bg);
    }
    header {
      padding: 28px 32px;
      border-bottom: 1px solid rgba(35,31,32,0.1);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    .title {
      font-size: 20px;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }
    .wrap {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 24px;
      padding: 24px 32px 40px;
    }
    .card {
      background: var(--card);
      border: 1px solid rgba(35,31,32,0.08);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 6px 20px rgba(35,31,32,0.06);
    }
    h2 { margin: 0 0 12px; font-size: 18px; }
    textarea {
      width: 100%;
      min-height: 120px;
      border: 1px solid rgba(35,31,32,0.2);
      border-radius: 12px;
      padding: 12px;
      font: inherit;
      background: #fffdf9;
    }
    .row { display: flex; gap: 12px; margin-top: 10px; }
    button {
      border: 0;
      border-radius: 10px;
      padding: 10px 14px;
      font: inherit;
      cursor: pointer;
    }
    .btn-primary { background: var(--accent); color: #fff; }
    .btn-secondary { background: var(--accent-2); color: #fff; }
    .log {
      background: #14110f;
      color: #fdf7f0;
      border-radius: 12px;
      padding: 12px;
      height: 300px;
      overflow: auto;
      font-family: "Iosevka", "Menlo", monospace;
      font-size: 13px;
    }
    .muted { color: rgba(35,31,32,0.6); font-size: 12px; }
    @media (max-width: 900px) {
      .wrap { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="title">mlxsmith serve</div>
    <div class="muted">OpenAI-compatible API + RLM monitor</div>
  </header>
  <div class="wrap">
    <section class="card">
      <h2>Chat</h2>
      <textarea id="prompt" placeholder="Ask the model anything..."></textarea>
      <div class="row">
        <button class="btn-primary" id="send">Send</button>
        <button class="btn-secondary" id="stream">Stream</button>
        <a class="btn-secondary" href="/rlm/monitor" style="text-decoration:none;">RLM Monitor</a>
      </div>
      <div class="log" id="output"></div>
    </section>
    <section class="card">
      <h2>Status</h2>
      <div id="status" class="muted">Loading…</div>
      <div class="muted" style="margin-top:12px;">Tip: enable <code>serve.ui</code> in mlxsmith.yaml to keep this page on by default.</div>
    </section>
  </div>
  <script>
    const output = document.getElementById('output');
    const statusEl = document.getElementById('status');
    const promptEl = document.getElementById('prompt');
    const baseBody = () => ({
      messages: [{role: 'user', content: promptEl.value}],
      max_tokens: 256
    });

    async function refreshStatus() {
      try {
        const res = await fetch('/internal/rlm/state');
        if (!res.ok) return;
        const data = await res.json();
        statusEl.textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        statusEl.textContent = 'Status unavailable';
      }
    }
    refreshStatus();
    setInterval(refreshStatus, 5000);

    document.getElementById('send').onclick = async () => {
      output.textContent = '';
      const res = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(baseBody())
      });
      const data = await res.json();
      output.textContent = data.choices?.[0]?.message?.content || '';
    };

    document.getElementById('stream').onclick = async () => {
      output.textContent = '';
      const body = baseBody();
      body.stream = true;
      const res = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
      });
      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buf = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, {stream: true});
        const parts = buf.split('\\n\\n');
        buf = parts.pop() || '';
        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith('data:')) continue;
          const payload = line.replace('data:', '').trim();
          if (payload === '[DONE]') return;
          try {
            const obj = JSON.parse(payload);
            const delta = obj.choices?.[0]?.delta?.content || '';
            output.textContent += delta;
          } catch (e) {}
        }
      }
    };
  </script>
</body>
</html>
"""


def _monitor_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RLM Monitor</title>
  <style>
    body {
      margin: 0;
      font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
      background: #f4f6f1;
      color: #222;
    }
    header {
      padding: 24px 32px;
      background: #1f3b2c;
      color: #f4f6f1;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .wrap { padding: 24px 32px; display: grid; gap: 16px; }
    .card {
      background: #fff;
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }
    canvas { width: 100%; height: 220px; }
    pre { margin: 0; font-size: 12px; }
  </style>
</head>
<body>
  <header>
    <div>RLM Monitor</div>
    <a href="/" style="color:#f4f6f1;text-decoration:none;">Back to Serve</a>
  </header>
  <div class="wrap">
    <section class="card">
      <canvas id="chart" width="900" height="220"></canvas>
    </section>
    <section class="card">
      <pre id="state">Loading…</pre>
    </section>
  </div>
  <script>
    const canvas = document.getElementById('chart');
    const ctx = canvas.getContext('2d');
    const stateEl = document.getElementById('state');

    function draw(history) {
      ctx.clearRect(0,0,canvas.width,canvas.height);
      if (!history.length) return;
      const scores = history.map(h => h.adapter_score || 0);
      const max = Math.max(...scores, 1);
      const min = Math.min(...scores, 0);
      const pad = 20;
      const w = canvas.width - pad * 2;
      const h = canvas.height - pad * 2;
      ctx.strokeStyle = '#1f3b2c';
      ctx.lineWidth = 2;
      ctx.beginPath();
      scores.forEach((s, i) => {
        const x = pad + (i / Math.max(1, scores.length - 1)) * w;
        const y = pad + (1 - (s - min) / (max - min || 1)) * h;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    async function refresh() {
      try {
        const h = await fetch('/internal/rlm/history').then(r => r.json());
        draw(h);
        const s = await fetch('/internal/rlm/state').then(r => r.json());
        stateEl.textContent = JSON.stringify(s, null, 2);
      } catch (e) {
        stateEl.textContent = 'History unavailable';
      }
    }
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""


def create_app(model_spec: str, cfg: ProjectConfig) -> FastAPI:
    """Create and configure the FastAPI application.
    
    Args:
        model_spec: Model specification (path or HF repo ID)
        cfg: Project configuration
    
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="mlxsmith",
        description="MLXSmith API server for local LLM inference and RLM training",
        version="0.1.0",
    )
    
    # Load LLM backend
    llm = get_llm_backend(cfg.model.backend)
    base_model, adapter_path, _meta = resolve_model_spec(Path.cwd(), model_spec, cfg)
    llm.load(
        base_model,
        max_seq_len=cfg.model.max_seq_len,
        dtype=cfg.model.dtype,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if adapter_path:
        llm.apply_adapter(str(adapter_path))
    current_adapter = str(adapter_path) if adapter_path else None
    
    # Add authentication middleware for internal endpoints
    app.add_middleware(
        InternalAuthMiddleware,
        api_token=None,  # Set via MLXSMITH_API_TOKEN env var
        internal_prefix="/internal",
        public_paths=["/health", "/v1/chat/completions", "/docs", "/openapi.json"],
    )
    
    # Create and include the API router
    router = create_router(
        llm_backend=llm,
        base_model=base_model,
        current_adapter=current_adapter,
        cfg=cfg,
    )
    app.include_router(router)
    
    # Add UI routes if enabled
    if cfg.serve.ui:
        @app.get("/")
        def ui_root():
            return HTMLResponse(_ui_html())

        @app.get("/rlm/monitor")
        def ui_monitor():
            return HTMLResponse(_monitor_html())
    
    return app
