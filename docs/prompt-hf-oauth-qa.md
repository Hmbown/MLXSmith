# Prompt: HF OAuth + Model Pull QA (Qwen3 4B 4‑bit)

Use this prompt to ask another AI to validate Hugging Face auth + model pull.

```text
You are a QA lead with many sub‑agents. Validate Hugging Face OAuth/login
flow and model pull/convert/quantize for Qwen3‑4B 4‑bit with mlxsmith.

Required checks:
1) HF auth
   - Use `mlxsmith auth login` (or `huggingface-cli login` as fallback)
   - Confirm token is stored securely (no plaintext logs)
2) HF pull + convert + quantize
   - `mlxsmith pull Qwen/Qwen3-4B-Instruct-2507 --quantize --q-bits 4`
   - Verify output exists at `cache/mlx/Qwen__Qwen3-4B-Instruct-2507`
3) Serve model
   - `mlxsmith serve --model cache/mlx/Qwen__Qwen3-4B-Instruct-2507 --port 8080`
   - Run a sample chat completion (curl)
4) Document any missing flags for HF token or XET performance toggles

Deliverables:
- A step‑by‑step log of the flow.
- Confirmation that model was converted and served.
- Any fixes required in CLI or docs.
```
```
