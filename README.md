# nim-to-openai-proxy

An OpenAI-compatible proxy in front of [NVIDIA NIM](https://build.nvidia.com), running on Cloudflare Workers. Point any OpenAI-client-compatible tool (Janitor AI, SillyTavern, etc.) at this proxy's `/v1` endpoint and it will transparently forward to NIM-hosted models, with automatic fallback, reasoning/thinking normalization, and recovery from a known upstream tool-call leak bug.

## Features

- **OpenAI-compatible endpoint** — `/v1/chat/completions`, streaming and non-streaming, plus `/v1/models`.
- **Model aliasing** — friendly names like `gpt-4`, `gpt-3.5-turbo`, `mistral`, `glm-5.3`, etc. map to specific NIM-hosted backend models (see `MODEL_MAPPING` in `src/index.js`).
- **Automatic fallback chain** — if the mapped model fails, the proxy retries against a fixed fallback list (`FALLBACK_MODELS`), with per-model cooldowns after auth/rate-limit errors.
- **Reasoning/"thinking" normalization** — each backend model has its own way of requesting and returning reasoning output (structured fields, delimiter tags, budgets, effort levels). `src/reasoning.js` normalizes all of this into a consistent `reasoning`/`reasoning_content` field, or inline `<thinking>` tags if requested.
- **Tool-call leak recovery** — some backend models occasionally leak native tool calls into `content` as raw `<tool_call>{...}</tool_call>` text instead of NIM's structured `tool_calls` field. `src/tools.js` detects and repairs this for both streaming and non-streaming responses.
- **Startup model validation** — on Worker isolate init, checks that every mapped model still exists in the live NIM catalog, and optionally posts a Discord alert if any are missing.
- **No sleep, no cold start** — runs on Cloudflare Workers' free tier (100,000 requests/day), unlike the original Render-based version.

## Requirements

- A Cloudflare account (Workers free tier is sufficient)
- An NVIDIA API key from [build.nvidia.com](https://build.nvidia.com)
- Node.js + npm (for local development only)

## Deploying

### 1. Push to GitHub

Everything the Worker needs lives under `src/`, plus `wrangler.jsonc` and `package.json` at the repo root.

### 2. Connect the repo to Cloudflare

1. Go to [dash.cloudflare.com](https://dash.cloudflare.com) → **Workers & Pages** → **Create** → **Import a repository**.
2. Sign in with GitHub and select the repo.
3. Cloudflare reads `wrangler.jsonc` automatically and deploys — no build command needed.

This enables **Workers Builds**: every push to your main branch auto-redeploys.

### 3. Set your secrets

In the Cloudflare dashboard: your Worker → **Settings** → **Variables and Secrets** → **Add**. Add as type **Secret**:

| Name | Value |
|---|---|
| `NIM_API_KEY` | Your NVIDIA API key from build.nvidia.com |
| `CLIENT_AUTH_KEY` | An auth key of your choosing — clients must send this as a Bearer token |

Optional environment variables:

| Name | Purpose |
|---|---|
| `SHOW_REASONING` | `true` to include model reasoning in responses |
| `ENABLE_THINKING_MODE` | `true` to send thinking/reasoning params to supported models by default |
| `DISCORD_WEBHOOK_URL` | Webhook URL for model-validation-failure alerts |
| `SKIP_VALIDATION` | `true` to disable the startup model catalog check |
| `NIM_API_BASE` | Override the NIM API base URL (defaults to `https://nvidia.com`) |
| `REQUEST_TIMEOUT_MS` | Timeout for non-reasoning requests (default 180000) |
| `REASONING_REQUEST_TIMEOUT_MS` | Timeout for reasoning requests (default 480000) |
| `ACCESS_DENIED_COOLDOWN_MS` | Cooldown after a 403 from a model (default 300000) |
| `RATE_LIMIT_COOLDOWN_MS` | Cooldown after a 429 from a model (default 30000) |

Changes to secrets/variables apply on the next request — no redeploy needed.

### 4. Point your client at the proxy

Your proxy will be live at:

```
https://nim-to-openai-proxy.<your-subdomain>.workers.dev
```

Set your client's API base URL to that address with `/v1` on the end, and use your `CLIENT_AUTH_KEY` as the API key / Bearer token.

## Local development

```bash
npm install
npx wrangler dev
```

Runs the proxy locally with live reload, using the same engine Cloudflare runs in production.

Useful scripts (see `package.json`):

```bash
npm run dev      # wrangler dev
npm run deploy   # wrangler deploy
npm run tail     # wrangler tail (live logs)
```

## API

### `GET /`
Simple health/landing page.

### `GET /health`
Returns `{ status: "ok", version: "..." }`.

### `GET /v1/models`
Lists available model aliases. Pass `?live=true` to also cross-check each alias's backend model against NIM's live catalog (requires `NIM_API_KEY`).

### `POST /v1/chat/completions`
OpenAI-compatible chat completions endpoint. Requires `Authorization: Bearer <CLIENT_AUTH_KEY>`. Supports `stream: true/false`, standard OpenAI fields (`messages`, `temperature`, `max_tokens`, `tools`, etc.), plus:

- `reasoning_effort` — `"off"`, `"on"`, or a model-specific effort level (e.g. `"low"`, `"high"`, `"max"`), to control thinking on a per-request basis.
- `x-reasoning-format: inline` header — returns reasoning as inline `<thinking>...</thinking>` tags in `content` instead of a separate field.

`max_tokens` is capped at 65536 regardless of what's requested.

## Model aliases

Aliases (left) map to specific NIM-hosted backend models (right) — see `MODEL_MAPPING` in `src/index.js` for the current list, e.g. `gpt-4` → `nvidia/nemotron-3-ultra-550b-a55b`, `claude-3-opus` → `google/diffusiongemma-26b-a4b-it`, `mistral` → `mistralai/mistral-large-2-instruct`, and so on. Any alias not in the mapping falls back to the default model.

If a request's primary model fails, the proxy retries in order through `FALLBACK_MODELS`, skipping any model currently in cooldown from a prior 403/429.

## Project structure

```
wrangler.jsonc       Cloudflare Workers config
package.json         Scripts and dependencies (wrangler only)
src/
  index.js           Request routing, auth, fallback logic, SSE handling
  reasoning.js        Per-model reasoning/thinking payload construction and normalization
  tools.js            Recovery for leaked <tool_call> tags in model output
```

## Notes

- `polyfill.js` from the original Render-based version was not ported — nothing in this codebase imports it.
- The model-validation check runs once per Cloudflare isolate rather than once per server process, so it may run more often under low traffic than it did on Render. This is harmless but can make `DISCORD_WEBHOOK_URL` alerts fire slightly more frequently if any models are down.
