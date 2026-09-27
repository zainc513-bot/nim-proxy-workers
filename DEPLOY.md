# Deploying this to Cloudflare Workers

This is the same proxy, rewritten to run on Cloudflare Workers instead of
Render. Same env vars, same `/v1/chat/completions` endpoint, same model
aliases — just no sleep, no cold start, and no card required (Workers'
free tier: 100,000 requests/day).

## 1. Push this to GitHub

Replace the contents of your existing fork with these files (or make a new
repo — either works). The old `server.js`, `package.json`, `polyfill.js`
are no longer used; everything now lives under `src/`.

> `polyfill.js` wasn't ported — nothing in the codebase actually imports it,
> so it looks like dead code from the original repo. If something breaks
> because of that, let me know and I'll dig into why it was there.

## 2. Connect the repo to Cloudflare

1. Go to [dash.cloudflare.com](https://dash.cloudflare.com) → **Workers & Pages** → **Create** → **Import a repository**.
2. Sign in with GitHub, pick this repo.
3. Cloudflare reads `wrangler.jsonc` automatically and deploys — no build command to configure.

This is "Workers Builds" — from here on, every push to your main branch
auto-redeploys, exactly like Render did.

## 3. Set your secrets

In the Cloudflare dashboard: your Worker → **Settings** → **Variables and
Secrets** → **Add**. Add these as type **Secret** (encrypted, not shown
again in the UI):

| Name | Value |
|---|---|
| `NIM_API_KEY` | your NVIDIA API key from build.nvidia.com |
| `CLIENT_AUTH_KEY` | the same auth key you used on Render — no need to regenerate it |

Optional, same as before — add only the ones you were already using:

| Name | Value |
|---|---|
| `SHOW_REASONING` | `true` to show model reasoning in responses |
| `ENABLE_THINKING_MODE` | `true` to send thinking params to supported models |
| `DISCORD_WEBHOOK_URL` | webhook URL for model-validation-failure alerts |
| `SKIP_VALIDATION` | `true` to disable the startup model check |

Changes apply on the next request — no redeploy needed, same as Render.

## 4. Point Janitor AI / SillyTavern at the new URL

Your proxy is now at `https://nim-to-openai-proxy.<your-subdomain>.workers.dev`
(Cloudflare shows you the exact URL after the first deploy). Update the API
base URL in your client to that, keeping `/v1` on the end — everything else
(auth header, model aliases, streaming) is unchanged.

## What's different from the Render version

- **No sleep, ever.** Workers doesn't spin down on idle, so there's nothing
  to keep warm and no cold-start delay.
- **Model-validation check timing:** the original ran once per server
  process (i.e., once per Render restart/deploy). Here it runs once per
  Cloudflare *isolate*, which can be recycled more often under low traffic.
  Harmless, but if you have `DISCORD_WEBHOOK_URL` set, you might see the
  validation-failure alert (if any models are down) fire a bit more often
  than before.
- Everything else — model mapping, fallback chain, reasoning-tag handling,
  tool-call leak recovery, streaming — works identically. I tested the
  auth check, the fallback/cooldown logic, both reasoning-display modes,
  and the SSE streaming transform against a simulated NIM backend before
  handing this over.

## Local testing (optional)

If you ever want to try changes before pushing:

```bash
npm install
npx wrangler dev
```

This runs the proxy locally with live reload, using the same engine
Cloudflare runs in production.
