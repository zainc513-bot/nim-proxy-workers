import { getReasoningPayload, resolveEffectiveThinking, StreamNormalizer, normalizeNonStreamChoice } from './reasoning.js';
import { extractLeakedToolCalls, ToolCallStreamRecovery } from './tools.js';

const MAX_TOKENS_LIMIT = 65536;
const VALIDATION_TIMEOUT_MS = 15000;
const MAX_BUFFER_SIZE = 1024 * 1024;

// Default NIM base URL — must match what server.js used
// ('https://integrate.api.nvidia.com/v1'). Everything below assumes this
// already includes the /v1 prefix, e.g. nimBase(env) + '/chat/completions'.
const DEFAULT_NIM_API_BASE = 'https://integrate.api.nvidia.com/v1';

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/nemotron-3-super-120b-a12b',
  'gpt-4': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-3.5': 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning',
  'gpt-4-turbo': 'moonshotai/kimi-k3',
  'claude-3-opus': 'google/diffusiongemma-26b-a4b-it',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'nvidia/llama-3.1-nemotron-70b-instruct',
  'gemini-turbo': 'nvidia/llama3-chatqa-1.5-70b',
  'gpt-3.5o': 'nvidia/nemotron-3.5-lightning-30b-a3b',
  'gpt-4-flash': 'deepseek-ai/deepseek-v4.1-flash',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro-0813',
  'mistral': 'mistralai/mistral-large-2-instruct',
  'mistral-turbo': 'nv-mistralai/mistral-nemo-12b-instruct',
  'mistral-pro': 'mistralai/mistral-7b-instruct-v0.3',
  'mistral-nemo': 'mistralai/mistral-nemotron',
  'mistral-fast': 'nvidia/mistral-nemo-minitron-8b-8k-instruct',
  'google-light': 'google/gemma-4-31b-it',
  'google-lightest': 'meta/muse-glimmer-30b',
  'google-lighter': 'poolside/laguna-xs-2.1',
  'glm-5.3': 'z-ai/glm-5.3',
  'glm-flash': 'z-ai/glm-5-3-flash'
};

const DEFAULT_MODEL = 'google/diffusiongemma-26b-a4b-it';

const FALLBACK_MODELS = [
  'google/diffusiongemma-26b-a4b-it',
  'google/gemma-4-31b-it',
  'mistralai/mistral-nemotron',
  'nvidia/nemotron-3-super-120b-a12b'
];

// In-memory per-isolate cooldown map. NOT shared across isolates/regions —
// this is a best-effort optimization, same caveat as the original
// process-local Map on Render/Railway (which also didn't share state across
// horizontally-scaled instances).
const modelCooldowns = new Map();
let didIsolateInit = false;

function isInCooldown(model) {
  const until = modelCooldowns.get(model);
  return typeof until === 'number' && Date.now() < until;
}

function setCooldown(model, ms) {
  modelCooldowns.set(model, Date.now() + ms);
}

function corsHeaders(extra = {}) {
  return {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-reasoning-format',
    'Access-Control-Max-Age': '86400',
    ...extra
  };
}

function jsonResponse(body, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: corsHeaders({ 'Content-Type': 'application/json', ...extraHeaders })
  });
}

function errorResponse(message, type, code) {
  return jsonResponse({ error: { message, type, code } }, code);
}

function extractBearerToken(authHeader) {
  if (!authHeader || typeof authHeader !== 'string') return null;
  const trimmed = authHeader.trim();
  if (!trimmed.startsWith('Bearer ')) return null;
  const token = trimmed.slice('Bearer '.length).trim();
  return token || null;
}

// Constant-time comparison via the Workers-runtime crypto.subtle extension.
// Falls back to `false` (never silently succeeds) if the API is ever
// unavailable in a given runtime.
async function safeTimingEqual(a, b) {
  const encoder = new TextEncoder();
  const aBytes = encoder.encode(a || '');
  const bBytes = encoder.encode(b || '');
  if (aBytes.byteLength !== bBytes.byteLength) {
    // Still do a same-length compare so the failure path costs about the
    // same time as a real one, rather than short-circuiting on length.
    await crypto.subtle.timingSafeEqual(aBytes, aBytes);
    return false;
  }
  return crypto.subtle.timingSafeEqual(aBytes, bBytes);
}

function nimBase(env) {
  return env.NIM_API_BASE || DEFAULT_NIM_API_BASE;
}

async function fetchLiveModelIds(env) {
  const res = await fetch(nimBase(env) + '/models', {
    headers: {
      Authorization: 'Bearer ' + env.NIM_API_KEY,
      'Content-Type': 'application/json'
    },
    signal: AbortSignal.timeout(VALIDATION_TIMEOUT_MS)
  });
  if (!res.ok) throw new Error('NIM /models returned ' + res.status);
  const data = await res.json();
  return new Set((data.data || []).map(m => m.id));
}

async function validateModels(env) {
  if (env.SKIP_VALIDATION === 'true' || !env.NIM_API_KEY) return;
  try {
    const availableModels = await fetchLiveModelIds(env);
    const invalid = [];
    for (const [alias, nimId] of Object.entries(MODEL_MAPPING)) {
      if (!availableModels.has(nimId)) {
        invalid.push({ alias, nimId, error: 'Model not found in NIM catalog' });
      }
    }
    if (invalid.length > 0) await sendDiscordAlert(env, invalid);
  } catch (err) {
    console.warn('[VALIDATION] Catalog endpoint failure: ' + err.message);
  }
}

async function sendDiscordAlert(env, invalidModels) {
  if (!env.DISCORD_WEBHOOK_URL) return;
  const embed = {
    title: '⚠️ NIM Proxy: Model Validation Failed',
    description: invalidModels.length + ' model(s) failed validation.',
    color: 0xff4444,
    timestamp: new Date().toISOString(),
    fields: invalidModels.map(m => ({
      name: '`' + m.alias + '`',
      value: 'Backend: `' + m.nimId + '`\nError: ' + m.error,
      inline: true
    }))
  };
  try {
    await fetch(env.DISCORD_WEBHOOK_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ embeds: [embed], username: 'NIM Proxy Monitor' }),
      signal: AbortSignal.timeout(5000)
    });
  } catch (err) {
    console.error('[DISCORD] Alert failed:', err.message);
  }
}

async function callWithFallback(env, baseRequest, models, enableThinking, clientReasoningEffort, hasTools) {
  let lastError = null;
  const debug = env.DEBUG_MODE === 'true';
  const showReasoning = env.SHOW_REASONING === 'true';
  const timeoutMs = resolveEffectiveThinking(enableThinking, clientReasoningEffort)
    ? (Number(env.REASONING_REQUEST_TIMEOUT_MS) || 480000)
    : (Number(env.REQUEST_TIMEOUT_MS) || 180000);

  const activeModels = models.filter(m => !isInCooldown(m));
  const attemptOrder = activeModels.length > 0 ? activeModels : models;

  for (const model of attemptOrder) {
    const reasoningPayload = getReasoningPayload(model, enableThinking, clientReasoningEffort, hasTools, showReasoning);
    const fullRequest = { ...baseRequest, model, ...reasoningPayload };

    if (debug) {
      console.log(`[DEBUG] Attempting ${model} with reasoning payload:`, JSON.stringify(reasoningPayload), `(timeout: ${timeoutMs}ms)`);
    }

    try {
      const upstreamRes = await fetch(nimBase(env) + '/chat/completions', {
        method: 'POST',
        headers: {
          Authorization: 'Bearer ' + env.NIM_API_KEY,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(fullRequest),
        signal: AbortSignal.timeout(timeoutMs)
      });
      if (!upstreamRes.ok) {
        const status = upstreamRes.status;
        let errBody = null;
        try { errBody = await upstreamRes.json(); } catch {}
        if (debug) console.log('[DEBUG] Upstream error body:', JSON.stringify(errBody));
        const err = new Error(errBody?.error?.message || 'NIM returned ' + status);
        err.status = status;
        throw err;
      }
      return { response: upstreamRes, model };
    } catch (err) {
      lastError = err;
      console.warn('[FALLBACK] Model failed:', model, err.status, err.message);
      // Same key for every attempt: a 401 means every remaining model would
      // fail identically, so bail out rather than burning the whole chain.
      if (err.status === 401) throw err;
      // 403: likely an access-tier issue, not a dead key — cooldown just
      // this model.
      if (err.status === 403) setCooldown(model, Number(env.ACCESS_DENIED_COOLDOWN_MS) || 300000);
      // 429: rate limited — short cooldown.
      if (err.status === 429) setCooldown(model, Number(env.RATE_LIMIT_COOLDOWN_MS) || 30000);
    }
  }
  throw lastError || new Error('All models failed');
}

class SSEReformatter {
  constructor(usedModel, inlineReasoning, showReasoning) {
    this.inlineReasoning = inlineReasoning;
    this.showReasoning = showReasoning;
    this.normalizer = new StreamNormalizer(usedModel);
    this.toolRecovery = new ToolCallStreamRecovery();
    this.reasoningOpen = false;
    this.doneSent = false;
  }

  processLine(line) {
    const out = [];
    if (!line.startsWith('data: ')) return out;

    // Exact match avoids false-positiving on model output that happens to
    // contain the literal substring "[DONE]".
    if (line.trim() === 'data: [DONE]') {
      if (!this.doneSent) {
        out.push('data: [DONE]\n\n');
        this.doneSent = true;
      }
      return out;
    }

    try {
      const data = JSON.parse(line.slice(6));
      const choice = (data.choices && data.choices[0]) ? data.choices[0] : null;
      const delta = choice ? choice.delta : null;

      if (delta) {
        const normalizedDelta = this.normalizer.processDelta(delta);
        let clientContent = this._composeContent(normalizedDelta.content, normalizedDelta.reasoning);

        const { content: recoveredContent, toolCallDeltas } = this.toolRecovery.process(clientContent);
        clientContent = recoveredContent;
        if (toolCallDeltas.length > 0) {
          delta.tool_calls = toolCallDeltas;
          if (choice) {
            choice.finish_reason = 'tool_calls';
          }
        }

        delta.content = clientContent;

        if (this.showReasoning && normalizedDelta.reasoning) {
          delta.reasoning = normalizedDelta.reasoning;
          delta.reasoning_content = normalizedDelta.reasoning;
        } else {
          delete delta.reasoning;
          delete delta.reasoning_content;
        }
      }

      out.push('data: ' + JSON.stringify(data) + '\n\n');
    } catch {
      console.warn('[STREAM] Invalid JSON line:', line.slice(0, 100));
      out.push('data: ' + JSON.stringify({
        error: { message: 'Upstream sent malformed chunk', type: 'stream_parse_error', details: line.slice(0, 100) }
      }) + '\n\n');
    }
    return out;
  }

  _composeContent(content, reasoning) {
    let clientContent = '';
    if (this.showReasoning && this.inlineReasoning) {
      if (reasoning && !this.reasoningOpen) {
        clientContent += '<thinking>\n' + reasoning;
        this.reasoningOpen = true;
      } else if (reasoning) {
        clientContent += reasoning;
      }
      if (content && this.reasoningOpen) {
        clientContent += '\n</thinking>\n\n' + content;
        this.reasoningOpen = false;
      } else if (content) {
        clientContent += content;
      }
    } else {
      clientContent = content || '';
    }
    return clientContent;
  }

  // Called once the upstream stream ends. Mirrors server.js's 'end' handler:
  // flush any buffered reasoning/content, close a dangling <thinking> tag,
  // and always terminate with [DONE] even if upstream never sent one.
  flush() {
    const out = [];
    const flushedDelta = this.normalizer.flush();
    const toolRecoveryLeftover = this.toolRecovery.flush();
    if (toolRecoveryLeftover) {
      console.warn('[TOOL_CALL_RECOVERY] Stream ended mid <tool_call> tag; flushing raw text instead of dropping it.');
      flushedDelta.content = (flushedDelta.content || '') + toolRecoveryLeftover;
    }

    if (flushedDelta.content || flushedDelta.reasoning) {
      const clientContent = this._composeContent(flushedDelta.content, flushedDelta.reasoning);
      const finalChunk = { choices: [{ delta: {} }] };

      if (clientContent) {
        finalChunk.choices[0].delta.content = clientContent;
      }

      if (this.showReasoning && !this.inlineReasoning && flushedDelta.reasoning) {
        finalChunk.choices[0].delta.reasoning = flushedDelta.reasoning;
        finalChunk.choices[0].delta.reasoning_content = flushedDelta.reasoning;
      }

      if (Object.keys(finalChunk.choices[0].delta).length > 0) {
        out.push('data: ' + JSON.stringify(finalChunk) + '\n\n');
      }
    }

    // Close an inline <thinking> tag left open if the model was cut off
    // mid-reasoning.
    if (this.showReasoning && this.inlineReasoning && this.reasoningOpen) {
      out.push('data: ' + JSON.stringify({ choices: [{ delta: { content: '\n</thinking>\n' } }] }) + '\n\n');
      this.reasoningOpen = false;
    }

    if (!this.doneSent) {
      out.push('data: [DONE]\n\n');
      this.doneSent = true;
    }

    return out;
  }
}

function createSSEBody(upstreamBody, reformatter) {
  const decoder = new TextDecoder('utf-8');
  const encoder = new TextEncoder();
  let buffer = '';

  return new ReadableStream({
    async start(controller) {
      const reader = upstreamBody.getReader();
      const flushLines = () => {
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';
        for (const line of lines) {
          for (const out of reformatter.processLine(line)) {
            controller.enqueue(encoder.encode(out));
          }
        }
      };

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          if (buffer.length > MAX_BUFFER_SIZE) {
            console.error('[STREAM] Buffer overflow, aborting stream');
            controller.enqueue(encoder.encode('data: ' + JSON.stringify({ error: { message: 'Stream buffer overflow', type: 'stream_error' } }) + '\n\n'));
            controller.enqueue(encoder.encode('data: [DONE]\n\n'));
            controller.close();
            await reader.cancel().catch(() => {});
            return;
          }
          flushLines();
        }
        buffer += decoder.decode();
        if (buffer.trim()) {
          flushLines();
          if (buffer.trim()) {
            for (const out of reformatter.processLine(buffer)) {
              controller.enqueue(encoder.encode(out));
            }
          }
        }
        for (const out of reformatter.flush()) {
          controller.enqueue(encoder.encode(out));
        }
        controller.close();
      } catch (err) {
        console.error('[STREAM] Upstream error:', err.message);
        try {
          controller.enqueue(encoder.encode('data: ' + JSON.stringify({ error: { message: 'Stream interrupted by upstream error', type: 'stream_error' } }) + '\n\n'));
          controller.enqueue(encoder.encode('data: [DONE]\n\n'));
          controller.close();
        } catch {}
      }
    },
    cancel(reason) {
      console.warn('[STREAM] Client disconnected:', reason);
    }
  });
}

function homepageResponse() {
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nim-to-openai-proxy</title>
<style>
  body {
    margin: 0;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #0b0f14;
    color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    text-align: center;
    padding: 24px;
  }
  .card { max-width: 480px; }
  h1 { font-size: 1.3rem; margin: 0 0 0.75rem; }
  p { color: #9aa7b2; line-height: 1.55; margin: 0.5rem 0; }
  code { background: #161b22; padding: 2px 6px; border-radius: 4px; color: #7ee787; }
</style>
</head>
<body>
  <div class="card">
    <svg width="44" height="44" viewBox="0 0 24 24" fill="none" style="margin: 0 auto 14px; display: block;">
      <circle cx="12" cy="12" r="11" stroke="#7ee787" stroke-width="1.5"/>
      <path d="M7 12.5l3 3 6-6.5" stroke="#7ee787" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
    </svg>
    <h1>it's up.</h1>
    <p>this proxies OpenAI-format chat requests to NVIDIA NIM, running on Cloudflare Workers. point any OpenAI-compatible client at it, pick a model with a plain alias (<code>gpt-4</code>, <code>mistral</code>, etc), and it handles model fallback, streaming, and each backend's own reasoning/thinking quirks for you.</p>
    <p>it's an API, not a website: nothing lives at this root path.</p>
    <p>send requests to <code>/v1/chat/completions</code> with an <code>Authorization: Bearer &lt;token&gt;</code> header.</p>
    <p>status check, no token needed: <code>/health</code></p>
  </div>
</body>
</html>`;
  return new Response(html, { headers: corsHeaders({ 'Content-Type': 'text/html; charset=utf-8' }) });
}

async function handleChatCompletions(request, env) {
  if (!env.NIM_API_KEY) return errorResponse('NIM_API_KEY is not set.', 'configuration_error', 500);

  let body;
  try {
    body = await request.json();
  } catch {
    return errorResponse('Invalid JSON in request body', 'invalid_request_error', 400);
  }

  try {
    const { model, max_tokens, temperature, stream, reasoning_effort } = body;
    let primaryModel = MODEL_MAPPING[model];
    if (!primaryModel) {
      console.warn(`[PROXY] Unknown model alias "${model}", falling back to default: ${DEFAULT_MODEL}`);
      primaryModel = DEFAULT_MODEL;
    }

    // De-dupe: avoids retrying the same model twice if it's also in
    // FALLBACK_MODELS.
    const modelChain = [...new Set([primaryModel, ...FALLBACK_MODELS])];

    // Forward all client fields except model (replaced per-attempt) and
    // reasoning_effort (translated per-model by getReasoningPayload).
    const { model: _m, reasoning_effort: _re, ...forwardedFields } = body;

    const baseRequest = {
      ...forwardedFields,
      temperature: temperature ?? 0.7,
      max_tokens: Math.min(max_tokens ?? 2048, MAX_TOKENS_LIMIT),
      stream: stream || false
    };

    const { response: upstreamRes, model: usedModel } = await callWithFallback(
      env,
      baseRequest,
      modelChain,
      env.ENABLE_THINKING_MODE === 'true',
      reasoning_effort,
      !!body.tools
    );

    console.log('[PROXY] Model used:', usedModel);

    const inlineReasoning = request.headers.get('x-reasoning-format') === 'inline';
    const showReasoning = env.SHOW_REASONING === 'true';

    if (stream) {
      const reformatter = new SSEReformatter(usedModel, inlineReasoning, showReasoning);
      const sseBody = createSSEBody(upstreamRes.body, reformatter);
      return new Response(sseBody, {
        status: 200,
        headers: corsHeaders({
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive'
        })
      });
    }

    const upstreamData = await upstreamRes.json();
    const openaiResponse = {
      id: 'chatcmpl-' + Date.now(),
      object: 'chat.completion',
      model: usedModel,
      created: Math.floor(Date.now() / 1000),
      choices: (upstreamData.choices || []).map((choice, i) => {
        const normalizedChoice = normalizeNonStreamChoice(choice, usedModel);
        let content = normalizedChoice.message?.content || '';
        const reasoning = normalizedChoice.message?.reasoning || '';
        const { content: cleanedContent, toolCalls: recoveredToolCalls } = extractLeakedToolCalls(content);
        content = cleanedContent;

        if (showReasoning && inlineReasoning && reasoning) {
          content = '<thinking>\n' + reasoning + '\n</thinking>\n\n' + content;
        }

        const finalMessage = { ...normalizedChoice.message, content };

        if (recoveredToolCalls.length > 0) {
          finalMessage.tool_calls = [...(normalizedChoice.message?.tool_calls || []), ...recoveredToolCalls];
          // null content on tool-call turns matches real OpenAI responses.
          if (!finalMessage.content || !finalMessage.content.trim()) finalMessage.content = null;
        }

        if (showReasoning && reasoning) {
          finalMessage.reasoning = reasoning;
          finalMessage.reasoning_content = reasoning;
        } else {
          delete finalMessage.reasoning;
          delete finalMessage.reasoning_content;
        }

        return {
          ...normalizedChoice,
          index: i,
          message: finalMessage,
          ...(recoveredToolCalls.length > 0 && { finish_reason: 'tool_calls' })
        };
      }),
      usage: upstreamData.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
    };

    return jsonResponse(openaiResponse);
  } catch (error) {
    console.error('[PROXY] Fatal error:', error.message);
    return errorResponse(error.message, 'invalid_request_error', error.status || 500);
  }
}

async function handleModels(request, env) {
  const url = new URL(request.url);

  if (url.searchParams.get('live') !== 'true') {
    return jsonResponse({
      object: 'list',
      data: Object.keys(MODEL_MAPPING).map(id => ({
        id,
        object: 'model',
        created: Math.floor(Date.now() / 1000),
        owned_by: 'nim-proxy'
      }))
    });
  }

  if (!env.NIM_API_KEY) return errorResponse('NIM_API_KEY is not set.', 'configuration_error', 500);

  try {
    const availableModels = await fetchLiveModelIds(env);
    const data = Object.entries(MODEL_MAPPING).map(([id, backend]) => ({
      id,
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'nim-proxy',
      backend,
      available: availableModels.has(backend)
    }));
    return jsonResponse({
      object: 'list',
      data,
      live_check: {
        checked_at: new Date().toISOString(),
        unavailable_aliases: data.filter(m => !m.available).map(m => m.id)
      }
    });
  } catch (err) {
    return errorResponse('Live check failed: ' + err.message, 'live_check_error', 502);
  }
}

const PUBLIC_PATHS = new Set(['/health', '/v1/models', '/']);

async function checkAuth(request, env) {
  const token = extractBearerToken(request.headers.get('authorization'));
  if (!token || !env.CLIENT_AUTH_KEY) return errorResponse('Forbidden: Missing credentials', 'authentication_error', 403);
  if (!(await safeTimingEqual(token, env.CLIENT_AUTH_KEY))) return errorResponse('Forbidden: Invalid credentials', 'authentication_error', 403);
  return null;
}

export default {
  async fetch(request, env, ctx) {
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: corsHeaders() });

    // Model-catalog validation runs once per isolate rather than once per
    // server process (there's no equivalent of a long-lived Node process
    // here) — a Worker isolate can be recycled more often than a Render
    // instance under low traffic, so this may fire somewhat more often.
    // Harmless; it only produces a Discord alert if something is actually
    // wrong.
    if (!didIsolateInit) {
      didIsolateInit = true;
      ctx.waitUntil(validateModels(env));
    }

    const url = new URL(request.url);
    const path = url.pathname;

    if (path === '/') return homepageResponse();
    if (path === '/health') return jsonResponse({ status: 'ok', version: '2.6.1-workers' });
    if (path === '/v1/models' && request.method === 'GET') return handleModels(request, env);

    if (!PUBLIC_PATHS.has(path)) {
      const authError = await checkAuth(request, env);
      if (authError) return authError;
    }

    if (path === '/v1/chat/completions' && request.method === 'POST') return handleChatCompletions(request, env);

    return errorResponse('Endpoint ' + request.method + ' ' + path + ' not found', 'invalid_request_error', 404);
  }
};
