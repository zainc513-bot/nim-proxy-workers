import { getReasoningPayload, resolveEffectiveThinking, StreamNormalizer, normalizeNonStreamChoice } from './reasoning.js';
import { extractLeakedToolCalls, ToolCallStreamRecovery } from './tools.js';

const MAX_TOKENS_LIMIT = 65536;
const VALIDATION_TIMEOUT_MS = 15000;
const MAX_BUFFER_SIZE = 1024 * 1024;

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

async function safeTimingEqual(a, b) {
  const encoder = new TextEncoder();
  const aBytes = encoder.encode(a || '');
  const bBytes = encoder.encode(b || '');
  if (aBytes.byteLength !== bBytes.byteLength) {
    await crypto.subtle.timingSafeEqual(aBytes, aBytes);
    return false;
  }
  return crypto.subtle.timingSafeEqual(aBytes, bBytes);
}

function nimBase(env) {
  return env.NIM_API_BASE || 'https://nvidia.com';
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
      name: '' + m.alias,
      value: 'Backend: ' + m.nimId + '\nError: ' + m.error,
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
  const showReasoning = env.SHOW_REASONING === 'true';
  const timeoutMs = resolveEffectiveThinking(enableThinking, clientReasoningEffort)
    ? (Number(env.REASONING_REQUEST_TIMEOUT_MS) || 480000)
    : (Number(env.REQUEST_TIMEOUT_MS) || 180000);

  const activeModels = models.filter(m => !isInCooldown(m));
  const attemptOrder = activeModels.length > 0 ? activeModels : models;

  for (const model of attemptOrder) {
    const reasoningPayload = getReasoningPayload(model, enableThinking, clientReasoningEffort, hasTools, showReasoning);
    const fullRequest = { ...baseRequest, model, ...reasoningPayload };
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
        const err = new Error(errBody?.error?.message || 'NIM returned ' + status);
        err.status = status;
        throw err;
      }
      return { response: upstreamRes, model };
    } catch (err) {
      lastError = err;
      if (err.status === 401) throw err;
      if (err.status === 403) setCooldown(model, Number(env.ACCESS_DENIED_COOLDOWN_MS) || 300000);
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

    if (line.trim() === 'data: [DONE]') {
      if (!this.doneSent) {
        out.push('data: [DONE]\n\n');
        this.doneSent = true;
      }
      return out;
    }

    try {
      const data = JSON.parse(line.slice(6));
      const delta = (data.choices && data.choices) ? data.choices.delta : null;

      if (delta) {
        const normalizedDelta = this.normalizer.processDelta(delta);
        let clientContent = this._composeContent(normalizedDelta.content, normalizedDelta.reasoning);

        const { content: recoveredContent, toolCallDeltas } = this.toolRecovery.process(clientContent);
        clientContent = recoveredContent;
        if (toolCallDeltas.length > 0) {
          delta.tool_calls = toolCallDeltas;
          if (data.choices && data.choices) {
            data.choices.finish_reason = 'tool_calls';
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
      out.push('data: ' + JSON.stringify({
        error: { message: 'Upstream sent malformed chunk', type: 'stream_parse_error' }
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

  flush() {
    const out = [];
    const flushedDelta = this.normalizer.flush();
    const toolRecoveryLeftover = this.toolRecovery.flush();
    if (toolRecoveryLeftover) {
      flushedDelta.content = (flushedDelta.content || '') + toolRecoveryLeftover;
    }

    if (flushedDelta.content || flushedDelta.reasoning) {
      const clientContent = this._composeContent(flushedDelta.content, flushedDelta.reasoning);
      const finalChunk = { choices: [{ delta: {} }] };
      
      if (clientContent) {
        finalChunk.choices[0].delta.content = clientContent;
      }

      if (this.showReasoning && !this.inlineReasoning && flushedDelta.reasoning) {
import { getReasoningPayload, resolveEffectiveThinking, StreamNormalizer, normalizeNonStreamChoice } from './reasoning.js';
import { extractLeakedToolCalls, ToolCallStreamRecovery } from './tools.js';

const MAX_TOKENS_LIMIT = 65536;
const VALIDATION_TIMEOUT_MS = 15000;
const MAX_BUFFER_SIZE = 1024 * 1024;

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

async function safeTimingEqual(a, b) {
  const encoder = new TextEncoder();
  const aBytes = encoder.encode(a || '');
  const bBytes = encoder.encode(b || '');
  if (aBytes.byteLength !== bBytes.byteLength) {
    await crypto.subtle.timingSafeEqual(aBytes, aBytes);
    return false;
  }
  return crypto.subtle.timingSafeEqual(aBytes, bBytes);
}

function nimBase(env) {
  return env.NIM_API_BASE || 'https://nvidia.com';
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
      name: '' + m.alias,
      value: 'Backend: ' + m.nimId + '\nError: ' + m.error,
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
  const showReasoning = env.SHOW_REASONING === 'true';
  const timeoutMs = resolveEffectiveThinking(enableThinking, clientReasoningEffort)
    ? (Number(env.REASONING_REQUEST_TIMEOUT_MS) || 480000)
    : (Number(env.REQUEST_TIMEOUT_MS) || 180000);

  const activeModels = models.filter(m => !isInCooldown(m));
  const attemptOrder = activeModels.length > 0 ? activeModels : models;

  for (const model of attemptOrder) {
    const reasoningPayload = getReasoningPayload(model, enableThinking, clientReasoningEffort, hasTools, showReasoning);
    const fullRequest = { ...baseRequest, model, ...reasoningPayload };
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
        const err = new Error(errBody?.error?.message || 'NIM returned ' + status);
        err.status = status;
        throw err;
      }
      return { response: upstreamRes, model };
    } catch (err) {
      lastError = err;
      if (err.status === 401) throw err;
      if (err.status === 403) setCooldown(model, Number(env.ACCESS_DENIED_COOLDOWN_MS) || 300000);
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

    if (line.trim() === 'data: [DONE]') {
      if (!this.doneSent) {
        out.push('data: [DONE]\n\n');
        this.doneSent = true;
      }
      return out;
    }

    try {
      const data = JSON.parse(line.slice(6));
      const delta = (data.choices && data.choices) ? data.choices.delta : null;

      if (delta) {
        const normalizedDelta = this.normalizer.processDelta(delta);
        let clientContent = this._composeContent(normalizedDelta.content, normalizedDelta.reasoning);

        const { content: recoveredContent, toolCallDeltas } = this.toolRecovery.process(clientContent);
        clientContent = recoveredContent;
        if (toolCallDeltas.length > 0) {
          delta.tool_calls = toolCallDeltas;
          if (data.choices && data.choices) {
            data.choices.finish_reason = 'tool_calls';
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
      out.push('data: ' + JSON.stringify({
        error: { message: 'Upstream sent malformed chunk', type: 'stream_parse_error' }
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

  flush() {
    const out = [];
    const flushedDelta = this.normalizer.flush();
    const toolRecoveryLeftover = this.toolRecovery.flush();
    if (toolRecoveryLeftover) {
      flushedDelta.content = (flushedDelta.content || '') + toolRecoveryLeftover;
    }

    if (flushedDelta.content || flushedDelta.reasoning) {
      const clientContent = this._composeContent(flushedDelta.content, flushedDelta.reasoning);
      const finalChunk = { choices: [{ delta: {} }] };
      
      if (clientContent) {
        finalChunk.choices[0].delta.content = clientContent;
      }

      if (this.showReasoning && !this.inlineReasoning && flushedDelta.reasoning) {
