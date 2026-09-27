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
return env.NIM_API_BASE || 'nvidia.com';
}
async function fetchLiveModelIds(env) {
const res = await fetch(${nimBase(env)}/models, {
headers: {
Authorization: Bearer ${env.NIM_API_KEY},
'Content-Type': 'application/json'
},
signal: AbortSignal.timeout(VALIDATION_TIMEOUT_MS)
});
if (!res.ok) throw new Error(NIM /models returned ${res.status});
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
console.warn([VALIDATION] Catalog endpoint failure: ${err.message});
}
}
async function sendDiscordAlert(env, invalidModels) {
if (!env.DISCORD_WEBHOOK_URL) return;
const embed = {
title: '⚠️ NIM Proxy: Model Validation Failed',
description: ${invalidModels.length} model(s) failed validation.,
color: 0xff4444,
timestamp: new Date().toISOString(),
fields: invalidModels.map(m => ({
name: ${m.alias},
value: Backend: ${m.nimId}\nError: ${m.error},
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
const upstreamRes = await fetch(${nimBase(env)}/chat/completions, {
method: 'POST',
headers: {
Authorization: Bearer ${env.NIM_API_KEY},
'Content-Type': 'application/json'
},
body: JSON.stringify(fullRequest),
signal: AbortSignal.timeout(timeoutMs)
});
if (!upstreamRes.ok) {
const status = upstreamRes.status;
let errBody = null;
try { errBody = await upstreamRes.json(); } catch {}
const err = new Error(errBody?.error?.message || NIM returned ${status});
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
out.push(data: ${JSON.stringify(data)}\n\n);
} catch {
out.push(data: ${JSON.stringify({ error: { message: 'Upstream sent malformed chunk', type: 'stream_parse_error' } })}\n\n);
}
return out;
}
_composeContent(content, reasoning) {
let clientContent = '';
if (this.showReasoning && this.inlineReasoning) {
if (reasoning && !this.reasoningOpen) {
clientContent += <thinking>\n${reasoning};
this.reasoningOpen = true;
} else if (reasoning) {
clientContent += reasoning;
}
if (content && this.reasoningOpen) {
clientContent += \n</thinking>\n\n${content};
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
finalChunk.choices[0].delta.reasoning = flushedDelta.reasoning;
finalChunk.choices[0].delta.reasoning_content = flushedDelta.reasoning;
}
if (Object.keys(finalChunk.choices[0].delta).length > 0) {
out.push(data: ${JSON.stringify(finalChunk)}\n\n);
}
}
if (this.showReasoning && this.inlineReasoning && this.reasoningOpen) {
out.push(data: ${JSON.stringify({ choices: [{ delta: { content: '\n</thinking>\n' } }] })}\n\n);
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
controller.enqueue(encoder.encode(data: ${JSON.stringify({ error: { message: 'Stream buffer overflow', type: 'stream_error' } })}\n\n));
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
try {
controller.enqueue(encoder.encode(data: ${JSON.stringify({ error: { message: 'Stream error encountered', type: 'stream_error' } })}\n\n));
controller.enqueue(encoder.encode('data: [DONE]\n\n'));
controller.close();
} catch {}
}
},
cancel(reason) {
console.warn('[STREAM] Disconnected:', reason);
}
});
}
function homepageResponse() {
const html = <h1>Proxy Online</h1><p>Point tools to <code>/v1/chat/completions</code></p>;
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
let primaryModel = MODEL_MAPPING[model] || DEFAULT_MODEL;
const modelChain = [...new Set([primaryModel, ...FALLBACK_MODELS])];
const { model: _m, reasoning_effort: _re, ...forwardedFields } = body;
const baseRequest = {
...forwardedFields,
temperature: temperature ?? 0.7,
max_tokens: Math.min(max_tokens ?? 2048, MAX_TOKENS_LIMIT),
stream: stream || false
};
const { response: upstreamRes, model: usedModel } = await callWithFallback(
env, baseRequest, modelChain, env.ENABLE_THINKING_MODE === 'true', reasoning_effort, !!body.tools
);
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
id: chatcmpl-${Date.now()},
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
content = <thinking>\n${reasoning}\n</thinking>\n\n${content};
}
const finalMessage = { ...normalizedChoice.message, content };
if (recoveredToolCalls.length > 0) {
finalMessage.tool_calls = [...(normalizedChoice.message?.tool_calls || []), ...recoveredToolCalls];
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
return errorResponse(error.message, 'invalid_request_error', error.status || 500);
}
}
async function handleModels(request, env) {
const url = new URL(request.url);
if (url.searchParams.get('live') !== 'true') {
return jsonResponse({
object: 'list',
data: Object.keys(MODEL_MAPPING).map(id => ({
id, object: 'model', created: Math.floor(Date.now() / 1000), owned_by: 'nim-proxy'
}))
});
}
if (!env.NIM_API_KEY) return errorResponse('NIM_API_KEY is not set.', 'configuration_error', 500);
try {
const availableModels = await fetchLiveModelIds(env);
const data = Object.entries(MODEL_MAPPING).map(([id, backend]) => ({
id, object: 'model', created: Math.floor(Date.now() / 1000), owned_by: 'nim-proxy', backend, available: availableModels.has(backend)
}));
return jsonResponse({
object: 'list', data, live_check: { checked_at: new Date().toISOString(), unavailable_aliases: data.filter(m => !m.available).map(m => m.id) }
});
} catch (err) {
return errorResponse(Live check failed: ${err.message}, 'live_check_error', 502);
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
if (!didIsolateInit) {
didIsolateInit = true;
ctx.waitUntil(validateModels(env));
}
const url = new URL(request.url);
const path = url.pathname;
if (path === '/') return homepageResponse();
if (path === '/health') return jsonResponse({ status: 'ok', version: '2.6.0-workers' });
if (path === '/v1/models' && request.method === 'GET') return handleModels(request, env);
if (!PUBLIC_PATHS.has(path)) {
const authError = await checkAuth(request, env);
if (authError) return authError;
}
if (path === '/v1/chat/completions' && request.method === 'POST') return handleChatCompletions(request, env);
return errorResponse(Endpoint ${request.method} ${path} not found, 'invalid_request_error', 404);
}
};

