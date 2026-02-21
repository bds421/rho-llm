# rho/llm — Architecture

> **Status:** Reflects the actual implementation as of February 2026.

---

## 1. Overview

`gitlab2024.bds421-cloud.com/bds421/rho/llm` is a Go package providing a **unified, provider-agnostic LLM client interface** that covers eleven providers across three distinct wire protocols.

**Key capabilities:**
- Single `Client` interface for all providers and protocols
- Streaming via Go 1.23 `iter.Seq2[StreamEvent, error]` iterators
- Tool use / function calling
- Extended thinking (Anthropic extended thinking, Gemini `thought_signature`)
- Auth pool rotation with exponential backoff and per-profile cooldown
- Structured error types enabling reliable retry classification
- Cost estimation from per-model pricing data
- Privacy-safe request/response logging middleware

---

## 2. Package Structure

```
gitlab2024.bds421-cloud.com/bds421/rho/llm/
├── types.go          # Core types: Message, Request, Response, StreamEvent, Client interface
├── config.go         # Config struct + DefaultConfig()
├── provider.go       # Provider presets, protocol resolution, URL/auth resolution
├── registry.go       # ModelRegistry, ModelAliases, cost estimation, ResolveModelAlias()
├── register.go       # RegisterProvider() + provider factory registry
├── factory.go        # NewClient() / NewClientWithKeys() — registry lookup entry point
├── pool.go           # AuthPool + PooledClient (rotation + retry for Complete and Stream pre-data failures)
├── middleware.go     # LoggingClient decorator
├── errors.go         # APIError type + Is*() helpers
├── backoff.go        # Exponential backoff with jitter
├── llm_test.go       # Integration and unit tests
│
└── provider/                        # Provider adapters (database/sql driver pattern)
    ├── all.go                       # Blank-imports all sub-packages
    ├── anthropic/anthropic.go       # Native Anthropic API adapter
    ├── gemini/gemini.go             # Native Google Gemini API adapter
    └── openaicompat/openaicompat.go # OpenAI-compatible adapter (11+ providers)
```

Provider implementations register themselves via `init()` using `llm.RegisterProvider()`. Consumers that call `llm.NewClient()` must add a blank import: `_ "gitlab2024.bds421-cloud.com/bds421/rho/llm/provider"`.

---

## 3. Architecture Overview

```
                      ┌────────────────────────────────┐
                      │  Application / Service         │
                      └───────────────┬────────────────┘
                                      │ llm.NewClient(cfg)
                                      │ llm.NewClientWithKeys(cfg, keys)
                                      ▼
                      ┌────────────────────────────────┐
                      │         factory.go             │
                      │  newSingleClient() /           │
                      │  newPooledClient()             │
                      └──────┬────────────────┬────────┘
               1 key         │                │  >1 keys
                             ▼                ▼
              ┌──────────────────┐   ┌──────────────────────┐
              │  Single Client   │   │   PooledClient       │
              │  (direct)        │   │  (auth rotation)     │
              └────────┬─────────┘   └──────────┬───────────┘
                       │                        │ wraps N SingleClients
                       ▼                        ▼
         ┌─────────────────────────────────────────────────┐
         │               Protocol Adapters                 │
         │  ┌──────────────┐ ┌──────────┐ ┌─────────────┐  │
         │  │ anthropic.go │ │gemini.go │ │openai_compat│  │
         │  │  (native)    │ │(native)  │ │  .go        │  │
         │  └──────────────┘ └──────────┘ └─────────────┘  │
         │                                  ↑ used by:     │
         │                                  │ openai, xai  │
         │                                  │ grok, groq   │
         │                                  │ cerebras     │
         │                                  │ mistral      │
         │                                  │ openrouter   │
         │                                  │ ollama, vllm │
         │                                  │ lmstudio     │
         └─────────────────────────────────────────────────┘
                       │
            (optional) │ cfg.LogRequests = true
                       ▼
         ┌─────────────────────────────────────────────────┐
         │          LoggingClient (middleware.go)          │
         │  Wraps any Client — logs metadata only,         │
         │  no message content                             │
         └─────────────────────────────────────────────────┘

### The Layers

The architecture is structured like an onion, sharply isolating concerns:

1. **Outer Layer:** Unified Types & Interfaces (`types.go`).
2. **Middle Layer:** Logging Middleware (`middleware.go`) wrapping the...
3. **Core Resilience Layer:** `PooledClient` and `AuthPool` (`pool.go`) handling circuit-breaking, failover routing, and thread-safe lock synchronization, which finally wraps the...
4. **Raw Translators:** Vendor-specific adapters making the raw network calls over native `http.Client`s.

---

## 4. Core Types (`types.go`)

### Client Interface

```go
type Client interface {
    Complete(ctx context.Context, req Request) (*Response, error)
    Stream(ctx context.Context, req Request) iter.Seq2[StreamEvent, error]
    Provider() string
    Model() string
    Close() error
}
```

All concrete adapters (`anthropic.Client`, `gemini.Client`, `openaicompat.Client`) implement this interface. `PooledClient` and `LoggingClient` also implement it as decorators.

### Message Model

```
Message
  ├── Role: Role (RoleUser | RoleAssistant | RoleSystem)
  └── Content: []ContentPart
        ├── {Type: ContentText, Text: "..."}
        ├── {Type: ContentImage, Source: ImageSource{base64, media_type}}
        ├── {Type: ContentToolUse, ToolUseID, ToolName, ToolInput, ThoughtSignature}
        └── {Type: ContentToolResult, ToolResultID, ToolResultContent, IsError}
```

`ThoughtSignature` on `tool_use` parts is a Gemini 3 requirement — the model returns an opaque signature that must be echoed back in the corresponding `tool_result` message. The adapters handle this automatically.

### Stream Events

```go
type StreamEvent struct {
    Type EventType // EventContent | EventToolUse | EventThinking | EventDone | EventError

    Text     string     // content
    ToolCall *ToolCall  // tool_use
    Thinking string     // thinking (Anthropic extended thinking)

    InputTokens  int    // usage / done (-1 = not reported)
    OutputTokens int    // usage / done (-1 = not reported)
    StopReason   string // done: "end_turn" | "tool_use" | "max_tokens"
    Error        string // error
}
```

**Token counts:** In streaming, token counts are `-1` if the provider didn't report them (connection dropped, provider quirk). This distinguishes "not reported" from a legitimate "zero tokens" response.

---

## 5. Provider & Protocol Resolution (`provider.go`, `factory.go`)

### Protocol Routing

Three wire protocols are supported:

| Protocol | Adapters | Notes |
|---|---|---|
| `anthropic` | `AnthropicClient` | Native SSE streaming, `x-api-key` header |
| `gemini` | `GeminiClient` | Native REST + SSE, API key as query param |
| `openai_compat` | `OpenAICompatClient` | Standard `/chat/completions` with SSE streaming |

Protocol selection happens in `factory.newSingleClient()` via `ResolveProtocol(cfg)`:

```
cfg.Provider ──→ Presets lookup ──→ preset.Protocol
                     │
                     ↓ (not found)
                cfg.BaseURL set? ──→ "openai_compat"
                     │ (not set)
                     ↓
                "openai_compat" (fallback)
```

### Provider Presets

Each known provider has a `ProviderPreset`:
```go
type ProviderPreset struct {
    BaseURL    string // Default API endpoint
    AuthHeader string // "Bearer" or "" (no auth)
    Protocol   string // Wire protocol
}
```

`Config.BaseURL` and `Config.AuthHeader` always take precedence, enabling proxy routing and custom deployments. Any unknown provider with a `BaseURL` is treated as OpenAI-compatible.

---

## 6. Factory (`factory.go`)

```
NewClient(cfg)
    └── NewClientWithKeys(cfg, nil)
            ├── len(keys) >= 1 → newPooledClient(cfg, keys)  // Even single-key gets retry/backoff
            └── else           → newSingleClient(cfg)        // Only when no keys provided
                                    ├── ResolveProtocol(cfg)
                                    ├── getProviderFactory(protocol)
                                    │     "anthropic"    → anthropic.New(cfg)
                                    │     "gemini"       → gemini.New(cfg)
                                    │     "openai_compat"→ openaicompat.New(cfg)
                                    └── cfg.LogRequests? → WithLogging(client)
```

Single-key clients now go through `PooledClient` to get exponential backoff on transient errors (429, 503, 502). Previously, single-key clients bypassed the pool entirely and had zero retry logic.

---

## 7. Auth Pool & Rotation (`pool.go`)

### AuthPool

`AuthPool` maintains an ordered list of `*AuthProfile` entries (one per API key). It uses **round-robin selection** with health-based skipping.

**Key format:** Keys can include a custom BaseURL using `API_KEY|BASE_URL`:
```go
keys := []string{
    "sk-primary",                          // Uses config BaseURL
    "sk-backup|https://proxy.example.com", // Uses custom endpoint
}
```

`NewAuthPool` parses each key at the `|` separator and populates `AuthProfile.APIKey` and `AuthProfile.BaseURL` accordingly. The `clientFunc` receives the full `AuthProfile` so it can use both fields.

```
GetAvailable():
  1. Try current profile → if available, mark used & return
  2. Rotate forward until available profile found
  3. If all in cooldown → return error with "next available in Xm"
```

Cooldown durations are error-type-dependent:

| Error | Cooldown |
|---|---|
| Rate limit (429) | 60 s |
| Overloaded (503) | 30 s |
| Any other retryable | 10 s |

### PooledClient

`PooledClient` wraps N single-provider clients (one per API key), all sharing the same `AuthPool`. Even single-key pools get retry/backoff for transient errors. On failure:

```
Complete():
    loop (maxRetries = max(pool.Count(), 3)):  // Minimum 3 for single-key resilience
        1. Call current client
        2. Success → MarkSuccess(), return
        3. Non-retryable, non-auth error (400) → return immediately
        4. Auth error (401/403) OR retryable error (429/503/502):
             MarkFailed(err):
               - Auth errors: IsHealthy = false (permanent)
               - Others: cooldown (temporary)
             rotateClient() → GetAvailable() → create new single client
             if rotation fails:
               - Auth error → return immediately (dead key is dead)
               - Transient error → Backoff(attempt, 1s, 30s) → sleep & retry same

Stream():
    loop (maxRetries = max(pool.Count(), 3)):
        1. Start streaming from current client
        2. If error BEFORE any event yielded (firstEvent == true):
             a. Non-retryable, non-auth error → return immediately
             b. Auth or retryable error → MarkFailed(err), rotateClient():
                  - rotation fails + auth error → return immediately
                  - rotation fails + transient → backoff & retry
                  - rotation succeeds → retry with new client
        3. If error AFTER events yielded (firstEvent == false):
             → pass through to caller (no retry — would duplicate content)
        4. Stream completes → MarkSuccess(), return
```

**Auth error handling:** When a 401/403 occurs, the key is marked permanently unhealthy (`IsHealthy = false`), not just put in cooldown. If rotation fails because no healthy keys remain, the error returns immediately — no point backing off with a dead key.

**Pre-data vs mid-stream retry:** A stream that fails on the initial HTTP connection (429/503 before any SSE events) is functionally identical to a failed `Complete` — no data has reached the caller, so retry with rotation is safe. Once any event has been yielded via `for-range`, retrying would replay content from scratch with no way for the caller to detect duplication, so mid-stream errors pass through immediately.

`rotateClient()` does NOT close the replaced client — doing so would race with in-flight requests still holding a reference. Orphaned clients are garbage collected. If adapters ever need real cleanup (persistent connections), proper reference counting would be required.

**Thundering herd prevention:** When 50 goroutines hit a 429 simultaneously, naive single-checked locking would let all 50 create new clients. `PooledClient` uses double-checked locking with a dedicated `rotateMu` mutex:

```go
func (pc *PooledClient) rotateClient(failedName string) error {
    pc.rotateMu.Lock()         // Gate: only one goroutine enters at a time
    defer pc.rotateMu.Unlock()

    pc.mu.RLock()
    currentName := pc.activeName
    pc.mu.RUnlock()

    if currentName != failedName {
        return nil             // Another goroutine already rotated
    }
    // ... create new client ...
}
```

Goroutine 1 rotates; goroutines 2-50 block at `rotateMu.Lock()`, then short-circuit when they see `currentName != failedName`.

### Backoff (`backoff.go`)

Exponential with ±25% jitter to prevent thundering herd:

```
attempt 0: base × 2⁰ = ~1s  (0.75–1.25s)
attempt 1: base × 2¹ = ~2s  (1.50–2.50s)
attempt 2: base × 2² = ~4s  (3.00–5.00s)
attempt 3: base × 2³ = ~8s  (6.00–10.0s)
...capped at maxDelay (default 30s)
```

---

## 8. Structured Errors (`errors.go`)

All adapters produce `*APIError` instead of opaque `fmt.Errorf` strings:

```go
type APIError struct {
    StatusCode int    // HTTP status
    Message    string // Response body
    Provider   string
    Retryable  bool
}
```

Classification helpers use `errors.As()`:

| Helper | Condition |
|---|---|
| `IsRateLimited(err)` | StatusCode == 429 |
| `IsOverloaded(err)` | StatusCode == 503 |
| `IsAuthError(err)` | StatusCode == 401 or 403 |
| `IsContextLength(err)` | StatusCode == 400 + context-length keywords in body |
| `IsRetryable(err)` | `APIError.Retryable == true` (429, 503, 500, 502, 408) |

`NewAPIErrorFromStatus()` is the shared constructor used by all adapters to map HTTP responses to the appropriate `*APIError` subtype.

---

## 9. Model Registry (`registry.go`)

Three data structures:

```
ModelRegistry    map[string]ModelInfo     — full model metadata, keyed by model ID
ModelAliases     map[string]string        — short alias → full model ID
DefaultModels    map[string]string        — provider → default model ID
```

`ModelInfo` fields:
```go
type ModelInfo struct {
    ID, Provider     string
    MaxTokens        int     // Model output limit (0 = use config)
    ContextWindow    int     // Max input tokens
    InputPricePer1M  float64 // USD pricing
    OutputPricePer1M float64
    SupportsThinking bool    // Anthropic extended thinking
    ThoughtSignature bool    // Gemini 3: must echo thought_signature in tool results
    Label            string  // Short display name
}
```

**Registered providers:**

| Provider | Models |
|---|---|
| Anthropic | claude-opus-4-6, claude-sonnet-4-6, claude-sonnet-4-5, claude-haiku-4-5 |
| xAI | grok-4-1-fast-{reasoning,non-reasoning}, grok-4-fast-{reasoning,non-reasoning}, grok-code-fast-1, grok-3, grok-3-mini |
| Gemini | gemini-3-{pro,flash}-preview, gemini-2.5-{pro,flash,flash-lite} |

`EstimateCost(model, inputTokens, outputTokens)` returns a USD float from registry pricing. Returns `0` if the model is unknown.

---

## 10. Logging Middleware (`middleware.go`)

`LoggingClient` is a **decorator** over any `Client`. It logs metadata only — never message content — making it safe to enable in production:

**On Complete:**
```
DEBUG: [LLM] Complete: provider=anthropic model=claude-sonnet-4-6 messages=5 tools=2 max_tokens=8192
DEBUG: [LLM] Complete done: ... elapsed=1.234s tokens_in=1200 tokens_out=350 stop=end_turn cost=$0.009150
```

**On Stream:**
```
DEBUG: [LLM] Stream: provider=anthropic model=claude-sonnet-4-6 messages=5 ...
DEBUG: [LLM] Stream done: ... elapsed=3.412s chunks=87 tokens_in=1200 tokens_out=450 stop=end_turn cost=$0.011100
```

Enable via config:
```go
cfg.LogRequests = true  // applied automatically in factory
```

Or wrap any existing client:
```go
client = llm.WithLogging(client)
client = llm.WithLoggingPrefix(client, "[MyService]")
```

---

## 11. Protocol Adapter Details

### Anthropic (`provider/anthropic/`)

- Endpoint: `https://api.anthropic.com/v1/messages`
- Auth: `x-api-key: <key>` + `anthropic-version: 2023-06-01`
- Streaming: SSE with `event: content_block_delta` / `event: message_delta`
- Extended thinking: enabled via `thinking: {type: enabled, budget_tokens: N}` — budget mapped from ThinkingLevel (`ThinkingNone`→disabled, `ThinkingLow`→1024, `ThinkingMedium`→4096, `ThinkingHigh`→16384)
- Tool use: native anthropic format with `type: tool_use` content blocks

### Gemini (`provider/gemini/`)

- Endpoint: `https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent`
- Auth: `?key=<apikey>` query parameter
- Streaming: SSE with JSON chunks
- `ThoughtSignature`: when a model has `ThoughtSignature: true` in the registry, function call responses include a `thought_signature` field that must be preserved and echoed in subsequent `tool_result` parts
- System prompt: mapped to `systemInstruction.parts[0].text`

### OpenAI-Compatible (`provider/openaicompat/`)

- Endpoint: configurable `BaseURL` + `/chat/completions`
- Auth: `Authorization: Bearer <key>` (or empty for no-auth providers)
- Streaming: OpenAI SSE format with `data: {delta: ...}` chunks
  - `stream_options: {include_usage: true}` is set automatically to receive token usage
  - OpenAI sends `finish_reason` and `usage` in **separate chunks**: the parser accumulates state across chunks and emits `EventDone` with complete data after `[DONE]`
  - Tool calls are flushed before `EventDone` even if `finish_reason` is missing (handles network drops and spec-violating servers like Ollama)
- Tool use: OpenAI function-calling format, translated to/from the shared `ToolCall` types. Multiple tool results in a single Anthropic-style message are expanded to separate `role: "tool"` messages as OpenAI requires.
- Works for: OpenAI, xAI/Grok, Groq, Cerebras, Mistral, OpenRouter, Ollama, vLLM, LM Studio, any custom proxy

---

## 12. Config Reference

```go
type Config struct {
    Provider      string        // Provider name (see presets)
    Model         string        // Model ID or alias
    APIKey        string        // API key (empty OK for no-auth providers)
    MaxTokens     int           // Max output tokens (default: 8192)
    Temperature   float64       // Sampling temperature (default: 1.0)
    ThinkingLevel ThinkingLevel  // ThinkingLow | ThinkingMedium | ThinkingHigh (zero = none)
    Timeout       time.Duration // HTTP timeout (default: 120s)
    BaseURL       string        // Override provider endpoint
    AuthHeader    string        // Override auth header ("Bearer", "x-api-key", "")
    ProviderName  string        // Override Client.Provider() return value
    LogRequests   bool          // Enable metadata logging
}
```

**Defaults (`DefaultConfig()`):**
```
Provider:      "anthropic"
Model:         "claude-sonnet-4-6"
MaxTokens:     8192
Temperature:   1.0
ThinkingLevel: "" (ThinkingNone)
Timeout:       120s
AuthHeader:    "Bearer"
```

---

## 13. Design Decisions

### Why a unified interface over provider SDKs?

Provider SDKs have incompatible types, inconsistent error models, and evolve independently. A thin HTTP adapter per protocol gives full control over retry logic, streaming, and error classification without taking on SDK dependency churn. The three protocols (Anthropic native, Gemini native, OpenAI-compat) cover the entire current provider landscape.

### Why `*APIError` instead of typed error vars?

`errors.As()` on a concrete `*APIError` allows callers to inspect the HTTP status code without parsing strings. The `Retryable bool` field moves the retry classification decision into the library, where context (provider, status code, body content) is available.

### Why `AuthPool` rotates rather than load-balances?

Round-robin rotation on failure (not on every request) minimizes unnecessary API calls. Using the same profile for consecutive requests maximizes cache hits and context locality on the provider side. Load balancing would require tracking concurrent request counts per profile, adding complexity with marginal benefit for most workloads.

### `ThoughtSignature` (Gemini 3)

Gemini 3 models return an opaque `thought_signature` field in function call responses. This signature encodes the model's internal reasoning state and must be echoed in the `tool_result` message, or the model will repeat the same tool call. The adapters preserve this field transparently through the `ContentPart.ThoughtSignature` field so calling code does not need to handle it explicitly.

