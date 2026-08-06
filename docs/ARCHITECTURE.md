# rho/llm — Architecture

> **Status:** Reflects the current implementation as of August 2026.

---

## 1. Overview

`github.com/bds421/rho-llm` is a Go package providing a **unified, provider-agnostic LLM client interface** that covers twenty-three providers across four distinct wire protocols (`anthropic`, `gemini`, `openai_compat`, `openai_responses`).

**Key capabilities:**
- Single `Client` interface for all providers and protocols
- Streaming via Go 1.23 `iter.Seq2[StreamEvent, error]` iterators
- Tool use / function calling
- Image/vision support (base64 images in all 3 adapters)
- Document/PDF support (`ContentDocument`: native on Gemini/Anthropic, data URI on OpenAI-compatible)
- Extended thinking (Anthropic extended thinking, Gemini `thought_signature`)
- Serializable conversations (`Conversation`) + a stateful `Session` driver with **cross-provider handoff** (`SwitchProvider`, `NormalizeForProvider`); pluggable persistence (`Store`: `MemoryStore`/`FileStore`)
- Structured output (JSON mode / JSON schema) on OpenAI-compatible + Gemini; tool-call validation; fine-grained streaming boundaries (`StreamWithBoundaries`)
- Registered `ModalityClient` adapters for embeddings, image generation, speech,
  and transcription; OpenAI-compatible is the first modality driver
- A `MockClient` test double and model/provider discovery (`Models`/`Providers`)
- Auth pool rotation with exponential backoff and per-profile cooldown
- Structured error types enabling reliable retry classification
- Cost estimation from per-model pricing data
- Privacy-safe request/response logging middleware

---

## 2. Package Structure

```
github.com/bds421/rho-llm/
├── types.go          # Core types: Message, Request, Response, StreamEvent, Client interface
├── config.go         # Config struct + DefaultConfig()
├── provider.go       # Provider presets, protocol resolution, URL/auth resolution
├── registry.go       # ModelRegistry, ModelAliases, cost estimation, ResolveModelAlias()
├── register.go       # RegisterProvider() + provider factory registry
├── modalityregister.go # RegisterModalityDriver() + modality registry
├── batch.go          # BatchClient interface + BatchItem/BatchResult/BatchHandle (async, serializable)
├── batchregister.go  # RegisterBatchProvider() + batch driver registry (parallels register.go)
├── factory.go        # NewClient() / NewModalityClient() / NewBatchClient()
├── conversation.go      # Conversation (serializable transcript) + Usage + versioned JSON
├── session.go           # Session (concurrency-safe driver) + provider handoff (SwitchProvider)
├── normalize.go         # NormalizeForProvider() — cross-provider transcript translation
├── store.go             # Store interface + MemoryStore / FileStore (conversation persistence)
├── streamevents.go      # StreamWithBoundaries() — fine-grained block events
├── mock.go              # MockClient test double + faux builders
├── discovery.go         # Models() / ModelsByProvider() / Providers()
├── simple.go            # CompleteSimple / StreamSimple
├── validate.go          # ValidateToolCall() against a tool's InputSchema
├── capabilities.go      # Provider-neutral modality types/interfaces/validation
├── oauth.go             # OAuth 2.0 device-flow helpers (RFC 8628)
├── transport.go         # Shared HTTP plumbing: bounded reads + APIError construction
├── pool.go              # AuthPool + PooledClient (rotation + retry for Complete and Stream pre-data failures)
├── retrypolicy.go       # RetryPolicy (configurable exponential backoff with jitter) + RetryHook
├── circuitbreaker.go    # CircuitBreaker (3-state: closed → open → half-open)
├── middleware.go        # LoggingClient decorator
├── errors.go            # APIError type + Is*() helpers
├── llm_test.go          # Integration and unit tests
├── retrypolicy_test.go  # RetryPolicy unit tests
├── circuitbreaker_test.go # CircuitBreaker unit tests
│
└── provider/                        # Provider adapters (database/sql driver pattern)
    ├── all.go                       # Blank-imports all sub-packages
    ├── anthropic/anthropic.go       # Native Anthropic API adapter
    ├── gemini/gemini.go             # Native Google Gemini API adapter
    ├── openaicompat/                # OpenAI-compatible chat + modality adapter
    ├── openairesponses/responses.go # OpenAI Responses API adapter (GPT-5 reasoning)
    └── openaibatch/openaibatch.go   # OpenAI Batch API driver (async; Files + Batches REST)
```

Provider implementations register themselves via `init()` using
`llm.RegisterProvider()`, `llm.RegisterModalityDriver()`, and/or
`llm.RegisterBatchProvider()`. Consumers that construct clients or invoke pure
adapter encodability validation must add a blank import:
`_ "github.com/bds421/rho-llm/provider"`.

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
                      │  → always newPooledClient()    │
                      └──────────────┬─────────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │       PooledClient            │
                      │  (retry + rotation + circuit  │
                      │   breaker + retry hooks)      │
                      └──────────────┬────────────────┘
                                 │ wraps N SingleClients (1 per key)
                                 ▼
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
        ├── {Type: ContentDocument, Document: DocumentSource{base64, media_type}}
        ├── {Type: ContentToolUse, ToolUseID, ToolName, ToolInput, ThoughtSignature}
        └── {Type: ContentToolResult, ToolResultID, ToolResultContent, IsError}
```

`ContentImage` parts are fully implemented across all three adapters. Each adapter validates images via `ValidateImageSource()` and serializes to its native wire format: Anthropic uses inline `image` blocks with a `source` object; Gemini uses `inlineData` parts; OpenAI-compatible switches content from string to array with `image_url` data URIs. Supported media types: `image/jpeg`, `image/png`, `image/gif`, `image/webp`.

`ContentDocument` parts carry inline PDFs (base64), validated via `ValidateDocumentSource()`. Gemini serializes them as `inlineData` and Anthropic as native `document` blocks (both parse the PDF text layer); OpenAI-compatible adapters emit an `image_url` data URI for vision models such as xAI Grok. The OpenAI Responses adapter returns an explicit error for documents rather than dropping them silently. Supported media types: `application/pdf`.

`ThoughtSignature` on `tool_use` parts is a Gemini 3 requirement — the model returns an opaque signature that must be echoed back in the corresponding `tool_result` message. The adapters handle this automatically.

### Stream Events

```go
type StreamEvent struct {
    Type EventType // EventContent | EventToolUse | EventThinking | EventDone | EventError

    Text     string     // content
    ToolCall *ToolCall  // tool_use
    Thinking string     // thinking (Anthropic extended thinking)

    InputTokens    int    // usage / done (-1 = not reported)
    OutputTokens   int    // usage / done (-1 = not reported)
    ThinkingTokens int    // Gemini: tokens consumed by thinking (0 for other providers)
    StopReason     string // done: "end_turn" | "tool_use" | "max_tokens"

    CacheCreationTokens int // Anthropic: tokens written to cache (EventDone)
    CacheReadTokens     int // Anthropic/Gemini: tokens read from cache (EventDone)

    Error        string // error
}
```

**Token counts:** In streaming, token counts are `-1` (`TokensNotReported`) if the provider didn't report them (connection dropped, provider quirk). This distinguishes "not reported" from a legitimate "zero tokens" response. All four adapters honor this — including Gemini, whose `usageMetadata` is now decoded through a pointer so an absent block reports `-1` rather than a fake `0`.

**Completion is explicit (v0.4.0):** every stream terminates in exactly one of two ways — an `EventDone` (stop reason + final usage) or an error. If the connection closes mid-turn without the wire protocol's final event (`message_delta` / a `finishReason` / `[DONE]` / `response.completed`), the adapter yields an explicit error wrapping `io.ErrUnexpectedEOF` instead of ending silently, so a truncated turn is never mistaken for a complete one — and `PooledClient` can classify the pre-data case as retryable. An explicit `[DONE]` without a `finish_reason` (a known local-server quirk) still synthesizes a stop reason rather than erroring.

---

## 5. Provider & Protocol Resolution (`provider.go`, `factory.go`)

### Protocol Routing

Four wire protocols are supported:

| Protocol | Adapters | Notes |
|---|---|---|
| `anthropic` | `AnthropicClient` | Native SSE streaming, `x-api-key` header |
| `gemini` | `GeminiClient` | Native REST + SSE, `x-goog-api-key` header |
| `openai_compat` | `OpenAICompatClient` | Standard `/chat/completions` with SSE streaming |
| `openai_responses` | `OpenAIResponsesClient` | OpenAI `/responses` for GPT-5 reasoning-effort control (auto-selected for `ResponsesAPI` models) |

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

`Config.BaseURL` and `Config.AuthHeader` always take precedence, enabling proxy routing and custom deployments. Unknown providers (not in presets) **must** set `BaseURL` — without it, `NewClient` returns an error to prevent typos from silently defaulting to an incorrect endpoint.

---

## 6. Factory (`factory.go`)

```
NewClient(cfg)
    └── NewClientWithKeys(cfg, []string{cfg.APIKey})
            └── newPooledClient(cfg, keys)  // All clients get retry/backoff
                    ├── AuthPool (1 profile, even if APIKey is empty)
                    └── clientFunc:
                          ├── ResolveProtocol(cfg)
                          ├── getProviderFactory(protocol)
                          │     "anthropic"    → anthropic.New(cfg)
                          │     "gemini"       → gemini.New(cfg)
                          │     "openai_compat"→ openaicompat.New(cfg)
                          └── cfg.LogRequests? → WithLogging(client)
```

All chat clients — including keyless local providers (Ollama, vLLM, LM Studio)
— go through `PooledClient` to get exponential backoff on transient errors (429,
503, 502). `NewClient` always delegates to `NewClientWithKeys`, and
`NewClientWithKeys` with a nil/empty key slice falls back to `cfg.APIKey` rather
than bypassing the pool.

### Modality factory

```
NewModalityClient(cfg)
    ├── resolve exact model + protocol
    ├── CheckBaseURL(cfg)
    ├── getModalityDriver(protocol)
    ├── driver.New(cfg) → durable protocol client
    └── capabilityValidatedModalityClient
```

`ModalityClient` exposes embeddings, image generation, speech synthesis, and
transcription without placing any OpenAI endpoint or format vocabulary in the
root package. The OpenAI-compatible driver is registered next to its chat
driver and implements those operations on the same concrete `openaicompat.Client`.
Each constructed modality client retains one `SafeHTTPClient` until `Close`, so
workers reuse connections instead of constructing a transport per dispatch.

Before every network effect, the factory wrapper performs generic structure and
reviewed-capability validation, then delegates pure encodability validation to
the registered driver. Provider-specific URL paths, JSON/multipart bodies,
output format tokens, and response parsing remain in that driver. Network calls
use `DoHTTP`, giving remote and local deployments the same explicit proxy,
retry-hook, backoff, caller-cancellation, bounded-read, and classified-error
behavior. Credential rotation/circuit breaking remain chat-pool concerns; a
modality client represents one exact deployment credential owned by its worker.

Image geometry/count/format, voice, and language are request values authored by
the application. Rho does not choose preferred values or impose product-level
image limits; it only rejects structurally invalid or unencodable combinations.
Canonical base64 plus signature/MIME agreement is required for inline image/PDF
inputs, and supported transcription audio signatures are checked before upload.

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

Cooldown durations are error-type-dependent (configurable via `Config`):

| Error | Default Cooldown | Config Field |
|---|---|---|
| Rate limit (429) | 60 s | `CooldownRateLimit` |
| Overloaded (503) | 30 s | `CooldownOverload` |
| Any other retryable | 10 s | `CooldownDefault` |

### PooledClient

`PooledClient` wraps N single-provider clients (one per API key), all sharing the same `AuthPool`. All clients — including keyless local providers — go through `PooledClient` for retry/backoff on transient errors. On failure:

```
Complete():
    loop (maxRetries = clamp(pool.HealthyCount(), 3, 10)):  // Min 3, capped at 10
        0. Circuit breaker gate: if open → return ErrCircuitOpen immediately
        1. Call current client
        2. Success → MarkSuccess(), breaker.RecordSuccess(), return
        3. Non-retryable, non-auth error (400) → return immediately
        4. Auth error (401/403) OR retryable error (429/503/502):
             breaker.RecordFailure() (auth errors exempt — bad key ≠ broken endpoint)
             MarkFailed(err):
               - Auth errors: IsHealthy = false (permanent)
               - Others: cooldown (temporary, configurable)
             rotateClient() → GetAvailable() → create new single client
             if rotation fails:
               - Auth error → return immediately (dead key is dead)
               - Transient error → retryPolicy.Delay(attempt) → sleep & retry same

Stream():
    loop (maxRetries = min(max(pool.Count(), 3), 10)):
        0. Circuit breaker gate: if open → return ErrCircuitOpen immediately
        1. Start streaming from current client
        2. If error BEFORE any event yielded (firstEvent == true):
             a. Non-retryable, non-auth error → return immediately
             b. Auth or retryable error → breaker.RecordFailure(), MarkFailed(err), rotateClient():
                  - rotation fails + auth error → return immediately
                  - rotation fails + transient → retryPolicy.Delay(attempt) & retry
                  - rotation succeeds → retry with new client
        3. If error AFTER events yielded (firstEvent == false):
             → pass through to caller (no retry — would duplicate content)
        4. Stream completes → MarkSuccess(), breaker.RecordSuccess(), return
```

**Caller cancellation (v0.4.0):** before marking a key failed or recording a breaker failure, both `Complete` and `Stream` check `ctx.Err()`. A request aborted by the caller's context surfaces through net/http as a `*url.Error` that satisfies `net.Error` — without this check it looked like a transient provider failure, so a cancelled request would cool down a healthy key and trip the breaker. Caller cancellation now returns immediately, touching neither key nor breaker health. The HTTP client's own timeout (`context.DeadlineExceeded`) is *not* exempted — it remains a retryable transient failure. TLS certificate-verification failures are classified non-retryable (no retry can fix endpoint identity).

**Auth error handling:** When a 401/403 occurs, the key is marked permanently unhealthy (`IsHealthy = false`), not just put in cooldown. If rotation fails because no healthy keys remain, the error returns immediately — no point backing off with a dead key.

**Pre-data vs mid-stream retry:** A stream that fails on the initial HTTP connection (429/503 before any SSE events) is functionally identical to a failed `Complete` — no data has reached the caller, so retry with rotation is safe. Once any event has been yielded via `for-range`, retrying would replay content from scratch with no way for the caller to detect duplication, so mid-stream errors pass through immediately.

`rotateClient()` does NOT close the replaced client — doing so would race with in-flight requests still holding a reference. Orphaned clients are garbage collected; their `Close()` method (which drains idle HTTP connections via `CloseIdleConnections()`) is called by the `refCountedClient` mechanism when the last reference is released.

**`Close()` limitation:** The `Close()` methods on all three adapters call `httpClient.CloseIdleConnections()`, which only drains idle connections. Active streaming connections are not forcefully terminated — they are cleaned up by context cancellation or HTTP timeout. This is the correct Go pattern: `http.Client` has no API for forceful connection termination. To cleanly abort an in-progress stream, cancel the context passed to `Stream()`.

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

### Retry Policy (`retrypolicy.go`)

`RetryPolicy` provides configurable exponential backoff with jitter. The `DefaultRetryPolicy` matches the original hardcoded behavior (1s base, 30s cap, 2x factor, ±25% jitter).

```
attempt 0: base × 2⁰ = ~1s  (0.75–1.25s)
attempt 1: base × 2¹ = ~2s  (1.50–2.50s)
attempt 2: base × 2² = ~4s  (3.00–5.00s)
attempt 3: base × 2³ = ~8s  (6.00–10.0s)
...capped at maxDelay (default 30s)
```

All parameters are configurable through `Config.RetryPolicy`: `BaseDelay`,
`MaxDelay`, `Factor`, and `Jitter`. `RetryPolicy.Delay` is the single public
backoff calculation; no compatibility wrapper is retained.

### Circuit Breaker (`circuitbreaker.go`)

Port of kit's 3-state machine, zero external dependencies:

```
CircuitClosed ──(threshold consecutive failures)──→ CircuitOpen
CircuitOpen   ──(cooldown elapsed, 1 probe)───────→ CircuitHalfOpen
CircuitHalfOpen ──(probe success)─────────────────→ CircuitClosed
CircuitHalfOpen ──(probe failure)─────────────────→ CircuitOpen
```

- **Nil-safe:** all methods are no-ops on nil receiver (circuit always allows)
- **Thread-safe:** `sync.Mutex` protects all state transitions
- **Callback-safe:** `WithOnStateChange` callbacks are invoked outside the mutex, so they may safely call back into the circuit breaker
- **Auth-aware:** the optional `WithSuccessPredicate` lets `Execute` exclude an error class (e.g. auth errors) from failure counting. The `PooledClient` doesn't use it — it drives the breaker manually and applies the auth exemption inline (bad key ≠ broken endpoint) — so no dead predicate is wired on the pool's breaker.
- **Fail-fast on open:** when the circuit is open, `Complete` and `Stream` return `ErrCircuitOpen` immediately instead of burning retry iterations
- **No half-open wedge (v0.4.0):** a half-open probe admits one request and rejects the rest until it reports back. If that probe is abandoned (iterator dropped, goroutine died) without recording success or failure, a fresh probe is admitted after another cooldown elapses rather than wedging the circuit half-open forever.
- **Probe re-arm (Unreleased):** a probe whose outcome is non-diagnostic — the caller cancelled, or it drew a client-side error (400) that says nothing about endpoint health — is re-armed immediately (`ReleaseProbe`) instead of stranding other traffic for an extra cooldown. Re-arming is probe-identity-scoped (`allow()` hands back a probe token; `ReleaseProbe(token)` re-arms only that exact probe), so a stale cancel/error from a superseded or never-admitted request can't reset an unrelated goroutine's in-flight probe.
- Enabled by default via `DefaultConfig()` with threshold=5, cooldown=30s

### Retry Hook (`retrypolicy.go`)

`RetryHook` is a `func(RetryEvent)` callback fired during retry lifecycle events:

| Event Type | When |
|---|---|
| `RetryAttemptFailed` | An attempt returned a retryable error |
| `RetryRotating` | Pool is rotating to a different auth profile |
| `RetryBackingOff` | Client is sleeping before next attempt |
| `RetryCircuitOpen` | Circuit breaker rejected an attempt |
| `RetryExhausted` | All retry attempts exhausted |

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
    ID, Provider         string
    MaxTokens            int     // Model output limit (0 = use config)
    ContextWindow        int     // Max input tokens
    InputPricePer1M      float64 // USD pricing
    OutputPricePer1M     float64
    CacheWritePricePer1M float64 // Anthropic: per 1M cache creation tokens
    CacheReadPricePer1M  float64 // Anthropic/Gemini: per 1M cached input tokens
    SupportsThinking     bool    // API-controlled thinking budgets (Anthropic)
    ThoughtSignature     bool    // Gemini 3: must echo thought_signature in tool results
    Thinking             bool    // Intrinsic reasoning models (DeepSeek, Grok)
    NoToolSupport        bool    // Model lacks function calling capabilities
    Label                string  // Short display name
}
```

### Thinking & Reasoning (Semantic Capabilities)

Different LLM providers implement chain-of-thought reasoning in fundamentally different ways. The registry abstracts these semantic capabilities:

1. **API-Controlled Budgets (`SupportsThinking: true`)**
   Models like Anthropic's Claude 4 series require the client to explicitly allocate a "thinking budget" in the API request payload. The config `ThinkingLevel` is mapped into this budget. Only the `anthropic` and `gemini` adapters support *requesting* thinking — the OpenAI-compatible adapter returns an explicit error if `ThinkingLevel` is set, since the OpenAI chat completions API has no equivalent parameter. However, all three adapters **parse** thinking from responses: Anthropic via `thinking` blocks, Gemini via `thought: true` parts, and OpenAI-compat via the `reasoning_content` field.

2. **Intrinsic Reasoning (`Thinking: true`)**
   Models like DeepSeek-R1 and Grok 4 Reasoning emit chain-of-thought intrinsically inside their standard output streams. They do not require specific API flags to enable this, but the registry flags them so your application knows they will consume output tokens for reasoning before answering.

**Registered providers:**

| Provider | Models |
|---|---|
| Anthropic | claude-opus-4-8, claude-opus-4-7, claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5 |
| xAI | grok-4.3, grok-4.20-beta, grok-4-1-fast-{reasoning,non-reasoning}, grok-4-fast-{reasoning,non-reasoning}, grok-code-fast-1, grok-3, grok-3-mini |
| Gemini | gemini-3.6-flash, gemini-3.5-{flash,flash-lite}, gemini-3.1-flash-lite, gemini-3.1-pro-preview, gemini-3-{pro,flash}-preview, gemini-2.5-{pro,flash,flash-lite} |

`EstimateCost(CostInput{...})` returns a USD float from registry pricing. Accepts all token types including `ThinkingTokens`, `CacheCreateTokens`, and `CacheReadTokens` for accurate cache-aware pricing. Returns `0` if the model is unknown. Negative token counts (e.g. `TokensNotReported = -1`) are clamped to 0.

**Runtime extension (v0.4.0):** built-in metadata is curated for 15 providers, but `RegisterModel(ModelInfo)` and `RegisterModelAlias(alias, modelID)` add or override entries at runtime — so unlisted/newly released models get cost estimation, capability flags, and discovery (and stale built-in pricing can be corrected) without a library release. The registry maps were "immutable after init"; they are now guarded by an `RWMutex` taken by every reader (`GetModelInfo`, `EstimateCost`, `ResolveModelAlias`, `Models`, …), so runtime registration is safe for concurrent use.

---

## 10. Logging Middleware (`middleware.go`)

`LoggingClient` is a **decorator** over any `Client`. It logs metadata only — never message content — making it safe to enable in production:

**On Complete:**
```
INFO: component=llm msg="complete request" provider=anthropic model=claude-sonnet-4-6 messages=5 tools=2 max_tokens=8192
INFO: component=llm msg="complete done" ... elapsed=1.234s tokens_in=1200 tokens_out=350 stop=end_turn cost=0.00915
```

**On Complete (with caching):**
```
INFO: component=llm msg="complete done" ... tokens_in=1200 tokens_out=350 ... cache_write=5000 cache_read=3000
```

**On Stream:**
```
INFO: component=llm msg="stream request" provider=anthropic model=claude-sonnet-4-6 messages=5 ...
INFO: component=llm msg="stream done" ... elapsed=3.412s chunks=87 tokens_in=1200 tokens_out=450 stop=end_turn cost=0.0111
```

`cache_write` and `cache_read` are only included in log output when > 0.

Errors are logged at `Warn` level (`"complete failed"`, `"stream failed"`).

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
- Extended thinking: enabled via `thinking: {type: enabled, budget_tokens: N}` — budget mapped from ThinkingLevel (`ThinkingNone`→disabled, `ThinkingLow`→4096, `ThinkingMedium`→16384, `ThinkingHigh`→65536)
- Tool use: native anthropic format with `type: tool_use` content blocks
- Context caching: `cache_control: {type: "ephemeral"}` on content blocks, system blocks, and tool definitions. Cache tokens reported in `usage` for both Complete and Stream responses.

### Gemini (`provider/gemini/`)

- Endpoint: `https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent`
- Auth: `x-goog-api-key` header (moved from URL query parameter in v0.1.9 to prevent key leakage)
- Streaming: SSE with JSON chunks
- `ThoughtSignature`: when a model has `ThoughtSignature: true` in the registry, function call responses include a `thought_signature` field that must be preserved and echoed in subsequent `tool_result` parts
- Thinking: parts with `thought: true` are routed to `resp.Thinking` / `EventThinking` (not mixed into `Content`). `thoughtsTokenCount` from usage metadata is exposed as `resp.ThinkingTokens` / `event.ThinkingTokens`, separate from `OutputTokens` (which maps to `candidatesTokenCount` only). Anthropic and OpenAI-compat bundle thinking tokens into `OutputTokens`; for those providers `ThinkingTokens` is 0.
- System prompt: mapped to `systemInstruction.parts[0].text`
- Context caching: `cachedContent` field in request references a pre-created cache by name. `cachedContentTokenCount` from response usage is mapped to `CacheReadTokens`.

### OpenAI-Compatible (`provider/openaicompat/`)

- Endpoint: configurable `BaseURL` + `/chat/completions`
- Auth: `Authorization: Bearer <key>` (or empty for no-auth providers)
- Streaming: OpenAI SSE format with `data: {delta: ...}` chunks
  - `stream_options: {include_usage: true}` is set automatically to receive token usage
  - OpenAI sends `finish_reason` and `usage` in **separate chunks**: the parser accumulates state across chunks and emits `EventDone` with complete data after `[DONE]`
  - Tool calls are flushed before `EventDone` even if `finish_reason` is missing (handles network drops and spec-violating servers like Ollama)
- Tool use: OpenAI function-calling format, translated to/from the shared `ToolCall` types. Multiple tool results in a single Anthropic-style message are expanded to separate `role: "tool"` messages as OpenAI requires.
- Thinking: `reasoning_content` field (used by Ollama Qwen3, DeepSeek-R1, etc.) is parsed into `resp.Thinking` / `EventThinking`. Requesting thinking via `ThinkingLevel` is still rejected — the adapter can *parse* thinking from models that think by default, but cannot *request* it.
- Works for: OpenAI, xAI/Grok, Groq, Cerebras, Mistral, OpenRouter, Ollama, vLLM, LM Studio, any custom proxy

---

## 12. Config Reference

```go
type Config struct {
    Provider         string         // Provider name (see presets)
    Model            string         // Model ID or alias
    APIKey           string         // API key (empty OK for no-auth providers)
    MaxTokens        int            // Max output tokens (default: 8192)
    Temperature      *float64       // Sampling temperature (nil = omit from wire, provider default)
    ThinkingLevel    ThinkingLevel  // ThinkingLow | ThinkingMedium | ThinkingHigh (zero = none)
    Timeout          time.Duration  // HTTP timeout (default: 120s)
    BaseURL          string         // Override provider endpoint
    AuthHeader       string         // Override auth header ("Bearer", "x-api-key", "")
    ProviderName     string         // Override Client.Provider() return value
    LogRequests      bool           // Enable metadata logging
    BlockPrivateBaseURL bool        // Opt-in SSRF guard: reject loopback/private BaseURL hosts

    // (Request carries ResponseFormat for structured output; per-Config safety
    //  limits — MaxErrorBodyBytes, MaxResponseBodyBytes, … — are read via the
    //  Effective*() accessors. See README "Config Reference" for the full set.)

    // Resilience
    RetryPolicy       *RetryPolicy  // Configurable backoff (nil = DefaultRetryPolicy)
    CircuitThreshold  int           // Consecutive failures to open circuit (default: 5)
    CircuitCooldown   time.Duration // Open→half-open cooldown (default: 30s)
    CooldownRateLimit time.Duration // Profile cooldown for 429 errors (default: 60s)
    CooldownOverload  time.Duration // Profile cooldown for 503 errors (default: 30s)
    CooldownDefault   time.Duration // Profile cooldown for other errors (default: 10s)
    RetryHook         RetryHook     // Observability hook for retry events (not serialized)
}
```

**Defaults (`DefaultConfig()`):**
```
Provider:         "anthropic"
Model:            "claude-sonnet-4-6"
MaxTokens:        8192
Temperature:      nil (omitted from wire)
ThinkingLevel:    "" (ThinkingNone)
Timeout:          120s
AuthHeader:       "Bearer"
CircuitThreshold: 5
CircuitCooldown:  30s
```

---

## 13. Conversations, Sessions & Provider Handoff

Added in the conversation/handoff work, this layer sits **above** the stateless `Client`
interface — it never changes how a single request is made, it just accumulates and
re-points history. The design follows a cross-library survey (pi, Vercel AI SDK,
LangGraph, OpenAI Agents SDK, and the Go ecosystem) and is tuned to Go idioms.

### Two types: a value and a driver

| Type | Role | Concurrency | Serialization |
|------|------|-------------|---------------|
| `Conversation` (`conversation.go`) | Plain, provider-neutral transcript: `SchemaVersion`, `System`, `Tools`, `Messages`, `Usage`. Owns **no** client. `Clone()` deep-copies it via the versioned JSON round-trip. | none (a value) | versioned JSON via explicit tags |
| `Session` (`session.go`) | Concurrency-safe driver: holds a `*Conversation` + current `Client` + a base `Request`; appends each turn; supports handoff. | two mutexes — one serializes turns, one briefly guards state (v0.4.0) | persist the underlying `Conversation` |

**Session locking (v0.4.0):** turns (`Send`/`Stream`) are serialized by `turnMu`, but snapshot reads (`Conversation`/`Usage`/`Provider`) and `SwitchProvider` take a separate, briefly-held `stateMu` — so they never block on a (possibly minutes-long) streaming turn. A turn is built from a state snapshot and **committed atomically** (input messages + assistant reply together) only after it succeeds; a failed, aborted, or abandoned turn records nothing, and an in-flight turn is invisible to snapshots until it completes. `Conversation()` returns a deep `Clone()`, so mutating the snapshot can never corrupt the live transcript.

This split is deliberate. The serializable data (`Conversation`) is decoupled from the
mutable runtime concerns (client, mutex) — so persistence is just `json.Marshal(conv)`, and
the stateful ergonomics live in `Session` without contaminating the stored form. Generation
stays **stateless**: `Client.Complete`/`Stream` remain `Request`-in / `Response`-out.

### Provenance drives handoff

Every assistant `Message` records `Provider`/`Model`/`StopReason` provenance (all `omitempty`,
so older blobs stay valid). This is what makes cross-provider handoff decidable from the data
alone: a thinking block is replayable verbatim **only** to the same provider that produced it.

### The handoff pass (`NormalizeForProvider`)

`Session` applies `NormalizeForProvider(messages, targetProvider)` before every request (so it
also self-heals same-provider transcripts); direct `Client` users can call it explicitly. It is
the single choke point for all cross-provider concerns and never mutates its input:

1. **Thinking degradation** — a thinking block is kept verbatim only when `msg.Provider ==
   targetProvider` *and* it carries a provider-valid signature (or is redacted). Otherwise the
   reasoning is preserved by converting it to a plain text block. (A foreign/unsigned thinking
   block would be rejected, e.g. Anthropic requires the original `signature` on replay.)
2. **Orphan tool-call repair** — a `tool_use` with no matching `tool_result` anywhere gets a
   synthetic error result appended after its turn (every provider rejects an unanswered call).
3. **Errored-turn drop** — assistant turns with `StopReason` `error`/`aborted` are removed,
   along with the tool results that referenced their now-gone calls.
4. **Empty-message drop** — messages left contentless by the above are removed.

Only the Anthropic adapter replays thinking blocks (it captures the `signature` from both the
non-streaming response and the streaming `signature_delta`, surfaced on `EventDone` as
`StreamEvent.ThinkingSignature`). The other three adapters skip thinking parts by
switch-omission; by the time history reaches them, normalization has already degraded any
cross-provider thinking to text. So **cross-provider handoff is intentionally lossy for
reasoning** (signatures don't transfer) but lossless for text, images, documents, and tools —
matching the reality that every surveyed library either drops reasoning on a switch or pins it
to one provider.

### Cost across a handoff

`Usage` sums **cost per response** using each response's own model (`EstimateCost`), so a
conversation that spans multiple providers/models still totals correctly — accumulating raw
tokens and pricing once at the end would be wrong when models differ.

### Persistence (`store.go`)

Persistence is a pluggable `Store` (`Save`/`Load`/`Delete` by id) kept **out** of the message
model: `MemoryStore` (mutex-guarded, stores serialized bytes so loads are independent copies)
and `FileStore` (one JSON file per id; ids are restricted to a single path segment, so traversal
like `../evil` is rejected; **writes are atomic and durable** — temp file + fsync + rename, then
a best-effort directory fsync — so neither a concurrent reader nor a crash ever sees a
half-written conversation, v0.4.0) ship in the library, both stdlib-only. Under the hood it is still just
`json.Marshal(conv)` / `LoadConversation(blob)`. `LoadConversation` validates
`schema_version` (tolerating pre-versioned blobs as v1, rejecting versions newer than the
build) — the format is keyed by explicit JSON tags and is **not** tied to Go type/package
names, so stored conversations stay portable across versions.

---

## 14. Batch API (async)

The Batch layer is the **asynchronous, bulk counterpart** to `Client`. It sits beside the
synchronous stack (it does not wrap or modify it) for offline processing at ~50% cost: submit
many requests, poll, then fetch results — possibly across process restarts.

### Interface & registry

`BatchClient` (`batch.go`) is a separate interface — `Submit` / `Get` / `Results` / `Cancel` /
`Close` — that deliberately does **not** satisfy `Client`, since batch is submit→poll→fetch, not
one-in/one-out. Drivers register per protocol via `RegisterBatchProvider` (`batchregister.go`), a
parallel of the `Client` driver registry; `NewBatchClient(cfg)` (`factory.go`) gates on
`ProviderPreset.SupportsBatch` (OpenAI only — most `openai_compat` resellers don't expose
`/v1/batches`), then resolves the driver by protocol. `WaitForBatch` polls `Get` to a terminal
status under caller-controlled `context`.

### Neutral types

- `BatchItem` — a neutral `ItemID` plus exactly one of a chat `Request` **xor** an `EmbeddingRequest`.
- `BatchResult` — correlated by `ItemID`; carries a `*Response`, an `*EmbeddingResponse`, or an
  `*APIError`.
- `BatchHandle` — serializable exact schema v2. It exposes only a neutral external `ID`, closed
  operation and lifecycle, request counts, recovery identity, and bounded opaque adapter state.
  Missing, older, and newer schemas fail closed. `Get`, `Results`, and `Cancel` consume the complete
  handle, so endpoint and remote-file vocabulary never leak into the root API.
- `BatchOptions.MaxTurnaround` — a required caller-authored duration. Root bounds it but never
  selects or defaults it; each adapter proves exact wire representability before I/O.

### OpenAI driver (`provider/openaibatch/`)

OpenAI batches go through the **Files API**: assemble a JSONL file (one `{custom_id, method, url,
body}` line per item), upload it (`purpose=batch`), create a batch referencing it, poll, then
**download** the result file. A batch targets a single endpoint, so `Submit` validates that all
items are homogeneous (all `/v1/chat/completions`, all `/v1/responses`, or all `/v1/embeddings`)
and rejects mixed/duplicate/empty sets **before any network effect**. The per-item chat-vs-responses
split reuses the live `ResolveProtocol` rule, so a GPT-5 item batches to `/v1/responses` exactly as
`Complete` would send it. The adapter maps exact `24h` turnaround to the OpenAI wire and owns the
endpoint plus input/output/error file IDs in strict adapter-state v1. It normalizes `validating` to
`queued`, `in_progress`/`finalizing` to `running`, and rejects unknown provider states. OpenAI's
50,000-request, 50,000-total-embedding-input, and 200 MB encoded-JSONL limits are
adapter-owned and enforced before the Files API.

### Single source of wire-format truth

Each batch line's `body` is built — and each result line parsed — by the **same** translation code
as a live call, exposed as thin same-package helpers (`openaicompat.BuildChatBatchLineBody` /
`ParseChatBatchResultBody`, the `openairesponses` equivalents, and root-package
`BuildEmbeddingsBatchLineBody` / `ParseEmbeddingsBatchResultBody`). A parity test asserts a batch
line body is **byte-identical** to what `Complete` POSTs, so the two paths can never drift.

### Cost & limits

`CostInput.Batch` applies the 50% discount in `EstimateCost`; `Usage.AddBatchResponse` folds batch
results at batch pricing (sharing the `TokensNotReported` clamp with `AddResponse`).
`Config.MaxBatchDownloadBytes` (default 256 MB) bounds result-file downloads — far above the 32 MB
sync-response cap, so a large completed batch is not silently truncated.

### Cross-provider shape

The interface is provider-agnostic by design: the Files-API transport is OpenAI-specific, while
Anthropic Message Batches (inline submit + `results_url`) would register a sibling driver reusing
the Anthropic adapter's translation the same way. Only the transport differs; results normalize to
the same `[]BatchResult`.

---

## 15. Design Decisions

### Why a unified interface over provider SDKs?

Provider SDKs have incompatible types, inconsistent error models, and evolve independently. A thin HTTP adapter per protocol gives full control over retry logic, streaming, and error classification without taking on SDK dependency churn. The three protocols (Anthropic native, Gemini native, OpenAI-compat) cover the entire current provider landscape.

### Why `*APIError` instead of typed error vars?

`errors.As()` on a concrete `*APIError` allows callers to inspect the HTTP status code without parsing strings. The `Retryable bool` field moves the retry classification decision into the library, where context (provider, status code, body content) is available.

### Why `AuthPool` rotates rather than load-balances?

Round-robin rotation on failure (not on every request) minimizes unnecessary API calls. Using the same profile for consecutive requests maximizes cache hits and context locality on the provider side. Load balancing would require tracking concurrent request counts per profile, adding complexity with marginal benefit for most workloads.

### `ThoughtSignature` (Gemini 3)

Gemini 3 models return an opaque `thought_signature` field in function call responses. This signature encodes the model's internal reasoning state and must be echoed in the `tool_result` message, or the model will repeat the same tool call. The adapters preserve this field transparently through the `ContentPart.ThoughtSignature` field so calling code does not need to handle it explicitly.

### Context Caching

Anthropic and Gemini offer context caching with fundamentally different models:

| | Anthropic `cache_control` | Gemini `cachedContent` |
|---|---|---|
| **Pattern** | Inline per-block annotation in the request | Separate resource created via dedicated API, then referenced by name |
| **Statefulness** | Stateless — annotation goes in same request | Stateful — two-stage: create cache, then use it |
| **Granularity** | Per content block, system block, or tool definition | All-or-nothing on a context prefix |

**Anthropic (full support):**
- `ContentPart.CacheControl = true` adds `cache_control: {type: "ephemeral"}` to that content block
- `Request.SystemCacheControl = true` sends the system prompt as structured content blocks (array of `{type: "text", text: "...", cache_control: {type: "ephemeral"}}`) instead of a plain string. The `cache_control` is applied to the last block.
- `Tool.CacheControl = true` adds `cache_control` to the tool definition JSON
- `Response.CacheCreationTokens` / `CacheReadTokens` report cache usage from the `usage` object
- `StreamEvent.CacheCreationTokens` / `CacheReadTokens` are extracted from `message_start` events

**Gemini (reference only):**
- `Request.CachedContent` sets `cachedContent` in the wire request, referencing a pre-created cache by name
- Cache lifecycle management (create/list/delete) is out of scope — callers manage it externally
- `Response.CacheReadTokens` is populated from `usageMetadata.cachedContentTokenCount`

**OpenAI-compatible:** All cache fields are silently ignored (no API support).
