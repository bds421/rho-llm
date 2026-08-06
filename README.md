# rho-llm

Multi-provider LLM client for Go. Streaming, tool use, image/vision + PDF/document input, extended thinking, structured output (JSON mode), serializable conversations with cross-provider handoff, embeddings, image generation, audio (speech/transcription), an async **Batch API** (~50% cheaper bulk processing), OAuth device flow, and auth pool rotation. Includes thread-safe concurrency management to prevent redundant HTTP client allocations during concurrent rate-limit failovers. The library imports only the Go standard library (the `examples/` use `joho/godotenv`).

**Requires Go 1.26.4+** (`go 1.26.4` in `go.mod`; 1.26.4 fixes stdlib CVEs in `net/textproto` and `crypto/x509`).

## Install

```bash
go get github.com/bds421/rho-llm
```

## Supported Providers

| Provider | Protocol | Auth | Default BaseURL |
|----------|----------|------|-----------------|
| Anthropic/Claude | Native | x-api-key | api.anthropic.com |
| Google Gemini | Native | x-goog-api-key | generativelanguage.googleapis.com |
| OpenAI | OpenAI-compat / Responses | Bearer | api.openai.com/v1 |
| xAI/Grok | OpenAI-compat | Bearer | api.x.ai/v1 |
| Groq | OpenAI-compat | Bearer | api.groq.com/openai/v1 |
| Cerebras | OpenAI-compat | Bearer | api.cerebras.ai/v1 |
| Mistral | OpenAI-compat | Bearer | api.mistral.ai/v1 |
| OpenRouter | OpenAI-compat | Bearer | openrouter.ai/api/v1 |
| DashScope/Qwen | OpenAI-compat | Bearer | dashscope-intl.aliyuncs.com/compatible-mode/v1 |
| DeepSeek | OpenAI-compat | Bearer | api.deepseek.com |
| Cohere | OpenAI-compat | Bearer | api.cohere.ai/compatibility/v1 |
| Together AI | OpenAI-compat | Bearer | api.together.xyz/v1 |
| Fireworks | OpenAI-compat | Bearer | api.fireworks.ai/inference/v1 |
| NVIDIA NIM | OpenAI-compat | Bearer | integrate.api.nvidia.com/v1 |
| Perplexity | OpenAI-compat | Bearer | api.perplexity.ai |
| DeepInfra | OpenAI-compat | Bearer | api.deepinfra.com/v1/openai |
| Z.ai/GLM | OpenAI-compat | Bearer | api.z.ai/api/openai/v1 |
| MiniMax | OpenAI-compat | Bearer | api.minimax.io/v1 |
| Moonshot/Kimi | OpenAI-compat | Bearer | api.moonshot.ai/v1 |
| Ollama | OpenAI-compat | None | localhost:11434/v1 |
| vLLM | OpenAI-compat | None | localhost:8000/v1 |
| LM Studio | OpenAI-compat | None | localhost:1234/v1 |

> **Counting:** 23 built-in provider presets across 4 wire protocols — the OpenAI
> row spans two (`openai_compat` and the auto-selected `openai_responses` for GPT-5
> reasoning models), and `claude`/`google`/`grok`/`qwen`/`z-ai`/`glm`/`kimi` are aliases
> of providers already listed. 18 of them ship curated **model metadata** (pricing + capabilities)
> for cost estimation and discovery; the rest work fully — any model ID can be passed
> directly — and you can add metadata for any model at runtime (see
> [Model Registry](#model-registry)). Unknown providers work too via `Config.BaseURL`.

## Quick Start

This example demonstrates a complete request using Google Gemini, but the code is identical for all 23 providers.

```go
import _ "github.com/bds421/rho-llm/provider" // required: register adapters

// 1. Configure and initialize
cfg := llm.Config{
    Provider: "gemini",
    Model:    "flash", // resolves to gemini-2.5-flash
    APIKey:   os.Getenv("GEMINI_API_KEY"),
}
client, err := llm.NewClient(cfg)
if err != nil {
    panic(err)
}
defer client.Close()

// 2. Send a message
req := llm.Request{
    Messages: []llm.Message{
        llm.NewTextMessage(llm.RoleUser, "Explain quantum entanglement in one sentence."),
    },
}
resp, err := client.Complete(context.Background(), req)

fmt.Println(resp.Content)
```

### Provider Recipes

#### Ollama (local, no API key)
```go
cfg := llm.Config{
    Provider: "ollama",
    Model:    "llama3", // or "mistral", "phi3", etc.
}
```

#### Custom OpenAI-compatible endpoint

Unknown providers (not in the presets list) **must** set `BaseURL`. Without it, `NewClient` returns an error to prevent typos from silently defaulting to an incorrect endpoint.

```go
cfg := llm.Config{
    Provider:   "custom",
    BaseURL:    "http://my-proxy:8080/v1", // required for unknown providers
    APIKey:     "my-key",
}
```

## Image / Vision Support

Send images to vision-capable models using `ContentImage` parts with base64-encoded data. All three protocol adapters serialize images to the correct wire format automatically.

```go
import (
    "encoding/base64"
    "os"
)

imgBytes, _ := os.ReadFile("photo.png")
imgData := base64.StdEncoding.EncodeToString(imgBytes)

req := llm.Request{
    Messages: []llm.Message{{
        Role: llm.RoleUser,
        Content: []llm.ContentPart{
            {Type: llm.ContentText, Text: "What do you see in this image?"},
            {Type: llm.ContentImage, Source: &llm.ImageSource{
                Type: "base64", MediaType: "image/png", Data: imgData,
            }},
        },
    }},
}
resp, err := client.Complete(ctx, req)

// Or use the convenience helper for single-image messages:
msg := llm.NewImageMessage(llm.RoleUser, "image/png", imgData)
```

Supported media types: `image/jpeg`, `image/png`, `image/gif`, `image/webp`.

## Document / PDF Support

Send PDFs to document-capable models using `ContentDocument` parts with base64-encoded data. Gemini and Anthropic parse PDFs natively (preserving text layout); OpenAI-compatible vision models (e.g. xAI Grok) receive the PDF as an `image_url` data URI. The Responses API adapter returns an explicit error rather than silently dropping a document.

```go
pdfBytes, _ := os.ReadFile("invoice.pdf")
pdfData := base64.StdEncoding.EncodeToString(pdfBytes)

req := llm.Request{
    Messages: []llm.Message{{
        Role: llm.RoleUser,
        Content: []llm.ContentPart{
            {Type: llm.ContentText, Text: "Extract the invoice total and date."},
            {Type: llm.ContentDocument, Document: &llm.DocumentSource{
                Type: "base64", MediaType: "application/pdf", Data: pdfData,
            }},
        },
    }},
}
resp, err := client.Complete(ctx, req)

// Or use the convenience helper for single-document messages:
msg := llm.NewDocumentMessage(llm.RoleUser, "application/pdf", pdfData)
```

Supported media types: `application/pdf`.

## Streaming

`client.Stream()` returns a Go 1.23 iterator (`iter.Seq2[StreamEvent, error]`) that yields events as the model generates tokens. This lets you display partial output in real time rather than waiting for the full response. Use `break` to abort early — the iterator cleans up the underlying HTTP connection automatically.

**Completion is explicit.** Every stream either yields an `EventDone` (carrying the stop reason and final token usage) or yields an error — never both, and never neither. If the server closes the connection mid-turn without sending its protocol-final event, the adapter yields an explicit error (wrapping `io.ErrUnexpectedEOF`) instead of ending silently, so a truncated turn can never be mistaken for a complete one. To abort an in-flight stream, cancel the `ctx` you passed to `Stream()` — a caller-cancelled stream is reported to you but is *not* counted as a provider failure (no key cooldown, no circuit-breaker trip).

```go
for event, err := range client.Stream(ctx, req) {
    if err != nil {
        fmt.Printf("Stream error: %v\n", err)
        break
    }
    switch event.Type {
    case llm.EventContent:
        fmt.Print(event.Text)           // Partial text token
    case llm.EventToolUse:
        // Model wants to call a tool (see Tool Use below)
        fmt.Printf("Tool call: %s(%v)\n", event.ToolCall.Name, event.ToolCall.Input)
    case llm.EventThinking:
        fmt.Print(event.Thinking)       // Extended thinking (Anthropic with ThinkingLevel set)
    case llm.EventDone:
        // Final metadata — stop reason and token usage
        fmt.Printf("\nDone: %s (in=%d, out=%d)\n",
            event.StopReason, event.InputTokens, event.OutputTokens)
    }
}
```

### Event types

| Event | Fields | Description |
|-------|--------|-------------|
| `EventContent` | `Text` | A chunk of generated text. Concatenate all chunks for the full response. |
| `EventToolUse` | `ToolCall` (ID, Name, Input) | The model is invoking a tool. Handle it and continue the conversation with the result. |
| `EventThinking` | `Thinking` | Extended thinking output (requires `ThinkingLevel` in config). |
| `EventDone` | `StopReason`, `InputTokens`, `OutputTokens` | Stream completed. `StopReason` is normalized across all providers: `end_turn`, `tool_use`, or `max_tokens`. |
| `EventError` | `Error` | An error occurred mid-stream. |

**Stream completion:** `EventDone` is emitted when the API sends a completion signal (finish reason + usage stats). If the connection drops or the API response is malformed, the iterator may exhaust without `EventDone`. Handle iterator exhaustion as the authoritative "stream ended" signal; treat `EventDone` as optional metadata. Token counts use the sentinel `llm.TokensNotReported` (-1) when the provider did not report usage; compare against this constant to distinguish "not reported" from "zero tokens" (0).

**Malformed events:** If a provider sends an SSE event with invalid JSON, the iterator yields an error for that event and continues parsing subsequent events. Callers should check `err` on every iteration and decide whether to `break` or continue. This ensures data corruption is never silent.

## Tool Use

Tool use (function calling) lets the model invoke functions you define. When a model wants to call a tool, `resp.StopReason` will be `"tool_use"`. You manage the conversation by executing the tool locally and feeding `llm.NewToolResultMessage` back into the message history.

Use `llm.NewAssistantMessage(resp)` to append the assistant's response — it preserves both text and `tool_use` blocks. Using `NewTextMessage(RoleAssistant, resp.Content)` would drop tool call blocks, causing the next request to fail.

```go
// In the tool use loop:
req.Messages = append(req.Messages, llm.NewAssistantMessage(resp))  // text + tool_use blocks
req.Messages = append(req.Messages, llm.NewToolResultMessage(tc.ID, result, false))
```

**For a full working example of an agentic Tool Use loop, see [`examples/tool_use/main.go`](examples/tool_use/main.go).**

If a tool execution fails, you can pass `isError: true` so the model knows the call failed and can attempt to recover:
```go
req.Messages = append(req.Messages, llm.NewToolResultMessage(tc.ID, "location not found", true))
```

## Thinking & Reasoning

Many modern models support reasoning (chain-of-thought) capabilities where they expose their internal thought processes before outputting the final answer. 

You can check if a given model natively supports extended thinking by checking the registry. ThinkingLevel is supported by `anthropic`, `gemini`, and `openai` (via the Responses API for GPT-5 family models, where it maps to reasoning effort). The OpenAI Chat Completions adapter (`openai_compat`) returns an error if ThinkingLevel is set — use the `openai` provider instead, which auto-routes GPT-5 models to the Responses API. All four adapters **parse** thinking content from responses: Anthropic via `thinking` blocks, Gemini via `thought: true` parts, OpenAI Responses via `reasoning_summary`, and OpenAI-compat via the `reasoning_content` field (used by Ollama Qwen3, DeepSeek-R1, etc.).

```go
info, ok := llm.GetModelInfo("claude-opus-4-6")

// 1. API-controlled thinking budgets (e.g. Anthropic)
if ok && info.SupportsThinking {
    fmt.Println("Model supports extended thinking budgets")
    
    // Opt-in via config
    cfg := llm.Config{
        Provider:      "anthropic",
        Model:         "claude-opus-4-6",
        ThinkingLevel: llm.ThinkingLow, // or llm.ThinkingMedium / llm.ThinkingHigh
    }
}

// 2. Intrinsic reasoning models (e.g. DeepSeek-R1, Grok 4 R)
if ok && info.Thinking {
    fmt.Println("Model uses intrinsic reasoning natively")
}
```

You can override the default token budget for a specific request using `ThinkingBudget`:

```go
req := llm.Request{
    Messages:       messages,
    ThinkingLevel:  llm.ThinkingMedium,
    ThinkingBudget: 8192, // overrides ThinkingMedium's default of 16384
}
```

**Note:** Anthropic's API requires `temperature = 1.0` when extended thinking is enabled. The adapter enforces this automatically. A warning-level log is emitted when a temperature override occurs.

If extended thinking is enabled, you can read it synchronously via `resp.Thinking` or asynchronously in a stream via `llm.EventThinking` and `event.Thinking`.

## Conversations & Provider Handoff

For multi-turn chat, a `Conversation` is a plain, serializable, provider-neutral transcript, and a `Session` is a concurrency-safe driver that appends each turn for you (with provider/model provenance and accumulated token/cost usage). Generation itself stays stateless — a `Session` just wires a `Conversation` to a `Client`.

```go
sess := llm.NewSession(client, llm.WithSystem("You are concise."))

resp, _ := sess.Send(ctx, "What's the capital of France?")
fmt.Println(resp.Content) // "Paris."

resp, _ = sess.Send(ctx, "And its population?") // history is carried automatically
fmt.Println(resp.Content)

fmt.Printf("conversation cost so far: $%.4f\n", sess.Usage().Cost)
```

**Persist & resume** — a `Conversation` round-trips losslessly through JSON (versioned with `schema_version` so the format can evolve safely):

```go
blob, _ := json.Marshal(sess.Conversation())   // save anywhere
// ... later ...
conv, _ := llm.LoadConversation(blob)           // validates schema_version
sess = llm.NewSession(client, llm.WithConversation(conv))
```

**Provider handoff** — switch the underlying provider mid-conversation; the accumulated history is translated into the new provider's format on the next turn:

```go
sess.SwitchProvider(anthropicClient) // continue the same chat on a different provider
resp, _ = sess.Send(ctx, "Continue.")
```

On a handoff, `NormalizeForProvider` (applied automatically by `Session`, and exported for direct `Client` users) prepares the transcript for the target provider: extended-thinking blocks are replayed verbatim only to the **same** provider that produced them (Anthropic requires the original signature) and otherwise **degrade to plain text** so the reasoning survives; orphaned tool calls get a synthetic error result (every provider rejects an unanswered tool call); dangling tool results are dropped; and errored/aborted turns are dropped. Text, images, documents, and tool calls carry over unchanged.

> **Thinking replay requires thinking enabled.** Anthropic only accepts replayed thinking blocks when extended thinking is on for the request, so keep it enabled across turns (e.g. `llm.NewSession(client, llm.WithBaseRequest(llm.Request{ThinkingLevel: llm.ThinkingLow}))`, or set `ThinkingLevel` on the client `Config`). With thinking off, historical thinking blocks are simply dropped (no error). A failed `Send`/`Stream` rolls back its appended input, so the transcript stays consistent and retryable.

> **Tool loops:** after a `Send` that returns `resp.ToolCalls`, run your tools and continue with `sess.SendMessages(ctx, llm.NewToolResultMessage(id, result, isError))`.

## Context Caching

Context caching reduces cost and latency by reusing previously processed input. Anthropic and Gemini support caching with different models.

### Anthropic (inline cache_control)

Mark content blocks, system prompts, or tool definitions as cacheable. Anthropic caches the marked prefix and reuses it on subsequent requests with the same prefix.

```go
req := llm.Request{
    System:             "You are a helpful assistant with extensive knowledge...",
    SystemCacheControl: true, // Cache the system prompt
    Messages: []llm.Message{{
        Role: llm.RoleUser,
        Content: []llm.ContentPart{
            {Type: llm.ContentText, Text: longDocument, CacheControl: true}, // Cache this block
            {Type: llm.ContentText, Text: "Summarize the above."},
        },
    }},
    Tools: []llm.Tool{{
        Name: "search", Description: "Search the web",
        InputSchema:  map[string]interface{}{"type": "object"},
        CacheControl: true, // Cache this tool definition
    }},
}

resp, _ := client.Complete(ctx, req)
fmt.Printf("Cache write: %d tokens, Cache read: %d tokens\n",
    resp.CacheCreationTokens, resp.CacheReadTokens)
```

Cache token usage is also available in streaming via `EventDone`:
```go
for event, err := range client.Stream(ctx, req) {
    if event.Type == llm.EventDone {
        fmt.Printf("Cache: write=%d read=%d\n",
            event.CacheCreationTokens, event.CacheReadTokens)
    }
}
```

### Gemini (cached content reference)

Gemini uses a two-stage model: create a cache resource externally, then reference it by name. Cache lifecycle (create/list/delete) is managed outside the SDK.

```go
req := llm.Request{
    CachedContent: "cachedContents/abc123", // Pre-created via Gemini API
    Messages:      []llm.Message{llm.NewTextMessage(llm.RoleUser, "Summarize.")},
}
resp, _ := client.Complete(ctx, req)
fmt.Printf("Cached tokens: %d\n", resp.CacheReadTokens)
```

### OpenAI-compatible

Cache fields are silently ignored — no error, no effect.

## Automatic Retry, Circuit Breaker & Auth Pool Rotation

All clients get automatic retry with exponential backoff (1s→2s→4s, capped at 30s) and a circuit breaker (opens after 5 consecutive failures, probes after 30s) — including keyless local providers like Ollama and vLLM. A solo developer hitting a transient 502 or 429 gets the same resilience as an enterprise with 10 keys.

The rotation engine is thread-safe. During concurrent rate-limit events, rotation is synchronized to prevent redundant HTTP client allocations, ensuring all in-flight requests seamlessly fail over to the next available endpoint.

**Caller cancellation is never a provider failure.** If you cancel the request `ctx` (or `Complete`/`Stream` returns because *your* context expired), the library returns immediately without rotating, putting the key in cooldown, or tripping the circuit breaker — cancelling a slow request can no longer poison the pool's health. The HTTP client's own `Timeout` (`context.DeadlineExceeded`) still counts as a transient failure and retries. Permanent transport errors that no retry can fix — TLS certificate verification failures — are classified non-retryable and returned at once.

```go
cfg := llm.Config{
    Provider:  "anthropic",
    Model:     "claude-sonnet-4-6",
    APIKey:    os.Getenv("ANTHROPIC_API_KEY"),
    MaxTokens: 8192,
    Timeout:   120 * time.Second,
}

// Single-key: gets retry/backoff + circuit breaker on transient errors
client, err := llm.NewClient(cfg)

// Multi-key: rotates between keys on failure
keys := []string{"key1", "key2", "key3"}
client, err := llm.NewClientWithKeys(cfg, keys)
```

### Circuit Breaker

When an endpoint is degraded (returning 503s), the circuit breaker prevents request storms by opening after consecutive failures and allowing a single probe after cooldown:

```go
cfg := llm.DefaultConfig()                     // circuit breaker enabled by default
cfg.CircuitThreshold = 3                        // open after 3 consecutive failures
cfg.CircuitCooldown  = 15 * time.Second         // probe after 15s
```

Auth errors (401/403) do not trip the circuit — a bad key is not a broken endpoint.

### Configurable Retry & Cooldowns

```go
cfg.RetryPolicy = &llm.RetryPolicy{
    BaseDelay: 500 * time.Millisecond,         // faster retries for local providers
    MaxDelay:  10 * time.Second,
    Factor:    2.0,
    Jitter:    0.25,
}
cfg.CooldownRateLimit = 30 * time.Second       // 429 cooldown (default: 60s)
cfg.CooldownOverload  = 15 * time.Second       // 503 cooldown (default: 30s)
cfg.CooldownDefault   = 5 * time.Second        // other errors (default: 10s)
```

### Retry Observability

```go
cfg.RetryHook = func(evt llm.RetryEvent) {
    metrics.Counter("llm_retries", "type", evt.Type.String()).Inc()
}
```

### Per-Profile Endpoints

Keys can include a custom BaseURL using the `API_KEY|BASE_URL` format. This enables failover across different backends:

```go
keys := []string{
    "sk-primary-key",                              // Uses cfg.BaseURL (or provider default)
    "sk-backup-key|https://azure-proxy.example.com/v1",  // Uses Azure proxy
    "local-key|http://localhost:8000/v1",          // Falls back to local vLLM
}
client, err := llm.NewClientWithKeys(cfg, keys)
```

When a key fails, the pool rotates to the next profile — which may use an entirely different endpoint.

**Error handling:**
- **Transient errors (429, 503, 502):** Backoff and retry, rotating to other keys if available
- **Auth errors (401, 403):** Key is permanently disabled; rotates to other keys or fails immediately if none remain
- **Bad request (400):** Returns immediately — the request is broken, not the key

## Structured Errors

All API errors are returned as `*APIError` with HTTP status code, enabling reliable classification. 
*(Note: If using `NewClientWithKeys` for Auth Pool Rotation, retries happen automatically. These helpers are useful for manual flow control with a single client or application-level retries).*

```go
resp, err := client.Complete(ctx, req)
if err != nil {
    switch {
    case llm.IsRateLimited(err):
        // 429 - back off and retry
    case llm.IsOverloaded(err):
        // 503 - server busy, retry later
    case llm.IsAuthError(err):
        // 401/403 - check API key
    case llm.IsContextLength(err):
        // 400 - input too long, truncate
    case llm.IsRetryable(err):
        // Any retryable error (429, 503, 500, 502, 408)
    default:
        // Non-retryable error
    }
}
```

## Cost Estimation

Estimate cost from token counts using registry pricing data:

```go
cost := llm.EstimateCost(llm.CostInput{
    Model:             "claude-sonnet-4-6",
    InputTokens:       resp.InputTokens,
    OutputTokens:      resp.OutputTokens,
    ThinkingTokens:    resp.ThinkingTokens,
    CacheCreateTokens: resp.CacheCreationTokens,
    CacheReadTokens:   resp.CacheReadTokens,
})
fmt.Printf("Cost: $%.6f\n", cost)

// Access pricing data directly
info, _ := llm.GetModelInfo("claude-opus-4-6")
fmt.Printf("Context: %d tokens, Input: $%.2f/1M\n", info.ContextWindow, info.InputPricePer1M)
```

## Request Logging (Middleware)

Enable metadata-only logging (no message content) via `LogRequests`:

```go
cfg := llm.Config{
    Provider:    "anthropic",
    Model:       "claude-sonnet-4-6",
    APIKey:      apiKey,
    LogRequests: true,  // Logs provider, model, tokens, cost, elapsed time
}
client, _ := llm.NewClient(cfg)
```

Or wrap an existing client manually:

```go
client = llm.WithLogging(client)
client = llm.WithLoggingPrefix(client, "[MyApp]")
```

## Exponential Backoff

The pool uses configurable exponential backoff with jitter (default: 1s base,
30s cap). Override it with `Config.RetryPolicy`; callers that need the same
calculation use that policy directly:

```go
p := llm.RetryPolicy{BaseDelay: 100*time.Millisecond, MaxDelay: 5*time.Second, Factor: 3.0, Jitter: 0.1}
delay := p.Delay(attempt)
```

## More Capabilities

### Structured output (JSON mode)

Set `Request.ResponseFormat` to constrain output to JSON. Wired for OpenAI-compatible providers (`response_format`) and Gemini (`responseMimeType`/`responseSchema`); Anthropic has no native JSON mode (use a tool).

```go
req.ResponseFormat = &llm.ResponseFormat{
    Type:   llm.ResponseFormatJSONSchema, // or llm.ResponseFormatJSONObject
    Name:   "person",
    Schema: map[string]any{"type": "object", "properties": map[string]any{"name": map[string]any{"type": "string"}}},
}
```

### Tool-call validation

Validate model tool arguments against the tool's `InputSchema` before executing (JSON-Schema subset: `required`, per-property `type`, `enum`):

```go
if err := llm.ValidateToolCall(tool, call); err != nil { /* reject / re-prompt */ }
```

### Fine-grained streaming events

Wrap any stream to get block boundaries (`EventTextStart/End`, `EventThinkingStart/End`, `EventToolStart/End`) for clean UI rendering — uniform across providers:

```go
for ev, err := range llm.StreamWithBoundaries(client.Stream(ctx, req)) { ... }
```

### Conversation persistence

Persist conversations with a pluggable `Store` (in-memory or file; both stdlib-only):

```go
store := llm.NewFileStore("./chats")        // or llm.NewMemoryStore()
store.Save(ctx, "chat-1", sess.Conversation())
conv, err := store.Load(ctx, "chat-1")       // llm.ErrConversationNotFound if absent
```

### Model & provider discovery

```go
for _, m := range llm.Models()             { /* m.ID, m.Provider, m.SupportsThinking, ... */ }
mods := llm.ModelsByProvider("anthropic")
provs := llm.Providers()
```

### Reviewed capability validation

Validate the exact provider/model operation before dispatch. Unknown models and
undeclared combinations fail closed instead of relying on a provider-family guess:

```go
err := llm.ValidateRequestCapabilities(cfg, req, false)
err = llm.RequireCapabilities(cfg, llm.CapabilityEmbeddings)
```

Deployment-owned Ollama, vLLM, or custom OpenAI-compatible models use the same
registry as hosted models; no separate local-provider switch is required:

```go
llm.RegisterModel(llm.ModelInfo{
    ID: "acme/qwen-local", Provider: "vllm",
    Capabilities: llm.Capabilities(
        llm.CapabilityChat,
        llm.CapabilityStream,
        llm.CapabilityTools,
        llm.CapabilityStructuredOutput,
    ),
})
```

The capability vocabulary also covers vision, document input, supported batch,
image generation, speech synthesis, transcription, and enforceable reasoning
effort. Intrinsic reasoning output alone does not satisfy `CapabilityReasoning`.
Provider protocol support
and reviewed model support are intersected, so metadata cannot claim an operation
the selected adapter cannot encode.

### One-shot helpers & a mock client

```go
resp, _ := llm.CompleteSimple(ctx, client, "What's 2+2?")

mock := llm.NewMockClient("anthropic", "claude").PushText("hi").PushStream(/* events */)
sess := llm.NewSession(mock)                 // drive Sessions/handoff in tests, no network
```

### Embeddings, image generation, audio

Non-chat operations use the same registered-adapter pattern as chat. Construct
one `ModalityClient` for an exact deployment and reuse it for the worker
lifetime; its safe HTTP transport, connection pool, retry policy, proxy policy,
bounded reads, caller cancellation, and classified errors are shared across
all four operations:

```go
cfg.Model = "text-embedding-3-small"
modalities, err := llm.NewModalityClient(cfg)
if err != nil { /* handle */ }
defer modalities.Close()

emb, _ := modalities.GenerateEmbeddings(ctx, llm.EmbeddingRequest{Model: cfg.Model, Input: []string{"hi"}})

// A different exact deployment/client owns image generation.
imageCfg := cfg; imageCfg.Model = "gpt-image-1"
imageClient, _ := llm.NewModalityClient(imageCfg)
defer imageClient.Close()
img, _ := imageClient.GenerateImages(ctx, llm.ImageRequest{
    Model: "gpt-image-1", Prompt: "a cat", WidthPixels: 1024, HeightPixels: 1024,
    MediaType: "image/png",
})
speechCfg := cfg; speechCfg.Model = "tts-1"
speechClient, _ := llm.NewModalityClient(speechCfg)
defer speechClient.Close()
speech, _ := speechClient.SynthesizeSpeech(ctx, llm.SpeechRequest{
    Model: "tts-1", Input: "hello", Voice: "alloy", MediaType: "audio/wav",
})
_ = speech.Audio
_ = speech.MediaType // verified from Content-Type plus bounded payload signature
transcriptionCfg := cfg; transcriptionCfg.Model = "whisper-1"
transcriptionClient, _ := llm.NewModalityClient(transcriptionCfg)
defer transcriptionClient.Close()
text, _ := transcriptionClient.TranscribeAudio(ctx, llm.TranscriptionRequest{
    Model: "whisper-1", Audio: bytes, MediaType: "audio/mpeg",
})
```

Exact image count, geometry, output media type, speech voice, and transcription
language are application choices. Rho neither selects preferred values nor
imposes product-level size/count defaults; it validates only generic structure,
reviewed model capability, and whether the registered adapter can encode the
request. Omitted optional values retain the endpoint default where the operation
contract permits omission.

Generated base64 images receive a media type only after their decoded signature
matches the requested format. URL-only responses fail because the request requires
exact `b64_json` bytes. Speech synthesis returns
`*SpeechResponse` rather than unlabeled bytes and fails closed on unknown or
mismatched response formats. `ValidateEmbeddingRequest`, `ValidateImageRequest`,
`ValidateSpeechRequest`, and `ValidateTranscriptionRequest` provide the same
request/capability checks without network I/O.

Speech requests use canonical output media types: `audio/mpeg`,
`audio/ogg; codecs=opus`, `audio/aac`, `audio/flac`, `audio/wav`, or
`audio/L16`. Wire/header aliases such as `audio/opus` and `audio/x-wav` are
normalized only while verifying provider responses; they are rejected as request
values so the returned media type always matches the admitted output contract.

Inline image and PDF inputs require canonical RFC 4648 base64 whose decoded
signature matches the declared media type. Transcription likewise verifies the
uploaded FLAC, MP4/M4A, MP3, Ogg, WAV, or WebM signature before dispatch, so
mislabeled bytes never reach a provider.

The OpenAI-compatible transcription wire accepts an optional lowercase
ISO-639-1 language code. The validator rejects regional or extended BCP47 tags
such as `de-AT` rather than silently truncating them to `de`.

### Batch API (async, ~50% cheaper)

For bulk work whose results you don't need immediately, `NewBatchClient` submits many requests at
once (OpenAI: chat, responses, and embeddings). It returns a serializable `BatchHandle` you can
persist and poll later — even after a restart. The interface is provider-agnostic; OpenAI is the
first driver.

```go
cfg := llm.DefaultConfig(); cfg.Provider = "openai"; cfg.APIKey = os.Getenv("OPENAI_API_KEY")
bc, _ := llm.NewBatchClient(cfg)           // gated by ProviderPreset.SupportsBatch
defer bc.Close()

handle, _ := bc.Submit(ctx, []llm.BatchItem{
    {ItemID: "q1", Request: &llm.Request{Model: "gpt-5.3-chat-latest",
        Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentPart{{Type: llm.ContentText, Text: "classify: ..."}}}}}},
    {ItemID: "q2", Request: &llm.Request{Model: "gpt-5.3-chat-latest", /* ... */ }},
}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour})

blob, _ := json.Marshal(handle)            // persist; resume after a restart with llm.LoadBatchHandle(blob)

done, _ := llm.WaitForBatch(ctx, bc, *handle, 30*time.Second) // polls until terminal; honors ctx
if done.Status == llm.BatchCompleted {
    results, _ := bc.Results(ctx, *done)       // correlated by ItemID
    for _, r := range results {
        if r.Error != nil { /* failed line */ } else { /* r.Response (or r.Embedding) */ }
    }
}
```

Each batch has one neutral operation kind (`completion` or `embedding`); the adapter owns its exact
wire endpoint and remote file identifiers in bounded, versioned opaque handle state. `Submit`
rejects mixed/duplicate/empty sets and an unrepresentable caller-authored `MaxTurnaround` before
any upload. Batch cost is estimated at 50% via `CostInput{Batch: true}` /
`Usage.AddBatchResponse`.

### OAuth (device flow)

For providers that issue tokens via OAuth, use the RFC 8628 device-authorization grant; the resulting access token becomes `Config.APIKey`:

```go
da, _  := llm.StartDeviceAuth(ctx, llm.DeviceAuthConfig{ClientID: "...", DeviceAuthURL: "...", TokenURL: "..."})
fmt.Printf("Visit %s and enter %s\n", da.VerificationURI, da.UserCode)
tok, _ := llm.PollDeviceToken(ctx, cfg, da)  // honors authorization_pending / slow_down
```

### SSRF hardening

`Config.BlockPrivateBaseURL` rejects a `BaseURL` (or per-key override) pointing at a loopback/private/link-local host (e.g. the cloud metadata IP) — opt-in, off by default so local providers keep working. `BaseURL` is otherwise a **trusted** value: the API key is sent to it, so never populate it from untrusted input.

## Config Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| Provider | string | "anthropic" | Provider name |
| Model | string | "claude-sonnet-4-6" | Model identifier |
| ModelCapabilities | CapabilitySet | 0 | Exact deployment-scoped reviewed capabilities for Model; overrides global registry metadata |
| APIKey | string | "" | API key (empty OK for local providers) |
| MaxTokens | int | 8192 | Max output tokens |
| Temperature | *float64 | nil | Sampling temperature (nil = provider default, omitted from wire) |
| ThinkingLevel | ThinkingLevel | "" | Extended thinking: ThinkingLow/ThinkingMedium/ThinkingHigh |
| Timeout | Duration | 120s | HTTP timeout |
| BaseURL | string | "" | Override provider endpoint |
| ProxyURL | string | "" | Explicit reviewed HTTP(S) forward proxy; overrides ambient proxy variables |
| DisableProxy | bool | false | Explicitly bypass configured and ambient proxies for local/private endpoints |
| AuthHeader | string | "Bearer" | Override auth header format |
| ProviderName | string | "" | Override Client.Provider() |
| LogRequests | bool | false | Enable request/response metadata logging |
| RetryPolicy | *RetryPolicy | nil | Configurable backoff (nil = DefaultRetryPolicy: 1s–30s, 2x, ±25% jitter) |
| CircuitThreshold | int | 5 | Consecutive failures to open circuit (0 = disabled) |
| CircuitCooldown | Duration | 30s | Open→half-open cooldown |
| CooldownRateLimit | Duration | 60s | Profile cooldown for 429 errors |
| CooldownOverload | Duration | 30s | Profile cooldown for 503 errors |
| CooldownDefault | Duration | 10s | Profile cooldown for other transient errors |
| RetryHook | RetryHook | nil | Observability hook for retry lifecycle events |
| MaxRetries | int | 10 | Cap on retry/rotation iterations (min effective: 3) |
| BetaFeatures | []string | ["interleaved-thinking-..."] | Provider beta flags (Anthropic: anthropic-beta header) |
| AnthropicVersion | string | "2023-06-01" | Override Anthropic API version header |
| MaxErrorBodyBytes | int | 1 MB | Cap on error response body reads |
| MaxSSELineBytes | int | 256 KB | Cap on per-line SSE buffer |
| MaxResponseBodyBytes | int | 32 MB | Cap on success response body reads |
| MaxToolInputBytes | int | 1 MB | Cap on accumulated tool input JSON |
| MaxErrorMessageLen | int | 4096 | Cap on stored error message length |
| BlockPrivateBaseURL | bool | false | Opt-in SSRF guard: reject loopback/private/link-local BaseURL hosts |
| ResponseFormat | *ResponseFormat | nil | Structured output (JSON mode / JSON schema) — OpenAI-compat + Gemini |

## Model Registry

Use `ResolveModelAlias()` for short aliases:

```go
model := llm.ResolveModelAlias("opus")   // -> "claude-opus-4-8"
model = llm.ResolveModelAlias("grok")    // -> "grok-4.20-beta"
model = llm.ResolveModelAlias("flash")   // -> "gemini-2.5-flash"
```

### Anthropic aliases

| Alias | Resolves to |
|-------|-------------|
| `opus` | `claude-opus-4-8` |
| `sonnet` | `claude-sonnet-4-6` |
| `haiku` | `claude-haiku-4-5-20251001` |
| `claude` | `claude-sonnet-4-6` |

### xAI / Grok aliases

| Alias | Resolves to |
|-------|-------------|
| `grok4.3`, `grok-4-3` | `grok-4.3` |
| `grok`, `grok4.2`, `grok4.20`, `grok4` | `grok-4.20-beta` |
| `grok4.1`, `grok-4.1` | `grok-4-1-fast-non-reasoning` |
| `grok-reasoning`, `grok-4-reasoning` | `grok-4-fast-reasoning` |
| `grok-4.1-reasoning` | `grok-4-1-fast-reasoning` |
| `grok-code` | `grok-code-fast-1` |
| `grok-mini` | `grok-3-mini` |

### OpenAI aliases

| Alias | Resolves to |
|-------|-------------|
| `gpt5.5` | `gpt-5.5` |
| `gpt5.5-pro` | `gpt-5.5-pro` |
| `gpt`, `gpt5.4`, `gpt5` | `gpt-5.4` |
| `gpt5.3`, `gpt-instant` | `gpt-5.3-chat-latest` |
| `gpt5.4-mini` | `gpt-5.4-mini` |
| `gpt5.4-nano` | `gpt-5.4-nano` |
| `gpt5.2` | `gpt-5.2` |
| `gpt5.1` | `gpt-5.1` |
| `gpt5-mini` | `gpt-5-mini` |
| `gpt5-nano` | `gpt-5-nano` |
| `gpt4.1` | `gpt-4.1` |

### Groq aliases

| Alias | Resolves to |
|-------|-------------|
| `groq` | `llama-3.3-70b-versatile` |
| `llama`, `llama-70b` | `llama-3.3-70b-versatile` |
| `llama-8b` | `llama-3.1-8b-instant` |
| `llama4`, `llama-4-scout` | `meta-llama/llama-4-scout-17b-16e-instruct` |
| `gpt-oss` | `openai/gpt-oss-120b` |
| `gpt-oss-20b` | `openai/gpt-oss-20b` |

### Mistral aliases

| Alias | Resolves to |
|-------|-------------|
| `mistral-large` | `mistral-large-2512` |
| `mistral-medium` | `mistral-medium-latest` |
| `mistral-small` | `mistral-small-2603` |
| `magistral` | `magistral-medium-2509` |
| `codestral` | `codestral-2508` |
| `devstral` | `devstral-2512` |
| `devstral-small` | `devstral-small-2-2512` |
| `ministral` | `ministral-8b-2512` |

### DeepSeek aliases

| Alias | Resolves to |
|-------|-------------|
| `deepseek-cloud`, `deepseek-v4` | `deepseek-chat` |

### Cohere aliases

| Alias | Resolves to |
|-------|-------------|
| `cohere`, `command-a` | `command-a-03-2025` |

### DashScope/Qwen aliases

| Alias | Resolves to |
|-------|-------------|
| `qwen-cloud` | `qwen3.6-plus` |
| `qwen-omni` | `qwen3.5-omni-plus` |

### Ollama aliases

| Alias | Resolves to |
|-------|-------------|
| `deepseek` | `deepseek-r1:14b` |
| `mistral-local` | `mistral-small3.2:24b` |
| `qwen-local` | `qwen3:8b` |
| `qwen-code` | `qwen3-coder:30b` |
| `gemma`, `gemma4` | `gemma4:e4b` |
| `gemma3` | `gemma3:12b` |

### Gemini aliases

| Alias | Resolves to |
|-------|-------------|
| `gemini`, `flash-lite`, `gemini3.5-lite` | `gemini-3.5-flash-lite` |
| `gemini3.6` | `gemini-3.6-flash` |
| `gemini3.5` | `gemini-3.5-flash` |
| `gemini3.1-lite` | `gemini-3.1-flash-lite` |
| `flash` | `gemini-2.5-flash` |
| `gemini-pro`, `gemini3.1`, `gemini3`, `gemini-3` | `gemini-3.1-pro-preview` |

> **Gemini 3 note:** `gemini-3-pro-preview` and `gemini-3-flash-preview` use
> `ThoughtSignature` — the model returns an opaque signature in tool call responses
> that must be echoed in the corresponding `tool_result`. The adapter handles this
> automatically; no changes to calling code are required.

`gemini-3.6-flash` and `gemini-3.5-flash-lite` deliberately do not advertise
the sampling-temperature capability: Google deprecates and ignores
`temperature`, `top_p`, and `top_k` for these models and states that future
generations reject them.

### Cost and metadata

```go
// Estimate cost from token counts (cache-aware)
cost := llm.EstimateCost(llm.CostInput{Model: "claude-sonnet-4-6", InputTokens: inputTokens, OutputTokens: outputTokens})

// Query per-model metadata (context window, pricing, thinking support)
info, ok := llm.GetModelInfo("grok-4.20-beta")
fmt.Printf("Context: %d tokens\n", info.ContextWindow)

// Detect provider from model ID
provider := llm.ProviderForModel("gemini-2.5-flash") // -> "gemini"

// Get the default model for a provider
model := llm.GetDefaultModel("xai") // -> "grok-4.20-beta"
```

**Extending the registry at runtime.** Unlisted models return a cost estimate of `0`
and carry no capability flags. Register metadata for any model — or correct stale
built-in pricing in place — without waiting for a release. Both calls are safe for
concurrent use:

```go
// Add (or override) a model's metadata — it now feeds EstimateCost, the
// capability flags adapters read, and the discovery API (Models/ModelsByProvider).
llm.RegisterModel(llm.ModelInfo{
    ID:               "my-self-hosted-llm",
    Provider:         "vllm",
    ContextWindow:    128000,
    InputPricePer1M:  0.20,
    OutputPricePer1M: 0.60,
})

// Point a short alias at it (the target model must already be registered).
llm.RegisterModelAlias("myllm", "my-self-hosted-llm")
```

## Package Structure

Provider implementations live in sub-packages under `provider/`, following the `database/sql` driver registration pattern:

```
llm/
  types.go, config.go, errors.go, ...   # Core types and interfaces
  register.go                            # RegisterProvider() registry
  modalityregister.go                    # RegisterModalityDriver() registry
  factory.go                             # NewClient() / NewModalityClient() / NewBatchClient()
  batch.go                               # BatchClient interface + BatchItem/BatchResult/BatchHandle (async)
  batchregister.go                       # RegisterBatchProvider() + batch driver registry
  conversation.go                        # Conversation + Usage + versioned serialization
  session.go                             # Session (stateful driver) + provider handoff
  normalize.go                           # NormalizeForProvider() cross-provider handoff pass
  store.go                               # Store interface + MemoryStore / FileStore
  streamevents.go                        # StreamWithBoundaries() fine-grained events
  mock.go                                # MockClient (test double) + faux builders
  discovery.go                           # Models() / ModelsByProvider() / Providers()
  simple.go                              # CompleteSimple / StreamSimple
  validate.go                            # ValidateToolCall()
  capabilities.go                        # Provider-neutral modality contracts + validation
  oauth.go                               # OAuth 2.0 device-flow helpers
  transport.go                           # Shared HTTP plumbing (bounded reads, error construction)
  retrypolicy.go                         # RetryPolicy + RetryHook (configurable backoff)
  circuitbreaker.go                      # CircuitBreaker (3-state machine)
  provider/
    all.go                               # Blank-imports all sub-packages
    anthropic/anthropic.go               # Anthropic Claude adapter
    gemini/gemini.go                     # Google Gemini adapter
    openaicompat/                         # Chat + modality OpenAI-compatible adapter
    openairesponses/responses.go         # OpenAI Responses API adapter (GPT-5 reasoning)
    openaibatch/openaibatch.go           # OpenAI Batch API driver (async; Files + Batches REST)
```

Consumers that call `llm.NewClient()`, `llm.NewModalityClient()`, or the pure
modality request validators must add a blank import in their `main.go`:

```go
import _ "github.com/bds421/rho-llm/provider"  // register all provider adapters
```

Consumers that only use types (`llm.Client`, `llm.Config`, `llm.Message`, etc.) need no blank import.

## Acknowledgements

The name **rho** is a tribute to [pi](https://github.com/badlogic/pi-mono/tree/main/packages/ai) — rho is the next letter in the Greek alphabet. pi's `ai` package is a great multi-provider LLM library; rho brings the same idea to Go.
