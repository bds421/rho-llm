# rho/llm

Multi-provider LLM client for Go. Streaming, tool use, extended thinking, auth pool rotation. Includes thread-safe concurrency management to prevent redundant HTTP client allocations during concurrent rate-limit failovers. Zero external dependencies (stdlib only).

**Requires Go 1.23+** (`go 1.23.4` in `go.mod`).

## Install

```bash
go get gitlab2024.bds421-cloud.com/bds421/rho/llm
```

## Supported Providers

| Provider | Protocol | Auth | Default BaseURL |
|----------|----------|------|-----------------|
| Anthropic/Claude | Native | x-api-key | api.anthropic.com |
| Google Gemini | Native | x-goog-api-key | generativelanguage.googleapis.com |
| OpenAI | OpenAI-compat | Bearer | api.openai.com/v1 |
| xAI/Grok | OpenAI-compat | Bearer | api.x.ai/v1 |
| Groq | OpenAI-compat | Bearer | api.groq.com/openai/v1 |
| Cerebras | OpenAI-compat | Bearer | api.cerebras.ai/v1 |
| Mistral | OpenAI-compat | Bearer | api.mistral.ai/v1 |
| OpenRouter | OpenAI-compat | Bearer | openrouter.ai/api/v1 |
| Ollama | OpenAI-compat | None | localhost:11434/v1 |
| vLLM | OpenAI-compat | None | localhost:8000/v1 |
| LM Studio | OpenAI-compat | None | localhost:1234/v1 |

## Quick Start

This example demonstrates a complete request using Google Gemini, but the code is identical for all 11 providers.

```go
import _ "gitlab2024.bds421-cloud.com/bds421/rho/llm/provider" // required: register adapters

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
```go
cfg := llm.Config{
    Provider:   "custom",
    BaseURL:    "http://my-proxy:8080/v1",
    APIKey:     "my-key",
}
```

## Streaming

`client.Stream()` returns a Go 1.23 iterator (`iter.Seq2[StreamEvent, error]`) that yields events as the model generates tokens. This lets you display partial output in real time rather than waiting for the full response. Use `break` to abort early — the iterator cleans up the underlying HTTP connection automatically.

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

You can check if a given model natively supports extended thinking by checking the registry. If you pass thinking options to a client targeting a non-reasoning model, they are silently ignored at the adapter level.

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

**Note:** Anthropic's API requires `temperature = 1.0` when extended thinking is enabled. The adapter enforces this automatically. A debug-level log is emitted when a temperature override occurs.

If extended thinking is enabled, you can read it synchronously via `resp.Thinking` or asynchronously in a stream via `llm.EventThinking` and `event.Thinking`.

## Automatic Retry & Auth Pool Rotation

All clients with an API key get automatic retry with exponential backoff (1s→2s→4s, capped at 30s). A solo developer hitting a transient 502 or 429 gets the same resilience as an enterprise with 10 keys. 

The rotation engine is thread-safe. During concurrent rate-limit events, rotation is synchronized to prevent redundant HTTP client allocations, ensuring all in-flight requests seamlessly fail over to the next available endpoint.

```go
cfg := llm.Config{
    Provider:  "anthropic",
    Model:     "claude-sonnet-4-6",
    APIKey:    os.Getenv("ANTHROPIC_API_KEY"),
    MaxTokens: 8192,
    Timeout:   120 * time.Second,
}

// Single-key: gets retry/backoff on transient errors
client, err := llm.NewClient(cfg)

// Multi-key: rotates between keys on failure
keys := []string{"key1", "key2", "key3"}
client, err := llm.NewClientWithKeys(cfg, keys)
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
cost := llm.EstimateCost("claude-sonnet-4-6", resp.InputTokens, resp.OutputTokens)
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

The pool uses exponential backoff with jitter (1s base, 30s cap) when retrying:

```go
// Available as a utility for custom retry logic
delay := llm.Backoff(attempt, 1*time.Second, 30*time.Second)
// attempt 0: ~1s, 1: ~2s, 2: ~4s, 3: ~8s, ...
```

## Config Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| Provider | string | "anthropic" | Provider name |
| Model | string | "claude-sonnet-4-6" | Model identifier |
| APIKey | string | "" | API key (empty OK for local providers) |
| MaxTokens | int | 8192 | Max output tokens |
| Temperature | float64 | 1.0 | Sampling temperature |
| ThinkingLevel | ThinkingLevel | "" | Extended thinking: ThinkingLow/ThinkingMedium/ThinkingHigh |
| Timeout | Duration | 120s | HTTP timeout |
| BaseURL | string | "" | Override provider endpoint |
| AuthHeader | string | "Bearer" | Override auth header format |
| ProviderName | string | "" | Override Client.Provider() |
| LogRequests | bool | false | Enable request/response metadata logging |

## Model Registry

Use `ResolveModelAlias()` for short aliases:

```go
model := llm.ResolveModelAlias("opus")   // -> "claude-opus-4-6"
model = llm.ResolveModelAlias("grok")    // -> "grok-4-fast-non-reasoning"
model = llm.ResolveModelAlias("flash")   // -> "gemini-2.5-flash"
```

### Anthropic aliases

| Alias | Resolves to |
|-------|-------------|
| `opus` | `claude-opus-4-6` |
| `sonnet` | `claude-sonnet-4-6` |
| `haiku` | `claude-haiku-4-5-20251001` |
| `claude` | `claude-sonnet-4-6` |

### xAI / Grok aliases

| Alias | Resolves to |
|-------|-------------|
| `grok`, `grok4`, `grok-4` | `grok-4-fast-non-reasoning` |
| `grok4.1`, `grok-4.1` | `grok-4-1-fast-non-reasoning` |
| `grok-reasoning`, `grok-4-reasoning` | `grok-4-fast-reasoning` |
| `grok-4.1-reasoning` | `grok-4-1-fast-reasoning` |
| `grok-code` | `grok-code-fast-1` |
| `grok-mini` | `grok-3-mini` |

### OpenAI aliases

| Alias | Resolves to |
|-------|-------------|
| `gpt`, `gpt5` | `gpt-5.2` |
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
| `gpt-oss` | `openai/gpt-oss-120b` |
| `gpt-oss-20b` | `openai/gpt-oss-20b` |

### Mistral aliases

| Alias | Resolves to |
|-------|-------------|
| `mistral-large` | `mistral-large-2512` |
| `mistral-medium` | `mistral-medium-latest` |
| `mistral-small` | `mistral-small-2506` |
| `magistral` | `magistral-medium-2509` |
| `codestral` | `codestral-2508` |
| `devstral` | `devstral-small-2-25-12` |
| `ministral` | `ministral-3-8b-25-12` |

### Ollama aliases

| Alias | Resolves to |
|-------|-------------|
| `deepseek` | `deepseek-r1:14b` |
| `mistral-local` | `mistral-small3.2:24b` |
| `qwen` | `qwen3:8b` |
| `qwen-code` | `qwen3-coder:30b` |
| `gemma` | `gemma3:12b` |

### Gemini aliases

| Alias | Resolves to |
|-------|-------------|
| `gemini`, `flash-lite` | `gemini-2.5-flash-lite` |
| `flash` | `gemini-2.5-flash` |
| `gemini-pro`, `gemini3`, `gemini-3` | `gemini-3-pro-preview` |

> **Gemini 3 note:** `gemini-3-pro-preview` and `gemini-3-flash-preview` use
> `ThoughtSignature` — the model returns an opaque signature in tool call responses
> that must be echoed in the corresponding `tool_result`. The adapter handles this
> automatically; no changes to calling code are required.

### Cost and metadata

```go
// Estimate cost from token counts
cost := llm.EstimateCost("claude-sonnet-4-6", inputTokens, outputTokens)

// Query per-model metadata (context window, pricing, thinking support)
info, ok := llm.GetModelInfo("grok-4-fast-non-reasoning")
fmt.Printf("Context: %d tokens\n", info.ContextWindow)

// Detect provider from model ID
provider := llm.ProviderForModel("gemini-2.5-flash") // -> "gemini"

// Get the default model for a provider
model := llm.GetDefaultModel("xai") // -> "grok-4-fast-non-reasoning"
```

## Package Structure

Provider implementations live in sub-packages under `provider/`, following the `database/sql` driver registration pattern:

```
llm/
  types.go, config.go, errors.go, ...   # Core types and interfaces
  register.go                            # RegisterProvider() registry
  factory.go                             # NewClient() -> registry lookup
  provider/
    all.go                               # Blank-imports all sub-packages
    anthropic/anthropic.go               # Anthropic Claude adapter
    gemini/gemini.go                     # Google Gemini adapter
    openaicompat/openaicompat.go         # OpenAI-compatible adapter (11+ providers)
```

Consumers that call `llm.NewClient()` must add a blank import in their `main.go`:

```go
import _ "gitlab2024.bds421-cloud.com/bds421/rho/llm/provider"  // register all provider adapters
```

Consumers that only use types (`llm.Client`, `llm.Config`, `llm.Message`, etc.) need no blank import.
