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
| Google Gemini | Native | query param | generativelanguage.googleapis.com |
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
package main

import (
	"context"
	"fmt"
	"os"
	"gitlab2024.bds421-cloud.com/bds421/rho/llm"
	_ "gitlab2024.bds421-cloud.com/bds421/rho/llm/provider" // register all provider adapters
	"time"
)

func main() {
	ctx := context.Background()

	// 1. Configure for your provider
	cfg := llm.Config{
		Provider: "gemini",
		Model:    "flash", // resolves to gemini-2.5-flash
		APIKey:   os.Getenv("GEMINI_API_KEY"),
		Timeout:  30 * time.Second,
	}

	// 2. Initialize client
	client, err := llm.NewClient(cfg)
	if err != nil {
		panic(err)
	}
	defer client.Close()

	// 3. Send a message
	resp, err := client.Complete(ctx, llm.Request{
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "Explain quantum entanglement in one sentence."),
		},
	})

	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Println(resp.Content)
}
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
        fmt.Print(event.ThinkingText)   // Extended thinking (Anthropic with ThinkingLevel set)
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
| `EventThinking` | `ThinkingText` | Extended thinking output (requires `ThinkingLevel` in config). |
| `EventDone` | `StopReason`, `InputTokens`, `OutputTokens` | Stream completed. `StopReason` is `end_turn`, `tool_use`, or `max_tokens`. |
| `EventError` | `Error` | An error occurred mid-stream. |

**Stream completion:** `EventDone` is emitted when the API sends a completion signal (finish reason + usage stats). If the connection drops or the API response is malformed, the iterator may exhaust without `EventDone`. Handle iterator exhaustion as the authoritative "stream ended" signal; treat `EventDone` as optional metadata.

## Tool Use

Tool use (function calling) lets the model invoke functions you define. The typical pattern is an agentic loop: send a request with tool definitions, check if the model wants to call a tool, execute it locally, feed the result back, and repeat until the model produces a final text response.

### 1. Define tools and send the request

Tools are defined with a name, description, and a JSON Schema for their input. The description matters — it's how the model decides when to use the tool.

```go
req := llm.Request{
    Messages: []llm.Message{
        llm.NewTextMessage(llm.RoleUser, "What's the weather in Berlin?"),
    },
    Tools: []llm.Tool{{
        Name:        "get_weather",
        Description: "Get the current weather for a city",
        InputSchema: map[string]interface{}{
            "type": "object",
            "properties": map[string]interface{}{
                "location": map[string]interface{}{
                    "type":        "string",
                    "description": "City name, e.g. 'Berlin'",
                },
            },
            "required": []string{"location"},
        },
    }},
}

resp, err := client.Complete(ctx, req)
```

### 2. Handle tool calls in a loop

When the model wants to call a tool, `resp.StopReason` is `"tool_use"` and `resp.ToolCalls` contains the invocations. Execute each tool locally, then send the results back as the next message.

```go
for resp.StopReason == "tool_use" {
    // Build tool result messages for each call
    var results []llm.Message
    for _, tc := range resp.ToolCalls {
        // Execute the tool (your application logic)
        output := executeMyTool(tc.Name, tc.Input)

        // Wrap the result — the tool use ID links it back to the request
        results = append(results, llm.NewToolResultMessage(tc.ID, output, false))
    }

    // Append the assistant's response and tool results, then continue
    req.Messages = append(req.Messages,
        llm.NewTextMessage(llm.RoleAssistant, resp.Content))
    req.Messages = append(req.Messages, results...)

    resp, err = client.Complete(ctx, req)
    if err != nil {
        break
    }
}

// resp.Content now has the final text answer
fmt.Println(resp.Content)
```

### 3. Error results

If a tool execution fails, pass `isError: true` so the model knows the call failed and can recover:

```go
llm.NewToolResultMessage(tc.ID, "location not found", true)
```

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
