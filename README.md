# rho-llm

Multi-provider LLM client for Go. Streaming, tool use, image/vision + PDF/document input, extended thinking, structured output (JSON mode), serializable conversations with cross-provider handoff, embeddings, image generation, audio (speech/transcription), an async **Batch API** (~50% cheaper bulk processing), OAuth device flow, and auth pool rotation. Includes thread-safe concurrency management to prevent redundant HTTP client allocations during concurrent rate-limit failovers. The library imports the Go standard library and one rho-kit package (`security/v2/netutil`, for the private-address check); the `examples/` use `joho/godotenv`.

**Requires Go 1.26.10+** (the minimum toolchain in `go.mod`, including the standard-library security updates described in the changelog).

## Install

```bash
go get github.com/bds421/rho-llm
```

### Upgrading to v0.13.0

v0.13.0 sends Anthropic API version `2026-10-21` by default
([migration guide](https://platform.claude.com/docs/en/api/migration-2026-10-21)). Check:

- **Sampling.** `Request.Temperature` and `temperature` / `top_p` / `top_k` in `SamplingParams`
  are refused on Anthropic (every model). Remove them, or set
  `Config.AnthropicVersion = llm.AnthropicVersion20230601` where a Claude 4.6 or 4.5 call needs
  a non-default temperature. Claude 4.7 and later took only the defaults already.
- **History.** A document in an assistant message, or citations on user text (copying
  Claude's cited blocks into a user turn), is refused; move the document to a user
  message and send the text without `Citations`.
- **Batches** created before the switch finish under the version they were created with.
- **Anthropic requests that now fail before sending:** `ResponseFormat` `json_object` (use a
  schema), `CacheTTL` without `CacheControl`, a 1-hour cache marker after a 5-minute one,
  `CachedContent`, an `Effort` that conflicts with the `ThinkingLevel`. A client built
  directly with `anthropic.New` checks capabilities like the factory, so an unregistered
  model needs `Config.ModelCapabilities`.
- **Other providers:** `ContainerID` and `Effort` are refused by Gemini and
  OpenAI-compatible adapters; Gemini `SAFETY` / `PROHIBITED_CONTENT` and OpenAI
  `content_filter` now stop with `StopRefusal` (`RawStopReason` keeps the value). A fallback
  chain that reuses one `Request` across providers must not set the provider-specific fields.
- **Thinking text.** `ReasoningSummaryAuto` now leaves Anthropic `display` to the API default,
  so Claude 4.7 and later return empty thinking text; leave `ReasoningSummary` unset for
  readable summaries.

## Supported Providers

| Provider | Protocol | Auth | Default BaseURL |
|----------|----------|------|-----------------|
| Anthropic/Claude | Native | x-api-key | api.anthropic.com |
| Google Gemini | Native | x-goog-api-key | generativelanguage.googleapis.com |
| OpenAI | OpenAI-compat / Responses | Bearer | api.openai.com/v1 |
| xAI/Grok | OpenAI-compat | Bearer | api.x.ai/v1 |
| Meta | OpenAI-compat | Bearer | api.meta.ai/v1 |
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
| Z.ai/GLM | OpenAI-compat | Bearer | api.z.ai/api/paas/v4 |
| MiniMax | OpenAI-compat | Bearer | api.minimax.io/v1 |
| Moonshot/Kimi | OpenAI-compat | Bearer | api.moonshot.ai/v1 |
| Ollama | OpenAI-compat | None | localhost:11434/v1 |
| vLLM | OpenAI-compat | None | localhost:8000/v1 |
| LM Studio | OpenAI-compat | None | localhost:1234/v1 |

> **Counting:** 23 providers across 4 wire protocols, reachable through 35 preset
> names — the 23 canonical names in the table, `openai_responses` (the OpenAI row
> spans two protocols: `openai_compat` and the auto-selected `openai_responses` for
> GPT-5 reasoning models), the aliases `claude`/`google`/`grok`/`qwen`/`z-ai`/`glm`/`kimi`,
> and the mainland-China endpoints `dashscope-cn`/`qwen-cn`/`moonshot-cn`/`kimi-cn`.
> Curated **model metadata** (pricing + capabilities) backs
> cost estimation, discovery, and fail-closed capability checks. Unlisted model IDs
> need `RegisterModel` (or `Config.ModelCapabilities`) before dispatch — see
> [Model Registry](#model-registry). Unknown providers work via `Config.BaseURL` once
> capabilities are declared the same way.

## Quick Start

This example demonstrates a complete request using Google Gemini, but the code is identical for all 23 providers.

```go
import _ "github.com/bds421/rho-llm/provider" // required: register adapters

// 1. Configure and initialize
cfg := llm.Config{
    Provider: "gemini",
    Model:    "flash", // resolves to gemini-3.8-flash
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
if err != nil {
    panic(err)
}

fmt.Println(resp.Content)
```

### Provider Recipes

#### Ollama (local, no API key)
```go
cfg := llm.Config{
    Provider:          "ollama",
    Model:             "llama3", // an installed Ollama model
    ModelCapabilities: llm.Capabilities(llm.CapabilityChat, llm.CapabilityStream),
}
```

#### Custom OpenAI-compatible endpoint

Unknown providers (not in the presets list) **must** set `BaseURL`. Without it, `NewClient` returns an error to prevent typos from silently defaulting to an incorrect endpoint.

```go
cfg := llm.Config{
    Provider:          "custom",
    Model:             "my-model",
    BaseURL:           "http://my-proxy:8080/v1", // required for unknown providers
    APIKey:            "my-key",
    ModelCapabilities: llm.Capabilities(llm.CapabilityChat, llm.CapabilityStream),
}
```

## Image / Vision Support

Send images to vision-capable models using `ContentImage` parts with base64-encoded data. All four protocol adapters serialize images to the correct wire format automatically.

```go
import (
    "encoding/base64"
    "os"
)

imgBytes, err := os.ReadFile("photo.png")
if err != nil {
    panic(err)
}
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
if err != nil {
    panic(err)
}

// Or use the convenience helper for single-image messages:
msg := llm.NewImageMessage(llm.RoleUser, "image/png", imgData)
```

Supported media types: `image/jpeg`, `image/png`, `image/gif`, `image/webp`.

## Document / PDF Support

Send PDFs to document-capable models using `ContentDocument` parts with base64-encoded data. Gemini and Anthropic parse PDFs natively (preserving text layout); OpenAI-compatible vision models (e.g. xAI Grok) receive the PDF as an `image_url` data URI. The Responses API adapter returns an explicit error rather than silently dropping a document.

```go
pdfBytes, err := os.ReadFile("invoice.pdf")
if err != nil {
    panic(err)
}
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
if err != nil {
    panic(err)
}

// Or use the convenience helper for single-document messages:
msg := llm.NewDocumentMessage(llm.RoleUser, "application/pdf", pdfData)
```

Supported media types: `application/pdf`.

**Plain-text documents (Anthropic).** `llm.NewTextDocument(title, text)` builds a
document part with a `text` source (`media_type: text/plain`, the text itself as
`Data`). `ContentPart.Title` and `ContentPart.Context` are optional on any document
and go to the model next to it; `CacheControl`/`CacheTTL` work as on other parts.
The text must be valid, non-empty UTF-8. Gemini and OpenAI-compatible adapters
refuse text documents (send the text as a text part there).

## Streaming

`client.Stream()` returns a Go 1.23 iterator (`iter.Seq2[StreamEvent, error]`) that yields events as the model generates tokens. This lets you display partial output in real time rather than waiting for the full response. Use `break` to abort early — the iterator cleans up the underlying HTTP connection automatically.

**Completion is explicit.** A successfully completed stream yields exactly one
`EventDone` carrying the stop reason and final token usage. If the stream ends
without its protocol-final event, the adapter yields a terminal error (wrapping
`io.ErrUnexpectedEOF`) instead of ending silently, so a truncated turn cannot be
mistaken for a complete one. A recoverable malformed-event error can occur before
a later `EventDone` if the caller chooses to continue iterating. To abort an
in-flight stream, cancel the `ctx` passed to `Stream()` — caller cancellation is
reported but does *not* count as a provider failure (no key cooldown or
circuit-breaker trip).

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
| `EventThinking` | `Thinking` | Thinking output returned by the provider, including models that reason by default. |
| `EventServerTool` | `Part` | Anthropic: a server tool call (once its input is complete) or its result. |
| `EventCitation` | `Citation` | Anthropic: a citation for the text block being streamed. |
| `EventDone` | `Model`, `StopReason`, `RawStopReason`, `InputTokens`, `OutputTokens`, `ServerToolUse`, `Iterations`, `Parts`, `ContainerID` | Stream completed. Common stop reasons normalize to `end_turn`, `tool_use`, or `max_tokens`; provider-specific reasons can remain unchanged. `RawStopReason` preserves the provider value when available. |
| `EventError` | `Error`, usage fields, `Parts`, `Model` | Yielded together with the iterator's `error` when an Anthropic stream breaks after `message_start`: the usage reported so far (input and cache tokens) and the content assembled so far, so the broken turn's cost can be recorded. Other failures yield an empty event with the error. |

**Stream completion:** Require `EventDone` to recognize a completed turn. An unexpected connection close produces an error; iterator exhaustion alone does not prove the response completed. Token counts can use the sentinel `llm.TokensNotReported` (-1) when the provider did not report usage; compare against this constant to distinguish "not reported" from "zero tokens" (0).

`llm.CollectStream(client.Stream(ctx, req))` folds a stream into a `*Response`
(the same fold `Session.Stream` uses, exported as `llm.StreamAccumulator`). On a
broken stream it returns the partial response with the usage seen so far together
with the error; a stream that ends without `EventDone` returns
`llm.ErrStreamIncomplete`. `EventDone.Model` (and so `Response.Model`) is the model
that answered, the fallback model after a server-side refusal fallback. A
mid-stream Anthropic `error` event becomes an `*llm.APIError` with the status the
same error has outside a stream (`overloaded_error` = 529, retryable).

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

Use `Request.ToolChoice` to constrain tool selection when `Tools` is non-empty:

```go
req.ToolChoice = llm.ForceTool("get_weather")
// Or llm.NewToolChoice(llm.ToolChoiceRequired / llm.ToolChoiceNone / llm.ToolChoiceAuto).
```

The adapters translate this neutral choice to their protocol's wire format. A nil
choice uses the provider default. Malformed choices fail before dispatch; choices
are omitted when no tools are supplied.

`Request.SamplingParams` carries provider-specific top-level body fields such as
OpenRouter routing (`"provider"`), vLLM `"priority"`, or supported sampling knobs.
Keys already emitted by the adapter and reserved structural keys are rejected.
Use typed request fields where available. The map is protocol-specific: for
example, a top-level `"top_p"` is not a Gemini `generationConfig` setting.

If a tool execution fails, you can pass `isError: true` so the model knows the call failed and can attempt to recover:
```go
req.Messages = append(req.Messages, llm.NewToolResultMessage(tc.ID, "location not found", true))
```

## Server Tools (Anthropic Web Search)

Server tools run at the provider: you declare them, and the call and its result
come back in the same assistant turn. `Request.ServerTools` (or
`WithServerTools` on a `Session`) takes a typed `ServerTool`; adapters other than
Anthropic return an error instead of silently dropping it, and so for a `ContainerID`.

```go
search := llm.ServerTool{
    Type:           llm.WebSearch20260209, // or WebSearch20250305 / WebSearch20260318
    Name:           llm.ServerToolWebSearch,
    MaxUses:        5,                               // searches per turn (0 = no cap)
    AllowedDomains: []string{"example.com"},         // or BlockedDomains, not both
    UserLocation:   &llm.UserLocation{Country: "US", Timezone: "America/New_York"},
    AllowedCallers: []string{llm.CallerDirect},      // no dynamic filtering (ZDR, older models)
    CacheControl:   true,
}
s := llm.NewSession(client, llm.WithServerTools(search))
```

`Params` carries further version-specific fields (web fetch's
`max_content_tokens`, web search's `response_inclusion`) without a new release;
it cannot override a typed field.

**Response.** `Response.Parts` holds the assistant content in provider order:

| Part | Fields |
|------|--------|
| `ContentServerToolUse` | `ToolUseID`, `ToolName`, `ToolInput` |
| `ContentServerToolResult` | `ToolResultID`, `BlockType` (`"web_search_tool_result"`, …), `WebSearchResults` or `ServerToolError` (`ErrorCode` such as `max_uses_exceeded`) |
| `ContentText` | `Text`, `Citations` (`web_search_result_location`, `char_location`, `page_location`, `content_block_location`, `search_result_location`) |
| `ContentProviderBlock` | `BlockType`, for blocks the neutral model does not describe |

Each server part keeps the block exactly as received in `ProviderBlock` (and
each citation in `Citation.Raw`), so `encrypted_content` and `encrypted_index`
reach the API unchanged on the next request, also after the conversation was
persisted and loaded. `NewAssistantMessage` uses `Parts` when set.
`NormalizeForProvider` drops server parts and citations when a conversation
moves to another provider. `Response.Content` still holds the concatenated
text.

**Streaming** emits `EventServerTool` (`Part`) for each call and result and
`EventCitation` (`Citation`) per citation; `EventDone.Parts` has the assembled
turn, the same as the non-streaming `Response.Parts`.

**`pause_turn`.** A long server-tool turn can stop with `llm.StopPauseTurn`.
Continue it by sending the paused content back unchanged, with the same tools:

```go
resp, err := s.Send(ctx, "When did the example widget ship?")
for i := 0; err == nil && resp.StopReason == llm.StopPauseTurn && i < 5; i++ {
    resp, err = s.Continue(ctx) // or StreamContinue
}

// Without a Session:
next, err := llm.ContinuePausedTurn(req, resp) // appends the paused turn, nothing else
resp, err = client.Complete(ctx, next)
```

The continuation leaves `Tools`, `ServerTools`, `System` and the cache flags
untouched, so the tool definitions encode to the same bytes and the prompt cache
stays valid. The container of the paused turn (`Response.ContainerID`, used by
dynamic filtering) is sent back. `Continue` returns `llm.ErrNotPaused` without
a request when the last turn did not pause.

**Thinking in a continuation.** Thinking blocks go back unchanged with the rest of the
turn, signature included, also when their text is empty (the default `display:
"omitted"` on Claude 4.7+), also without a `ThinkingLevel` on always-thinking models and on
Sonnet 5.5 (`between_tools`).
Only a handoff to another provider drops them.

**Refusals and server-side fallback.** A declined request stops with `llm.StopRefusal`;
`Response.StopDetails` (also on `EventDone`) has `Category`, `Explanation`,
`RecommendedModel`, the fallback-credit fields and `Raw`. Branch on `StopReason`.
Gemini (`SAFETY`, `PROHIBITED_CONTENT`) and OpenAI (`content_filter`) safety stops map to
`StopRefusal` too, with the provider value in `RawStopReason`.
Server-side fallback is a body field plus a beta header:

```go
cfg.BetaFeatures = []string{"server-side-fallback-2026-07-01"} // sent on every request
req.SamplingParams = map[string]any{"fallbacks": "default"}
```

The `fallback` block that marks the switch comes back as a `ContentProviderBlock` and is
replayed in place, and `Response.Iterations` prices the attempts.

**Usage.** `Response.ServerToolUse` counts `WebSearchRequests` (and web fetch and
code execution requests). `Response.Iterations` is Anthropic's
`usage.iterations` when the API sends it (compaction steps, refusal fallbacks);
see [Cost Estimation](#cost-estimation).

## Thinking & Reasoning

Many modern models support reasoning (chain-of-thought) capabilities where they expose their internal thought processes before outputting the final answer. 

You can check if a given model natively supports extended thinking by checking the registry. ThinkingLevel is supported by `anthropic`, `gemini`, and `openai` (via the Responses API for GPT-5 family models, where it maps to reasoning effort). The OpenAI Chat Completions adapter (`openai_compat`) returns an error if ThinkingLevel is set — use the `openai` provider instead, which auto-routes GPT-5 models to the Responses API. All four adapters **parse** thinking content from responses: Anthropic via `thinking` blocks, Gemini via `thought: true` parts, OpenAI Responses via `reasoning_summary`, and OpenAI-compat via the `reasoning_content` field (used by Ollama Qwen3, DeepSeek-R1, etc.).

```go
info, ok := llm.GetModelInfo("claude-opus-4-6")

// 1. API-controlled thinking budgets (e.g. Anthropic)
if ok && info.SupportsThinking {
    fmt.Println("Model supports extended thinking budgets")
    
    // Opt-in via config
    thinkingCfg := llm.Config{
        Provider:      "anthropic",
        Model:         "claude-opus-4-6",
        ThinkingLevel: llm.ThinkingLow, // or llm.ThinkingMedium / llm.ThinkingHigh
    }
    fmt.Printf("Configure %s with %s thinking\n", thinkingCfg.Model, thinkingCfg.ThinkingLevel)
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

For Claude 4.7 and later (Opus 4.7/4.8/5/5.5, Fable 5/5.1, Sonnet 5/5.5,
Haiku 5.5), `ThinkingLevel` selects adaptive thinking effort instead of a token
budget (`minimal` maps to `low`; `ThinkingMax` sends effort `max`) — Anthropic rejects manual `budget_tokens` on
these models, so a positive `ThinkingBudget` with reasoning enabled is rejected
before dispatch. `ThinkingNone` is sent the way each model accepts it: thinking
`disabled` (Opus 4.7/4.8/5, Sonnet 5, Haiku 5.5), `between_tools` (Sonnet 5.5),
or omitted on Fable 5/5.1 and Opus 5.5, whose thinking is always on. These
models do not advertise custom-temperature support, and Fable 5.1, Opus 5.5 and
Sonnet 5.5 reject a forced `ToolChoice` (`Required`/`Tool`) before dispatch.
Opus 4.6, Sonnet 4.6 and the 4.5 models keep manual budgets.

With a `ThinkingLevel` on these models the adapter asks for `display: "summarized"`, so
`resp.Thinking` holds readable text. `ReasoningSummary: llm.ReasoningSummaryOmitted` sends
`"omitted"`; `llm.ReasoningSummaryAuto` leaves `display` off and gets the API default, which is
`"omitted"` on Claude 4.7 and later (thinking blocks with empty text, a shorter
pause before the answer when streaming). Display changes only what comes back:
thinking is billed the same either way. `ResponseFormat` is sent as
`output_config.format` beside the effort.

**Effort without thinking.** `Request.Effort` (`llm.EffortLow` ... `llm.EffortMax`) sets
Anthropic `output_config.effort` and OpenAI Responses `reasoning.effort` without asking for
thinking. With a `ThinkingLevel` that already sets the effort, the two must agree. The
adapter refuses what the API rejects: effort above `high` with thinking `disabled` (Haiku
5.5, Opus 5) or `between_tools` (Sonnet 5.5), effort on Sonnet 4.5 and Haiku 4.5, `xhigh` on
Opus/Sonnet 4.6, `xhigh`/`max` on Opus 4.5. Gemini and OpenAI-compatible adapters refuse
`Effort`. The adapter owns `output_config`: pass effort here. `SamplingParams["output_config"]` for
effort is deprecated; it still passes through when the adapter builds no `output_config`.

**Note:** On the default Anthropic API version `2026-10-21` an explicit temperature
(or `top_p` / `top_k` in `SamplingParams`) is refused on every model before dispatch. With
`Config.AnthropicVersion = llm.AnthropicVersion20230601` the Claude 4.6 and 4.5 models take
it, and for manual extended thinking a temperature other than `1.0` is rejected. An omitted temperature stays omitted;
the library does not rewrite an application's sampling value.

If extended thinking is enabled, you can read it synchronously via `resp.Thinking` or asynchronously in a stream via `llm.EventThinking` and `event.Thinking`.

## Conversations & Provider Handoff

For multi-turn chat, a `Conversation` is a plain, serializable, provider-neutral transcript, and a `Session` is a concurrency-safe driver that appends each turn for you (with provider/model provenance and accumulated token/cost usage). Generation itself stays stateless — a `Session` just wires a `Conversation` to a `Client`.

```go
sess := llm.NewSession(client, llm.WithSystem("You are concise."))

resp, err := sess.Send(ctx, "What's the capital of France?")
if err != nil {
    panic(err)
}
fmt.Println(resp.Content) // "Paris."

resp, err = sess.Send(ctx, "And its population?") // history is carried automatically
if err != nil {
    panic(err)
}
fmt.Println(resp.Content)

fmt.Printf("conversation cost so far: $%.4f\n", sess.Usage().Cost)
```

**Persist & resume** — a `Conversation` round-trips losslessly through JSON (versioned with `schema_version` so the format can evolve safely):

```go
blob, err := json.Marshal(sess.Conversation()) // save anywhere
if err != nil {
    panic(err)
}
// ... later ...
conv, err := llm.LoadConversation(blob)        // validates schema_version
if err != nil {
    panic(err)
}
sess = llm.NewSession(client, llm.WithConversation(conv))
```

**Provider handoff** — switch the underlying provider mid-conversation; the accumulated history is translated into the new provider's format on the next turn:

```go
sess.SwitchProvider(anthropicClient) // continue the same chat on a different provider
resp, err = sess.Send(ctx, "Continue.")
if err != nil {
    panic(err)
}
```

On a handoff, `NormalizeForModel` (applied automatically by `Session`; it and `NormalizeForProvider` are exported for direct `Client` users) prepares the transcript for the target model: thinking from a **different model** is dropped, because a model shown another model's reasoning tends to imitate it; thinking from the same model is replayed verbatim only to the **same** provider with a valid signature (Anthropic requires the original signature) and otherwise **degrades to plain text**; orphaned tool calls get a synthetic error result (every provider rejects an unanswered tool call); dangling tool results are dropped; and errored/aborted turns are dropped. Text, images, documents, and tool calls carry over unchanged.

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

resp, err := client.Complete(ctx, req)
if err != nil {
    panic(err)
}
fmt.Printf("Cache write: %d tokens, Cache read: %d tokens\n",
    resp.CacheCreationTokens, resp.CacheReadTokens)
```

Cache token usage is also available in streaming via `EventDone`:
```go
for event, err := range client.Stream(ctx, req) {
    if err != nil {
        panic(err)
    }
    if event.Type == llm.EventDone {
        fmt.Printf("Cache: write=%d read=%d\n",
            event.CacheCreationTokens, event.CacheReadTokens)
    }
}
```

**1-hour TTL.** Each cache marker takes an optional TTL: `CacheTTL: llm.CacheTTL1h` on a
content part, tool or server tool, `SystemCacheTTL` for the system prompt (empty or
`llm.CacheTTL5m` is the default five minutes). The API requires 1-hour markers to come
before 5-minute ones. The adapter checks both rules before sending: a `CacheTTL` without
`CacheControl`, or a 1-hour marker after a 5-minute one (in render order: tools, server
tools, system, messages), is an error. `Response.CacheCreation1hTokens` is the 1-hour share of
`CacheCreationTokens`; `EstimateCost` prices it at `CacheWrite1hPricePer1M` (2x input on
Anthropic; a model without one gets 2x its input price) and the rest at the 5-minute
write rate.

### Gemini (cached content reference)

Gemini uses a two-stage model: create a cache resource externally, then reference it by name. Cache lifecycle (create/list/delete) is managed outside the SDK.

```go
req := llm.Request{
    CachedContent: "cachedContents/abc123", // Pre-created via Gemini API
    Messages:      []llm.Message{llm.NewTextMessage(llm.RoleUser, "Summarize.")},
}
resp, err := client.Complete(ctx, req)
if err != nil {
    panic(err)
}
fmt.Printf("Cached tokens: %d\n", resp.CacheReadTokens)
```

`resp.InputTokens` is always the **uncached** input: for Gemini (and the
OpenAI-style adapters, from `cached_tokens`) the cached share is subtracted from
the provider's total prompt count, matching Anthropic's disjoint
`input_tokens` / `cache_read_input_tokens`. Total prompt tokens =
`InputTokens + CacheReadTokens + CacheCreationTokens` for every provider.

### OpenAI-compatible

Cache fields are silently ignored — no error, no effect.

## Automatic Retry, Circuit Breaker & Auth Pool Rotation

All clients get automatic retry with exponential backoff — including keyless local providers like Ollama and vLLM. A solo developer hitting a transient 502 or 429 gets the same resilience as an enterprise with 10 keys.

| | Default | Notes |
|---|---|---|
| Attempts (single key, and modality/batch calls) | 3 | `Config.MaxRetries` raises it (min 3). Multi-key pools make one attempt per healthy key, capped at `MaxRetries` (default 10). |
| Backoff | `RetryPolicy`: 1s → 2s → 4s … capped at 30s, ±25% jitter | A provider `Retry-After` (seconds, HTTP-date, or `retry-after-ms`) stretches a shorter backoff, capped at `llm.MaxRetryAfter` (60s). A multi-key pool whose keys are all cooling down waits for the soonest key instead. No sleep after the final attempt. |
| Total time | unbounded (attempts × backoff, and `ctx`) | `Config.RetryBudget` stops the sequence before a backoff that would overrun it, and a backoff that would reach the `ctx` deadline is not started either — the last provider error (e.g. a 429 with its `RetryAfter`) is returned at once instead of `context.DeadlineExceeded`. In-flight attempts are not cut — use `ctx` for a hard deadline. |
| Circuit breaker | **only with `DefaultConfig()`** (opens after 5 consecutive failures, probes after 30s) | A struct-literal `Config` has `CircuitThreshold: 0` = no breaker. Set it explicitly to get one. |
| Non-idempotent creates (batch create, image generation, speech synthesis) | retried only on 429/503 and connection (dial/DNS) failures | A 500/502/504/408 may come after the provider committed the job — resending would duplicate it and its charge. |

Before v0.9.5 a single-key client slept the 30–60s key *cooldown* between attempts (ignoring `RetryPolicy`), modality/batch calls made 10 attempts by default, and this section claimed every client had a breaker.

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

// Single-key: gets retry/backoff on transient errors. This struct-literal
// config has no circuit breaker — add CircuitThreshold (or start from
// llm.DefaultConfig()) if you want one.
cfg.CircuitThreshold = 5
cfg.RetryBudget = 30 * time.Second // optional: cap total time spent retrying
singleClient, err := llm.NewClient(cfg)
if err != nil {
    panic(err)
}
defer singleClient.Close()

// Multi-key: rotates between keys on failure
keys := []string{"key1", "key2", "key3"}
pooledClient, err := llm.NewClientWithKeys(cfg, keys)
if err != nil {
    panic(err)
}
defer pooledClient.Close()
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
cfg.RetryBudget = 20 * time.Second            // stop retrying after ~20s total (0 = no budget)
// Per-key cooldowns drive multi-key rotation (a cooling key is skipped);
// a single-key client backs off with RetryPolicy instead.
cfg.CooldownRateLimit = 30 * time.Second       // 429 cooldown (default: 60s)
cfg.CooldownOverload  = 15 * time.Second       // 503 cooldown (default: 30s)
cfg.CooldownDefault   = 5 * time.Second        // other errors (default: 10s)
```

### Streaming and `Config.Timeout`

`Config.Timeout` (default 120s) bounds a `Complete` call end to end. A `Stream` is **not** cut off after `Timeout` in total — long thinking turns may run for minutes. For streams, `Timeout` bounds the wait for response headers and each *silent gap* between body bytes (a stalled stream fails with `*llm.StreamIdleTimeoutError`, a `net.Error` timeout). The gap is measured in bytes, not tokens: Anthropic sends keep-alive pings during long thinking, but Gemini and the OpenAI Responses API may reason silently for minutes — **raise `Timeout` for silent long reasoning** (e.g. `ThinkingHigh`) and bound the total with `ctx`. Bound a stream's total duration with the request context. (Before v0.9.5 the whole-body `http.Client.Timeout` killed any stream longer than `Timeout`.)

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
- **Transient errors (429, 503, 502, 500, 408, network):** Backoff and retry, rotating to other keys if available
- **Auth errors (401, 403):** Key is permanently disabled; rotates to other keys or fails immediately if none remain
- **Bad request (400):** Returns immediately — the request is broken, not the key

## Structured Errors

All API errors are returned as `*APIError` with HTTP status code, enabling reliable classification. 
*(Note: If using `NewClientWithKeys` for Auth Pool Rotation, retries happen automatically. These helpers are useful for manual flow control with a single client or application-level retries).* `APIError.RetryAfter` carries the provider's `Retry-After` hint (0 when absent, capped at `llm.MaxRetryAfter`).

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
// InputTokens/CacheReadTokens/CacheCreationTokens are disjoint for every
// provider, so each token is priced once. A model without a cache-read price
// bills cache reads at its input price.

// Audio-billed models: audio tokens use AudioInputPricePer1M, duration-billed
// speech-to-text uses AudioPricePerMinute; an unknown rate prices at 0.
sttCost := llm.EstimateCost(llm.CostInput{
    Model: "gemini-3.5-transcribe", InputTokens: 1500, AudioInputTokens: 1500, OutputTokens: 60,
})
whisperCost := llm.EstimateCost(llm.CostInput{Model: "whisper-1", AudioSeconds: 90})
fmt.Printf("STT: $%.6f / $%.6f\n", sttCost, whisperCost)

// Anthropic server tools and per-iteration usage. Searches are priced at the
// model's WebSearchPricePer1K ($10 per 1,000 on the Claude API), without the
// batch discount. With Iterations set, each iteration is priced at its own
// model (a refusal fallback at the fallback model's rates) and the top-level
// token fields are ignored; an iteration on an unpriced model adds 0. Each
// iteration's 1-hour cache share is priced at the 1-hour rate. With an
// unregistered top-level model, searches use the first priced iteration's rate.
searchCost := llm.EstimateCost(llm.CostInput{
    Model:             resp.Model,
    InputTokens:       resp.InputTokens,
    OutputTokens:      resp.OutputTokens,
    CacheCreateTokens: resp.CacheCreationTokens,
    CacheReadTokens:   resp.CacheReadTokens,
    WebSearchRequests: resp.ServerToolUse.WebSearchRequests,
    Iterations:        resp.Iterations,
})
// Conversation/Session Usage does the same per turn: Usage.Cost and
// Usage.ServerToolUse.

// Access pricing data directly
info, _ := llm.GetModelInfo("claude-opus-4-6")
fmt.Printf("Context: %d tokens, Input: $%.2f/1M\n", info.ContextWindow, info.InputPricePer1M)
```

## Request Logging (Middleware)

Enable metadata-only logging (no message content) via `LogRequests`:

Independently of `LogRequests`, the library's own operational notes (key
rotation, backoff, ignored parameters) are logged at `slog.LevelDebug` on the
default logger, so they stay silent unless you enable debug logging. Only a key
disabled by an auth error and a recovered hook panic log at `Warn`.

```go
cfg := llm.Config{
    Provider:    "anthropic",
    Model:       "claude-sonnet-4-6",
    APIKey:      apiKey,
    LogRequests: true,  // Logs provider, model, tokens, cost, elapsed time
}
client, err := llm.NewClient(cfg)
if err != nil {
    panic(err)
}
```

Or wrap an existing client manually (choose one wrapper):

```go
client = llm.WithLogging(client)
// Alternatively: client = llm.WithLoggingPrefix(client, "[MyApp]")
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

Set `Request.ResponseFormat` to constrain output to JSON. Wired for OpenAI-compatible providers (`response_format`), OpenAI Responses (`text.format`), Gemini (`responseMimeType`/`responseSchema`) and Anthropic (`output_config.format`, next to the effort a `ThinkingLevel` sets; also in Message Batches). Anthropic takes `json_schema` only: `json_object` or a schema-less request is refused, and the schema must set `additionalProperties: false` on objects. Citations cannot be combined with it.

```go
req.ResponseFormat = &llm.ResponseFormat{
    Type:   llm.ResponseFormatJSONSchema, // or llm.ResponseFormatJSONObject
    Name:   "person",
    Schema: map[string]any{
        "type":                 "object",
        "properties":           map[string]any{"name": map[string]any{"type": "string"}},
        "required":             []string{"name"},
        "additionalProperties": false,
    },
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
if err := store.Save(ctx, "chat-1", sess.Conversation()); err != nil {
    panic(err)
}
conv, err := store.Load(ctx, "chat-1") // llm.ErrConversationNotFound if absent
if err != nil {
    panic(err)
}
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

Non-chat operations use the same registered-adapter pattern as chat.
**OpenAI-compatible** drivers cover embeddings, image generation, speech, and
transcription; **Gemini** covers embeddings, image generation and transcription
(speech synthesis is unsupported). Construct one `ModalityClient` for an exact
deployment and reuse it for the worker lifetime; its safe HTTP transport,
connection pool, retry policy, proxy policy, bounded reads, caller cancellation,
and classified errors are shared across operations:

```go
cfg.Model = "text-embedding-3-small"
modalities, err := llm.NewModalityClient(cfg)
if err != nil { /* handle */ }
defer modalities.Close()

emb, _ := modalities.GenerateEmbeddings(ctx, llm.EmbeddingRequest{Model: cfg.Model, Input: []string{"hi"}})

// A different exact deployment/client owns image generation.
imageCfg := cfg; imageCfg.Model = "gpt-image-2"
imageClient, _ := llm.NewModalityClient(imageCfg)
defer imageClient.Close()
img, _ := imageClient.GenerateImages(ctx, llm.ImageRequest{
    Model: "gpt-image-2", Prompt: "a cat", WidthPixels: 1024, HeightPixels: 1024,
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

// Gemini chat models are natively audio-capable: transcription runs through
// generateContent with the same API key, no separate speech service.
geminiCfg := llm.Config{Provider: "gemini", Model: "gemini-3.5-flash-lite", APIKey: key}
geminiSTT, _ := llm.NewModalityClient(geminiCfg)
defer geminiSTT.Close()
transcript, _ := geminiSTT.TranscribeAudio(ctx, llm.TranscriptionRequest{
    Audio: webm, MediaType: "audio/webm", Language: "de-AT",
    // Optional vocabulary/context hint (Whisper `prompt` on OpenAI-compatible).
    Prompt: "Faschierter Braten, Weckerl, Kraut",
})
```

For dedicated speech-to-text, `gemini-3.5-transcribe` takes no prompt; bias it
with `Vocabulary` instead, which every adapter supports (Gemini Transcribe:
`customVocabulary`; Gemini chat models: instruction; OpenAI: Whisper `prompt`;
xAI `grok-voice-transcribe-2.0`: `keyterm`):

```go
stt, _ := llm.NewModalityClient(llm.Config{Provider: "gemini", Model: "gemini-3.5-transcribe", APIKey: key})
text, _ := stt.TranscribeAudio(ctx, llm.TranscriptionRequest{
    Audio: webm, MediaType: "audio/webm", Language: "de-AT",
    Vocabulary: []string{"Weckerl", "faschierter Braten", "Meal Prep"},
})
```

Dedicated or preview models can break without notice. Chain a fallback so a
provider failure on one deployment is answered by the next:

```go
stt, _ := llm.NewFallbackModalityClient(
    llm.Config{Provider: "gemini", Model: "gemini-3.5-transcribe", APIKey: key},
    llm.Config{Provider: "gemini", Model: "gemini-3.5-flash-lite", APIKey: key},
)
```

Failover happens only on provider failures (an `APIError` or a transport
error), never on validation errors or cancellation, and without backing off on
the earlier deployments. The final error classifies as the last attempt.

By default *any* `APIError` fails over — including a 400/413/422 for a payload
the provider rejected as malformed or oversized, which is then sent (and
billed) again at the next vendor. When that egress or cost matters, pick the
recommended policy, which fails over only on transport errors, 408/429/5xx and
401/403/404:

```go
stt, _ := llm.NewFallbackModalityClientWithPolicy(llm.FailoverTransientOnly,
    llm.Config{Provider: "gemini", Model: "gemini-3.5-transcribe", APIKey: key},
    llm.Config{Provider: "openai", Model: "whisper-1", APIKey: openaiKey},
)
```

A `FailoverPolicy` is a plain `func(ctx, err) bool`, so it can be extended (for
example to also fail over on a 400 from a model you know is broken).
`NewFallbackModalityClient` equals `NewFallbackModalityClientWithPolicy(nil, …)`
(`FailoverAnyProviderError`).

To show what each call costs, set `Config.UsageHook`. It receives one
`UsageEvent` per provider attempt — failed attempts included, with `Err` set —
carrying the operation, provider, model, provider-reported tokens (with the
audio share in `AudioInputTokens`), billed audio seconds for duration-billed
speech-to-text, latency, and `CostUSD` estimated from registry list prices
(0 when a price is unknown; never guessed). In a fallback chain `Attempt` is
the deployment's position and `Fallback` is true for every deployment but the
primary, so a failover is two events. A fallback without its own hook uses the
primary's. `InputTokens` excludes cached prompt tokens, which are reported in
`CacheReadTokens` and priced at the cache-read rate (same contract as
`Response`).

To attribute events to the request that caused them on a client shared by many
callers, set `Config.UsageHookCtx` instead (or as well): it receives the same
events plus the caller's `ctx`, so a user or tenant ID carried as a context
value reaches the hook. When both hooks are set, each sees every event
(`UsageHook` first). The ctx may already be cancelled — read its values, don't
do I/O with it.

```go
type tenantKey struct{}
cfg.UsageHookCtx = func(ctx context.Context, e llm.UsageEvent) {
    tenant, _ := ctx.Value(tenantKey{}).(string)
    ledger.Record(tenant, e.Operation, e.CostUSD)
}
text, err := stt.TranscribeAudio(context.WithValue(ctx, tenantKey{}, userID), req)
```

```go
cfg := llm.Config{Provider: "gemini", Model: "gemini-3.5-transcribe", APIKey: key,
    UsageHook: func(e llm.UsageEvent) {
        log.Printf("%s %s/%s attempt=%d fallback=%v $%.6f %v err=%v",
            e.Operation, e.Provider, e.Model, e.Attempt, e.Fallback, e.CostUSD, e.Latency, e.Err)
    }}
stt, _ := llm.NewFallbackModalityClient(cfg,
    llm.Config{Provider: "gemini", Model: "gemini-3.5-flash-lite", APIKey: key})
```

The hook runs synchronously after each attempt (keep it fast), a panic in it is
recovered, and it never changes the call's result. Requests rejected by
validation before dispatch are not reported. Usage sources: Gemini
`usageMetadata` (transcription, image generation), OpenAI transcription `usage`
(tokens or duration), xAI `/v1/stt` `duration`, and OpenAI-compatible embedding
`prompt_tokens`. Gemini embeddings, image generation on OpenAI-compatible
endpoints and speech synthesis report no usage (tokens and cost 0; before
v0.9.5 Gemini embeddings returned a `len(text)/4` guess in
`EmbeddingResponse.InputTokens`).

`TranscriptionRequest.Prompt` is a spelling hint of at most
`MaxTranscriptionPromptRunes` runes, never content to transcribe. Gemini accepts
WAV, MP3, AIFF, AAC, OGG, FLAC, WebM and MP4/M4A inline up to 14 MiB (its 20 MB
request ceiling after base64); the declared media type must match the bytes'
signature (`llm.AudioMediaTypeFromSignature`), and `Language` must be a tag such
as `de` or `de-AT`.

Exact image count, geometry, output media type, speech voice, and transcription
language are application choices. Rho neither selects preferred values nor
imposes product-level size/count defaults; it validates only generic structure,
reviewed model capability, and whether the registered adapter can encode the
request. Omitted optional values retain the endpoint default where the operation
contract permits omission.

Generated base64 images receive a media type only after their decoded signature
matches the requested format. URL-only responses fail because rho requires inline
bytes: GPT Image models return base64 by contract, while legacy compatible models
are sent `response_format: "b64_json"`. Speech synthesis returns
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
once. Drivers: **OpenAI** (chat, responses, embeddings), **Anthropic Message Batches** (chat), and
**Gemini Batch** (chat + embeddings). It returns a serializable `BatchHandle` you can
persist and poll later — even after a restart. The interface is provider-agnostic; each vendor
registers a batch driver for its wire protocol.

```go
cfg := llm.DefaultConfig(); cfg.Provider = "openai"; cfg.APIKey = os.Getenv("OPENAI_API_KEY")
bc, _ := llm.NewBatchClient(cfg)           // gated by ProviderPreset.SupportsBatch
defer bc.Close()

handle, _ := bc.Submit(ctx, []llm.BatchItem{
    {ItemID: "q1", Request: &llm.Request{Model: "gpt-4.1-mini",
        Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentPart{{Type: llm.ContentText, Text: "classify: ..."}}}}}},
    {ItemID: "q2", Request: &llm.Request{Model: "gpt-4.1-mini", /* ... */ }},
}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour})

blob, _ := json.Marshal(handle)            // persist; resume after a restart with llm.LoadBatchHandle(blob)

// Anthropic / Gemini use the same API (Provider: "anthropic" or "gemini", SupportsBatch).

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

### Mistral & Chinese hosts (beyond chat)

These use the **openai_compat** wire. Chat works with the named presets. **Embeddings**
work when the host exposes OpenAI-shaped `/embeddings` and the model is registered
with `CapabilityEmbeddings` (e.g. Mistral `mistral-embed`, DashScope
`text-embedding-v3`):

```go
cfg := llm.Config{Provider: "mistral", Model: "mistral-embed", APIKey: os.Getenv("MISTRAL_API_KEY")}
mc, _ := llm.NewModalityClient(cfg)
emb, _ := mc.GenerateEmbeddings(ctx, llm.EmbeddingRequest{Model: "mistral-embed", Input: []string{"query"}})

// Mainland DashScope (compatible-mode):
cfgCN := llm.Config{Provider: "dashscope-cn", Model: "text-embedding-v3", APIKey: os.Getenv("DASHSCOPE_API_KEY")}
```

**Not** first-class in rho today: MiniMax proprietary speech/video APIs, DashScope
native multimodal-embed (non-OpenAI shape), DeepSeek embeddings (vendor has no
public vector endpoint). Use chat + external embedders, or `RegisterModel` + custom
`BaseURL` when a gateway exposes an OpenAI-shaped path.

### Realtime (OpenAI reference)

Long-lived bidirectional sessions are a **separate** surface from `Complete`/`Stream`.
Inject a `RealtimeDialer` (typically a WebSocket) to open a session; offline unit
tests drive framing through an in-memory conn:

```go
session, err := llm.OpenRealtimeSession(ctx, cfg, myWebSocketDialer)
if err != nil { /* handle */ }
defer session.Close()
_ = session.SendAudio(ctx, base64PCM)
_ = session.CommitAudio(ctx)
_ = session.RequestResponse(ctx)
ev, _ := session.Recv(ctx) // session.created, response.audio.delta, ...
```

### OAuth (device flow)

For providers that issue tokens via OAuth, use the RFC 8628 device-authorization grant; the resulting access token becomes `Config.APIKey`:

```go
authCfg := llm.DeviceAuthConfig{ClientID: "...", DeviceAuthURL: "...", TokenURL: "..."}
da, err := llm.StartDeviceAuth(ctx, authCfg)
if err != nil {
    panic(err)
}
fmt.Printf("Visit %s and enter %s\n", da.VerificationURI, da.UserCode)
tok, err := llm.PollDeviceToken(ctx, authCfg, da) // honors authorization_pending / slow_down
if err != nil {
    panic(err)
}
```

### SSRF hardening

`Config.BlockPrivateBaseURL` rejects a `BaseURL` (or per-key override) whose host is `localhost` or a private or reserved IP literal (e.g. the cloud metadata IP), including IPv6 forms that carry one (IPv4-mapped and -compatible, 6to4, Teredo, NAT64; in `64:ff9b::/96` the embedded IPv4 address decides), and refuses at connect time every resolved address that is private or reserved (`ErrPrivateAddress`, not retryable), so host names and DNS rebinding are covered too; the configured or ambient proxy is exempt — opt-in, off by default so local providers keep working. `BaseURL` is otherwise a **trusted** value: the API key is sent to it, so never populate it from untrusted input.

## Config Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| Provider | string | "anthropic" | Provider name |
| Model | string | "claude-sonnet-5-5" | Model identifier |
| ModelCapabilities | CapabilitySet | 0 | Exact deployment-scoped reviewed capabilities for Model; overrides global registry metadata |
| APIKey | string | "" | API key (empty OK for local providers) |
| MaxTokens | int | 8192 | Max output tokens (zero is floored to the default; see the note below on Gemini thinking models) |
| Temperature | *float64 | nil | Sampling temperature (nil = provider default, omitted from wire) |
| ThinkingLevel | ThinkingLevel | "" | Extended thinking: ThinkingMinimal/Low/Medium/High/XHigh/Max. Providers without a level get their deepest one: `ThinkingMax` → OpenAI `xhigh`, Gemini `high` |
| Timeout | Duration | 120s | Non-stream request timeout; for streams, the header wait and max silent gap between bytes (see [Streaming and `Config.Timeout`](#streaming-and-configtimeout)) |
| BaseURL | string | "" | Override provider endpoint |
| ProxyURL | string | "" | Explicit reviewed HTTP(S) forward proxy; overrides ambient proxy variables |
| DisableProxy | bool | false | Explicitly bypass configured and ambient proxies for local/private endpoints |
| AuthHeader | string | "Bearer" | Override auth header format |
| ProviderName | string | "" | Override Client.Provider() |
| LogRequests | bool | false | Enable request/response metadata logging |
| RetryPolicy | *RetryPolicy | nil | Configurable backoff (nil = DefaultRetryPolicy: 1s–30s, 2x, ±25% jitter) |
| CircuitThreshold | int | 5 (`DefaultConfig`), 0 (struct literal) | Consecutive failures to open circuit (0 = disabled) |
| CircuitCooldown | Duration | 30s | Open→half-open cooldown |
| CooldownRateLimit | Duration | 60s | Profile cooldown for 429 errors |
| CooldownOverload | Duration | 30s | Profile cooldown for 503 errors |
| CooldownDefault | Duration | 10s | Profile cooldown for other transient errors |
| RetryHook | RetryHook | nil | Observability hook for retry lifecycle events |
| UsageHook | UsageHook | nil | Per-attempt modality usage + cost events |
| UsageHookCtx | ContextUsageHook | nil | Same events plus the caller's ctx, for per-request attribution |
| MaxRetries | int | 10 (`DefaultConfig`), 0 (struct literal) | Cap on retry/rotation iterations (min effective: 3); 0 = 3 attempts for single-key and modality/batch calls |
| RetryBudget | Duration | 0 | Total wall-clock bound across one call's retries (0 = none) |
| DisableRetries | bool | false | Exactly one transport attempt (caller owns retry) |
| BetaFeatures | []string | ["interleaved-thinking-..."] | Provider beta flags (Anthropic: anthropic-beta header, sent on every request and on Message Batch creation; the interleaved-thinking flag only with budget thinking) |
| AnthropicVersion | string | "2026-10-21" | Anthropic API version header; `llm.AnthropicVersion20230601` keeps temperature/top_p/top_k on the Claude 4.6 and 4.5 models |
| MaxErrorBodyBytes | int | 1 MB | Cap on error response body reads |
| MaxSSELineBytes | int | 256 KB | Cap on per-line SSE buffer |
| MaxResponseBodyBytes | int | 32 MB | Cap on success response body reads |
| MaxToolInputBytes | int | 1 MB | Cap on accumulated tool input JSON |
| MaxErrorMessageLen | int | 4096 | Cap on stored error message length |
| BlockPrivateBaseURL | bool | false | Opt-in SSRF guard: reject private or reserved BaseURL hosts and dial targets |

Structured output is selected per call with `Request.ResponseFormat`; it is not a
client `Config` field.

## Model Registry

Use `ResolveModelAlias()` for short aliases:

```go
model := llm.ResolveModelAlias("opus")   // -> "claude-opus-5-5"
model = llm.ResolveModelAlias("grok")    // -> "grok-4.7"
model = llm.ResolveModelAlias("flash")   // -> "gemini-3.8-flash"
```

### Anthropic aliases

| Alias | Resolves to |
|-------|-------------|
| `opus` | `claude-opus-5-5` |
| `sonnet` | `claude-sonnet-5-5` |
| `haiku` | `claude-haiku-5-5` |
| `fable` | `claude-fable-5` |
| `claude` | `claude-sonnet-5-5` |

### xAI / Grok aliases

| Alias | Resolves to |
|-------|-------------|
| `grok`, `grok4`, `grok-4`, `grok4.7`, `grok-4-7` | `grok-4.7` |
| `grok4.6`, `grok-4-6` | `grok-4.6` |
| `grok4.5`, `grok-4-5` | `grok-4.5` |
| `grok4.3`, `grok-4-3` | `grok-4.3` |
| `grok4.2`, `grok4.20` | `grok-4.20-0309-reasoning` |
| `grok4.1`, `grok-4-1` | `grok-4-1-fast-non-reasoning` |
| `grok-reasoning`, `grok-4-reasoning` | `grok-4-fast-reasoning` |
| `grok-4-1-reasoning` | `grok-4-1-fast-reasoning` |
| `grok-code` | `grok-code-fast-1` |
| `grok-mini` | `grok-3-mini` |

### OpenAI aliases

| Alias | Resolves to |
|-------|-------------|
| `gpt`, `gpt6.1`, `gpt-6.1` | `gpt-6.1-sol` |
| `gpt6`, `gpt-6` | `gpt-6-sol` |
| `gpt6-luna` | `gpt-6-luna` |
| `gpt6-astra` | `gpt-6-astra` |
| `gpt5`, `gpt5.6`, `gpt-5.6` | `gpt-5.6-sol` |
| `gpt5.5` | `gpt-5.5` |
| `gpt5.5-pro` | `gpt-5.5-pro` |
| `gpt5.4` | `gpt-5.4` |
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
| `groq`, `gpt-oss` | `openai/gpt-oss-120b` |
| `llama`, `llama-70b` | `llama-3.3-70b-versatile` (enterprise tier only since 2026-08-16) |
| `llama-8b` | `llama-3.1-8b-instant` (enterprise tier only since 2026-08-16) |
| `gpt-oss-20b` | `openai/gpt-oss-20b` |

### Mistral aliases

| Alias | Resolves to |
|-------|-------------|
| `mistral-large` | `mistral-large-2512` |
| `mistral-medium` | `mistral-medium-latest` |
| `mistral-small` | `mistral-small-2603` |
| `codestral` | `codestral-2508` |
| `ministral` | `ministral-8b-2512` |

### DeepSeek aliases

| Alias | Resolves to |
|-------|-------------|
| `deepseek-cloud`, `deepseek-v4` | `deepseek-flash` |

### Z.ai / GLM aliases

| Alias | Resolves to |
|-------|-------------|
| `glm`, `glm5.3`, `glm-5-3` | `glm-5.3` |
| `glm5.2`, `glm-5-2` | `glm-5.2` |
| `glm4.7`, `glm-4-7` | `glm-4.7` |

Aliases that name a model the provider has retired (`pixtral`, `magistral`,
`devstral`, `llama4`, `gpt5.3`, `gpt-instant`, `deepseek-reasoner`) are kept so
they fail with the retired-model error and its replacement (see below).

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
| `flash` | `gemini-3.8-flash` |
| `gemini-pro`, `gemini3.1`, `gemini3`, `gemini-3` | `gemini-3.1-pro-preview` |

> **Gemini 3 note:** `gemini-3.1-pro-preview` and `gemini-3-flash-preview` use
> `ThoughtSignature` — the model returns an opaque signature in tool call responses
> that must be echoed in the corresponding `tool_result`. The adapter handles this
> automatically; no changes to calling code are required.

`gemini-4-argon`, `gemini-3.8-flash`, `gemini-3.7-flash`, `gemini-3.6-flash` and
`gemini-3.5-flash-lite` deliberately do not advertise
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
provider := llm.ProviderForModel("gemini-3.8-flash") // -> "gemini"

// Get the default model for a provider
model := llm.GetDefaultModel("xai") // -> "grok-4.7"
```

**`MaxTokens` on Gemini thinking models.** Gemini 2.5 models reason internally and
charge that reasoning against `maxOutputTokens`, but they do not accept a
`thinkingConfig` to bound it. A small `MaxTokens` would therefore be consumed by
thinking before any visible text is produced, so the Gemini adapter pads
`maxOutputTokens` by the `ThinkingLow` budget (capped at the model's ceiling) and
logs a warning naming the original and padded values. The visible answer can
therefore be longer than the `MaxTokens` you set on such a model; other providers
and non-thinking Gemini models honour it exactly.

**Retired models.** IDs the provider has sunset are removed from the registry, so a
request naming one fails at dispatch with an actionable error rather than a generic
"no reviewed capability metadata" message:

```go
_, err := client.Complete(ctx, req) // Model: "claude-sonnet-4-20250514"
// llm: model "claude-sonnet-4-20250514" was retired by its provider; use "claude-sonnet-4-6" instead
```

Query the mapping directly with `RetiredModelReplacement`, which resolves aliases
before looking up the replacement:

```go
if replacement, retired := llm.RetiredModelReplacement(model); retired {
    log.Printf("%s is retired; switching to %s", model, replacement)
    model = replacement
}
```

If you still need a retired ID (a proxy or a vendor that keeps serving it), re-add it
with `RegisterModel` — the registry is extensible at runtime, below.

**Extending the registry at runtime.** Unlisted models return a cost estimate of `0`
and cannot dispatch until they have reviewed capability metadata. Register metadata
for any model — or correct stale built-in pricing in place — without waiting for a
release. Both calls are safe for concurrent use. A zero `Capabilities` field gets
the same conservative chat defaults as built-ins; set explicit bits for vision,
tools, embeddings, etc., or use `Config.ModelCapabilities` on the client.

```go
// Add (or override) a model's metadata — it now feeds EstimateCost, capability
// validation, and the discovery API (Models/ModelsByProvider).
llm.RegisterModel(llm.ModelInfo{
    ID:               "my-self-hosted-llm",
    Provider:         "vllm",
    ContextWindow:    128000,
    InputPricePer1M:  0.20,
    OutputPricePer1M: 0.60,
    // Optional: Capabilities defaults to chat+stream(+tools) for local hosts.
    // Capabilities: llm.Capabilities(llm.CapabilityChat, llm.CapabilityStream, llm.CapabilityTools),
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
  normalize.go                           # NormalizeForProvider/NormalizeForModel() handoff pass
  servertools.go                         # ServerTool, citations, server tool results + usage types
  pauseturn.go                           # ContinuePausedTurn() (pause_turn continuation)
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
