# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.7] - 2026-02-21

### Added

- **OpenAI model registry** — Added 16 OpenAI models to the model registry: GPT-5.2/5.1/5 (with `-chat-latest` variants), GPT-5 Mini/Nano, GPT-4.1/Mini/Nano, and O-series (o3, o3-mini, o4-mini). Includes default model (`gpt-5.2`), available models list, and aliases (`gpt`, `gpt5`, `gpt4.1`, etc.).

- **Grok reasoning model flags** — Added `Thinking: true` to `grok-4-1-fast-reasoning` and `grok-4-fast-reasoning` in the model registry.

- **Registry thinking flag tests** — Added tests validating that `Thinking` and `SupportsThinking` flags are correctly set across all models in the registry.

### Fixed

- **`ProviderForModel()` returned empty for OpenAI models** — No OpenAI models were registered in the model registry, so provider auto-detection failed for any `gpt-*` or `o*` model ID.

- **`GetDefaultModel("openai")` returned `claude-sonnet-4-6`** — No `"openai"` entry in the `defaultModels` map caused the function to fall through to the hardcoded Anthropic fallback. Now returns `gpt-5.2`.

## [0.1.6] - 2026-02-21

### Added

- **Per-profile BaseURL support** — Keys can now include a custom endpoint using the `API_KEY|BASE_URL` format. This enables failover across different backends (e.g., primary Anthropic API → Azure proxy → local vLLM fallback). The `clientFunc` callback now receives the full `AuthProfile` instead of just the API key string.

### Fixed

- **rotateClient() race condition** — Previously, `rotateClient()` called `Close()` on the old client after swapping. This raced with in-flight requests still using the old client reference. Current adapters' `Close()` is a no-op so this was latent, but would break if adapters ever did real cleanup. Now orphaned clients are garbage collected instead of explicitly closed.

- **Thundering herd on rotation** — When 50 goroutines hit a 429 simultaneously, all 50 would pass the "already rotated?" check (single-checked outside the lock), create 50 new HTTP clients, and queue up to overwrite `pc.client` 50 times. Now uses proper double-checked locking with a dedicated `rotateMu` mutex: goroutine 1 rotates while goroutines 2-50 block at the gate, then short-circuit when they see the rotation already happened.

- **Zero tokens indistinguishable from "not reported"** — OpenAI and Anthropic streaming parsers initialized token counts to 0. If a stream ended without a usage chunk (connection dropped, provider quirk), `EventDone` reported 0 tokens — indistinguishable from a legitimate zero-token response. Now token counts initialize to -1; callers can distinguish "not reported" (-1) from "zero tokens" (0).

- **Unflushed tool call on abrupt stream end** — If an OpenAI-compatible stream ended without `finish_reason` (network drop, spec-violating servers like Ollama dumping `[DONE]` without finish), any accumulated `currentToolCall` was silently lost. Now tool calls are flushed before emitting `EventDone`, regardless of whether `finish_reason` arrived.

### Changed

- `NewPooledClient` signature changed: `clientFunc` is now `func(profile AuthProfile) (Client, error)` instead of `func(apiKey string) (Client, error)`.

- `NewAuthPool` now parses keys with `|` separator: the portion before `|` is the API key, the portion after is the BaseURL. Keys without `|` work as before.

## [0.1.5] - 2026-02-21

### Fixed

- **Critical: Auth error poison loop** — A single revoked API key (401/403) would permanently poison the `PooledClient`, causing 100% of subsequent requests to fail until process restart. The bug: auth errors returned early without calling `MarkFailed()` or `rotateClient()`, leaving the dead key active. Now auth errors trigger rotation to try other keys, and permanently disable the bad key (`IsHealthy = false`) instead of applying a temporary cooldown.

- **Critical: Single-key clients had no retry/backoff** — `NewClientWithKeys` with exactly 1 key bypassed `PooledClient` entirely, returning a raw adapter with zero retry logic. A transient 502/429 would fail immediately. Now single-key clients go through `PooledClient` and get the same exponential backoff (1s→2s→4s...) as multi-key pools.

- **Missing backoff on rotation failure** — When all keys were in cooldown, `PooledClient` gave up immediately instead of backing off. Now uses `Backoff()` with exponential delay (1-30s) and retries, respecting context cancellation. Minimum 3 retries for single-key resilience.

- **Parallel tool results truncated in OpenAI adapter** — When a user message contained multiple tool results (parallel tool execution), only the first was sent to OpenAI; the rest were silently dropped. This caused 400 Bad Request errors ("missing tool result for call_id"). Now each `ContentToolResult` in a message emits a separate `role: "tool"` message as OpenAI's format requires.

- **Auth errors wasted retries with dead keys** — When a 401/403 occurred and rotation failed (single-key pool or all keys revoked), the code would backoff and retry 3 times with the same dead key. Now auth errors return immediately when no healthy keys remain — backoff is only for transient errors (429, 503) where waiting helps.

- **OpenAI streaming never reported token usage** — Missing `stream_options: {include_usage: true}` in requests meant OpenAI/Groq/xAI streams never sent usage stats. Now set automatically for all streaming requests. Additionally, the parser was fixed to handle OpenAI's two-chunk format: `finish_reason` arrives in one chunk (with empty usage), then usage arrives in a separate chunk (with empty choices). The parser now accumulates state across chunks and emits `EventDone` with complete data after `[DONE]`.

- **Dead profiles corrupted cooldown calculation** — When finding the soonest available key, dead profiles (`IsHealthy=false`) with `Cooldown=ZeroTime` (year 0001) would be selected as "soonest," producing `time.Until()` results of -17 million hours. Now dead profiles are filtered out before comparing cooldowns. If all keys are permanently dead, returns a distinct error.

- **Retry loop ignored pool cooldowns** — When rotation failed due to cooldown, the retry loop used its own exponential backoff (1s→2s→4s) instead of the pool's authoritative wait time (e.g., 60s for rate limits). This caused the loop to blast the rate-limited API 3+ times during the penalty period. Now `GetAvailable()` returns a `CooldownError` with the exact wait duration, and the retry loop honors it.

- **`NewClient(cfg)` bypassed PooledClient** — Despite documentation claiming retry/backoff support, `NewClient(cfg)` passed `nil` to `NewClientWithKeys`, which fell through to `newSingleClient` (zero resilience). Now `NewClient(cfg)` injects `cfg.APIKey` into a single-element slice, routing through `PooledClient` for automatic retry.

### Changed

- `GetAvailable()` now returns `*CooldownError` (with `Wait` duration) instead of a formatted string error when all healthy keys are in cooldown. Callers can extract the exact wait time via `errors.As()`.

- Gemini adapter: synthetic tool call IDs now use `makeToolCallID()` helper, keeping format definition in one place with its inverse `resolveToolName()`.

- Scanner buffer allocation simplified across all adapters: removed wasteful pre-allocation, using `scanner.Buffer(nil, 1MB)` instead.

## [0.1.4] - 2026-02-21

### Changed

- **Breaking**: `Client.Stream` now returns `iter.Seq2[StreamEvent, error]` (Go 1.23 iterator) instead of taking a `StreamCallback` — callers use `for event, err := range client.Stream(ctx, req)` and `break` to abort cleanly; `defer resp.Body.Close()` inside the yield closure handles HTTP cleanup
- **Breaking**: `StreamCallback` type removed — replaced by the iterator pattern above
- **Breaking**: Introduced named string types for stringly-typed domains:
  - `type Role string` (`RoleUser`, `RoleAssistant`, `RoleSystem`) — replaces raw `string` on `Message.Role`
  - `type ContentType string` (`ContentText`, `ContentImage`, `ContentToolUse`, `ContentToolResult`) — replaces raw `string` on `ContentPart.Type`
  - `type EventType string` (`EventContent`, `EventToolUse`, `EventThinking`, `EventDone`, `EventError`) — replaces raw `string` on `StreamEvent.Type`
  - `type ThinkingLevel string` (`ThinkingNone`, `ThinkingLow`, `ThinkingMedium`, `ThinkingHigh`) — replaces raw `string` on `Request.ThinkingLevel`
- `NewTextMessage` first parameter changed from `string` to `Role` (untyped string literals still auto-convert)
- `ThinkingNone` is `""` (zero value) instead of `"none"` — simplifies checks to `req.ThinkingLevel != ThinkingNone`

### Fixed

- `PooledClient.Stream` now retries on **pre-data** connection failures (429, 503, etc.) — mirrors `Complete`'s rotation/backoff logic for errors that occur before any events are yielded to the caller; mid-stream errors still pass through immediately (retrying would duplicate content)
- **Data race fix**: `AuthPool.GetCurrent()` and `AuthPool.GetAvailable()` now return by value (`AuthProfile`, not `*AuthProfile`) — prevents callers from holding unprotected pointers to mutable pool state after the mutex unlocks
- **Global state corruption fix**: `GetAvailableModels()` now returns a copy of the internal slice — previously returned a direct reference, allowing callers to mutate the global registry array in place
- **Upstream DDoS fix**: `PooledClient.Complete` and `PooledClient.Stream` now return immediately when rotation fails (all profiles in cooldown) — previously ignored the pool's own cooldown state, waited 1s via exponential backoff, then fired a second request at the same rate-limited API key
- **Logging black hole fix**: `LoggingClient.Stream` "stream done" log is now in a `defer` — previously skipped entirely when the caller broke early from the iterator (`yield` returning false)
- Anthropic and Gemini adapters now respect `cfg.BaseURL` — previously hardcoded their endpoints, silently ignoring any override; `ResolveBaseURL()` is now called at construction time, consistent with the OpenAI-compat adapter
- `EventUsage` constant removed — was never emitted by any adapter (token usage arrives on `EventDone`); keeping it invited callers to write dead `case` branches
- `PooledClient.rotateClient()` now logs `Close()` errors on the replaced client instead of silently discarding them (`_ = oldClient.Close()` → `slog.Warn`)

## [0.1.3] - 2026-02-22

### Changed

- Renamed `ContentPart.Content` → `ContentPart.ToolResultContent` to eliminate naming confusion with `ContentPart.Text` (breaking change — both fields meant "the text content" but for different `Type` values; the new name makes the purpose unambiguous)
- JSON wire format unchanged (`"content"` tag preserved)
- `PooledClient.Stream` no longer retries on failure — streams cannot be safely retried because the callback may have already received partial events; retrying would silently duplicate content

### Fixed

- `Temperature: 0.0` is no longer silently omitted from API requests — removed `omitempty` from `Temperature float64` JSON tags in `Request` and all adapter request types (Go's `omitempty` treats 0.0 as empty, causing deterministic temperature to be sent as provider default)
- Anthropic streaming now reports token usage in `done` event — `InputTokens` captured from `message_start`, `OutputTokens` from `message_delta` (previously always 0, breaking cost estimation and logging)
- Anthropic streaming `StopReason` now correctly read from `delta.stop_reason` (was reading from non-existent `message.stop_reason`, always empty)
- Fixed 3 remaining `io.ReadAll` errors silently discarded in streaming error paths (inconsistent with non-streaming paths fixed in v0.1.2)
- `PooledClient.Complete` retry backoff now respects context cancellation — uses `select` with `ctx.Done()` instead of blocking `time.Sleep`
- Gemini tool call IDs are now unique — format changed from `call_<name>` to `call_<index>_<name>` so duplicate calls to the same tool get distinct IDs; `resolveToolName()` handles both formats

### Added

- Unit tests for `pool.go`: AuthPool (creation, empty pool, rotation, all-in-cooldown, mark success, status) and PooledClient (complete success, retry on retryable, no retry on non-retryable, stream no-retry, stream success, context cancellation, no keys, provider/model delegation, pool status) — 15 tests total

## [0.1.2] - 2026-02-22

### Changed

- Unexported all mutable package-level maps (`ModelRegistry`, `ModelAliases`, `DefaultModels`, `AvailableModels`, `Presets`) — accessed via read-only functions only
- Removed `sync.RWMutex` from `registry.go`, `provider.go`, `register.go` — data is immutable after init, no locking needed
- Removed dead `sync.RWMutex` fields from `openaicompat.Client` and `gemini.Client` (never used)
- Removed duplicate auth rotation from Anthropic adapter (`profiles`, `currentIndex`, `getNextProfile`) — `PooledClient` handles this
- Simplified `anthropic.Complete` / `anthropic.Stream` — direct calls, no retry loop
- License changed from MIT to Apache 2.0

### Fixed

- Fixed 14 swallowed errors across all 3 provider adapters:
  - 6x `io.ReadAll` errors silently discarded in error response paths — now logged via `slog.Warn`
  - 3x `json.Unmarshal` errors silently skipped in SSE stream parsing — now logged via `slog.Warn`
  - 4x `json.Unmarshal` errors silently discarded in tool input parsing — now logged + raw string preserved as fallback
  - 1x `json.Marshal` error silently discarded in tool input serialization — now logged + empty object fallback

### Added

- Read-only accessor functions: `GetModelInfo`, `GetDefaultModel`, `GetAvailableModels`, `ResolveModelAlias`, `ProviderForModel`, `PresetFor`

### Removed

- Exported maps: `Presets`, `ModelRegistry`, `DefaultModels`, `AvailableModels`, `ModelAliases` (breaking change — use accessor functions)
- `RegisterModel`, `RegisterAlias`, `RegisterPreset`, `AllPresets` (unnecessary write/copy functions for static data)
- `anthropic.Client.AuthProfiles` support (use `NewClientWithKeys` for multi-key rotation)
- Dead `ToolResult` type from `types.go` (never referenced)
- Dead `Config.Cooldown` field (pool uses hardcoded per-error-type cooldowns)

## [0.1.1] - 2026-02-10

### Changed

- Replaced `shared/logger` dependency with stdlib `log/slog` — zero external dependencies
- `LoggingClient` now stores `*slog.Logger` instead of string prefix
- Standalone module path: `gitlab2024.bds421-cloud.com/bds421/rho/llm`

## [0.1.0] - 2026-01-30

### Added

- Unified `Client` interface: `Complete`, `Stream`, `Provider`, `Model`, `Close`
- Provider adapters: Anthropic (native), Gemini (native), OpenAI-compatible (11+ providers)
- `database/sql`-style driver registration via `RegisterProvider` / `init()`
- Streaming with typed `StreamEvent` callbacks (content, tool_use, thinking, usage, done, error)
- Tool use / function calling across all three wire protocols
- Extended thinking support (Anthropic budget, Gemini ThoughtSignature)
- Auth pool rotation with per-profile cooldown (`PooledClient`)
- Structured errors: `APIError` with `IsRateLimited`, `IsOverloaded`, `IsRetryable`, `IsAuthError`, `IsContextLength`
- Exponential backoff with jitter (`Backoff`)
- Logging middleware (`LoggingClient`) -- metadata only, no message content
- Model registry with 65+ models, 23 short aliases, per-model pricing
- Cost estimation (`EstimateCost`)
- Zero external dependencies (stdlib only, `log/slog` for logging)
