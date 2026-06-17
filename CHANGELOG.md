# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.12] - 2026-06-17

Hardening pass 11 — verified the Anthropic / OpenAI-compatible / Responses adapters use fixed
URL paths and cannot be URL-injected via a hostile model name (no defect; regression coverage).

## [0.4.11] - 2026-06-17

Hardening pass 10 — defense-in-depth.

### Security

- **The Gemini adapter URL-escapes the model name** — the model is interpolated into the
  request URL path (`/{model}:generateContent`); an unescaped model containing URL
  metacharacters (`?`, `#`, `/`) could inject a query parameter or manipulate the path
  (same-host; no auth bypass — Gemini auth is header-based). The model is now `url.PathEscape`d.

## [0.4.10] - 2026-06-17

Hardening pass 9 — direct adversarial coverage for `ValidateImageSource` (no defect found).
The adversarial attack surface is now exhausted across resilience, streaming, persistence,
handoff, serialization, secret redaction, registry, request-building (all adapters), session,
OAuth, capabilities, middleware, the SSRF guard, and structured output.

## [0.4.9] - 2026-06-17

Hardening pass 8 — structured output.

### Fixed

- **A `json_schema` response format with no schema degrades to plain JSON mode** — it
  previously serialized to `"schema": null`, which the provider rejects. (No serious bug found
  in structured-output handling.)

## [0.4.8] - 2026-06-17

Hardening pass 7 — verified the OpenAI-compatible and Responses adapters do not emit an
empty-content message (no defect found; regression coverage added).

## [0.4.7] - 2026-06-17

Hardening pass 6 — adapter `buildRequest` message integrity.

### Fixed

- **The Anthropic adapter drops an empty-content message** — a message whose content all
  filtered out (an empty-text part, or a thinking block dropped because thinking is off) was
  emitted as `"content": null`, which Anthropic rejects. It is now skipped, matching
  `NormalizeForProvider`. (The Gemini adapter already skipped empties.)

## [0.4.6] - 2026-06-17

Hardening pass 5 — SSRF guard (`CheckBaseURL`).

### Security

- **The `BlockPrivateBaseURL` SSRF guard now normalizes a trailing-dot host** — `localhost.`
  / `127.0.0.1.` is an absolute FQDN that resolvers still map to loopback, but it evaded both
  the literal `localhost` comparison and `net.ParseIP`, slipping past the guard. The host is
  now stripped of a trailing dot before the loopback/private-IP checks.

## [0.4.5] - 2026-06-17

Hardening pass 4 — `LoggingClient` (`middleware.go`).

### Fixed

- **`LoggingClient.Complete` no longer panics on a `(nil, nil)` inner response** — it checked
  the returned error but then dereferenced `resp` to log token/cost metadata, so a misbehaving
  inner `Client` returning `(nil, nil)` crashed the decorator (the `Session` already guards
  this). It now logs and passes the nil response through for the caller to handle.

## [0.4.4] - 2026-06-17

Hardening pass 3 — non-chat capabilities (`capabilities.go`).

### Fixed

- **A truncated capabilities response no longer passes as success** — `doRaw` swallowed the
  HTTP body-read error, so a connection drop mid-read (or a `Content-Length` mismatch) returned
  a truncated body as complete. For `SynthesizeSpeech` (raw audio, with no later JSON decode to
  catch it) this was silent data corruption; the read error is now surfaced.

## [0.4.3] - 2026-06-17

Hardening pass 2 — OAuth 2.0 device flow (`oauth.go`). Each fix ships a regression test that
fails without it; `make ci` clean.

### Fixed

- **`PollDeviceToken` enforces `expires_in`** — it now stops polling once the device code's
  lifetime elapses (as its doc already promised), instead of relying solely on the server
  returning `expired_token` plus the caller's context. A server stuck on
  `authorization_pending` with no context deadline could otherwise poll forever (a hang).
- **Poll interval is clamped overflow-safely** — a hostile/huge server-supplied `interval`
  overflowed `time.Duration(interval) * time.Second` and could wrap to a tiny (or negative)
  delay, defeating the minimum; it is now clamped to `[1s, 300s]` (and `expires_in` is capped
  before the deadline math).
- **OAuth `error_description` is surfaced** — device-authorization and token errors now
  include the RFC 8628 human-readable `error_description`, not just the bare `error` code.

## [0.4.2] - 2026-06-17

Two more break-the-system campaigns (resilience/streaming/persistence, then
handoff/serialization/request-building) plus an adversarial hardening pass over the v0.4.x
surface found and fixed the issues below; each ships a regression test that fails without the
fix. `make ci` is clean (race + gosec + govulncheck).

### Security

- **API keys no longer leak into logs or serialized state** — a provider error body that
  echoes the request key (e.g. a 401 "invalid key sk-… rejected") was stored raw in
  `AuthProfile.LastError` and logged by the pool, and `AuthProfile.MarshalJSON` redacted
  `APIKey` but not `LastError`. The library now scrubs its own key — which it knows, so the
  redaction is exact rather than heuristic — from error text before storing, logging, or
  marshaling it. (The error returned to the caller is left intact: the caller already holds
  the key, and scrubbing would lose debugging detail.)
- **Credentials embedded in a `BaseURL` are redacted on marshal** — `Config.MarshalJSON` and
  `AuthProfile.MarshalJSON` redacted `APIKey` but serialized `BaseURL` verbatim, leaking a
  credential carried in the URL (`https://user:pass@host`, or a `?key=`/`token=` query param).
  Both now scrub URL userinfo and secret query params while keeping host/path visible.

### Fixed

- **`Conversation.Clone` no longer crashes on a cyclic `ToolInput`** — a circular `map`/slice
  made `json.Marshal` fail and the structural-clone fallback recurse forever into a
  stack-overflow (an unrecoverable crash). The deep-copy is now depth-bounded.
- **Nil tool schema/input is sent as an object, not `null`** — a no-arg tool
  (`InputSchema: nil`) or a replayed `tool_use` (`ToolInput: nil`) serialized to
  `"input_schema":null` / `"input":null`, which providers (Anthropic) reject; all four
  adapters now coerce nil to `{"type":"object"}` / `{}`.
- **`EstimateCost` resolves model aliases** — it looked up the registry without resolving an
  alias first, so an alias model name returned a silent `$0` cost.
- **Streamed `openai_responses` tool calls / `Gemini` thought signatures handled on handoff**
  — a Gemini `ThoughtSignature` on a `tool_use` no longer crosses to a foreign provider (same
  rule the thinking handler already applied to `ThinkingSignature`), and a thinking block that
  carries a signature but empty text is no longer replayed verbatim (a provider rejects it).
- **A negative cooldown no longer silently disables resilience** — a negative
  `CircuitCooldown` / key cooldown (e.g. a computed duration gone negative) made
  `time.Since(...) >= cooldown` always true, so an open circuit never blocked and a failed key
  was never actually cooled; negative values are now clamped to the default.
- **A `Client` returning an empty `Response.Model` no longer zeroes the cost** — the session
  attributes the turn to the model it actually ran against, so `EstimateCost` is no longer
  called with `""` (which returns `$0`).
- **An empty SSE `data:` line no longer aborts a stream** — `SSEData` returned an empty
  payload as a parseable event, so a keep-alive/padding `data: ` line mid-stream surfaced as
  a spurious malformed-event error that dropped an otherwise-complete turn. Empty data values
  are now skipped, matching `SSEData`'s own documented contract.
- **Circuit-breaker probe token can never collide with a reserved value** — the probe
  generation now skips `0` ("no probe") and the `anyProbe` wildcard when it advances, so a
  real probe token stays a distinct identity even across a (practically impossible) 2^64 wrap
  that would otherwise let a wrapped probe's recorder act as the wildcard.
- **`RegisterModel` rejects a whitespace-only `ID` or `Provider`** — previously only the empty
  string was rejected, so a blank-after-trim value registered under a meaningless key.
- **`FileStore` rejects `.` as a conversation id** — the reserved current-directory segment
  slipped through id validation (`..` was already rejected) and wrote a `..json` file.

## [0.4.1] - 2026-06-17

Patch release. A review pass over the v0.4.0 hardening — and an adversarial self-review of
that pass — found and fixed a crash, several streaming/registry correctness gaps, and a
circuit-breaker probe-identity concurrency flaw. Each fix ships with a regression test that
fails without it; `make ci` is clean (race + gosec + govulncheck).

### Fixed

- **`Close()` racing key rotation no longer panics or leaks a client** — `rotateClient`
  swapped `pc.rc` under the pool mutex but never checked whether `Close()` had concurrently
  niled it; the window between creating the new client and re-taking the lock let `Close()`
  interleave, after which `old.Release()` dereferenced a nil `*refCountedClient` (panic) and
  the just-created client was resurrected into `pc.rc` and never closed (leak). `rotateClient`
  now bails out as `ErrClientClosed` (closing the new client) when the pool was closed, and
  `refCountedClient.Acquire`/`Release` are nil-safe. Close-during-rotation is inside the
  intended concurrency envelope, so this was reachable in normal teardown.
- **Streamed `openai_responses` tool calls report `tool_use`** — the streaming
  `response.completed` handler hardcoded `end_turn`, so a streamed turn that emitted function
  calls disagreed with the non-streaming path (which reports `tool_use`). An agent loop keying
  on `resp.StopReason == StopToolUse` stopped after a streamed tool call despite a populated
  `ToolCalls`. The stream now reports `tool_use` when it emitted a function call.
- **`openai_compat` trailing noise no longer masks a completed turn** — unlike the other
  adapters it cannot return on the terminal event (usage arrives in a later chunk), so a
  malformed SSE line after `finish_reason` but before/without `[DONE]` (common from
  Ollama/vLLM/LM Studio) surfaced as a spurious error and made a `Session` drop the turn. A
  malformed line after `finish_reason` is now ignored as trailing noise.
- **Anthropic flushes the *terminal* tool call on a missing `content_block_stop`** — the
  v0.4.0 flush-on-new-block fix did not cover the final tool block: a dropped
  `content_block_stop` immediately followed by `message_delta` silently lost the accumulated
  call while still reporting a clean `tool_use` turn. `message_delta` now flushes a pending
  tool call before emitting `EventDone`.
- **`StreamWithBoundaries` honors a consumer that stops on the error path** — on an
  underlying-stream error it closed the open block and then called `yield` again
  unconditionally; if the consumer stopped during the injected block-`End`, the extra `yield`
  violated the `iter.Seq2` contract and could panic composed iterators. It now returns when
  `closeBlock()` reports the consumer stopped.
- **`RegisterModel` overrides reach discovery** — the append to `availableModels` was gated on
  the model being new, so correcting a built-in already in the registry but absent from the
  curated list (20 such built-ins) never surfaced it in `GetAvailableModels`, and an override
  that changed a model's provider left a stale entry. The discovery list is now kept in sync
  on every override (add to the new provider, remove from the old, never duplicate).
- **Caller cancellation / client-side errors can't wedge a half-open circuit** — a half-open
  probe that returned via caller cancellation or a non-retryable client error (e.g. 400)
  recorded neither success nor failure, stranding other traffic until a second cooldown
  elapsed. The breaker now re-arms the probe (`CircuitBreaker.ReleaseProbe(token)`) on those
  paths so a fresh probe is admitted immediately.
- **The half-open probe is now strictly single-owner** — the breaker's admission path hands
  back a probe token, and probe re-arm *and* outcome recording are scoped to it: `ReleaseProbe`,
  and (internally) the pool's success/failure recording, act only on the request holding the
  current probe. A stale or foreign outcome — from a request admitted while the circuit was
  closed (token 0), or whose probe a newer one already superseded — can no longer re-arm,
  close, or re-open an unrelated goroutine's in-flight probe (which could otherwise admit a
  second concurrent probe or flip the circuit state). The public `RecordSuccess`/`RecordFailure`
  keep their original state-based behavior for direct callers (e.g. `Execute`).

### Changed

- **`RegisterModelAlias` rejects an alias key that equals a real model ID** — such an alias
  would silently shadow the model (aliases resolve first), redirecting every request for it.
  It is now rejected with an error.
- **`FileStore.Save` is durable, not just reader-atomic** — it now `fsync`s the temp file
  before the rename and best-effort `fsync`s the directory after, so the documented
  crash-safety holds across a power loss, not only for concurrent readers.
- **Removed dead circuit-breaker configuration** — `NewPooledClient` wired a
  `WithSuccessPredicate` that the pool's manual `Allow`/`RecordSuccess`/`RecordFailure` drive
  never consults (the auth-error exemption is applied inline); the misleading dead config is
  gone.

### Documentation

- **Backfilled two v0.4.0 changelog omissions** (the `[0.4.0]` section itself is left frozen
  to match its tag): **R-M3** — `IsRetryable` classifies a `*tls.CertificateVerificationError`
  as non-retryable (an endpoint-identity/configuration problem no retry or key rotation can
  fix; it previously surfaced as a `net.Error` and was retried); and **R-L6** — `SSEData(line)`
  is the shared SSE `data:` parser hoisted out of the four adapters. Both shipped in v0.4.0;
  only their changelog entries were missing.

## [0.4.0] - 2026-06-10

This release lands the fixes from the 2026-06-10 architecture review (finding ids
`R-*`). Each fix ships with a break-the-system regression test (red without the fix).

### Fixed

- **Caller cancellation no longer poisons the resilience stack** (R-H1) — a request
  aborted via the caller's `context` surfaces through net/http as a `*url.Error`
  satisfying `net.Error`, which the retry path treated as a transient provider
  failure: it put the (healthy) key in cooldown, recorded a circuit-breaker failure,
  and burned retries against a dead context — so a few user cancellations could open
  the circuit and reject real traffic. `PooledClient.Complete`/`Stream` now detect
  `ctx.Err() != nil` and return immediately without touching key or breaker health,
  and `IsRetryable` classifies `context.Canceled` as non-retryable. The HTTP client's
  own timeout (`context.DeadlineExceeded`) stays retryable.
- **Streams signal completion explicitly — no more silent truncation** (R-H2) — when a
  server closed the connection mid-turn without its protocol-final event, the
  anthropic, gemini, and openai_responses adapters ended the iterator with neither an
  `EventDone` nor an error (the anthropic adapter additionally dropped a
  half-accumulated tool call). All four adapters now yield an explicit error wrapping
  `io.ErrUnexpectedEOF` on a truncated stream, so a partial turn can never be mistaken
  for a complete one (and the pool can classify it as retryable). An explicit `[DONE]`
  without a `finish_reason` (a known local-server quirk) still synthesizes a stop
  reason rather than erroring.
- **Gemini honors the `TokensNotReported` (-1) convention** (R-M8) — a Gemini response
  or stream chunk carrying no `usageMetadata` previously reported `0` tokens,
  indistinguishable from a genuine zero; it now reports the `-1` sentinel like every
  other adapter, and streamed usage arriving on an earlier chunk than the
  `finishReason` is preserved.
- **Anthropic stop reasons are normalized** (R-M4) — the anthropic adapter passed raw
  API values (`stop_sequence`, …) straight through while the other adapters normalized
  to the unified vocabulary; `stop_sequence` now maps to `end_turn` in both `Complete`
  and `Stream`. New exported `Stop*` constants (`StopEndTurn`, `StopToolUse`,
  `StopMaxTokens`, `StopError`, `StopAborted`) document the unified set.
- **`Session.Conversation()` returns a true deep copy** (R-M1) — the snapshot
  previously shared each message's `Content` slice and each tool's `InputSchema` map
  with the live transcript, so mutating the snapshot corrupted the session. It now
  returns an independent deep copy via the new `Conversation.Clone()`.
- **`Session` no longer holds its lock for an entire streaming turn** (R-M2) — snapshot
  reads (`Conversation`/`Usage`/`Provider`) and `SwitchProvider` blocked until a
  (possibly minutes-long) stream finished. Turns are now serialized by one mutex while
  state access uses a separate, briefly-held mutex; a turn is built from a snapshot and
  committed (input + reply together) only on success, so reads never block and an
  in-flight turn is simply invisible until it completes.
- **`FileStore.Save` is atomic** (R-M7) — it wrote in place via `os.WriteFile`, so a
  crash or a concurrent reader could observe a half-written conversation. It now writes
  to a temp file in the same directory and renames.
- **Mid-stream provider failures count against the circuit breaker** (R-L7) — a
  retryable error that arrived *after* the first stream event marked the key failed but
  skipped `breaker.RecordFailure()`, asymmetric with the pre-data path; it now records
  the failure (still exempting caller cancellation).
- **Circuit breaker can't wedge half-open** (R-L2) — if a half-open probe was abandoned
  (iterator dropped, goroutine died) without reporting success or failure, the circuit
  stayed half-open forever, rejecting everything. A fresh probe is now admitted after
  another cooldown elapses.
- **`NewClientWithKeys(cfg, nil)` no longer bypasses the resilience stack** (R-L1) — an
  empty/nil key slice silently returned a bare adapter with no retry/rotation/breaker;
  it now falls back to `cfg.APIKey` and goes through `PooledClient`, matching `NewClient`
  and its own documentation.

The following two were surfaced by an adversarial break-the-system test pass over the
above changes (each ships with a test that fails without the fix):

- **`Conversation.Clone()` deep-copies even on its fallback path** — when a message carries
  a non-JSON-serializable `ToolInput` (e.g. a channel), `Clone`'s JSON round-trip fails and
  it took a shallow fallback that still shared the `Message.Content` backing arrays, nested
  `ToolResultParts`, `Tool.InputSchema` maps, and `*ImageSource`/`*DocumentSource` pointers
  with the original — so mutating the snapshot corrupted the live transcript (defeating the
  R-M1 fix for that case). The fallback now does a structural deep copy; only a truly-opaque
  `ToolInput` value (unreachable by reflection-free copy, and non-persistable anyway) stays
  shared.
- **Adapters stop reading after the terminal stream event** — anthropic/gemini/openai_responses
  kept scanning after emitting `EventDone`, so trailing bytes from a misbehaving server
  (including a malformed SSE line) surfaced as a spurious error *after* a complete turn —
  which made a `Session` drop the otherwise-complete turn. Each adapter now returns right
  after the terminal event, so post-completion noise can't mask a finished turn.

A second adversarial round found and fixed:

- **A misbehaving `Client` can no longer panic a `Session`** — a `Client.Complete` that
  returned `(nil, nil)` (no response, no error) reached `NewAssistantMessage(nil)` and
  panicked, taking down the caller. `Session` now surfaces it as an error and leaves the
  transcript untouched.
- **`ValidateToolCall` is panic-safe on hostile schemas** — an `enum` containing
  uncomparable values (slices/maps) panicked on `==`; it now compares with
  `reflect.DeepEqual`. It also rejects non-finite floats (`+Inf`) for an `integer` field
  (which previously passed, since `math.Trunc(Inf)==Inf`).
- **Anthropic flushes a pending tool call before a new `content_block_start`** — a
  spec-violating stream that omitted `content_block_stop` between two tool blocks silently
  dropped the first call's accumulated input. The adapter now flushes on a new block start
  (matching the OpenAI-compatible adapter), so no tool call is lost.
- **`StreamWithBoundaries` closes an open block before propagating an error** — an error
  mid-thinking/text-block previously left a dangling `…Start` with no matching `…End`,
  breaking block-based UIs. Every `Start` now gets its `End` even on error.

### Added

- **Runtime model-registry extension** (R-M5) — `RegisterModel(ModelInfo)` and
  `RegisterModelAlias(alias, modelID)` add or override model metadata and aliases at
  runtime, so unlisted or newly released models get cost estimation, capability flags,
  and discovery (and stale built-in pricing can be corrected) without a library release.
  The registry maps are now guarded by an `RWMutex` across all readers
  (`GetModelInfo`, `EstimateCost`, `Models`, `ResolveModelAlias`, …), making extension
  safe for concurrent use.
- **`Conversation.Clone()`** — a deep copy via the versioned JSON round-trip (the same
  machinery persistence uses), so no backing arrays, maps, or tool-input values are
  shared with the original.
- **`SSEData(line)`** — shared server-sent-events `data:` line parsing, hoisted out of
  the four adapters that each reimplemented it.

### Changed

- **`make test` (`-short`) actually skips live-API integration tests** (R-M6) — the
  Makefile passed `-short` and the docs promised it skipped integration tests, but no
  test consulted `testing.Short()`, so a developer with provider keys in their
  environment triggered real API calls (and spend). The Anthropic/xAI/Gemini streaming
  tests and the multi-key pool test now skip under `-short` regardless of env keys.
- **Documentation** — README gains a stream-completion guarantee, caller-cancellation
  semantics, the runtime registry-extension API, and a provider-count clarification;
  `docs/ARCHITECTURE.md` reflects the new streaming/cancellation/registry behavior.

## [0.3.3] - 2026-06-08

### Added

- **Model registry entries for the 5 newest providers** — `perplexity` (Sonar family), `fireworks`, `together`, `deepinfra`, and `nvidia` (NIM) now have curated, source-dated entries, so they appear in `Models()`/`ModelsByProvider()` and get cost estimation (NVIDIA's hosted catalog is free dev-tier, so its prices are left 0). Representative subsets — the registry only affects cost estimation + discovery; any model ID can still be passed directly to the provider.

### Fixed

- **GitHub CI pipeline** — the `make ci` security gates (`gosec`/`govulncheck`) had never actually run (they aren't installed locally), so the first GitHub Actions run went red. Fixed: `FileStore` directory perms `0o755` → `0o750` (gosec G301); justified `#nosec` on the validated `FileStore.Load` read (G304) and the cache-API example's keyed URL (G107); and **Go bumped `1.26.0` → `1.26.4`** to clear two stdlib CVEs (`net/textproto` GO-2026-5039, `crypto/x509` GO-2026-5037) reported by `govulncheck`.

### Changed

- **Go 1.26.4 minimum** (`go.mod` + README).
- **Pre-push checklist** (CLAUDE.md) gains a third gate — CI green (`make ci`) — alongside no-credentials and README-current.

## [0.3.2] - 2026-06-08

### Changed

- **Documentation pass for v0.3.1 (all `.md` files)** — corrected stale counts across README, ARCHITECTURE, and CLAUDE.md: 15 → 20 providers and "three" → four wire protocols (the `openai_responses` protocol is now listed in ARCHITECTURE's protocol table), the OpenAI-compatible adapter provider count 13+ → 18+, and `BlockPrivateBaseURL` / `ResponseFormat` are noted in the ARCHITECTURE config reference. No code changes.

## [0.3.1] - 2026-06-08

### Added

- **`MockClient`** (roadmap P6) — an in-memory `Client` for tests and examples (no network). Script it with `PushResponse`/`PushText`/`PushStream`/`PushError` or `SetResponseFunc`, inspect received requests via `Requests()`/`LastRequest()`, and pass it anywhere a `Client` is expected (e.g. `NewSession(mock)`). Scripting is symmetric — a queued `Response` is synthesized into stream events and queued events are assembled into a `Response`, so one script drives `Complete` or `Stream`. Plus `FauxResponse`/`FauxToolCall` builders.
- **Model-discovery API** (P9) — `Models()`, `ModelsByProvider(provider)`, and `Providers()` enumerate the registry programmatically (filter by capability via `ModelInfo` fields).
- **`CompleteSimple` / `StreamSimple`** (P10) — one-shot convenience wrappers taking a prompt string instead of a full `Request` (generation settings come from the client `Config`).
- **`Config.BlockPrivateBaseURL` + `CheckBaseURL`** (F2) — opt-in SSRF hardening: when set, client construction rejects a `BaseURL` whose host is loopback/private/link-local/unspecified (e.g. the `169.254.169.254` metadata IP) or `localhost`. Covers the per-key `"apikey|baseurl"` override too (wired through `newSingleClient`). Off by default, so local providers keep working.
- **Conversation persistence** (P1.3, completes roadmap P1) — a `Store` interface (`Save`/`Load`/`Delete` by id) with two stdlib-only implementations: `MemoryStore` (mutex-guarded, stores serialized bytes so loads are independent copies) and `FileStore` (one JSON file per id under a directory; ids are restricted to a single path segment, so traversal like `../evil` is rejected). `Load` of an unknown id returns `ErrConversationNotFound`.
- **`ValidateToolCall(tool, call)`** (P8) — best-effort structural validation of a tool call's arguments against the tool's `InputSchema` (a JSON-Schema subset: object `required`, per-property `type`, and `enum`). Accepts arguments as a `map[string]any` or a raw JSON string; returns nil when the tool has no schema.
- **`StreamWithBoundaries`** (P5) — wraps any coarse event stream and injects block-boundary events (`EventTextStart`/`End`, `EventThinkingStart`/`End`, `EventToolStart`/`End`) around contiguous runs. Derived uniformly over the neutral stream, so every provider gets identical fine-grained semantics without per-adapter changes.
- **Tool results with image content** (P3) — `ContentPart.ToolResultParts` + `NewToolResultParts(...)` let a tool return text and/or image blocks. Anthropic serializes them natively as a `tool_result` content array; the other providers degrade to the text via `ContentPart.ToolResultText()` (images dropped).
- **Five more OpenAI-compatible providers** (P7) — `together`, `fireworks`, `nvidia` (NIM), `perplexity`, and `deepinfra` presets (Bearer auth, chat-completions). Amazon Bedrock (AWS SigV4), Google Vertex AI (GCP OAuth), Azure OpenAI, and Cloudflare/Vercel gateways are documented as requiring custom auth or account-specific `BaseURL`s — their signing/addressing is outside the stdlib-only preset model.
- **Structured output / JSON mode** — `Request.ResponseFormat` (`ResponseFormatJSONObject` / `ResponseFormatJSONSchema` + optional `Schema`/`Name`) wired to OpenAI-compatible (`response_format`) and Gemini (`responseMimeType`/`responseSchema`). Anthropic has no native JSON mode (use a tool); the Responses adapter is not wired.
- **Embeddings** — `GenerateEmbeddings(ctx, cfg, EmbeddingRequest)` for OpenAI-compatible `/embeddings`.
- **Image generation** (P2) — `GenerateImages(ctx, cfg, ImageRequest)` for OpenAI-compatible `/images/generations` (base64 output). Gemini Imagen (a different API shape) is out of scope here.
- **Audio in / out** — `SynthesizeSpeech` (`/audio/speech` → raw bytes) and `TranscribeAudio` (`/audio/transcriptions`, multipart → text) for OpenAI-compatible providers.
- **OAuth 2.0 device flow** (P4) — `StartDeviceAuth` + `PollDeviceToken` implement the RFC 8628 Device Authorization Grant (stdlib only, honoring `authorization_pending`/`slow_down`); the resulting access token is used as `Config.APIKey`. Interactive, provider-specific `loginX()` flows remain the caller's responsibility (a stdlib library has no browser). The standalone capability functions all reuse `SafeHTTPClient` and honor `BlockPrivateBaseURL`.

### Security

- **Auth headers stripped on TLS-downgrade redirects** (F4) — `SafeHTTPClient` now strips `Authorization`/`x-api-key`/`x-goog-api-key` when a redirect changes scheme **or** host (previously host-only), closing an `https→http` same-host downgrade that would have sent the key in plaintext. `sameHost` → `sameOrigin`.

### Changed

- **`BaseURL` trust boundary documented** (F2) — `Config.BaseURL` godoc and the README now state explicitly that `BaseURL` is a trusted developer-supplied value (the API key is sent to it). The misleading scheme-validation test is corrected to note that non-HTTP schemes are rejected by `net/http`, not by this library.
- **Logging privacy caveat documented** (F5) — `LoggingClient` godoc now notes that logged errors carry the truncated upstream body (`APIError.Message`), which can echo request fragments; the "privacy-safe" guarantee covers message content, not error bodies.
- **Provider adapters de-duplicated** (Q1) — the security-critical HTTP plumbing (bounded error read + `APIError` construction, bounded JSON decode, JSON request building) is centralized in `transport.go` (`ErrorFromResponse`, `DecodeJSONResponse`, `NewJSONRequest`) and shared by all four adapters; wire-format translation stays per-adapter.

### Fixed

- **README dependency claim corrected** (Q2) — the library imports only the Go standard library; `joho/godotenv` is used solely by `examples/`. The previous "zero external dependencies" wording was inaccurate at the module level.

## [0.3.0] - 2026-06-07

### Added

- **Conversations + cross-provider handoff** (roadmap P1) — multi-turn chat with a serializable, provider-neutral transcript and a stateful driver:
  - **`Conversation`** (`conversation.go`) — a plain value holding `SchemaVersion`, `System`, `Tools`, `Messages`, and accumulated `Usage`. Round-trips losslessly through JSON; `LoadConversation([]byte)` validates `schema_version` (tolerates pre-versioned blobs as v1, rejects newer-than-build). The format is keyed by explicit JSON tags, not Go type names, so stored conversations stay portable.
  - **`Session`** (`session.go`) — a concurrency-safe (`sync.Mutex`) driver around a `Conversation`: `Send`/`SendMessages`/`Stream`/`StreamMessages` auto-append each turn with provider/model provenance and fold in usage; `SwitchProvider` performs a **provider handoff**; constructed with `WithSystem`/`WithTools`/`WithBaseRequest`/`WithConversation`. Generation stays stateless — `Client.Complete`/`Stream` are unchanged.
  - **`NormalizeForProvider`** (`normalize.go`) — the single cross-provider translation pass (applied automatically by `Session`, exported for direct `Client` users): degrades non-native thinking blocks to text, repairs orphaned tool calls with a synthetic error result, and drops errored/aborted turns. Never mutates its input.
  - **`Usage`** — accumulates token counts and sums cost **per response** (each response's own model), so totals stay correct across a multi-provider conversation.

- **`ContentThinking` content part + thinking round-trip** — extended-thinking blocks now survive multi-turn conversations. `Message` gained `Provider`/`Model`/`StopReason` provenance and `ContentPart` gained `Thinking`/`ThinkingSignature`/`Redacted` (all `omitempty`; older serialized messages remain valid). The Anthropic adapter captures the thinking signature (non-streaming response + streaming `signature_delta`, surfaced on `StreamEvent.ThinkingSignature`) and replays the thinking block on the next same-provider turn — required for correct Anthropic extended-thinking + tool-use across turns. `NewAssistantMessage` now preserves thinking (it previously dropped `resp.Thinking`).

### Changed

- **`NewAssistantMessage(resp)`** now emits a leading `ContentThinking` part when the response carried thinking (with its signature), and stamps `Model`/`StopReason` provenance onto the message. Existing text + tool-call behavior is unchanged.

### Fixed

Three defects in the new conversation/handoff stack found by adversarial testing (each has a regression test that fails without the fix):

- **Anthropic no longer replays thinking blocks when extended thinking is disabled** for the request — the API rejects thinking blocks unless thinking is enabled, so the adapter now drops historical thinking blocks when `ThinkingLevel` is none (replay requires thinking enabled, e.g. via `WithBaseRequest`).
- **`Usage` no longer accumulates the `TokensNotReported` (-1) sentinel** — a streamed turn whose provider never reported usage previously produced negative token totals (and could yield a negative cost). Token counts are now clamped to `>= 0` before accumulation and cost estimation.
- **openai-compat no longer emits an empty `{role:"assistant","content":null}` message** for an assistant turn that carried only a thinking block (which the protocol drops) — OpenAI rejects a message with null content and no tool calls; such messages are now skipped.

Additional hardening of `NormalizeForProvider` and `Session` from the same testing:

- **Tool-use IDs are deduplicated** on handoff — duplicate IDs (e.g. Gemini's per-turn synthetic `call_0_*` IDs colliding across turns) are rewritten to unique IDs and the paired `tool_result` is remapped, so a handoff to a provider that requires unique IDs (Anthropic) stays valid.
- **Tool results are ordered & pruned** — a `tool_result` is kept only if its `tool_use` already appeared earlier in the transcript; a result before its call, or a dangling result with no matching call, is dropped (every provider rejects those).
- `Session.Send`/`Stream` **roll back the appended input when a turn fails** (so a failed turn never leaves a dangling user message that breaks alternation on retry).
- `Session.Conversation()` returns a **snapshot copy** that is safe to read/marshal concurrently with in-flight turns.

Documented (not a defect): integers inside `ToolInput` (typed `any`) deserialize as `float64` after a JSON reload — standard Go behavior. The re-serialized wire bytes are byte-identical, so a resend is unaffected; only in-memory type assertions see the drift.

## [0.2.10] - 2026-06-07

### Added

- **Continuous integration workflow** (`.github/workflows/ci.yml`) — runs the full `make ci` pipeline (`fmt-check → vet → build → test-race → gosec → govulncheck`) on every push to `main` and every pull request, resolving the Go toolchain from `go.mod`. The `ci`/`security`/`vulncheck` Makefile targets already existed, but nothing executed them automatically — the repo had no CI runner config. *(audit finding F3)*

- **`NewAPIErrorFromStatusWithLimit(provider, status, body, maxMsgLen)`** — like `NewAPIErrorFromStatus` but truncates the stored error message to a caller-supplied limit. Adapters pass `Config.EffectiveMaxErrorMessageLen()` so the per-client limit is honored.

### Fixed

- **Per-`Config` safety limits are now actually honored** — `MaxErrorBodyBytes`, `MaxResponseBodyBytes`, `MaxSSELineBytes`, `MaxToolInputBytes`, and `MaxErrorMessageLen` set on a `Config` were silently ignored: every adapter read the process-global `llm.Max*` vars and `truncateErrorBody` hardcoded the `DefaultMaxErrorMessageLen` constant, so the documented per-client overrides did nothing. All four adapters now consult `Config.EffectiveMax*()` for every limit (error-body reads, response-body decoding, SSE line buffer, tool-input accumulation, and error-message truncation). The package-level `llm.Max*` vars remain as the process-wide fallback for backward compatibility (the `Effective*()` accessors fall back to them when the `Config` field is zero). New `TestPerConfigErrorMessageLimit` proves a per-`Config` limit changes behavior. *(audit finding F1)*

### Changed

- **`.mailmap`** added — unifies two `Rene Heinzl` commit identities (`reneheinzl@macbookpro.lan` → `rene@bds421.com`) for `git shortlog`/blame.

- **Repository reformatted with `gofmt`** — corrected pre-existing struct-alignment drift across several files (`retrypolicy.go`, `types.go`, examples, internal tests) so the new `fmt-check` CI gate passes. Formatting-only; no behavioral change.

## [0.2.9] - 2026-06-03

### Added

- **Claude Opus 4.8 (`claude-opus-4-8`)** — Anthropic's most capable model for complex reasoning and agentic coding. 1M context, 128K max output, extended thinking. $5.00/$25.00 per 1M tokens (cache write $6.25, cache read $0.50). The `opus` alias now resolves to this model.

- **Claude Opus 4.7 (`claude-opus-4-7`)** — Previous Opus generation, still available. 1M context, 128K max output. $5.00/$25.00 per 1M tokens.

- **OpenAI GPT-5.5 (`gpt-5.5`)** — Flagship GPT-5.5 reasoning model (Responses API). 1M context, 128K max output. $5.00/$30.00 per 1M tokens. Alias: `gpt5.5`.

- **OpenAI GPT-5.5 Pro (`gpt-5.5-pro`)** — Highest-effort GPT-5.5 variant (Responses API). 1M context, 128K max output. $30.00/$180.00 per 1M tokens. Alias: `gpt5.5-pro`.

- **Google Gemini 3.5 Flash (`gemini-3.5-flash`)** — GA Flash model that surpasses 3.1 Pro on coding and agentic benchmarks. 1M context, 64K max output, `ThoughtSignature`. $1.50/$9.00 per 1M tokens (cache read $0.15). Alias: `gemini3.5`.

- **xAI Grok 4.3 (`grok-4.3`)** — xAI's cost-efficient flagship with built-in reasoning. 1M context. $1.25/$2.50 per 1M tokens. Aliases: `grok4.3`, `grok-4-3`.

## [0.2.8] - 2026-06-02

### Added

- **Document / PDF content support** — New `ContentDocument` content type with a `DocumentSource{Type, MediaType, Data}` payload, `ValidateDocumentSource()`, and a `NewDocumentMessage()` constructor. Lets callers pass PDFs inline as base64 (supported media type: `application/pdf`).
  - **Gemini** serializes documents as `inlineData` parts (native PDF parsing — text layout preserved).
  - **Anthropic** serializes documents as native `document` content blocks (with `cache_control` support).
  - **OpenAI-compatible** (xAI Grok, etc.) emits an `image_url` data URI so vision models read the PDF, matching the legacy decoder's behavior.
  - **OpenAI Responses** returns an explicit error for document content instead of silently dropping it.

## [0.2.7] - 2026-04-04

### Added

- **DeepSeek provider preset** — New `deepseek` provider pointing to `api.deepseek.com` (OpenAI-compatible). Default model: `deepseek-chat`. Requires `DEEPSEEK_API_KEY`. Aliases: `deepseek-cloud`, `deepseek-v4`.

- **Cohere provider preset** — New `cohere` provider pointing to `api.cohere.ai/compatibility/v1` (OpenAI-compatible). Default model: `command-a-03-2025`. Requires `COHERE_API_KEY`. Aliases: `cohere`, `command-a`.

- **DeepSeek V4 (`deepseek-chat`)** — Flagship DeepSeek model with hybrid reasoning. 1M context, 16K max output, intrinsic thinking. $0.30/$0.50 per 1M tokens.

- **Cohere Command A (`command-a-03-2025`)** — Latest Cohere generation with reasoning support. 256K context, 4K max output. $2.50/$10.00 per 1M tokens.

- **Meta Llama 4 Scout (`meta-llama/llama-4-scout-17b-16e-instruct`)** — 17B active params, 16 experts (MoE). Available on Groq. 327K context, 16K max output. $0.11/$0.34 per 1M tokens. Aliases: `llama4`, `llama-4-scout`.

- **Mistral Devstral Small 2 (`devstral-small-2-2512`)** — Compact 24B coding model. 262K context. $0.10/$0.30 per 1M tokens. Alias: `devstral-small`.

- **Qwen 3.5 Omni variants** — Three multimodal models added to DashScope provider: `qwen3.5-omni-plus`, `qwen3.5-omni-flash`, `qwen3.5-omni-light`. All support 256K context and intrinsic thinking. Pricing in free preview (not yet published). Alias: `qwen-omni` resolves to `qwen3.5-omni-plus`.

## [0.2.5] - 2026-04-03

### Added

- **`Config.MaxRetries`** — Configurable retry cap for `PooledClient`. Replaces the hardcoded `maxRetryAttempts = 10` constant. `DefaultConfig()` sets it to `DefaultMaxRetries` (10). Minimum effective value is 3.

- **`Config.BetaFeatures`** — Configurable provider beta feature flags. For Anthropic, joined with `,` and sent as the `anthropic-beta` header. Default: `["interleaved-thinking-2025-05-14"]`. Other providers ignore this field.

- **`Config.AnthropicVersion`** — Override the Anthropic API version header. Default: `DefaultAnthropicVersion` (`"2023-06-01"`).

- **Configurable safety limits** — `Config.MaxErrorBodyBytes`, `Config.MaxSSELineBytes`, `Config.MaxResponseBodyBytes`, `Config.MaxToolInputBytes`, `Config.MaxErrorMessageLen` — all zero by default (use `Default*` constants). Override per-client for custom deployments.

- **`NewSystemMessage(text)`** — Convenience constructor for system messages, completing the symmetry with `NewTextMessage`, `NewImageMessage`, `NewToolResultMessage`, and `NewAssistantMessage`.

- **`ClampThinkingBudget(provider, model, budget, maxTokens)`** — Centralized thinking budget clamping with `slog.Warn` on clamp. Used by Anthropic and Gemini adapters (replaces duplicated inline clamping).

### Fixed

- **Anthropic sends empty tools array** — The Anthropic adapter sent `"tools": []` when no tools were provided. Now guarded with `len(req.Tools) > 0`, matching Gemini and OpenAI adapters.

- **Gemini output token padding silent** — Gemini 2.5 models silently padded `maxOutputTokens` for native thinking. Now logs `slog.Warn` with original and padded values.

### Changed

- **`interface{}` → `any`** — All production Go files now use the modern `any` alias. Test files unchanged.

- **All hardcoded production constants → configurable** — `maxRetryAttempts`, `maxErrorMessageLen`, `anthropicVersion`, and the `Max*Bytes` safety limits are now configurable via `Config` fields with `Default*` constants. No production-relevant value remains as a hardcoded constant.

## [0.2.4] - 2026-04-03

### Added

- **DashScope/Qwen cloud provider** — New `dashscope` and `qwen` provider presets pointing to `dashscope-intl.aliyuncs.com/compatible-mode/v1` (OpenAI-compatible). Requires `DASHSCOPE_API_KEY`.

- **Qwen 3.6 Plus** — `qwen3.6-plus` in the model registry. 1M context, 65K max output, intrinsic thinking. Default model for the `dashscope`/`qwen` provider.

- **OpenAI o3-pro** — `o3-pro` added to the O-series reasoning models. $20/$80 per 1M tokens, 200K context, 100K max output.

- **Gemma 4 Ollama models** — `gemma4:31b` (dense, max quality), `gemma4:26b` (MoE, 3.8B active), `gemma4:e4b` (edge). All three support native tool/function calling and configurable thinking — a significant upgrade from Gemma 3 which had `NoToolSupport: true`.

- **Qwen 3.5 Ollama models** — `qwen3.5:latest` (9B default), `qwen3.5:27b` (dense). Both support intrinsic thinking.

### Fixed

- **Gemini pricing errors** — Five models had incorrect pricing copied from older Gemini 2.5 entries when they were first added. Corrected to match official `ai.google.dev/gemini-api/docs/pricing`:
  - `gemini-3.1-pro-preview`: $1.25/$10.00 → **$2.00/$12.00**
  - `gemini-3-pro-preview`: $1.25/$10.00 → **$2.00/$12.00**
  - `gemini-3-flash-preview`: $0.15/$0.60 → **$0.50/$3.00**
  - `gemini-2.5-flash`: $0.15/$0.60 → **$0.30/$2.50**
  - `gemini-2.5-flash-lite`: $0.00/$0.00 → **$0.10/$0.40**

### Changed

- **Alias updates** — `gemma` now resolves to `gemma4:e4b` (was `gemma3:12b`). New aliases: `gemma3` → `gemma3:12b`, `gemma4` → `gemma4:e4b`, `qwen-local` → `qwen3:8b`, `qwen-cloud` → `qwen3.6-plus`. The `qwen` default model is now `qwen3.6-plus` (cloud API) instead of `qwen3:8b` (Ollama).

## [0.2.3] - 2026-03-24

### Fixed

- **Gemini `thinkingConfig` at wrong JSON level** — Was a top-level field on the request body; Gemini API expects it nested inside `generationConfig`. Caused `400: Unknown name "thinkingConfig"` on any request with `ThinkingLevel` set.

- **Gemini `thinkingConfig` sent to native thinking models** — Gemini 2.5 models (`Thinking: true`) think natively and reject explicit `thinkingConfig`. Setting `ThinkingLevel` on these models now correctly skips `thinkingConfig` while still padding `MaxOutputTokens`.

### Added

- **Gemini `thinkingLevel` string support** — Standard levels (`low`, `medium`, `high`) now send Gemini 3's native `thinkingLevel` string instead of converting to `thinkingBudget` tokens. Non-standard levels (`minimal`, `xhigh`) and custom budgets fall back to `thinkingBudget`. The two fields are mutually exclusive per Gemini API requirements.

## [0.2.2] - 2026-03-24

### Fixed

- **`ThinkingXHigh` budget exceeded Anthropic max** — Changed from 131,072 to 128,000 tokens. The previous value exceeded Anthropic's `max_tokens` limit for `claude-opus-4-6` (128,000) and would be rejected by the API.

- **Thinking budget not clamped to model limits** — Anthropic and Gemini adapters now clamp `budget_tokens` / `ThinkingBudget` to the model's `MaxTokens` with a `slog.Warn` when clamping occurs. Previously, using `ThinkingXHigh` (128,000) with a model like `claude-opus-4-5` (max 64,000) or any Gemini model (max 65,536) would cause an API rejection.

- **Temperature silently ignored by Responses API and Chat Completions adapters** — Both `openairesponses` and `openaicompat` adapters now emit `slog.Warn` when a caller-provided `Temperature` is dropped for reasoning models, matching the Anthropic adapter's existing warning pattern.

### Added

- **`TestThinkingBudgetTokens` coverage for all levels** — Added missing `ThinkingMinimal` (1,024) and `ThinkingXHigh` (128,000) cases to the table-driven test in `llm_test.go`.

- **Anthropic adapter unit tests** — New `anthropic_internal_test.go` with budget clamping tests (`TestBuildRequestThinkingBudgetClampedToModelMax`, `TestBuildRequestThinkingBudgetNotClampedWhenWithinLimit`).

- **Gemini adapter budget clamping test** — `TestBuildRequestThinkingBudgetClampedToModelMax` in `gemini_internal_test.go`.

- **Responses API temperature warning test** — `TestBuildRequestTemperatureWarning` now verifies both the wire format (Temperature omitted) and the `slog.Warn` output.

## [0.2.1] - 2026-03-18

### Breaking Changes

- **`Config.Temperature` / `Request.Temperature` changed to `*float64`** — `nil` means "omit from wire" (use provider default). Previously `float64` with default `1.0`. Callers must use a pointer: `temp := 0.7; req.Temperature = &temp`. `DefaultConfig()` no longer sets `Temperature: 1.0` — it is `nil` (omitted).

- **`EstimateCost` signature changed** — Now accepts a single `CostInput` struct instead of `(model, inputTokens, outputTokens)`. The new struct includes `ThinkingTokens`, `CacheCreateTokens`, and `CacheReadTokens` for accurate cache-aware and thinking-aware pricing.

- **Unknown providers require `BaseURL`** — `NewClient` with an unrecognized provider name and no `BaseURL` now returns an error instead of silently defaulting to openai_compat with no endpoint. Set `BaseURL` for custom providers.

### Added

- **`CostInput` struct** — Holds all token types for accurate cost estimation: `InputTokens`, `OutputTokens`, `ThinkingTokens`, `CacheCreateTokens`, `CacheReadTokens`.

- **Cache pricing in `ModelInfo`** — New `CacheWritePricePer1M` and `CacheReadPricePer1M` fields. Populated for all Anthropic models (write = 1.25× input, read = 0.1× input per Anthropic docs). `EstimateCost` now includes cache token costs in its calculation.

- **`EstimateCost` thinking token support** — Thinking tokens (Gemini: separate from output) are priced at the output rate. Anthropic bundles thinking in `OutputTokens`, so `ThinkingTokens` is 0 for Anthropic — no double-counting.

### Fixed

- **Context length false positives** — Removed overly broad `"token limit"` and `"too many tokens"` patterns from `isContextLengthMessage()`. These matched rate-limit messages (e.g. "token limit temporarily reached"), causing `IsContextLength()` to misclassify them. Remaining patterns are specific: `"context length"`, `"context_length"`, `"maximum context"`, `"prompt is too long"`, `"input too long"`, `"request too large"`, `"token count exceeds"`.

- **Tool input overflow corrupted stream** — Anthropic and OpenAI-compatible adapters continued parsing after tool input exceeded `MaxToolInputBytes`, producing corrupted tool JSON. Now the stream terminates cleanly with an error on overflow instead of dropping deltas and continuing.

- **Gemini stream tool input unbounded** — Added explicit `MaxToolInputBytes` check in the Gemini streaming adapter for consistency with Anthropic and OpenAI-compatible adapters.

- **Circuit breaker callback panic crashed caller** — A panicking `OnStateChange` callback propagated to the caller. Now recovered with `slog.Error` logging.

- **Anthropic thinking temperature override logged at wrong level** — Changed from `slog.Debug` to `slog.Warn` when overriding user-requested temperature to 1.0 for extended thinking. A silent override of a user-specified parameter should be visible without enabling debug logging.

## [0.1.22] - 2026-03-18

### Added

- **`Response.ThinkingTokens` / `StreamEvent.ThinkingTokens`** — Gemini models that think by default (gemini-2.5-*) report `thoughtsTokenCount` separately from `candidatesTokenCount`. This was previously parsed but discarded. Now exposed so callers can see the full token breakdown. Anthropic and OpenAI-compat bundle thinking tokens into `OutputTokens`; for those providers `ThinkingTokens` remains 0.
- **Logging middleware** now includes `tokens_thinking` when > 0, matching the pattern used for cache tokens.

## [0.1.21] - 2026-03-18

### Added

- **Image/vision support for all 3 adapters** — `ContentImage` parts are now fully serialized to the correct wire format for Anthropic (inline `image` blocks with `source`), Gemini (`inlineData` parts), and OpenAI-compatible providers (content array with `image_url` data URIs). Previously all three adapters rejected image content with an error.
- **`ValidateImageSource()`** — exported validation function checks for nil source, empty data, unsupported media types (`image/jpeg`, `image/png`, `image/gif`, `image/webp` allowed), and unsupported source types (only `base64`). Used by all adapters before serialization.
- **`NewImageMessage()`** — convenience constructor parallel to `NewTextMessage`, creates a single-image message from role, media type, and base64 data.
- **Anthropic `cache_control` on image blocks** — image content blocks respect `CacheControl: true` on `ContentPart`, adding `cache_control: {type: "ephemeral"}` to the wire format.

## [0.1.20] - 2026-03-18

### Fixed

- **Ollama reasoning field not parsed** — Ollama (Qwen3) uses `"reasoning"` instead of `"reasoning_content"` for thinking output. The OpenAI-compatible adapter now parses both field names in complete and streaming responses.
- **Gemini 2.5 thinking tokens starve maxOutputTokens** — Gemini 2.5 models think by default but do not support `thinkingConfig` — the API rejects it. Thinking tokens silently consume the `maxOutputTokens` budget, causing premature `MAX_TOKENS` truncation. The adapter now pads `maxOutputTokens` with a thinking overhead (4096 tokens, capped at model max) for models with intrinsic thinking.

### Changed

- **Gemini 2.5 models marked `Thinking: true`** — `gemini-2.5-pro`, `gemini-2.5-flash`, and `gemini-2.5-flash-lite` now have the `Thinking` flag set in the model registry, reflecting their intrinsic chain-of-thought behavior.

## [0.1.19] - 2026-03-18

### Fixed

- **Gemini thinking content was lost or mixed into Content** — Gemini models that think by default (e.g. `gemini-2.5-flash`) return `thought: true` parts alongside content parts. The adapter now routes thought parts to `resp.Thinking` (non-streaming) and emits `EventThinking` events (streaming) instead of mixing them into `Content`. Also parses `thoughtsTokenCount` from usage metadata.
- **OpenAI-compat reasoning models returned empty Content** — Models like Qwen3 (`qwen3:4b`) return `reasoning_content` alongside `content` (which may be null). The adapter now parses `reasoning_content` into `resp.Thinking` (non-streaming) and emits `EventThinking` events (streaming). Null `content` no longer causes empty responses.

### Changed

- **Documentation** — README and ARCHITECTURE.md updated to reflect that all three adapters now parse thinking content from responses (Anthropic via `thinking` blocks, Gemini via `thought: true` parts, OpenAI-compat via `reasoning_content`).

## [0.1.18] - 2026-03-18

### Fixed

- **Mistral/Groq/Ollama reasoning models sent wrong max tokens parameter** — The OpenAI-compatible adapter used `max_completion_tokens` (and omitted temperature) for all reasoning models (`Thinking: true`), but this is an OpenAI/xAI-specific API quirk. Mistral (`mistral-small-2603`, `magistral-medium-2509`, `magistral-small-2509`), Groq (`deepseek-r1-distill-*`), and Ollama (`qwen3:*`, `deepseek-r1:*`) reasoning models use standard `max_tokens` and accept temperature. Now scoped to `info.Provider == "openai" || info.Provider == "xai"` only.

## [0.1.17] - 2026-03-18

### Added

- **Acknowledgements section in README** — Reference to the [pi](https://github.com/badlogic/pi-mono/tree/main/packages/ai) library and explanation that the name rho is a tribute (next Greek letter after pi).

## [0.1.16] - 2026-03-18

### Added

- **OpenAI: GPT-5.4 Mini and Nano** — `gpt-5.4-mini` ($0.75/$4.50, 400K context, reasoning) and `gpt-5.4-nano` ($0.20/$1.25, 400K context, reasoning). Released 2026-03-17.
- **OpenAI: GPT-5 Pro, GPT-5.2 Pro, GPT-5 Codex** — `gpt-5-pro` ($15/$120, 272K max output), `gpt-5.2-pro` ($21/$168), `gpt-5-codex` ($1.25/$10).
- **xAI: Grok 4.20 dated variants** — `grok-4.20-beta-0309-reasoning`, `grok-4.20-beta-0309-non-reasoning`, `grok-4.20-multi-agent-beta-0309` (all $2/$6, 2M context).
- **Mistral: Mistral Small 4** — `mistral-small-2603` ($0.15/$0.60, 256K context, toggleable reasoning). Now the default Mistral model.
- **Mistral: Ministral 14B** — `ministral-14b-2512` ($0.20/$0.20, 256K context).
- **OpenAI aliases** — `gpt5.4-mini`, `gpt5.4-nano`.

### Fixed

- **Anthropic pricing corrections** — Opus 4.6 and 4.5 were listed at $15/$75 (the Opus 4.0/4.1 price); corrected to $5/$25. Haiku 4.5 corrected from $0.80/$4 to $1/$5.
- **Anthropic context windows** — Opus 4.6 and Sonnet 4.6 now show 1M context (were 200K). These are the first Anthropic models with native 1M context.
- **Anthropic max output** — Opus 4.5 corrected to 64K (was 128K), Opus 4.1/4.0 corrected to 32K (were 128K), Haiku 4.5 corrected to 64K (was 8192).
- **Haiku 4.5 extended thinking** — Now correctly marked `SupportsThinking: true` per official docs.
- **xAI pricing overhaul** — All Grok `-fast-` models corrected from $3/$15 to $0.20/$0.50. `grok-4.20-beta` corrected from $3.50/$18 to $2/$6. `grok-code-fast-1` corrected from $3/$15 to $0.20/$1.50.
- **xAI context windows** — All `-fast-` and 4.20 models corrected to 2M (were 131K). `grok-4-0709` and `grok-code-fast-1` corrected to 256K.
- **xAI reasoning flags** — Added `Thinking: true` to `grok-4-0709`, `grok-code-fast-1`, and `grok-3-mini`.
- **OpenAI `gpt-5.4-pro` pricing** — Corrected from $5/$20 to $30/$180.
- **OpenAI `gpt-5.3-codex` pricing** — Corrected from $1.50/$12 to $1.75/$14.
- **OpenAI `gpt-5.3-instant` renamed** — Model ID did not exist in OpenAI's API. Renamed to `gpt-5.3-chat-latest` with corrected pricing ($1.75/$14).
- **Gemini `gemini-3.1-flash-lite-preview`** — Pricing corrected from $0/$0 to $0.25/$1.50. Added `ThoughtSignature: true` (required for all Gemini 3.x models).
- **Mistral `mistral-large-2512` pricing** — Corrected from $2/$6 to $0.50/$1.50. Context window corrected to 256K.
- **Mistral `magistral-medium-2509` pricing** — Corrected from $0.40/$2 to $2/$5. Context corrected from 40K to 128K.
- **Mistral `magistral-small-2509` pricing** — Corrected from $0.10/$0.30 to $0.50/$1.50. Context corrected from 32K to 128K.
- **Mistral broken model IDs** — `devstral-small-2-25-12` (invalid) replaced with `devstral-2512` ($0.40/$2, 256K). `ministral-3-8b-25-12` (invalid) replaced with `ministral-8b-2512` ($0.15/$0.15, 256K).
- **Mistral context windows** — All models updated to match official docs (most were undersized).

### Changed

- **Default Mistral model** — Changed from `mistral-small-2506` to `mistral-small-2603` (Mistral Small 4).
- **`mistral-small` alias** — Now resolves to `mistral-small-2603` (was `mistral-small-2506`).
- **`devstral` alias** — Now resolves to `devstral-2512` (was invalid `devstral-small-2-25-12`).
- **`ministral` alias** — Now resolves to `ministral-8b-2512` (was invalid `ministral-3-8b-25-12`).

## [0.1.15] - 2026-03-18

### Added

- **New models** — Grok 4.20, Gemini 3.1 Flash Lite, GPT 5.4, GPT 5.3, new LLM examples (cached interactions for Gemini/Anthropic, Go code review system prompt).
- **OpenAI reasoning model parameters** — Fixed parameter handling for reasoning models.

### Changed

- **Module path migration** — Migrated from `gitlab2024.bds421-cloud.com/bds421/rho/llm` to `github.com/bds421/rho-llm`.

## [0.1.14] - 2026-03-06

### Added

- **Context caching support** — Anthropic inline caching (`cache_control`) and Gemini cached content references. New fields:
  - `ContentPart.CacheControl` — mark individual content blocks as cacheable (Anthropic)
  - `Request.SystemCacheControl` — cache the system prompt (Anthropic, sends structured content blocks with `cache_control: {type: "ephemeral"}`)
  - `Request.CachedContent` — reference a pre-created Gemini cache by name
  - `Tool.CacheControl` — mark tool definitions as cacheable (Anthropic)
  - `Response.CacheCreationTokens` / `Response.CacheReadTokens` — cache token usage from Anthropic
  - `StreamEvent.CacheCreationTokens` / `StreamEvent.CacheReadTokens` — cache token usage in streaming (Anthropic, Gemini)
  - `LoggingClient` logs `cache_write` and `cache_read` when > 0
  - OpenAI-compatible adapters silently ignore all cache fields (no API support)

### Fixed

- **OpenAI-compat stream: spurious error after successful completion** — The OpenAI-compatible adapter reported scanner errors even after the stream completed successfully (`finishReason != ""`). Unlike the Anthropic and Gemini adapters which had a `doneEmitted` guard, the OpenAI adapter would yield a trailing scanner error to the caller, causing the pool to spuriously mark profiles as failed. Now guarded by `finishReason == ""`, matching the other two adapters.

- **Circuit breaker `onStateChange` callback deadlock** — The `WithOnStateChange` callback was invoked while holding `cb.mu.Lock()`. Any callback that called back into the circuit breaker (e.g., `cb.State()`) would deadlock. Now the callback is invoked after releasing the lock via a `setStateLocked`/`fireCallback` split.

- **Circuit breaker + retry loop burned dead iterations** — When the circuit breaker was open, the retry loop would back off and `continue` for each remaining iteration, sleeping 1-8s per attempt without making any HTTP call. After a circuit trip, every request burned through all retry iterations doing nothing. Now `Complete` and `Stream` return `ErrCircuitOpen` immediately when the circuit is open, letting the circuit's own cooldown timer control recovery.

- **GPT-4.1 ContextWindow off by 1000** — `gpt-4.1`, `gpt-4.1-mini`, and `gpt-4.1-nano` had `ContextWindow: 1047576` instead of the correct `1048576` (1024 * 1024).

- **Double logging in pooled clients** — When `LogRequests: true` was set, each rotated inner client got its own `LoggingClient` wrapper, and the pool-level wrapper doubled the output. Now `LogRequests` is cleared on inner client configs and applied once at the pool level.

### Changed

- **`RetryPolicy.MaxRetries` removed** — This field was never read by any code path. The pool's retry count is determined by `HealthyCount()` clamped to `[3, 10]`. Removing prevents users from setting a value that silently has no effect.

- **`RetryPolicy` struct** — `BaseDelay`, `MaxDelay`, `Factor`, `Jitter` remain. `MaxRetries` is removed (see above).

- **Configurable retry policy** description in v0.1.13 changelog updated — `RetryPolicy` struct fields are `BaseDelay`, `MaxDelay`, `Factor`, `Jitter`.

## [0.1.13] - 2026-02-26

### Added

- **Circuit breaker** — 3-state machine (`closed → open → half-open`) that opens after a configurable threshold of consecutive failures, prevents request storms into degraded endpoints, and auto-recovers via probe requests after cooldown. Enabled by default (`CircuitThreshold: 5`, `CircuitCooldown: 30s`). Auth errors (401/403) do not trip the circuit — bad key ≠ broken endpoint. All methods are nil-safe and thread-safe. New file: `circuitbreaker.go`.

- **Configurable retry policy** — `RetryPolicy` struct with `BaseDelay`, `MaxDelay`, `Factor`, `Jitter`. `Config.RetryPolicy` overrides the default (1s base, 30s max, 2x factor, ±25% jitter). The `Backoff()` function remains backward-compatible. New file: `retrypolicy.go` (replaces `backoff.go`).

- **Configurable cooldowns** — `Config.CooldownRateLimit` (default 60s), `Config.CooldownOverload` (default 30s), `Config.CooldownDefault` (default 10s) replace hardcoded magic numbers. Exported constants: `DefaultCooldownRateLimit`, `DefaultCooldownOverload`, `DefaultCooldownDefault`, `DefaultCircuitThreshold`, `DefaultCircuitCooldown`.

- **Retry observability hook** — `Config.RetryHook` receives `RetryEvent` structs with typed events (`RetryAttemptFailed`, `RetryRotating`, `RetryBackingOff`, `RetryCircuitOpen`, `RetryExhausted`) for metrics, alerting, or debugging.

- **`MarkFailedByNameWithCooldown()`** — New method on `AuthPool` accepting explicit cooldown durations. Original `MarkFailedByName()` delegates to it with defaults.

### Changed

- **`DefaultConfig()` enables circuit breaker** — `CircuitThreshold: 5`, `CircuitCooldown: 30s`. All clients through `DefaultConfig()` or `NewClient()` benefit automatically.

- **`NewPooledClient` wires resilience from config** — Circuit breaker and retry hook are initialized from `Config` fields directly in `NewPooledClient`, not just in the factory path. Direct callers of `NewPooledClient` get the same behavior as `NewClient`.

### Removed

- **`backoff.go`** — Replaced by `retrypolicy.go`. The exported `Backoff()` function is preserved as a backward-compatible wrapper around `RetryPolicy.Delay()`.

## [0.1.12] - 2026-02-26

### Fixed

- **Gemini API 400 on empty text parts** — `ContentText` parts with empty strings (`""`) produced `geminiPart{Text: ""}` which serialized to `{}` due to the `omitempty` JSON tag. The Gemini API requires each part to have exactly one `data` oneof field set; an empty object violates this, returning: `"required oneof field 'data' must have one initialized field"`. Now empty text parts are skipped in both message content and system instructions. The existing `len(content.Parts) > 0` guard then drops messages that become entirely empty after filtering.

- **Anthropic and OpenAI adapters also sent empty text parts** — Same root cause across all three adapters. Anthropic sent `{"type":"text","text":""}` content blocks; OpenAI-compatible joined empty strings with `\n`, producing whitespace-only messages. Now all three adapters skip `ContentText` parts where `Text == ""` in every code path (system messages, user messages, assistant messages, tool-result-adjacent text).

## [0.1.11] - 2026-02-25

### Security

- **TLS 1.2+ minimum enforced** — `SafeHTTPClient` now sets `TLSClientConfig: &tls.Config{MinVersion: tls.VersionTLS12}` on the HTTP transport. Defense-in-depth against TLS downgrade attacks, explicit rather than relying on Go runtime defaults.

- **`gosec` static analysis added to CI** — New `security` Makefile target runs `gosec -exclude=G117,G704 ./...`. G117 (secret field pattern) and G704 (SSRF taint) are excluded with documented justifications — both are inherent to an HTTP client library with `MarshalJSON` redaction. G404 (weak RNG) suppressed via `#nosec` on backoff jitter where crypto randomness is unnecessary.

- **`govulncheck` added to CI** — New `vulncheck` Makefile target runs `govulncheck ./...`. Both `security` and `vulncheck` targets are now part of `make ci`.

- **Config JSON redaction tests** — `TestConfigMarshalJSONRedactsAPIKey` and `TestAuthProfileMarshalJSONRedactsAPIKey` verify that `MarshalJSON` replaces API keys with "REDACTED" and that raw keys never appear in serialized output.

- **BaseURL scheme validation test** — `TestBaseURLSchemeValidation` verifies that `http://` and `https://` schemes work while `file://`, `javascript:`, `ftp://`, and `data:` schemes produce clear errors. Defense-in-depth against SSRF-adjacent misuse.

### Changed

- **Go 1.26 minimum** — `go.mod` upgraded from `go 1.23.4` to `go 1.26.0`. Resolves 15 known stdlib CVEs in crypto/tls, crypto/x509, net/http, and related packages that affected Go 1.23.x.

- **`DefaultTimeout` constant exported** — `llm.DefaultTimeout` (120s) is available for callers who want to reference the library's default HTTP timeout.

### Added

- **`ThoughtSignature` registry test** — `TestThoughtSignatureFlags` validates that `ThoughtSignature` is true only for Gemini 3.x models, false for older Gemini models, and false for all non-Gemini models. Prevents silent model registry corruption from reaching production.

### Fixed

- **`ThinkingLevel` silently ignored for OpenAI-compatible providers** — Setting `ThinkingLevel` on an OpenAI-compat provider (xAI, Groq, Cerebras, Mistral, OpenRouter, Ollama, vLLM, LM Studio) was silently dropped with no feedback. Now returns an explicit error: `"<provider> adapter does not support ThinkingLevel"`. ThinkingLevel is only supported by the `anthropic` and `gemini` adapters.

- **`PooledClient.Provider()` and `Model()` bypassed ref-counting** — Both methods accessed the underlying client directly without calling `Acquire()`/`Release()`, creating a race with `rotateClient()` which could close the client mid-call. Now uses the same Acquire/Release pattern as `Complete()` and `Stream()`.

- **`Config.Timeout` zero-value created unbounded HTTP client** — `NewClient(Config{...})` without an explicit `Timeout` created an HTTP client with no timeout, risking goroutine hangs on unresponsive endpoints. Now defaults to `DefaultTimeout` (120s) when `Timeout <= 0`.

- **Negative `MaxTokens` and `Temperature` accepted silently** — `newSingleClient()` passed negative values through to provider APIs, which returned opaque errors. Now validates early: `MaxTokens` and `Temperature` must be >= 0.

- **`maxRetries` unbounded for large auth pools** — `maxRetries = max(pool.Count(), 3)` scaled linearly with pool size. A 50-key pool allowed 50 retry attempts. Now capped at `maxRetryAttempts` (10) in both `Complete()` and `Stream()`.

- **`IsRetryable` false positives on `"request failed"` prefix** — The string fallback matched any error containing `"request failed"`, including non-retryable errors like `"request failed: 400 bad request"`. Removed. Added `"broken pipe"` to match connection resets by peer. In production, wrapped `net.Error` types trigger the type-based check path instead.

- **Anthropic stream scanner error inconsistency** — If the Anthropic SSE stream completed successfully (`message_delta` emitted) but the underlying reader returned a trailing error (e.g., connection close after EOF), the error was propagated to the caller despite the stream being complete. Now suppresses trailing scanner errors after `EventDone`, matching the Gemini adapter's behavior.

- **`NewAssistantMessage(nil)` panicked with opaque nil dereference** — Now panics with an explicit message: `"llm.NewAssistantMessage: resp must not be nil"`.

- **`ThinkingBudgetTokens` returned 4096 for `ThinkingNone`** — The default case returned 4096 instead of 0 when called with `ThinkingNone`. If ever called by mistake with the none level, it would silently produce a 4096-token thinking budget. Now returns 0.

- **"No healthy profiles" error inconsistent with `ErrNoAvailableProfiles`** — When all auth profiles were permanently disabled (e.g., by 401 errors), `AuthPool.GetAvailable()` returned a bare `fmt.Errorf` that did not wrap `ErrNoAvailableProfiles`. Callers using `errors.Is(err, ErrNoAvailableProfiles)` would not match. Now wraps `ErrNoAvailableProfiles` via `%w`.

- **Local providers had no retry/backoff** — `NewClient()` bypassed `PooledClient` for keyless providers (Ollama, vLLM, LM Studio), giving them zero transient-error resilience. A single 502 from a local model server caused immediate failure. Now `NewClient()` always routes through `NewClientWithKeys()`, wrapping all providers — including keyless ones — in `PooledClient` for automatic retry with exponential backoff.

- **Adapter `Close()` leaked HTTP transport connections** — All three protocol adapters (`anthropic`, `gemini`, `openaicompat`) had no-op `Close()` methods that did not drain idle HTTP connections. During auth pool rotation, replaced clients left orphaned transport connections in the pool. Now `Close()` calls `httpClient.CloseIdleConnections()` to release them.

- **`make cover` under-reported provider sub-package coverage** — The `cover` Makefile target ran `go test -coverprofile=coverage.out ./...` which only reports coverage for code within the same package as the test file. Tests in `llm_test.go` and `security_test.go` that exercise provider adapter code (via `httptest` servers) showed 0% for `provider/anthropic`, `provider/gemini`, and `provider/openaicompat`. Added `-coverpkg=./...` to enable cross-package coverage tracking.

- **`fmt.Errorf` non-constant format string** — `llm_test.go:1840` used `fmt.Errorf(tc.errMsg)` which Go 1.26's stricter vet rejects. Changed to `fmt.Errorf("%s", tc.errMsg)`.

## [0.1.10] - 2026-02-23

### Added

- **`TokensNotReported` constant** — Named sentinel (`-1`) for token counts when the provider did not report usage. Replaces magic `-1` literals in Anthropic and OpenAI-compatible stream parsers. Callers can compare `event.InputTokens == llm.TokensNotReported` instead of checking against a raw `-1`.

- **`Request.ThinkingBudget` field** — Custom token budget that overrides `ThinkingLevel` defaults when > 0. Allows per-request control over thinking budget without changing `ThinkingLevel` presets (e.g., `ThinkingBudget: 8192` with `ThinkingLevel: ThinkingMedium`).

- **`ErrClientClosed` sentinel error** — Returned by `PooledClient.Complete()` and `Stream()` when called after `Close()`. Replaces nil-pointer panic.

- **Type-based network error detection in `IsRetryable`** — Now checks `net.Error`, `io.EOF`, `io.ErrUnexpectedEOF`, `syscall.ECONNRESET`, and `syscall.ECONNREFUSED` via `errors.As`/`errors.Is` before falling back to string matching. Strictly additive — all existing string-based detection preserved for backward compatibility.

- **3 new context-length error patterns** — `isContextLengthMessage` now matches `context_length` (underscore variant, Groq/OpenAI error codes), `input too long` (Gemini), and `request too large` (generic).

### Changed

- **Temperature override logging** — Anthropic adapter now emits a `slog.Debug` log when extended thinking forces `temperature = 1.0`, including the originally requested temperature. Previously silent.

### Fixed

- **BLOCKER: `t.Context()` broke Go 1.23 compilation** — `security_test.go` used `t.Context()` (Go 1.24+) in all 14 test call sites. Replaced with `context.Background()` to restore Go 1.23 compatibility.
- **BLOCKER: `ContentImage` silently dropped** — All three adapters silently ignored `ContentImage` parts in messages, causing data loss. Now returns a clear error: `"image content not yet supported by <provider> adapter"`.
- **Gemini hardcoded default model** — `gemini.New()` silently defaulted to `"gemini-2.0-flash"` when no model was provided, bypassing the factory's `ResolveModelAlias()` and inconsistent with anthropic/openaicompat. Removed; callers must set the model explicitly.
- **Panic on `Complete`/`Stream` after `Close`** — `PooledClient.Complete()` and `Stream()` would panic with nil pointer dereference if called after `Close()`. Now returns `ErrClientClosed`.
- **Missing model validation** — `newSingleClient()` accepted empty model strings, passing them to provider APIs which returned opaque errors. Now validates early: `"model is required"`.
- **OpenAI streaming missed zero-token usage** — Usage detection checked `PromptTokens > 0 || CompletionTokens > 0`, missing legitimate zero-token responses. Changed `Usage` to a pointer type — nil means absent, non-nil trusts the values including zero.
- **`Backoff()` returned zero on `delay=0`** — Guard only checked `delay < 0`; a zero delay (possible with very small base values) passed through. Changed to `delay <= 0`.
- **Documentation: thinking budget values** — README and ARCHITECTURE.md listed stale ThinkingLevel budgets (Low→1024, Medium→4096, High→16384). Corrected to match code: Low→4096, Medium→16384, High→65536.
- **Documentation: Gemini auth method** — ARCHITECTURE.md still described Gemini auth as `?key=` query parameter. Corrected to `x-goog-api-key` header (changed in v0.1.9).
- **Documentation: Ollama `mistral` alias** — README listed `mistral` as an Ollama alias but the actual alias is `mistral-local` (avoids collision with the Mistral provider name).
- **Internal codenames in comments** — Removed `RH/bds421` and `clawdbot/openclaw` references from `types.go` comments.

## [0.1.9] - 2026-02-22

### Security

- **Gemini API key moved from URL to header** — Previously sent as `?key=` query parameter, leaking into server logs, proxy logs, and HTTP referer headers. Now sent via `x-goog-api-key` header, matching other adapters' auth patterns.

- **Bounded error response reads** — All three adapters (`io.ReadAll` on error bodies) now use `io.LimitReader` capped at 1 MB. A malicious or broken endpoint returning a multi-GB error body previously caused OOM.

- **Error messages truncated at 4 KB** — `APIError.Message` is now capped to prevent unbounded strings from propagating through error chains, log systems, and serialization.

- **Redirect auth header stripping** — All HTTP clients now use `CheckRedirect` to strip sensitive headers (`Authorization`, `x-api-key`, `x-goog-api-key`) on cross-domain redirects. Previously, Anthropic's `x-api-key` and OpenAI's `Bearer` token would leak to a redirect target on a different host.

- **Smaller SSE scanner buffer** — Reduced from 1 MB to 256 KB per stream. Limits per-stream memory pressure from malicious endpoints sending oversized SSE lines.

- **Bounded success response body decoding** — `json.NewDecoder` on success response bodies now wraps `resp.Body` with `io.LimitReader` capped at 32 MB. Previously, a malicious endpoint returning a multi-GB JSON response caused OOM via unbounded `json.Decoder` allocation.

- **Bounded tool input accumulation in streams** — `inputBuffer` (Anthropic `input_json_delta`, OpenAI `function.arguments`) now capped at 1 MB. A malicious stream sending thousands of small fragments could previously accumulate without bound.

- **Malformed SSE events yield errors to callers** — Previously, `json.Unmarshal` failures on SSE events were silently logged and skipped (`continue`), causing data loss without the caller's knowledge. Now yielded as errors via the iterator, letting callers decide whether to continue or abort.

### Added

- **Security test suite** (`security_test.go`) — 15 tests verifying: Gemini header auth (Complete + Stream), error body truncation, bounded error reads, redirect header stripping (Gemini/Anthropic/OpenAI), bounded success body decoding (Gemini/Anthropic/OpenAI), bounded tool input accumulation (Anthropic/OpenAI), malformed SSE error propagation (Gemini/Anthropic/OpenAI).

- **`SafeHTTPClient(timeout)`** — Shared HTTP client constructor with redirect-safe auth header handling. Used by all three adapters.

- **`MaxErrorBodyBytes`** / **`MaxSSELineBytes`** / **`MaxResponseBodyBytes`** / **`MaxToolInputBytes`** constants — Centralized size limits for error reads (1 MB), SSE line parsing (256 KB), success response decoding (32 MB), and tool input accumulation (1 MB).

## [0.1.8] - 2026-02-21

### Added

- **`NewAssistantMessage(resp)` constructor** — Creates an assistant message from a `Response`, preserving both text content and `tool_use` blocks (with `ThoughtSignature` for Gemini 3). Use this instead of `NewTextMessage(RoleAssistant, resp.Content)` in tool use loops — `NewTextMessage` drops tool call blocks, causing providers to reject the next request.

- **Groq model registry** — Added 6 Groq models: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `deepseek-r1-distill-llama-70b`, `deepseek-r1-distill-qwen-32b`. Default: `llama-3.3-70b-versatile`. Aliases: `groq`, `llama`, `llama-70b`, `llama-8b`, `gpt-oss`.

- **Mistral model registry** — Added 8 Mistral models: `mistral-large-2512`, `mistral-medium-latest`, `mistral-small-2506`, `magistral-medium-2509`, `magistral-small-2509`, `codestral-2508`, `devstral-small-2-25-12`, `ministral-3-8b-25-12`. Default: `mistral-small-2506`. Aliases: `mistral-large`, `mistral-medium`, `mistral-small`, `magistral`, `codestral`, `devstral`, `ministral`.

### Fixed

- **Anthropic rejected `RoleSystem` messages** — The Anthropic adapter passed `role: "system"` messages in the `messages` array, which the API rejects. System messages are now extracted into the top-level `system` field (same pattern as the Gemini adapter).

- **StopReason values differed across providers** — Gemini returned `"STOP"` / `"FUNCTION_CALLING"` / `"MAX_TOKENS"`, OpenAI-compatible returned `"stop"` / `"tool_calls"` / `"length"`. Now all providers normalize to `"end_turn"` / `"tool_use"` / `"max_tokens"`. Anthropic already uses the normalized values natively.

## [0.1.7] - 2026-02-21

### Added

- **OpenAI model registry** — Added 16 OpenAI models to the model registry: GPT-5.2/5.1/5 (with `-chat-latest` variants), GPT-5 Mini/Nano, GPT-4.1/Mini/Nano, and O-series (o3, o3-mini, o4-mini). Includes default model (`gpt-5.2`), available models list, and aliases (`gpt`, `gpt5`, `gpt4.1`, etc.).

- **Grok reasoning model flags** — Added `Thinking: true` to `grok-4-1-fast-reasoning` and `grok-4-fast-reasoning` in the model registry.

- **Registry thinking flag tests** — Added tests validating that `Thinking` and `SupportsThinking` flags are correctly set across all models in the registry.

### Fixed

- **`ProviderForModel()` returned empty for OpenAI models** — No OpenAI models were registered in the model registry, so provider auto-detection failed for any `gpt-*` or `o*` model ID.

- **`GetDefaultModel("openai")` returned `claude-sonnet-4-6`** — No `"openai"` entry in the `defaultModels` map caused the function to fall through to the hardcoded Anthropic fallback. Now returns `gpt-5.2`.

- **`LogRequests: true` produced no visible output** — `LoggingClient` logged request/response metadata at `slog.Debug` level, which is suppressed by the default `slog.Info` threshold. Users who explicitly opted in via `LogRequests: true` saw only the pool creation `Info` line, not the actual per-request metadata (provider, model, tokens, cost, elapsed). Changed success-path logs from `Debug` to `Info`.

- **`Config.ThinkingLevel` had no effect** — The Anthropic adapter only read `Request.ThinkingLevel`, ignoring the config-level setting documented in the README. Now `doRequest` and `doStreamRequest` fall back to `c.config.ThinkingLevel` when the per-request field is empty.

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
- Standalone module path: `github.com/bds421/rho-llm`

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
