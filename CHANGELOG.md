# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.17.2] - 2026-11-12

### Changed

- **Batch and modality calls retry 3 times, like chat** (B15b). `DoHTTP` used
  `MaxRetries` as its attempt count, so under `DefaultConfig()` (`MaxRetries`
  10) a failing batch or modality call made 10 attempts while a single-key chat
  call made 3. Both now make 3; `MaxRetries` 1 or 2 still lowers them. A chat
  client with more healthy keys still makes one attempt per key.

### Fixed

- **Anthropic tool shapes the API rejects** (B31b, checked live 2026-11-12).
  Images in an `is_error` tool result ("all content must be type `text` if
  `is_error` is true") are left out and the text kept. A tool schema root
  without `type` ("Field required") gets `"type": "object"`, and a root
  `anyOf`/`oneOf`/`allOf` is refused before sending instead of by a 400.

## [0.17.1] - 2026-11-12

### Security

- **An API key was sent to a redirect target on the second hop** (B1). The
  redirect policy compared each hop with the previous one, but net/http copies
  the original request's headers onto every hop. A chain A → B → B/other
  stripped the key on the first hop and sent `x-api-key` / `x-goog-api-key`
  (and `Authorization`, when B shared A's hostname on another port) to B on the
  second. Every hop is now compared with the original request.
- **Batch handles loaded from storage could steer requests to another
  endpoint** (B29). The Anthropic batch ID and the Gemini batch name went into
  the URL path as is, so `../`, `?` or another resource name reached a
  different endpoint with the deployment's key. Both are now checked against
  the provider's ID shape (`msgbatch_…`, `batches/<id>`) before any request.
- **`Config` and `AuthProfile` printed the API key outside JSON** (B32). Only
  `MarshalJSON` redacted it; `%v`, `%+v`, `%#v` and slog's text handler printed
  the key and BaseURL credentials. Both types now implement `String`,
  `GoString` and `slog.LogValuer` with the same redaction.
- **`BlockPrivateBaseURL` allowed Azure's wire server** `168.63.129.16`
  (platform metadata on Azure VMs) (B36).
- **`examples/cache_gemini` sent the key as `?key=`** with a plain
  `http.Client` (B37), so it appeared in every transport error and proxy log.
  It now uses the `x-goog-api-key` header and `SafeHTTPClient`, and checks the
  cache name before building the delete URL.

- **A key in the BaseURL reached transport errors** (B32). net/http prints the
  request URL in every transport error and strips only a password, so a key
  carried as the URL's username or in `?key=` appeared in the error a caller
  logs. Every adapter, batch driver and `DoHTTP` now scrubs it; the error keeps
  its type, so retry classification is unchanged.

### Fixed

- **openai_compat: a provider error in the stream ended as a successful turn**
  (B2). An error chunk (`data: {"error":…}`, as OpenRouter, vLLM and Together
  send after the 200), with or without `[DONE]` after it, ended as `end_turn`
  with truncated content, or as a retryable "unexpected EOF" that re-sent a
  400. `finish_reason: "error"` ended as stop reason `"error"` without an
  error. Both now end the stream with an `*APIError` whose status comes from the
  error's numeric code or type (`invalid_api_key` → 401, `rate_limit_exceeded`
  → 429, unknown → 500), so `IsRetryable` and failover classify it like the
  same error sent as an HTTP status. A 200 `Complete` response that carries
  only an error object returns that error instead of an empty response.
- **Gemini: a blocked prompt, an error chunk and a failed finish were
  misreported** (B3). A prompt Gemini blocks has no candidates, only
  `promptFeedback.blockReason`: `Complete` returned an empty response with no
  stop reason, and `Stream` ran to EOF and surfaced a retryable truncation, so
  the pool sent the blocked prompt again. It is now a refusal (`StopRefusal`,
  the block reason in `RawStopReason`). An error chunk after the 200
  (`{"error":{"code":429,…}}`) became "stream ended without completion" with
  the status lost; it is now an `*APIError` with that status, and a 200
  `Complete` body with only an error returns it. Finish reasons: `RECITATION`,
  `BLOCKLIST`, `SPII` and the image filters are refusals like `SAFETY`;
  `MALFORMED_FUNCTION_CALL`, the other tool-call failures, `OTHER` and unknown
  reasons are `StopError` (they passed through as the raw string, so a tool
  loop ended on a failed turn as if it had completed). `RawStopReason` keeps
  the original. **Behaviour change:** `RECITATION` was returned as is.
- **OpenAI Responses: refusals were dropped, failures had no detail, and
  `Complete` joined text differently from `Stream`** (B4). A `refusal` content
  block (and `response.refusal.delta`) was ignored, giving an empty `end_turn`;
  it is now the content, with `StopRefusal`. A `failed` response was a Done
  event or response with stop reason `"error"` and no error; `Complete` and
  batch results now return an `*APIError` from `response.error` (`server_error`
  → 500, `rate_limit_exceeded` → 429, `invalid_*` → 400), and `Stream` ends
  with an `EventError` carrying the reported usage. A stream `error` event is
  classified the same way (it was a plain error). `Complete` joined text blocks
  with `"\n"`; it now concatenates them as `Stream` does.
- **`ValidateToolCall` skipped `required` and `enum` for schemas written in
  Go** (B9). It read only the JSON-decoded shapes (`[]any`, `float64`), so a
  schema with `"required": []string{"city"}` — as the README and examples
  write it — accepted a call without `city`. The schema is now normalized to its
  JSON form first; a schema that cannot be encoded is reported.
- **Breaking out of `Session.Stream` on `EventDone` dropped the completed
  turn** (B19). The usual `if ev.Type == EventDone { break }` returned before the
  turn was recorded, so the prompt, the reply and the usage vanished without an
  error. A break on the terminal event now records the turn; a break before it
  still leaves the conversation unchanged.
- **`SwitchProvider` kept a `Model` set through `WithBaseRequest`** (B21). The
  new client was asked for the previous provider's model on every turn. The
  switch now clears the base `Model`, so the new client's own model applies;
  other base settings carry over.
- **A burst of 429s on one key opened the circuit breaker for the whole pool**
  (B10). Five concurrent rate limits on key a (threshold 5) opened the circuit
  although key b was healthy, and every request was refused for the cooldown.
  A rate limit no longer counts against the endpoint while the pool has another
  healthy key; with a single key it still does. Auth errors now also re-arm a
  half-open probe they consumed instead of holding it.
- **A fallback chain never failed over when the primary hit its own
  `Config.Timeout`** (B11). net/http's timeout error matches
  `context.DeadlineExceeded`, which was read as caller cancellation, so the
  healthy fallback was never tried. The caller's context now decides: a
  deployment's own timeout is a provider failure, the caller's deadline or
  cancellation still stops the chain.
- **One request-scoped 403 permanently disabled every key** (B12). A 403 for a
  model or region the organization may not use disabled each key the pool
  rotated through, and they never recovered. A 401 still disables the key for
  good; a 403 now rests it for five minutes, and a second 403 for the same
  request is returned without trying more keys.
- **After waiting for a cooldown, the pool retried on the key still cooling**
  (B13). With every key cooling, the pool waited for the soonest one and then
  retried on the key that had failed last. It now rotates to the key it waited
  for.
- **`DisableRetries` silently turned off the circuit breaker** (B14). The
  pool returned before recording the failure, and `NewClient` bypassed the pool
  entirely. With `CircuitThreshold` set, a retries-disabled client now goes
  through the pool and every failure is recorded; it still makes one attempt.
- **`DoHTTP` did not retry Anthropic's 529 overload** (B16). The chat path
  retries it; batch and file calls failed on the first overload. 529 is now
  retried, and like a 503 it also allows resending a create
  (`HTTPCallOptions.NonIdempotent`): the request was not processed.
- **A text document in the history broke the session after a provider
  switch** (B20). Text documents exist only on the Anthropic protocol and every
  other adapter refuses them, so after `SwitchProvider` every turn failed and
  the history could not be edited. `NormalizeForProvider` now turns a text
  document into a text part (title, blank line, text) for any other provider.
- **Long OpenAI Responses answers failed at the end of the stream, and long
  Anthropic batch answers failed the download** (B26). `response.completed`
  repeats the whole response on one line, and an Anthropic batch result line
  holds a whole message; both were held to the 256 KB SSE line cap
  (`bufio.Scanner: token too long`). Without an explicit `MaxSSELineBytes`, a
  Responses stream line may now be as large as a `Complete` body
  (`MaxResponseBodyBytes`), and an Anthropic batch result line as large as the
  download. The Anthropic batch download is now held to `MaxBatchDownloadBytes`,
  which it ignored.
- **Anthropic batch: errored items lost the provider's message** (B27). The
  API nests the error (`{"type":"error","error":{"type":…,"message":…}}`); only
  a flat shape was read, so every errored item said "batch item failed". The
  message now reads `errored: invalid_request_error: <message>`.
- **`MaxRetries` of 1 or 2 was ignored by batch and modality calls** (B15).
  Chat clients honour it as a cap; `DoHTTP` raised it back to three attempts.
  It now makes exactly `MaxRetries` attempts (3 when unset). The `MaxRetries`
  docs (Config comment, README, ARCHITECTURE) said "minimum 3", which was true
  for neither path; they now describe both.
- **Cost was $0 for dated model IDs** (B22). Providers answer with snapshot
  IDs (`gpt-5.4-2026-03-05`, `claude-opus-5-5-20260901`), and `EstimateCost`
  looked up the exact ID, so `Session.Usage`, `LoggingClient` and usage hooks
  reported those turns as free. An unregistered ID that is a registered model
  plus a numeric snapshot suffix is now priced as that model.
- **Gemini thinking never came back** (B25). Without
  `thinkingConfig.includeThoughts`, Gemini returns no thought parts, so a
  request with `ThinkingLevel` never produced `EventThinking` or
  `Response.Thinking`. It is now sent with every thinking config (per Google's
  thinking docs; not yet confirmed against the live API with this library).
- **The stream idle timeout counted the caller's time** (B17). The watchdog
  ran from one read to the next, so a loop body slower than `Config.Timeout`
  (a tool call run mid-stream, a slow writer) killed a stream whose server
  never paused, with a retryable timeout. It now times only the wait for the
  server.
- **OpenAI Responses refused PDFs** (B20). The adapter returned an error for
  every `ContentDocument`, so a session that had read a PDF on Anthropic or
  Gemini failed every turn after a handoff to a GPT-5 model. PDFs are now sent
  as `input_file` items (`file_data` data URL; `ContentPart.Title` as the
  filename), as the Responses API documents.
- **Normalization paired results with the wrong call** (B7). A message that
  repeated a tool-call ID (servers that number every call `call_0`) paired both
  results with the second call; a tool call without an ID kept the call,
  dropped its result and skipped orphan repair. Results now pair with calls in
  order, and an ID-less call gets an ID (`call_noid_<n>`).
- **The pool's `Stream` leaked the client when the caller's loop body
  panicked** (B18). The in-flight reference was released only on normal
  paths, so a rotated-out client was never closed.
- **Saved conversations did not round-trip exactly, and sessions shared the
  caller's conversation** (B23). A tool argument holding an integer beyond
  ±2^53 (a 64-bit ID) changed on `LoadConversation` and `Clone`; such integers
  now load as `json.Number` (other numbers stay `float64`). A stored negative
  token count or cost loaded as is and corrupted later totals; it is clamped to
  zero, and trailing data after the object is refused. `ToRequest` returned the
  conversation's own backing array, so appending to the request could overwrite
  the conversation's next message; the slices are now clipped. `WithConversation`
  kept the caller's pointer and wrote to it on every turn, a data race with a
  caller saving it; the session now works on a copy (read it through
  `Session.Conversation`, as before).
- **The model read its own reasoning as part of its answer** (B24).
  Normalization turned unsigned thinking from the same provider into a text
  block, so every Gemini thought summary (returned since B25) and every
  DeepSeek `reasoning_content` came back on the next turn as text the model had
  supposedly said. It is now dropped: no adapter replays it. Thinking from
  another provider is still converted to text on a handoff, and signed
  Anthropic thinking is replayed as before.
- **Gemini batch results in the documented shape failed to decode** (B28).
  Google's batch reference nests inline results in an object
  (`inlinedResponses: {inlinedResponses: […]}`) under `response`,
  `response.output` or `metadata.output`; the driver read only a bare array
  under `dest` or `response`. All these shapes now decode, embedding results
  are also read from `inlinedResponses`, and results written to a
  `responsesFile` are reported as an error instead of an empty result set.
  (Checked against the reference docs, not a live batch.)
- **Gemini: a failed tool result was sent as a success** (B30). `IsError` was
  ignored, so the model read the error text as the tool's result. A failed
  result now goes under `error` in the `functionResponse` (the key Gemini's
  reference names for error details); a successful one stays under `result`.
- **Anthropic requests the API rejects went out** (B31). Extended thinking with
  a budget accepts only `tool_choice` auto or none; the guard covered only the
  adaptive models, so a Claude 4.6/4.5 request with thinking and a forced tool
  came back as a 400. It is now refused before sending. Whitespace-only text
  blocks (rejected by the API) are left out, and a rich tool result whose parts
  were all empty no longer sends `"content": null`.
- **Registry API edges** (B33). `GetDefaultModel` returned `claude-sonnet-5` (a
  stale literal) for every preset without a default: Together, OpenRouter,
  Fireworks, DeepInfra, NVIDIA, Cerebras, Perplexity, vLLM and LM Studio got a
  Claude ID their host rejects. Those presets now return `""` (name the model),
  `openai_responses` returns the OpenAI default, and an unknown provider falls
  back to the current Anthropic default. `RegisterModel` refused nothing it
  could never find: an ID that is an alias key (`sonnet`) or has surrounding
  whitespace is now an error, as is an alias with surrounding whitespace.
- **The logging middleware logged the `-1` token sentinel** (B34). A stream the
  caller stopped reading before it finished was logged as `stream done` with
  `tokens_in=-1`, and a turn without reported usage logged `-1` as a token
  count. An unfinished stream now logs `stream abandoned`, and missing usage
  logs `0` with `usage_reported=false`.
- **Realtime sessions: concurrent sends, wrapped EOF, no full `session.update`**
  (B35). Concurrent `Send*` calls wrote to the connection at the same time
  (WebSocket connections allow one writer); writes are now serialized. A
  wrapped `io.EOF` from the transport was returned as an error instead of a
  clean `session.closed`. `EncodeRealtimeEvent` kept only four fields, so a
  `session.update` with instructions, voice or tools could not be sent; a
  non-empty `RealtimeEvent.Raw` is now sent verbatim (it must be a JSON object
  with a `type`). The doc comment named a "registered dialer" that does not
  exist; the surface is now documented as experimental, with a caller-supplied
  dialer.
- **SSE lines without a space after `data:` were ignored by every adapter**
  (B5). The space is optional in the SSE spec; a stream from a gateway that
  omits it ended as a retryable "unexpected EOF".
- **Gemini tool results carried the wrong function name** (B6). Gemini numbers
  its synthetic tool-call IDs from 0 in every response, so a session that called
  the same tool first in two turns produced the same ID twice; normalization
  renamed the second, and `functionResponse.name` — parsed back out of the ID —
  became `get_weather#1`. After a handoff the other provider's ID (`toolu_…`)
  was sent as the name. The name now comes from the matching tool call in the
  transcript; parsing the synthetic ID is only the fallback.
- **Renamed duplicate tool-call IDs were invalid on Anthropic** (B7).
  `NormalizeForProvider` renamed a repeated ID to `<id>#<n>`; Anthropic accepts
  only `[a-zA-Z0-9_-]`. The suffix is now `_dup<n>`.
- **A tool call without arguments was stored as the string `""`** (B8). An
  empty Anthropic `partial_json` (or none) and `"arguments":""` from OpenAI-
  compatible servers and the Responses API became `ToolCall.Input == ""`, and the
  next request replayed `"input":""` / `"arguments":"\"\""`. Empty and `null`
  arguments now decode to `{}`, and every adapter sends an object when it replays
  a tool call, including from transcripts persisted before this fix. One
  internal decoder now does this for every adapter.

### Changed

- **Docs** (D2, D4, part of D1). README: an "Upgrading to v0.17.0" note
  (`GetAvailableModels` → `ModelsByProvider`, the removed `llm.Max*` vars), Fable 5.2
  in the adaptive-thinking notes, the retry and key-pool rules above, and the pi link
  (`earendil-works/pi`) with how the two libraries relate now. ARCHITECTURE: the
  `DefaultConfig` model (`claude-sonnet-5-5`) and `MaxRetries`, batch drivers for
  OpenAI, Anthropic and Gemini, the pool flow. CLAUDE.md: tutorial module count, pi
  link, Fable 5.2.
- **`tasks/todo.md`** rebuilt around the 2026-11-12 review: numbered bugs, the pi-ai
  1.5.2 comparison, registry dates in order; stale status lines and internal host,
  user and project names removed.

## [0.17.0] - 2026-11-11

### Removed

- **`GetAvailableModels` and the hand-written `availableModels` list.** The registry
  (`modelRegistry`) is the one model index; `Models()` / `ModelsByProvider(provider)`
  enumerate it, sorted by ID. The curated list had drifted from the registry (46
  registered models were missing from it, and it listed regional presets the registry
  files under their family). `ModelsByProvider` now lists a family's catalog for a
  preset that shares it (`dashscope-cn`, `qwen`, `kimi-cn`, …), as capability
  validation already did. Callers of `GetAvailableModels(p)` map
  `ModelsByProvider(p)` to `m.ID`.
- **The package-level `llm.MaxErrorBodyBytes`, `MaxSSELineBytes`, `MaxResponseBodyBytes`,
  `MaxBatchDownloadBytes` and `MaxToolInputBytes` vars.** A limit is set on `Config`
  and read through its `Effective*` accessor, which falls back to the `Default*`
  constant; the unsynchronised process-wide fallback is gone. A program that set a
  var sets the `Config` field instead.

### Changed

- **Tests are organised by concept.** The 82 root test files named after hardening
  passes, audit waves and versions (`harden_*`, `breaksystem*`, `v095_*`,
  `*_adversarial`, `llm_test.go`) are regrouped into 48 files named after the source
  concept they cover (`retry_test.go`, `pool_test.go`, `batch_test.go`, …;
  package-internal tests in `<concept>_internal_test.go`, shared fakes in
  `fixtures_test.go`). Two tests of the removed `GetAvailableModels` were dropped;
  tests that used it now use `ModelsByProvider`, which also matches the provider
  case- and whitespace-insensitively and lists runtime-registered `openai_responses`
  models under `openai`.

## [0.16.1] - 2026-11-11

### Fixed

- **Gemini tool calls with JSON-Schema keywords failed with a 400.** The adapter
  copied `Tool.InputSchema` into the function declaration's `parameters` field, which
  takes only Gemini's OpenAPI subset, so a schema with `additionalProperties`,
  `$defs`/`$ref` or `$schema` was rejected (`Unknown name "additionalProperties" at
  'tools[0].function_declarations[0].parameters'`, seen live in a tutorial run on
  v0.15.0). The schema now goes in `parametersJsonSchema`, which takes full JSON
  Schema (ai.google.dev/api, `FunctionDeclaration`), unchanged. Schemas that worked
  before keep working; the same schema now works on Anthropic, OpenAI and Gemini.

### Security

- **Go 1.26.11 minimum** (`go.mod`). govulncheck reports GO-2026-6859 in `net/http`
  for go1.26.10, fixed in go1.26.11.

## [0.16.0] - 2026-11-10

Registry and endpoint refresh for the providers v0.15.0 did not cover (Moonshot,
Meta, Cohere, Mistral, Groq, DeepInfra, Fireworks, Together), checked against each
provider's pages and public model lists on 2026-11-10.

> **Default and alias changes.** `moonshot` / `kimi` → `kimi-k3.1-pro` (was
> `kimi-k2.7-code`; the K2 series retires in late December 2026); `meta`, `muse`,
> `muse-spark` → `muse-spark-1.3` (was `muse-spark-1.2`). `moonshot-cn` / `kimi-cn`
> keep `kimi-k2.7-code` until K3.1 Pro is confirmed on api.moonshot.cn.
>
> **`mistral-medium-latest` costs more.** Mistral points it at Medium 3.5 now:
> $1.50 / $7.50 (was $0.40 / $2.00 for Medium 3.1, retired 2026-08-31).

### Added

- **`kimi-k3.1-pro`** (Moonshot's flagship) at $2 / $10, cache read $0.20, cache
  write $2 (1h $4), and **`kimi-k3`** at $3 / $15 / $0.30, write $3 (1h $6); 1M
  context, thinking always on (platform.kimi.ai/docs/pricing/chat). Also on Fireworks
  (`accounts/fireworks/models/kimi-k3p1-pro`, live-probed) and Together
  (`moonshotai/Kimi-K3.1-Pro`), both $2 / $10 / $0.20. Alias `kimi-k3.1`.
- **Meta:** `muse-spark-1.3` ($1.25 / $4.25, cache $0.15) and
  `muse-spark-1.3-contributor` ($0.10 / $0.20 / $0.002); `muse-wonder-1.0` (preview,
  $4 / $20 / $0.20); `muse-ember-1.0` ($0.10 / $0.50 / $0.004, 524,288 output; prompts
  over 200K tokens cost 3×, not modeled). All 1M context (dev.meta.ai pricing).
- **Cohere `command-a-plus-05-2026`** (Command A+): $2.50 / $10, 128K input, 64K
  output, reasoning.
- **Mistral `ministral-3b-2512`** ($0.10 / $0.10, 256K) and **Groq `qwen/qwen3.8-27b`**
  (preview, $0.80 / $4.00, 131,072 context, 16,384 output).
- **DeepInfra `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`** ($0.02 / $0.04).

### Changed

- **Together preset uses `https://api.together.ai/v1`**, the host its docs now
  document. `api.together.xyz` still answers; pass it as `BaseURL` if you need it.
- **Kimi K3 models refuse an explicit temperature** before the request is sent:
  Moonshot fixes temperature, top_p, n and the penalties ("omit them from requests").
- Prices and limits, from each provider's page:
  - `mistral-medium-latest`: Medium 3.5, $1.50 / $7.50, 256K, reasoning.
  - `codestral-2508` / `codestral-latest`: context 128K (was 256K).
  - `kimi-k2.7-code`: cache read $0.19 (was $0.16).
  - DeepInfra `deepseek-ai/DeepSeek-V4-Flash`: $0.09 / $0.18, cache read $0.018
    (was $0.10 / $0.20), context 1,048,576 (was 163,840).
  - Groq Llama 3.1 8B / 3.3 70B: context 131,072; these two are enterprise-only
    since 2026-08-16 (free and developer tiers moved to `openai/gpt-oss-*`).

### Removed

- **Retired:** `kimi-k2.5` and `moonshot-v1-128k` (Moonshot returns 404 since
  August 2026) → use `kimi-k3.1-pro`; DeepInfra
  `meta-llama/Meta-Llama-3.1-8B-Instruct` (deprecated 2026-07-16) → use the
  `-Turbo` variant. Requests for them now fail with the "retired by its provider"
  error naming the replacement.

## [0.15.0] - 2026-11-10

Endpoint refresh against the live model lists (`GET /v1/models` on Anthropic,
OpenAI and Mistral, `GET /v1/language-models` on xAI, `GET /v1beta/models` on
Gemini) and the providers' docs, 2026-11-10.

> **Alias changes.** `fable` → `claude-fable-5-2` (was `claude-fable-5`, set in
> v0.7.0); `mistral-large` → `mistral-large-4-0` (Mistral deprecated Large 3
> on 2026-10-31); `grok-code` → `grok-build-0.1`; `grok-mini`, `grok4.1`,
> `grok-4-1`, `grok-reasoning`, `grok-4-reasoning`, `grok-4-1-reasoning` →
> `grok-4.3`.
>
> **Retired xAI models now fail before dispatch.** `grok-3`, `grok-3-mini`,
> `grok-4-0709`, `grok-4-fast-{reasoning,non-reasoning}`,
> `grok-4-1-fast-{reasoning,non-reasoning}` → use `grok-4.3`;
> `grok-code-fast-1` → use `grok-build-0.1`. xAI no longer lists them and
> answers requests for them with grok-4.3 / grok-build-0.1 (the response's
> `model`), billed at that model's price, so `EstimateCost` was wrong by up to
> 6× and the caller never learned the model had changed.

### Added

- **`claude-fable-5-2`** (released 2026-11-09): $8 / $40, cache write
  $10 (1h $16), cache read $0.20; 1M context, 128K output; adaptive thinking
  only (always on), forced `tool_choice` and sampling parameters rejected — the
  same rules as Fable 5.1. Live: answers through rho-llm.
- **`mistral-large-4-0`** (alias `mistral-large-4`): $0.70 / $2.10, cached
  input $0.07, 1M context, reasons by default
  (docs.mistral.ai/models/mistral-large-4-0).
- **`gpt-image-2.6-flare`, `gpt-image-2.6-sunburst`, `gpt-image-2.5-flare`,
  `gpt-image-2.5-sunburst`** image-generation rows (no price, like
  `gpt-image-2`). Live: `gpt-image-2.6-flare` returns a PNG.

### Changed

- The `grok-4.20-beta*` slugs are aliases of the 0309 IDs instead of their own
  rows. Their rows priced them at $2 / $6 with a 2M context; xAI serves them as
  the 0309 models at $1.25 / $2.50 with 1M.

### Fixed

- **Mistral reasoning models returned an empty answer, and streams failed.**
  `mistral-large-4-0` sends `content` as typed chunks (a `thinking` chunk, then
  a `text` chunk) in both responses and stream deltas. The OpenAI-compatible
  adapter read `content` only as a string, so `Complete` returned `""` and
  every streamed chunk failed to decode ("cannot unmarshal array"). Text chunks
  now become `Content`, thinking chunks `Thinking` / `EventThinking`. Live
  2026-11-10: `Complete` and `Stream` both return `391` with the reasoning.

## [0.14.0] - 2026-11-07

`BlockPrivateBaseURL` checked against everything a private address can hide
behind (audit G3), and a caller-provided `*http.Client`.

### Security

- **`BlockPrivateBaseURL` covers IPv6 forms, non-canonical IPv4 hosts, host
  names, DNS rebinding and proxies** (audit G3). `CheckBaseURL` accepted IPv6
  forms of private addresses, such as `[64:ff9b::a9fe:a9fe]` (NAT64 of
  169.254.169.254), 6to4 (`2002::/16`) and IPv4-compatible (`[::a9fe:a9fe]`),
  and never resolved names. The check (standard library only) now unmaps
  IPv4-mapped addresses, judges the NAT64 well-known prefix `64:ff9b::/96` by
  the IPv4 address it carries (so IPv6-only hosts behind DNS64 still reach
  public providers), and refuses 6to4, Teredo, IPv4-compatible, CGNAT
  (`100.64.0.0/10`), benchmarking, documentation, multicast and reserved
  ranges. Hosts that a parser may read as IPv4 (`0x7f.1`, `2130706433`,
  `0177.0.0.1`) and `*.localhost` are refused too. With the option set the
  dialer checks every address it is about to connect to, each redirect hop
  included, while still racing IPv6 and IPv4 (Happy Eyeballs), so DNS
  rebinding cannot get through. `HTTP_PROXY` / `HTTPS_PROXY` are then ignored.
  With an explicit `ProxyURL` the client resolves the target itself, refuses
  it when the lookup fails, returns nothing or returns any private address,
  and opens CONNECT tunnels to the checked IP addresses (raced 300 ms apart,
  families interleaved), keeping the name for TLS and `Host`, so the proxy
  cannot resolve it differently; a proxy that filters by domain name
  sees only IP addresses. Refusals wrap the new `ErrPrivateAddress`, which
  `IsRetryable` reports as not retryable.
- **OAuth device flow uses the configured transport.** `DeviceAuthConfig.HTTP`
  (a `Config`) sets the timeout, proxy, `HTTPClient` and `BlockPrivateBaseURL`
  for both endpoints; before, it always used a 30-second default client.

### Added

- **`Config.HTTPClient`** — a caller-provided `*http.Client`. rho-llm uses a
  copy whose `CheckRedirect` strips the API-key headers on a cross-origin
  redirect and, with `BlockPrivateBaseURL`, refuses redirects to `localhost`,
  private IP literals and non-canonical IPv4 hosts, then applies the caller's
  `CheckRedirect`; the caller's client and transport are not changed. The
  caller's transport is responsible for proxying and egress filtering,
  including names. `ProxyURL` / `DisableProxy` are refused alongside it.

### Changed

- With `BlockPrivateBaseURL` and no `ProxyURL`, a failed name lookup is a
  `*net.DNSError` (retryable), not `ErrPrivateAddress`: the dialer resolves
  and checks each address it connects to, so nothing is dialed. Through `ProxyURL` a failed lookup
  is still refused with `ErrPrivateAddress`.
- A `Config.HTTPClient` without its own `CheckRedirect` keeps net/http's
  default: after 10 redirects the request fails with "stopped after 10
  redirects", as with any `http.Client`. rho-llm's own client still returns
  the last response at 10 redirects.

## [0.13.0] - 2026-11-06

Anthropic API version `2026-10-21` by default, following the migration guide
(platform.claude.com/docs/en/api/migration-2026-10-21), and the Anthropic
adapter checked field by field against the generic `Request` / `Response` and
the current API (structured outputs, citations and documents, adaptive thinking
and effort, prompt caching, streaming; 2026-11-06), plus the gaps a review of
v0.12.0/0.12.1 found. Live-tested on `2026-10-21` (Haiku 5.5 mostly; Sonnet 5.5,
Opus 5.5 and Haiku 4.5 where the behaviour needs them): streaming, tools, web
search streamed and not, caching, structured output, effort, text documents,
refusal fallback, Message Batch creation and results, and a pinned `2023-06-01`.
See "Upgrading to v0.13.0" in the README.
The guide's Admin and Analytics API changes do not apply (rho-llm calls none of
those endpoints), nor its SDK changes: the adapter is raw HTTP and keeps
`x-api-key`.

### Breaking

- **`anthropic-version: 2026-10-21` is the default** (`DefaultAnthropicVersion`;
  `AnthropicVersion20261021`, `AnthropicVersion20230601`). On it the adapter
  refuses, before sending, what the version rejects with a 400:
  `Request.Temperature` and `temperature` / `top_p` / `top_k` in
  `SamplingParams` (every model), a document in an assistant message, and
  citations on user text (also inside a tool result). `Config.AnthropicVersion =
  AnthropicVersion20230601` keeps the old rules; Claude 4.6 and 4.5 then take a
  temperature again. `ValidateRequestCapabilities` checks the version.
  Batches created before the switch finish under their old version.
- **`ResponseFormat` `json_object` on Anthropic is refused** (and a json_schema
  without a schema); the API constrains output only to a schema.
- **Direct clients check capabilities.** `anthropic.New` and the Anthropic batch
  client run `ValidateRequestCapabilities`; an unregistered model needs
  `Config.ModelCapabilities`.
- **`Effort` and `ThinkingLevel` must agree** where the level sets the effort
  (Claude 4.7 and later).
- **`ContainerID`** is refused by adapters without server tools, and
  **`CachedContent`** (a Gemini cache name) by Anthropic: a fallback chain that
  reuses one `Request` across Gemini and Anthropic with either set now fails.
- **`CacheTTL` without `CacheControl`** is an error, and so is a 1-hour cache
  marker after a 5-minute one.
- **`ReasoningSummaryAuto` no longer sends `display: "summarized"`** on
  Anthropic: Claude 4.7 and later then return empty thinking text (the API
  default). Leave `ReasoningSummary` unset for readable summaries.
- **Gemini and OpenAI refusals arrive as `StopRefusal`** (`SAFETY`,
  `PROHIBITED_CONTENT`, `content_filter`); read `RawStopReason` for the
  provider value.
- **`Session` records the answering model** (`EventDone.Model`) as a streamed
  turn's model, the fallback model after a server-side refusal fallback.

### Deprecated

- `SamplingParams["output_config"]` for Anthropic effort: use `Request.Effort`.
  It still passes through when the adapter builds no `output_config`.

### Added

- **Structured output on Anthropic.** `Request.ResponseFormat` (json_schema) is
  sent as `output_config.format`, next to the effort a `ThinkingLevel` sets, on
  the live and the Message Batch path. `json_object` and a schema-less request
  are refused (the API has no schema-less JSON mode). Anthropic models now
  advertise `CapabilityStructuredOutput`. Before, the schema was dropped on the
  direct `anthropic.New` and batch paths and refused through the factory.
- **`Request.Effort`** (`EffortLow` ... `EffortMax`): Anthropic
  `output_config.effort` and OpenAI Responses `reasoning.effort` without asking
  for thinking. It must agree with a `ThinkingLevel` that sets the effort.
  Refused where the API returns a 400: above `high` with thinking `disabled`
  (Haiku 5.5, Opus 5) or `between_tools` (Sonnet 5.5), on Sonnet 4.5 and
  Haiku 4.5, `xhigh` on the 4.6 models, `xhigh`/`max` on Opus 4.5. Gemini and
  OpenAI-compatible adapters refuse it.
- **Plain-text documents.** `DocumentSourceText` (`text/plain`, the text as
  `Data`), `NewTextDocument`, `IsTextDocument`, and optional `ContentPart.Title`
  / `Context` on any Anthropic document; cache markers as on other parts.
  Gemini and OpenAI-compatible adapters refuse text documents.
- **Thinking display choice.** `ReasoningSummaryOmitted` sends `display:
  "omitted"`, `ReasoningSummaryAuto` leaves it off (the API default, "omitted"
  on Claude 4.7 and later); unset keeps "summarized". Display changes what
  comes back, not what is billed (live: 31 output tokens in all three cases).
  OpenAI Responses treats `Omitted` as no summary.
- **`StreamEvent.Model`** on `EventDone`: the model that answered (from
  `message_start`; the fallback model after a server-side refusal fallback).
  `Session.Stream` records it as the turn's model.
- **`StreamAccumulator` and `CollectStream`.** The one fold from stream events
  to a `Response`, used by `Session.Stream`, the mock and callers.
  `ErrStreamIncomplete` when a stream ends without `EventDone`.
- **`Response.SetParts`** derives `Content`, `Thinking`, its signature and
  `ToolCalls` from `Parts`, so the two views cannot disagree; the Anthropic
  parser and the stream fold use it.
- **`ValidateCacheMarkers`**: a `CacheTTL` without `CacheControl`, or a 1-hour
  marker after a 5-minute one (tools, server tools, system, messages), is
  refused before the request instead of being dropped or returning a 400.
- **`UsageIteration.CacheCreation1hTokens`**, parsed from each iteration's
  `cache_creation`.

### Fixed

- **Streamed input tokens with server tools.** `message_delta` usage is
  cumulative and carries every sampling step's input and cache counts; the
  stream took them from `message_start` (the first step only). Live, a streamed
  web search reported 2,850 input tokens where the turn had 20,846.
- **A billed attempt is not retried away.** When a stream fails after the
  provider reported usage but before any content, the auth pool no longer
  retries silently: the `EventError` with the usage reaches the caller with the
  error.
### Fixed

- **Usage on a broken stream.** An Anthropic stream that breaks after
  `message_start` (read error, early EOF, oversized tool input, a mid-stream
  `error` event) now yields its error together with an `EventError` carrying
  the input and cache tokens, `message_start`'s output count and the content so
  far. `Session.Stream`, the auth pool, logging and `StreamWithBoundaries` pass
  that event on; the turn is still not recorded.
- **Mid-stream `error` events** were ignored and surfaced as "stream ended
  without completion". They are now an `*APIError` with the status the same
  error has outside a stream (`overloaded_error` = 529, retryable), its message scrubbed of the API key like an HTTP error body.
- **HTTP 529** (Anthropic overloaded) is retryable and `IsOverloaded`; 504 is
  retryable.
- **Capabilities on every path.** The direct `anthropic.New` client and the
  Anthropic batch client now run `ValidateRequestCapabilities` (and the batch
  capability per item), like the factory and the OpenAI batch client. A
  direct client with an unregistered model now needs `Config.ModelCapabilities`.
- **Batch betas.** Message Batch creation sends `Config.BetaFeatures` as
  `anthropic-beta` (interleaved thinking only when an item uses budget
  thinking).
- **Pricing.** Each usage iteration's 1-hour cache share is priced at the
  1-hour rate; web searches are priced at the first priced iteration's model
  when the top-level model is unregistered.
- **Refusals across providers.** Gemini `SAFETY` / `PROHIBITED_CONTENT` and
  OpenAI `content_filter` (Chat Completions and Responses) map to
  `StopRefusal`; `RawStopReason` keeps the provider value.
- **`ContainerID`** is refused by adapters without server tools, like
  `ServerTools`.
- **`CachedContent`** (a Gemini cache name) is refused by the Anthropic
  adapter instead of being dropped.
- A v0.12.0 test sent a 1-hour marker after a 5-minute one; corrected.

### Not changed

- `RoleSystem` messages are still hoisted into the top-level `system`. Sending
  them in place as `{"role":"system"}` (accepted by the API on Opus 4.8/5/5.5,
  Fable, Sonnet 5.5, Haiku 5.5) was tried and dropped: live, Haiku 5.5 declined
  an instruction placed there three times out of three and followed the same
  instruction in the top-level system prompt.
## [0.12.1] - 2026-11-06

### Fixed

- **Thinking disabled at effort `xhigh` / `max` on Claude Haiku 5.5 and
  Claude Opus 5.** Both accept `thinking: {"type": "disabled"}` only at effort
  `low`, `medium` or `high` and return a 400 otherwise; the docs say to lower
  the effort or use adaptive thinking. Without a `ThinkingLevel` the adapter
  sends `disabled` and no effort (the model default, valid), so the 400 came
  only from an `output_config.effort` of `xhigh` or `max` passed through
  `SamplingParams`. Such a request now fails before it is sent, with an error
  that names both fixes. The choice is the caller's, so the adapter does not
  switch thinking on or lower the effort itself. Other models are unchanged.

## [0.12.0] - 2026-11-06

Anthropic server tools (web search first), `pause_turn` continuations,
per-iteration usage, `stop_details`, the 1-hour cache TTL and unchanged replay
of empty signed thinking blocks, checked against platform.claude.com (web search, server
tools, citations, prompt caching, pricing, refusals and fallback, and the
preserved-thinking guidance; 2026-11-06). Tested with
hand-built fixtures in the documented shapes; no live calls.

### Added

- **Server tools.** `Request.ServerTools` / `Conversation.ServerTools` /
  `WithServerTools` take a `ServerTool` (`Type`, `Name`, `MaxUses`,
  `AllowedDomains` / `BlockedDomains`, `UserLocation`, `AllowedCallers`,
  `CacheControl`, plus `Params` for version-specific fields). Constants
  `WebSearch20250305`, `WebSearch20260209`, `WebSearch20260318`,
  `ServerToolWebSearch`, `CallerDirect`; `NewWebSearchTool`. The Anthropic
  adapter appends them after `Tools` as sorted-key objects, so the encoding is
  byte-identical on every request. The other adapters return an error
  (`UnsupportedServerTools`) instead of dropping them, and
  `RequestCapabilities` requires `CapabilityTools`.
- **Ordered assistant content.** `Response.Parts` (and `EventDone.Parts`) hold one
  part per Anthropic content block, in order: text with `Citations`, each
  thinking block with its own signature, tool calls, `ContentServerToolUse`,
  `ContentServerToolResult` (`BlockType`, `WebSearchResults` or
  `ServerToolError`) and `ContentProviderBlock` for any other block type.
  Server parts keep the block as received in `ProviderBlock`, citations in
  `Citation.Raw`; the adapter replays both verbatim, so `encrypted_content` and
  `encrypted_index` survive, also through `LoadConversation`.
  `NewAssistantMessage` uses `Parts` when set. Citation types:
  `char_location`, `page_location`, `content_block_location`,
  `search_result_location`, `web_search_result_location` (unknown types kept in
  `Raw`).
- **Streaming:** `EventServerTool` (`Part`) per server tool call and result,
  `EventCitation` (`Citation`) per `citations_delta`.
- **`pause_turn`:** `StopPauseTurn` (and `StopRefusal`) constants;
  `ContinuePausedTurn(req, resp)` appends the paused content unchanged and
  leaves every other request field alone; `Session.Continue` /
  `StreamContinue` do the same for a session and return `ErrNotPaused` without
  a request when the last turn did not pause. The paused turn's container
  (`Response.ContainerID` → `Request.ContainerID`, Anthropic `container`) is sent
  back.
- **Usage:** `Response.ServerToolUse` (`WebSearchRequests`, `WebFetchRequests`,
  `CodeExecutionRequests`) and `Response.Iterations` (Anthropic
  `usage.iterations`), also on `EventDone`; `Usage.ServerToolUse` sums them.
- **Cost:** `ModelInfo.WebSearchPricePer1K` ($10 per 1,000 searches on every
  Anthropic model) and `CostInput.WebSearchRequests`, priced without the batch
  discount (searches cost the same in a batch). `CostInput.Iterations` prices
  each iteration at its own model, a fallback at the fallback model's rates,
  and replaces the top-level token fields, which cover only the attempt that
  produced the message. An iteration on an unpriced model adds 0. A refusal
  before any output is counted although it is not always billed, so the
  estimate is an upper bound there.
- **`stop_details`:** `Response.StopDetails` and `EventDone.StopDetails`
  (`Type`, `Category`, `Explanation`, `RecommendedModel`,
  `FallbackCreditToken`, `FallbackHasPrefillClaim`, and `Raw` with the object
  as received); nil when the API sends null. `StopDetailsFromRaw`.
- **1-hour cache TTL:** `CacheTTL` on `ContentPart`, `Tool` and `ServerTool`,
  `Request.SystemCacheTTL` (`CacheTTL1h`; empty or `CacheTTL5m` is the default),
  sent as `cache_control: {"type":"ephemeral","ttl":"1h"}`; any other value is
  an error. `Response.CacheCreation1hTokens` (from
  `usage.cache_creation.ephemeral_1h_input_tokens`, also on `EventDone` and
  `Usage`) is the 1-hour share of `CacheCreationTokens`.
  `ModelInfo.CacheWrite1hPricePer1M` (2x input, set on every Anthropic model)
  and `CostInput.CacheCreate1hTokens` price it; a model without a registered
  1-hour price gets the documented 2x input price, or 0 when its input price is
  unknown too.

### Changed

- `NewAssistantMessage` builds the message from `Response.Parts` when set (every
  Anthropic response now sets it). Edits to `Response.Content` or `ToolCalls`
  made before the call no longer reach the message; edit `Parts`, or clear it.
  Multiple thinking blocks now replay as separate blocks with their own
  signatures instead of one merged block.
- `NormalizeForProvider` / `NormalizeForModel` drop server tool parts and
  provider blocks and strip citations when a message goes to a different
  provider.
- Anthropic `tool_choice` is also sent when only server tools are defined.
- **Anthropic `anthropic-beta`** is sent on every request when
  `Config.BetaFeatures` is set, not only with a `ThinkingLevel`, so server-side
  fallback (`SamplingParams{"fallbacks": "default"}`) and other betas work
  without thinking. The deprecated `interleaved-thinking-2025-05-14` flag
  (`DefaultConfig`'s default) is the exception: it goes out only with budget
  thinking (Claude 4.5 and older), the only case where it does anything.
- **Signed thinking blocks with empty text** (the default `display: "omitted"`
  on Claude 4.7+) are kept for the same provider and sent back unchanged,
  signature included, by `NormalizeForProvider` / `NormalizeForModel` and the
  Anthropic adapter; before they were dropped. The API docs say to keep them
  (an empty block is still a block; dropping it fails the preserved-thinking
  check and the continuation). Thinking blocks from the API are replayed as
  received. Without a `ThinkingLevel`, models whose thinking is always on
  (Fable 5.1, Opus 5.5) and Sonnet 5.5, which is sent `between_tools` and still
  writes thinking blocks between tool calls, replay them too; a model sent
  `{"type":"disabled"}` drops them. A handoff to another provider still drops
  empty blocks.

### Fixed

- Anthropic stream parser: removed a dead `doneEmitted` assignment
  (ineffassign).

## [0.11.0] - 2026-11-04

First two items from the consolidated backlog (`tasks/todo.md`), both taken
from pi-ai 1.5.0.

> **Behaviour change.** `Session` now drops thinking produced by a different
> model when you switch models (including Opus → Haiku on the same provider).
> Before, a signed same-provider block was replayed verbatim and cross-provider
> thinking was converted to text. The visible answers are unchanged.
> `NormalizeForProvider` keeps the old behaviour.

### Added

- **`ThinkingMax`** — the deepest thinking level. Claude 4.7+ get adaptive
  effort `max` (all of them list it in `GET /v1/models`); the OpenAI Responses
  API, which has no `max`, gets `xhigh`; Gemini gets `high`; budget-based Claude
  models get the `ThinkingXHigh` budget (128,000, clamped to the model limit).
  Live 2026-11-04: `claude-haiku-5-5` and `gpt-6-luna` accept it.
- **`NormalizeForModel(msgs, provider, model)`** — `NormalizeForProvider` plus
  dropping thinking from other models. Model IDs that differ only by a snapshot
  suffix of three or more digits count as the same model.

### Changed

- `Session` normalizes each turn with `NormalizeForModel` for the turn's model.
  Live 2026-11-04: a session switched from `claude-sonnet-5-5` (thinking) to
  `claude-haiku-5-5` continues normally.

### Fixed

- **`ThinkingXHigh` on Gemini 3+ returned 400.** It fell back to a token
  budget clamped to the model's 65,536-token output limit, but the API accepts
  `thinkingBudget` only up to 65,535. `ThinkingXHigh` and `ThinkingMax` now send
  `thinkingLevel: "high"` (Gemini's deepest level), and every `thinkingBudget`
  (including a custom `ThinkingBudget`) is capped at 65,535.

## [0.10.1] - 2026-11-03

Gemini registry follow-ups from the 2026-10-31 refresh, checked against
`GET /v1beta/models` with a key and aistudio.google.com/docs (2026-11-03).

> **Alias change.** `flash` now resolves to `gemini-3.8-flash` (was
> `gemini-2.5-flash`, which Google shuts down 2026-12-01). `gemini-3.8-flash`
> rejects `temperature`/`top_p`/`top_k`; pin `Model: "gemini-2.5-flash"` until
> the shutdown if you need sampling controls.

### Fixed

- **Gemini 3+ replies cut off by their own thinking.** Gemini 3 and later models
  think by default even without a `ThinkingLevel`, and thinking tokens count
  against `maxOutputTokens`, but the adapter only padded `maxOutputTokens` for
  Gemini 2.5. Live on 2026-11-03, `gemini-3.8-flash` with `MaxTokens: 100`
  spent 93 tokens thinking and returned 3 visible tokens (`stop_reason:
  max_tokens`). The padding (`MaxTokens` + the `ThinkingLow` budget, capped at
  the model limit) now applies to every Gemini model with `SupportsThinking`;
  the same request returns the full answer.
- **`gemini-4-argon` max output** is 32,768 tokens (`outputTokenLimit`), not
  65,536.
- **`gemini-3` alias** pointed at the shut-down `gemini-3-pro-preview`; it now
  resolves to `gemini-3.1-pro-preview`, like `gemini3`.

### Added

- `gemini-3.1-pro-preview-customtools` (the Gemini 3.1 Pro endpoint tuned to
  prefer custom tools; same limits and prices as `gemini-3.1-pro-preview`).
- Image generation: `gemini-3.1-flash-image` (Nano Banana 2),
  `gemini-3.1-flash-lite-image` (Nano Banana 2 Lite), `gemini-nano-banana-2.1`.

### Removed (retired by the provider → `retiredModels`)

- **Gemini:** `gemini-2.0-flash` (shut down 2026-06-01; use `gemini-3.6-flash`)
  and `gemini-3-pro-preview` (shut down 2026-03-09; use
  `gemini-3.1-pro-preview`). Neither appears in `GET /v1beta/models` any more.

## [0.10.0] - 2026-10-31

Registry and endpoint refresh against every provider's primary docs (2026-10-31),
plus a live unauthenticated probe of each preset endpoint and of the hosted model
IDs where the host checks the model before auth (Fireworks, NVIDIA).

> **Behaviour changes for callers who rely on defaults or aliases.** The provider
> defaults and the `opus`/`sonnet`/`haiku`/`claude`/`gpt`/`grok`/`glm`/`groq`
> aliases now resolve to newer models (see Changed). Pin an explicit `Model` to
> keep the old one. IDs whose provider shut them down now fail with a
> `retired by its provider; use "…"` error instead of a provider 404.

### Fixed

- **Anthropic thinking on Claude 4.7+ (Opus 4.7, 4.8, 5, 5.5; Fable 5, 5.1;
  Sonnet 5, 5.5; Haiku 5.5).** Only `claude-sonnet-5` was sent adaptive
  thinking; every other 4.7+ model got `thinking.type: "enabled"` with
  `budget_tokens`, which Anthropic rejects with a 400 ("Claude 4.7 and later
  models do not support it"). Any non-zero `ThinkingLevel` on those models
  failed. They now send `{type: "adaptive"}` + `output_config.effort`, and
  `ThinkingNone` is encoded the way each model accepts it: `disabled` (Opus
  4.7/4.8/5, Sonnet 5, Haiku 5.5), `between_tools` (Sonnet 5.5, which rejects
  `disabled`), or omitted (Fable 5/5.1, Opus 5.5 — thinking is always on and an
  explicit off is a 400). `ThinkingBudget` on these models returns an error
  before the request is sent.
- **Sampling temperature on Claude 4.7+.** These models reject
  `temperature`/`top_p`/`top_k` (Sonnet 5.5 and Haiku 5.5 reject non-default
  values), but only Sonnet 5 dropped `CapabilityTemperature`; an explicit
  `Temperature` now fails capability validation for all of them.
- **Forced `tool_choice` on Fable 5.1, Opus 5.5 and Sonnet 5.5.** `ToolChoiceRequired`
  / `ToolChoiceTool` return an error naming the model instead of a provider 400.
- **Z.ai preset endpoint.** `zai`/`z-ai`/`glm` pointed at
  `https://api.z.ai/api/openai/v1`, which does not exist (it answers HTTP 200
  with `{"code":500,"msg":"404 NOT_FOUND"}`). Now the documented
  `https://api.z.ai/api/paas/v4`.

### Added

- **Anthropic:** `claude-opus-5-5` ($4/$20, cache read $0.20), `claude-sonnet-5-5`
  ($2/$10, cache read $0.10), `claude-haiku-5-5` ($0.10/$0.50; prompts over 100K
  tokens cost 5×, not modeled).
- **OpenAI:** GPT-6 family — `gpt-6.1-sol` ($2/$10), `gpt-6.1-terra` ($1/$5),
  `gpt-6-sol`, `gpt-6-luna` ($0.10/$0.50), `gpt-6-astra` ($10/$50); all Responses
  API (tool calling requires it), 922K input / 128K output.
- **xAI:** `grok-4.7`, `grok-4.6` ($2/$6, cache read $0.50, 500K context).
- **Gemini:** `gemini-4-argon` ($2/$10 introductory through 2026-11-30, then
  $4/$20); `gemini-3-pro-image` (stable ID of Nano Banana Pro).
- **DeepSeek:** `deepseek-flash` (V4.1 Flash, 1M context, 384K output).
- **Z.ai:** `glm-5.3`, `glm-5.3-flash`. **MiniMax:** `MiniMax-M3.1-Flash`.
  **DashScope:** `qwen3.8-max`, `qwen3.8-flash`. **NVIDIA:**
  `nvidia/nemotron-3-ultra-550b-a55b`.
- **Fireworks:** live IDs `accounts/fireworks/models/{kimi-k3,glm-5p3,glm-5p3-flash,minimax-m3,deepseek-v4p1-flash,qwen3p8-max}`.
  **Together:** `moonshotai/Kimi-K3`, `zai-org/GLM-5.3`, `zai-org/GLM-5.3-Flash`,
  `deepseek-ai/DeepSeek-V4.1-Flash`, `Qwen/Qwen3.8-Flash`.
- Aliases `gpt6`/`gpt-6`, `gpt6.1`/`gpt-6.1`, `gpt6-luna`, `gpt6-astra`,
  `grok4.7`/`grok-4-7`, `grok4.6`/`grok-4-6`, `glm5.3`/`glm-5-3`.

### Changed

- **Defaults:** anthropic/claude → `claude-sonnet-5-5`; openai/gpt →
  `gpt-6.1-sol`; xai/grok → `grok-4.7`; deepseek → `deepseek-flash` (the legacy
  `deepseek-chat`/`deepseek-reasoner` names were discontinued 2026-07-24); zai →
  `glm-5.3`; groq → `openai/gpt-oss-120b` (Llama 3.3/3.1 left Groq's free and
  developer tiers on 2026-08-16; the rows stay for enterprise contracts).
- **Aliases:** `opus` → `claude-opus-5-5`, `sonnet`/`claude` → `claude-sonnet-5-5`,
  `haiku` → `claude-haiku-5-5`, `grok`/`grok4`/`grok-4` → `grok-4.7`, `glm` →
  `glm-5.3`, `groq` → `openai/gpt-oss-120b`, `deepseek-cloud`/`deepseek-v4` →
  `deepseek-flash`. Aliases that name a retired model (`pixtral`, `magistral`,
  `devstral`, `llama4`, `gpt5.3`, `gpt-instant`, `deepseek-reasoner`) now produce
  the retired-model error with its replacement.
- **Prices:** `gpt-5.6-sol` $5/$30 → $4/$20 (promotional "at least through
  November 21, 2026"); `gemini-3.6-flash` $1.50/$7.50 → $0.75/$3.75 (through
  2026-12-31); `gemini-embedding-2` $0.15 → $0.20; `MiniMax-M3` $0.60/$2.40 →
  $0.30/$1.20 (the ≤512K tier); `qwen3.6-plus` $0.29/$1.74 → $0.50/$3.00 and
  `qwen3.6-flash` $0.05/$0.40 → $0.25/$1.50 (≤256K tier); `deepseek-v4-pro`
  → $1.32/$3.96 with 1M context / 384K output; `deepseek-v4-flash` (legacy name,
  routed to V4.1 Flash) → V4.1 Flash rates. Cache-read prices added for
  `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite`, `gemini-3-flash-preview`,
  `gemini-2.5-pro`/`-flash`/`-flash-lite`.
- **Capabilities:** `claude-sonnet-4-6` max output 64K → 128K;
  `command-a-03-2025` max output 4K → 8K and no longer flagged as reasoning
  (Cohere's reasoning model is `command-a-reasoning-08-2025`).

### Security

- **Go 1.26.10 minimum (`go.mod`).** govulncheck reports ten standard-library
  vulnerabilities reachable from this module on Go 1.26.8 — `net/http`
  (GO-2026-6603, -6605, -6610, -6611, -6612, -6613, -6617), `net/textproto`
  (GO-2026-6608) and `crypto/tls` (GO-2026-6607), fixed in 1.26.9, and
  `crypto/x509` (GO-2026-6809), fixed in 1.26.10. Consumers building with an
  older toolchain get it via `GOTOOLCHAIN=auto`.

### Removed (retired by the provider → `retiredModels`)

- **OpenAI** (shutdown passed): `gpt-5.3-chat-latest`, `gpt-5.2-chat-latest`,
  `gpt-5.1-chat-latest`, `gpt-5-chat-latest`, `gpt-5-codex`, `gpt-4.1-nano`,
  `o4-mini`, `o3-mini`, `gpt-image-1`.
- **Mistral:** `mistral-small-2506`, `magistral-small-2509`,
  `magistral-medium-2509`, `devstral-2512`, `devstral-small-2-2512`,
  `pixtral-large-2411`, `pixtral-12b-2409`.
- **DeepSeek:** `deepseek-chat`, `deepseek-reasoner`.
- **Groq:** `deepseek-r1-distill-llama-70b`, `deepseek-r1-distill-qwen-32b`,
  `meta-llama/llama-4-scout-17b-16e-instruct`.
- **Fireworks:** all seven `accounts/fireworks/models/fireworks/*` IDs (the API
  answers 404 "Model not found").
- **Together:** `deepseek-ai/DeepSeek-V3`, `meta-llama/Llama-3.3-70B-Instruct-Turbo`.
- **NVIDIA:** `meta/llama-3.1-8b-instruct` (410 "end of life on 2026-08-26").

## [0.9.5] - 2026-10-08

Reliability patch from the 2026-10-07 architecture review (ids H*/M*/L refer to
it). Backwards compatible: no exported symbol was removed or renamed and no
exported signature changed; everything new is additive.

> **Consumers must check — openai_responses usage accounting changed.** For the
> `openai_responses` adapter (GPT-5 family via `Provider: "openai"`),
> `OutputTokens` now **excludes reasoning tokens**, which are reported in
> `ThinkingTokens` instead (before, `ThinkingTokens` was always 0 and
> `OutputTokens` included reasoning). If you bill, meter or budget from the
> usage fields yourself, charge **`OutputTokens + ThinkingTokens`** as output.
> `EstimateCost` already prices `ThinkingTokens` at the output rate when you
> pass both fields; totals are unchanged when both are summed.

### Added

- **`Config.RetryBudget` (H1).** Total wall-clock bound across one call's retry
  sequence, for pooled chat clients (`Complete` and pre-data `Stream` retries)
  and `DoHTTP` (modality + batch). Before each backoff the client checks whether
  sleeping and trying again still fits; if not it returns the last error. 0
  (default) = no budget. In-flight attempts are never cut — use `ctx` for that.
  A backoff that would reach the `ctx` deadline is skipped the same way (see
  Fixed).
- **`Retry-After` support (H1).** `APIError.RetryAfter` carries the provider's
  hint (delta-seconds, HTTP-date, or OpenAI's `retry-after-ms`), clamped to the
  new `MaxRetryAfter` (60s). `ErrorFromResponse` fills it, and both retry
  engines use it as a floor for the next backoff.
- **`DoHTTPWithOptions` + `HTTPCallOptions{NonIdempotent}` (H5).** A
  non-idempotent create is resent only after 429/503 or a dial/DNS/proxy-connect
  failure — never after 408/500/502/504 or a post-send transport error, which may
  follow a committed job. `DoHTTP` is unchanged (idempotent).
- **`NewStreamingHTTPClient` and `StreamIdleTimeoutError` (H3)** — see Fixed.
- **`Config.UsageHookCtx` / `ContextUsageHook` (M1).** Same per-attempt modality
  `UsageEvent`s as `UsageHook`, plus the caller's `context.Context`, so one shared
  `ModalityClient` can attribute usage per request (e.g. a user ID context value)
  without a client per caller. Both hooks fire when set (`UsageHook` first),
  each panic-recovered; fallback deployments inherit the primary's.
- **`NewFallbackModalityClientWithPolicy` + `FailoverPolicy` (M3).**
  `FailoverAnyProviderError` is the default (= `NewFallbackModalityClient`,
  unchanged). `FailoverTransientOnly` — recommended when a rejected payload must
  not be re-sent to another vendor — fails over on transport errors, 408/429/5xx
  and 401/403/404, never on 400/413/415/422. Caller cancellation always stops
  the chain.
- **`UsageEvent.CacheReadTokens` / `ModalityUsage.CacheReadTokens` (M5).** Cached
  prompt tokens, disjoint from `InputTokens` (the v0.9.4 chat contract), priced at
  the cache-read rate in `CostUSD`.
- `DefaultHTTPMaxAttempts` (3), `RedactSecrets(s, cfg)`.

### Changed

- **One retry model for chat and modality/batch (H1).** `DoHTTP` now makes
  `DefaultHTTPMaxAttempts` = 3 attempts when `Config.MaxRetries` is 0 (was
  `DefaultMaxRetries` = 10, ≈151s of backoff — the 2m22s modality stall); an
  explicit `MaxRetries` (including `DefaultConfig()`'s 10) is still honoured.
  A single-key chat client now backs off with `Config.RetryPolicy` (1s, 2s, …)
  instead of sleeping the 30s/60s key cooldown, so `RetryPolicy` is no longer dead
  configuration for it. Pools with more than one healthy key keep waiting for the
  soonest cooling key. No backoff is slept after the final attempt.
- **Breaker documentation matches behaviour (H2).** The README claimed every
  client gets a circuit breaker; only `DefaultConfig()` sets `CircuitThreshold`.
  Chosen fix: correct the docs rather than floor `CircuitThreshold` for struct
  literals — a floor would silently change failure behaviour (fail-fast
  `ErrCircuitOpen`) for every existing struct-literal consumer and leave no way to
  express "no breaker". README/Config docs now state it, and the README example
  sets it explicitly.
- **Creates are not retried on 408/500/502/504 (H5).** Anthropic and Gemini
  batch submit, OpenAI-compatible image generation and speech synthesis, and
  Gemini image generation use `NonIdempotent`: they are still retried on 429 and
  503 (the provider rejected the request before acting on it) and on
  dial/DNS/proxy-connect failures, but not on 408/500/502/504 or a post-send
  transport error, where the job may already exist. (OpenAI batch creates
  already used a single attempt.)
- **openai_responses usage (M6).** `reasoning_tokens` is read from
  `usage.output_tokens_details` (it was read from a top-level field the API never
  sends, so `ThinkingTokens` was always 0). Because it is a subset of
  `output_tokens`, it is moved from `OutputTokens` to `ThinkingTokens` — the
  Gemini-style "thinking separate from output" contract. Total cost is unchanged
  and reasoning is billed once; `OutputTokens` for reasoning models is now the
  visible output only.
- **Library logging (M8).** Pool/factory `Info` logs and per-attempt /
  per-request `Warn` notices (rate limited, overloaded, temperature ignored,
  thinking budget clamped, tool-input JSON unparsable, …) are now `Debug`. `Warn`
  remains only for a key disabled by an auth error and for recovered hook/callback
  panics.
- **Gemini embeddings report no usage (M5).** `EmbeddingResponse.InputTokens` is 0
  instead of a fabricated `len(text)/4`; `embedContent` reports no usage.
- **`NewSafeHTTPClient` transport bounds (non-stream too).** The transport now
  sets a dial timeout (≤30s), a **TLS-handshake timeout of 10s** (or `Timeout`
  if smaller), `ResponseHeaderTimeout` (= `Timeout`) and an **`IdleConnTimeout`
  of 90s** for pooled keep-alive connections. Non-stream calls are still bounded
  end to end by `Timeout`, but a TLS handshake slower than 10s now fails (as a
  retryable timeout) even when `Timeout` is larger, and idle connections are
  closed after 90s instead of being kept indefinitely.
- Docs: provider count corrected to 23 providers / 35 preset names (README,
  CLAUDE.md, ARCHITECTURE.md); `tasks/todo.md` header refreshed and the review's
  remaining items (M2, M4, M9, …) recorded as open work.

### Fixed

- **Streams were killed by `http.Client.Timeout` (H3).** It covers reading the
  whole body, so any stream longer than `Config.Timeout` (default 120s) — a
  `ThinkingHigh` turn, a long generation — died mid-response after content was
  yielded. Streams now use `NewStreamingHTTPClient`: same transport, no
  whole-body timeout, and an idle watchdog that fails a body silent for more than
  `Timeout` with `StreamIdleTimeoutError` (a `net.Error` timeout). Total stream
  duration is bounded by `ctx`. **Caveat — silent reasoning:** the watchdog
  counts bytes, not tokens. Anthropic sends `ping` events during long thinking,
  which keep the stream alive; Gemini and the OpenAI Responses API can reason
  for minutes without sending any bytes, and such a turn still fails with
  `StreamIdleTimeoutError` once the silence exceeds `Timeout`. **Raise
  `Config.Timeout` for silent long reasoning** (e.g. `ThinkingHigh` on Gemini or
  GPT-5) and bound the total with `ctx`.
- **A backoff no longer sleeps past the caller's deadline.** Before every
  backoff (`DoHTTP`, pooled `Complete` and pre-data `Stream` retries) the client
  checks `ctx.Deadline()`: if `now + delay` reaches it, the retry is abandoned
  and the last error/response is returned at once — the same exit as an
  exhausted `RetryBudget`. A 429 with `Retry-After: 2` under a 500ms deadline
  now returns the 429 `*APIError` (with `RetryAfter` set) immediately instead of
  `context.DeadlineExceeded` 500ms later.
- **Gemini image-generation panic (H4).** An 8–11 byte inline payload labelled
  `image/webp` indexed `raw[8:12]` past the end and crashed the caller.
- **Gemini modality usage double-counted cached tokens (M5).** `InputTokens` now
  excludes `cachedContentTokenCount` (reported in `CacheReadTokens`), and
  `AudioInputTokens` excludes cached audio: the `AUDIO` entries of
  `cacheTokensDetails` are subtracted from the `promptTokensDetails` audio total
  (previously total audio was only clamped to the uncached input).
- **Interleaved streamed tool calls were spliced together (M7).** openai_compat
  now assembles calls per delta `index` (index-less servers keep the old
  id-starts-a-call rule; a reused index with a new id starts a new call);
  openai_responses per `item_id`/`output_index`, taking `name`/`call_id` from
  `response.output_item.added` when the done event omits them. The tool-input
  cap now bounds all in-flight calls together.

### Security

- `redactSecret` (and so `ErrorFromResponse`/`RedactSecrets`) no longer skips
  keys of 4–7 bytes: they are scrubbed where they stand as a whole token (so
  `Bearer abc1` is redacted without mangling words that merely contain the key).
  Keys of 8+ bytes keep the v0.9.4 substring scrub. Keys under 4 bytes, and
  well-known placeholder keys that local servers accept (`none`, `EMPTY`,
  `ollama`, `dummy`, `test`, … — case-insensitive), are left alone: they are not
  secrets, and scrubbing them would corrupt ordinary error text ("must be none").
  A real 5–7 byte key is still redacted.
- openai_responses in-stream `error` events are redacted (`RedactSecrets`) and
  truncated to `EffectiveMaxErrorMessageLen` like HTTP error bodies.
- `TestAllProviders` (live, paid calls) now skips in `-short` mode.

## [0.9.4] - 2026-10-07

### Fixed

- **Gemini cached tokens were charged twice (C5).** Gemini's
  `usageMetadata.promptTokenCount` already includes `cachedContentTokenCount`
  ("this includes the number of tokens in the cached content" —
  ai.google.dev/api/generate-content), but the adapter passed the full prompt
  count as `InputTokens` *and* the cached count as `CacheReadTokens`, so
  `EstimateCost` billed the cached tokens at the input price and again at the
  cache-read price (1,000 uncached + 9,000 cached on a $10/$1 model: $0.109
  instead of $0.019). `InputTokens` is now `promptTokenCount −
  cachedContentTokenCount` (Complete, Stream and Gemini Batch results).
- **OpenAI-style adapters ignored cached tokens.** `openai_compat`
  (`prompt_tokens_details.cached_tokens`) and `openai_responses`
  (`input_tokens_details.cached_tokens`) now report the cached share in
  `CacheReadTokens` and subtract it from `InputTokens`, so the registry's
  OpenAI/xAI/Z.ai/MiniMax/Moonshot/Meta cache-read prices actually apply
  instead of billing cached input at the full input price.
- Hostile usage is clamped: a negative wire count reads as 0 and a cached
  count above the reported prompt total is clamped to it, so `InputTokens` is
  never negative and the buckets never exceed the prompt.

### Changed

- **Documented one cache-token contract for every adapter** (on
  `Response`, `StreamEvent`, `Usage`, `CostInput`): `InputTokens`,
  `CacheReadTokens` and `CacheCreationTokens` are disjoint — `InputTokens` is
  the uncached input only and total prompt = their sum (Anthropic's native
  shape). Callers that read `Response.InputTokens` as the *total* prompt size
  for Gemini/OpenAI must now add `CacheReadTokens`.
- `EstimateCost`: a model with no registered `CacheReadPricePer1M` now bills
  cache-read tokens at `InputPricePer1M` instead of $0, so moving cached tokens
  out of `InputTokens` is cost-neutral for models without a cache discount
  (most Gemini/OpenAI/DeepSeek entries) and cached input is never free. A zero
  `CacheWritePricePer1M` still means no write surcharge.

## [0.9.3] - 2026-10-07

### Fixed

- **`EstimateCost` returned a negative cost for extreme token counts (C4).**
  `OutputTokens + ThinkingTokens` was summed as `int` before the float
  conversion, so `math.MaxInt` in both wrapped to -2 and priced at about
  -3e-05 USD. Every token count is now converted to `float64` before any
  arithmetic. The result is guaranteed finite, `>= 0` and monotonic in every
  input: negative counts count as 0 (as before), a negative/NaN/±Inf
  registered price counts as unknown (0) instead of yielding a negative or NaN
  cost, and a total that would overflow (e.g. `math.MaxFloat64` audio seconds
  at a per-minute price above 60) saturates at `math.MaxFloat64` instead of
  `+Inf`.
- **`Usage.AddResponse` / `AddBatchResponse`** (and so `Conversation` and
  `Session` usage): running token totals saturate at `math.MaxInt` instead of
  wrapping negative, and the running `Cost` saturates at `math.MaxFloat64`
  (a negative or NaN `Cost` loaded from JSON is reset to 0 on the next add).
- **`ReportModalityUsage`**: repeated reports for one operation summed
  `AudioSeconds` to `+Inf`, which `EstimateCost` reads as "non-finite = 0",
  pricing more audio at $0 in `UsageEvent.CostUSD`. The sum now saturates at
  `math.MaxFloat64`.
- **Gemini modality usage**: candidates + thinking tokens (and the AUDIO
  prompt share) that would overflow now saturate at `math.MaxInt` instead of
  silently dropping the thinking tokens.

### Tests

- `cost_overflow_break_test.go`: `TestEstimateCostMaxIntInEachField`,
  `TestEstimateCostSummedFieldsDoNotWrap`,
  `TestEstimateCostNegativeInputsClampToZero`,
  `TestEstimateCostAudioSecondsHostile`, `TestEstimateCostHostilePrices`,
  `TestEstimateCostZeroPriceAndUnknownModel`, `TestUsageAccumulationSaturates`,
  `TestSessionUsageSaturatesAcrossTurns`, `TestUsageHookGeminiThoughtsSaturate`,
  and the fuzz target `FuzzEstimateCostMonotonic` (cost finite, `>= 0`, and
  non-decreasing when any single input is raised).
- `usage_overflow_internal_test.go`: `TestReportModalityUsageSaturatesSums`.

## [0.9.2] - 2026-10-07

### Added

- **`Config.UsageHook`** (`func(UsageEvent)`, not serialized): one
  `UsageEvent` per provider attempt of every modality operation
  (transcription, embeddings, image generation, speech) on clients from
  `NewModalityClient` / `NewFallbackModalityClient`, failed attempts included
  with `Err` set. Fields: `Operation`, `Provider`, `Model`, `InputTokens`,
  `OutputTokens`, `AudioInputTokens`, `AudioSeconds`, `CostUSD`, `Latency`,
  `Err`, `Attempt`, `Fallback`. `CostUSD` is estimated from registry list
  prices and is 0 when a price is unknown. A panicking hook is recovered and
  never changes the call's result. Requests rejected by validation before
  dispatch are not reported. Lets applications show a live cost display and
  the cost of each task.
- **Failover is visible**: `NewFallbackModalityClient` reports each attempt
  with its chain position (`Attempt` 0-based, `Fallback` true for every
  deployment but the primary), so a failover is two events. Each deployment
  reports to its own `UsageHook`; a fallback without one uses the primary's.
  Closes the v0.9.1 gap that failovers were neither logged nor counted.
- **Usage capture**: Gemini transcription and image generation read
  `usageMetadata` (prompt, candidates + thinking, `promptTokensDetails` AUDIO
  share), which the transcription path previously discarded. OpenAI-compatible
  transcription reads `usage` (`type: "tokens"` with `input_token_details.
  audio_tokens`, or `type: "duration"` with `seconds`); xAI `/v1/stt` reads
  `duration`; OpenAI-compatible embeddings report `prompt_tokens`. Malformed,
  negative or overflowing usage yields zero tokens and never fails the call.
- `ReportModalityUsage(ctx, ModalityUsage)` for adapters (including custom
  `ModalityDriver`s) to report provider usage; `ModalityOperation` constants.
- **Audio pricing**: `ModelInfo.AudioInputPricePer1M`,
  `ModelInfo.AudioPricePerMinute`, and `CostInput.AudioInputTokens` /
  `CostInput.AudioSeconds` in `EstimateCost` (audio tokens are never billed at
  the text rate when the audio rate is unknown). Prices set from the providers'
  pricing pages, fetched 2026-10-07: Gemini (ai.google.dev/gemini-api/docs/pricing)
  `gemini-3.5-transcribe` $2.00/1M audio in + $12.00/1M out,
  `gemini-3.5-flash-lite` $0.30, `gemini-3.5-flash` $1.50,
  `gemini-3.1-flash-lite` $0.50, `gemini-3-flash-preview` $1.00,
  `gemini-2.5-flash` $1.00, `gemini-2.5-flash-lite` $0.30 per 1M audio tokens;
  OpenAI (developers.openai.com/api/docs/models, /pricing) `gpt-4o-transcribe`
  $2.50/$10.00, `gpt-4o-mini-transcribe` $1.25/$5.00 per 1M,
  `gpt-transcribe` $0.0045/min, `whisper-1` $0.006/min; xAI
  (docs.x.ai/developers/models) `grok-voice-transcribe-2.0` $0.10/hour.

### Tests

- `usage_hook_break_test.go`: hook panic leaves result and error unchanged;
  nil hook; failed attempt reported with `Err` and zero cost; billed failure
  (MAX_TOKENS) keeps its cost; failover emits Attempt 0/1, Fallback
  false/true; per-deployment hooks; hostile Gemini `usageMetadata` (absent,
  null, wrong types, 1e300, negative, audio > input, MaxInt); hostile OpenAI /
  xAI usage shapes; unknown price → 0 and NaN/Inf/negative seconds; pinned
  registry audio prices; embeddings usage; no event for validation errors;
  40 concurrent failing-over calls under `-race` with no cross-call usage
  bleed; `Config` JSON round trip drops the hook. Mutation-checked: each of
  24 guards, when removed, turns its test red.

## [0.9.1] - 2026-10-07

### Added

- **`NewFallbackModalityClient(primary, fallbacks...)`**: model/provider
  failover for every modality operation (transcription, embeddings, images,
  speech). A later deployment is tried only when the previous one failed at the
  provider (an `APIError` of any status, or a transport error). Validation errors
  and caller cancellation are returned without another call. Every deployment
  except the last runs with retries disabled, so failover is immediate. Without
  that, a single 503 kept a request in the default backoff for 2 min 22 s
  (measured: ten attempts). Each request's `Model` is rewritten to the
  deployment being tried. When all deployments fail, the error wraps the LAST
  attempt, so `IsRateLimited`, `IsAuthError` and `errors.As` classify the final
  outcome; earlier attempts are kept as text.
  Motivation: on 2026-10-07 `gemini-3.5-transcribe` began rejecting every
  request with HTTP 400 ("Thinking is not enabled for this model"), even with no
  generation config, while `gemini-3.5-flash-lite` kept working.

### Tests

- Break tests for the fallback client: the 400 outage, 503/429 failover without
  backoff, no second call for invalid input or cancellation, three-deep chain
  order, last-attempt classification, invalid deployments failing construction,
  and concurrent use with double `Close` under `-race`. Mutation-checked (primary
  retries, missing model rewrite, first-attempt classification each turn a test
  red).

## [0.9.0] - 2026-10-06

### Added

- **`TranscriptionRequest.Vocabulary`**: provider-neutral recognition biasing.
  Dedicated Gemini transcription models receive it as `customVocabulary`,
  Gemini chat models in the instruction, OpenAI-compatible models in Whisper's
  `prompt`, and xAI as repeated `keyterm` fields. Bounded to 1,000 non-blank,
  single-line terms of at most 100 runes, validated before dispatch.
- **Gemini 3.5 Transcribe** (`gemini-3.5-transcribe`), Google's dedicated
  speech-to-text model. It takes audio only: language and vocabulary travel in
  `generationConfig.audioTranscriptionConfig`, the transcript comes back in
  `audioTranscription` parts, and a free-text `Prompt` is rejected. Live-verified
  on German speech with an API key; with vocabulary it fixed dialect terms the
  bare model misheard.
- **xAI speech-to-text** on the OpenAI-compatible adapter: `POST /v1/stt`
  (xAI's endpoint is not OpenAI-shaped), vocabulary as `keyterm`, free-text
  prompt rejected. Registry: `grok-voice-transcribe-2.0`.
- Registry entries for OpenAI `gpt-transcribe`, `gpt-4o-transcribe` and
  `gpt-4o-mini-transcribe` (transcription-only capability).
- `TranscriptionVocabularyHint` helper for adapters that take free text.

### Fixed

- **Truncated Gemini transcripts returned as success.** A candidate that
  finished with `MAX_TOKENS`, `SAFETY` or `RECITATION` but carried partial text
  was returned as a complete transcript. Any finish other than `STOP` is now an
  error and no partial text is returned.
- **OpenAI-compatible transcription: missing `text` field read as silence.** A
  200 body without `text` (an error object, a wrong shape, `null`) returned an
  empty transcript with no error. This predates 0.8.0 (Whisper path). A missing
  field is now an error; an explicit `""` is still genuine silence.
- **Control characters in vocabulary terms.** Only CR/LF were rejected; CR-only
  splits, NUL, tab, ESC, DEL, NEL and U+2028/U+2029 reached provider form fields
  and instructions. Every Unicode control character and line/paragraph separator
  is now rejected before dispatch.

### Security

- **API keys echoed in provider error bodies are redacted.** `ErrorFromResponse`,
  which every adapter uses to turn a non-2xx response into an `APIError` (chat,
  modality, batch and file calls on Anthropic, Gemini, OpenAI-compatible and
  Responses), built the error from the raw body. A provider that echoed the
  request key put it into the returned error, which callers log or display;
  only errors passing through the pooled client's own scrub were protected. The deployment's `APIKey` and BaseURL secrets are now scrubbed with
  the existing `redactProfileSecrets` before the error is constructed.

### Tests

- Break-the-system suites for transcription: dispatch-prevention on every
  malformed input (Gemini, OpenAI, xAI), mid-stream disconnects, response-body
  cap, error classification (429/401/403/503) with key-echo checks, per-request
  model overrides validated as the target model, cancellation, concurrent use of
  one client under `-race`, double `Close`, and two fuzz targets
  (`FuzzAudioMediaTypeFromSignature`,
  `FuzzTranscriptionValidationNeverAdmitsBreakingInput`). Each fixed bug's test
  was confirmed red before the fix.

### Changed

- Registry thinking-flag tests treat transcription-only Gemini models like the
  other modality-only models (no thinking, no thought signatures).

## [0.8.0] - 2026-10-06

### Added

- **Gemini transcription.** `TranscribeAudio` now works on the Gemini protocol:
  audio is sent inline to `generateContent` with a verbatim-transcript
  instruction, so any Gemini chat model transcribes with the existing API key.
  Gemini chat models advertise `CapabilityTranscription` (dedicated embedding and
  image models do not). Input is validated before dispatch: supported media type
  (WAV, MP3, AIFF, AAC, OGG, FLAC, WebM, MP4/M4A), declared type must match the
  byte signature, 14 MiB inline cap under Gemini's 20 MB request limit, and a
  `de`/`de-AT` style language tag (anything else is rejected rather than
  interpolated into the instruction). Thought parts are dropped; a non-`STOP`
  finish with no text is an error, while silence returns an empty transcript.
  Verified live against `gemini-3.5-flash-lite` (German audio, 1.1–1.5 s).
- **`TranscriptionRequest.Prompt`**: an optional vocabulary/context hint, bounded
  to `MaxTranscriptionPromptRunes` (1000) runes and valid UTF-8 for every
  provider. Gemini places it after a guard sentence in the instruction;
  OpenAI-compatible providers forward it as Whisper's `prompt` field. Live, the
  hint fixed dialect words a bare transcription got wrong ("dünne" heard as
  "Döner").
- **`AudioMediaTypeFromSignature`** in the root package, shared by the Gemini and
  OpenAI-compatible drivers. It now also recognizes ADTS AAC (previously
  misclassified as MP3) and AIFF/AIFC.

## [0.7.7] - 2026-09-19

Ships the unreleased 0.7.6 work (never tagged publicly) together with a model
registry refresh. See the 0.7.6 section below for the Gemini tool-call,
thinking-budget, max_tokens-floor and retired-model changes it contains.

### Added

- **Gemini 3.8 Flash and 3.7 Flash** (`gemini-3.8-flash`, `gemini-3.7-flash`).
  Both are live on the Gemini API but were missing from the registry, so a
  request naming one failed with "no reviewed capability metadata". Verified
  against ai.google.dev/gemini-api/docs/pricing and the API's own
  `models/{id}` metadata (2026-09-19): 1,048,576-token context, 65,536 max
  output, thinking supported, $0.75 in / $3.75 out per 1M tokens with context
  caching at $0.075 (promotional pricing through 2026-12-31; it doubles to
  $1.50 / $7.50 / $0.15 on 2027-01-01 — revisit then).
- **Claude Fable 5.1** (`claude-fable-5-1`), Anthropic's most capable widely
  released model. 1M context, 128K max output, $10 in / $50 out, cache reads at
  $0.25/MTok. Same tier and per-token price as Fable 5.
- `geminiWithoutSamplingControls`, replacing a hardcoded two-model comparison
  in `supportsSamplingTemperature`. Google removed sampling controls from the
  July 2026 GA models onward, and the old check said so in a comment while
  only naming `gemini-3.6-flash` and `gemini-3.5-flash-lite` — so 3.7 and 3.8
  would have advertised a temperature parameter their endpoints reject.
- Guard tests (`registry_gemini_sampling_internal_test.go`) that fail when a
  Gemini chat model is registered without a sampling-controls decision, when
  the deny list names an unregistered ID, or when the list stops driving
  `supportsSamplingTemperature`. Verified non-vacuous by removing
  `gemini-3.8-flash` from the list and confirming the suite goes red.

### Changed

- Refreshed the Anthropic and Gemini registry snapshot dates to 2026-09-19.

## [0.7.6] - 2026-09-18

### Added

- **Mutation-audited test coverage for three previously unpinned invariants.**
  Flipping each in production code left the entire suite green, so the guarantee
  was documented but unenforced:
  - the **TLS 1.2 floor** (`SafeHTTPClient` and `NewSafeHTTPClient`) — lowering it
    to TLS 1.0 went unnoticed, despite "TLS 1.2+" being an advertised guarantee;
  - the **redirect hop cap** — raising it from 10 to 1000 went unnoticed, leaving
    clients walkable through an unbounded redirect chain;
  - the Gemini **`TokensNotReported` sentinel** — initializing usage counters to
    `0` instead of `-1` went unnoticed, collapsing "provider omitted usage" into
    "this turn was free" and under-reporting accumulated spend.
- Adversarial regression suites for every v0.7.6 fix
  (`break_v076_test.go`, `provider/gemini/break_stopreason_internal_test.go`,
  `provider/anthropic/break_thinking_internal_test.go`): hostile Gemini
  candidates, a 68-case thinking-budget boundary sweep, concurrent retirement
  lookups under `-race`, and negative/zero config values. Each suite is verified
  non-vacuous by reverting its fix and confirming the tests go red — an audit
  that caught the Gemini suite asserting only that `tool_use` is never invented
  (which passes with the fix removed); it now also asserts that a real function
  call *is* reported, across single, parallel, text-prefixed and thought-
  prefixed turns on both the buffered and streaming paths.
- Documented the Gemini adapter's `maxOutputTokens` padding for native thinking
  models, which can make a response longer than the `MaxTokens` the caller set.
- `RetiredModelReplacement` reports whether a model ID has been sunset by its
  provider and names the registered replacement. Registry lookup is fail-closed,
  so a retired ID now fails at dispatch with an actionable error
  (`model %q was retired by its provider; use %q instead`) instead of the generic
  "no reviewed capability metadata" message.

### Changed

- Removed seven Anthropic model IDs the provider no longer serves, each verified
  as HTTP 404 against the live models endpoint: `claude-3-haiku-20240307`,
  `claude-opus-4-0`, `claude-opus-4-1`, `claude-opus-4-1-20250805`,
  `claude-opus-4-20250514`, `claude-sonnet-4-0`, and `claude-sonnet-4-20250514`.
  `claude-sonnet-4-5`, `claude-opus-4-5` and `claude-haiku-4-5` remain registered
  — they are unlisted aliases the API still resolves. Callers that need a retired
  ID can re-add it with `RegisterModel`.

### Fixed

- **Gemini tool calls were silently dropped.** Gemini reports `finishReason:
  "STOP"` even for a turn that requests a function call, so the adapter mapped it
  to `end_turn`. The documented agentic loop (`for resp.StopReason == "tool_use"`)
  never ran and callers saw an empty response with the tool call discarded. Both
  the non-streaming (`parseResponse`) and streaming (`parseStream`) paths now
  infer `tool_use` from the emitted function calls; `MAX_TOKENS` and other
  terminal reasons are preserved, since a truncated turn is not a usable call.
- **The `MaxTokens` floor now reaches every constructor.** `NewBatchClient` and
  `NewModalityClient` applied the `Timeout` floor but skipped `MaxTokens`, and
  the Anthropic batch encoder reuses the live request builder — so every batch
  entry went out as `"max_tokens": 0` and was rejected exactly like the
  single-request path. All three constructors now share one `applyConfigFloors`
  normalizer, so a floor can no longer be added to one path and forgotten on the
  others.
- **`Config.MaxTokens` zero value produced invalid requests.** `MaxTokens: 8192`
  existed only in `DefaultConfig()`, so a `Config` built as a struct literal — the
  form the README and every example use — sent `max_tokens: 0` and Anthropic
  rejected it with HTTP 400 (`max_tokens cannot be 0`). `NewClient` now applies a
  `DefaultMaxTokens` floor alongside the existing `Timeout` floor, and
  `DefaultConfig()` sources the same constant so the two cannot drift.
- **Extended thinking rejected whenever `MaxTokens` was below the model
  ceiling.** The Anthropic adapter clamped `thinking.budget_tokens` to the
  model's registry ceiling but not to the request's own `max_tokens`, so
  `ThinkingHigh` (65536) with `MaxTokens: 16000` sent `budget_tokens: 64000` and
  the API rejected it with HTTP 400 (``max_tokens` must be greater than
  `thinking.budget_tokens``). The budget is now also clamped to the effective
  `max_tokens`, reserving a quarter of it for the visible answer, and respects
  Anthropic's 1024-token minimum — when `max_tokens` is too small to satisfy
  both bounds, thinking is disabled with a warning rather than sending a request
  the API would reject.
- `examples/cache_anthropic` requested `claude-sonnet-4-20250514`, retired by
  Anthropic and returning HTTP 404; it now uses `claude-sonnet-4-6`, matching
  `examples/streaming`.

## [0.7.5] - 2026-09-17

### Added

- `RetryEventType.String` exposes stable retry lifecycle labels for logs and
  metrics (`attempt_failed`, `rotating`, `backing_off`, `circuit_open`, and
  `exhausted`).

### Fixed

- OpenAI-compatible image generation omits `response_format` for GPT Image
  models, whose API rejects that legacy parameter and already guarantees
  base64 output. Other compatible image endpoints still request `b64_json`.
- Shipped examples now use a reviewed failover model, cap tool-loop duration and
  rounds, and give Gemini cache create/delete calls deadlines, bounded response
  reads, and strict response validation.
- Corrected all defects found by compiling and locally exercising every README
  Go fence: stale aliases, the misplaced `ResponseFormat` config entry, invalid
  strict JSON Schema, incomplete local/custom provider recipes, ignored errors,
  duplicate logging wrappers, OAuth config ambiguity, and streaming prose.
- Corrected the embedded cache-review prompt's claim about `http.Get`: the
  default client reuses its transport but does not impose a request timeout.

## [0.7.4] - 2026-09-17

### Fixed

- Sonnet 5 uses adaptive thinking and effort instead of the removed manual
  thinking-budget format. Explicit manual budgets with reasoning enabled fail
  before transport, and custom temperature is no longer advertised. The effective
  `ThinkingNone` explicitly disables thinking on this model.
- OpenAI Responses streams recognize `response.incomplete` and `response.failed`
  as terminal events instead of misclassifying them as unexpected EOF. Both
  completion paths preserve `RawStopReason` from incomplete details or status.
- Batch request codecs preserve `SamplingParams` and enforce the same key
  protections as live completion requests across all four protocols.
- `MergeSamplingParams` rejects JSON-null request bodies (including typed nil
  pointers and maps) instead of panicking when assigning a passthrough key.
- Batch cancellation regression coverage now cancels after the first successful
  poll has been decoded, removing a timing assumption that intermittently failed
  the race-enabled CI suite.

### Changed

- README minimum Go version matches `go.mod` (1.26.8), describes `ToolChoice`
  and `SamplingParams`, and corrects stale streaming-completion and temperature
  override guidance.

## [0.7.3] - 2026-09-17

### Added

- **`Request.ToolChoice`** — provider-neutral tool choice (`auto` / `none` / `required` /
  a named tool), wired through all four wire protocols and translated to each one's own
  shape: OpenAI's `"required"`, Anthropic's `{"type":"any"}`, Gemini's
  `toolConfig.functionCallingConfig` mode `ANY` (+ `allowedFunctionNames` to force one tool),
  and the Responses API's flatter `{"type":"function","name":…}`. Expressed once, so a forced
  tool call survives a `SwitchProvider` handoff. A malformed choice (mode `tool` with no name,
  an unknown mode, a name on a mode that takes none) fails at the library boundary instead of
  as an opaque provider 400, and a constraint is never sent without tools.
- **`Request.SamplingParams`** — a `map[string]any` merged into the request body at the top
  level, for provider-specific knobs the neutral `Request` does not model (OpenRouter's
  `provider` routing object, vLLM's `priority`, `top_p`/`seed`). **Keys that would hijack the
  request are refused, not silently applied**: anything the adapter already set, plus a
  reserved list of structural fields (`model`, `messages`, `tools`, `system`, `contents`,
  `input`, …) refused even when absent from that particular body — so a stray `model` key
  cannot redirect a request to a different model, and `messages` cannot rewrite its history.
- **`Response.RawStopReason` / `StreamEvent.RawStopReason`** — the provider's own stop reason,
  preserved verbatim alongside the normalized one, so a normalized `max_tokens` can be traced
  back to OpenAI's `length` or Gemini's `MAX_TOKENS`. When a stop reason is *synthesized*
  (a `[DONE]` with no `finish_reason`), the raw field is left empty rather than fabricating
  provenance the provider never sent.

### Fixed

- **`claude-sonnet-5` was priced 50% too high** — $3.00/$15.00 against Anthropic's published
  $2.00/$10.00. The introductory price became standard; the scheduled 2026-09-01 rise to
  $3/$15 was cancelled. It is the `anthropic`/`claude` default, so every out-of-the-box cost
  estimate was wrong. Max output also corrected to 128K.
- **Groq `openai/gpt-oss-120b` was priced 20x too high** — $3.00/$8.00 against a published
  $0.15/$0.60, on the `groq` default model. `openai/gpt-oss-20b` likewise ($0.30/$0.80 →
  $0.075/$0.30). Context/max-output corrected to 131072/65536.
- **`DefaultConfig().Model` had drifted from the registry** — it returned `claude-sonnet-4-6`
  while `GetDefaultModel("anthropic")` returned `claude-sonnet-5`. It now reads from the
  registry, so the two cannot diverge again.

### Security

- **Minimum Go bumped 1.26.5 → 1.26.8.** v0.7.1 moved to 1.26.5 for GO-2026-5856, but
  `govulncheck` still reported four reachable Go standard-library vulnerabilities on that
  toolchain, all fixed in 1.26.6: GO-2026-6218 (`net/url`), GO-2026-6090 (`crypto/tls`),
  GO-2026-5972 (`encoding/asn1`) and GO-2026-5026 (`net/http`). They are reachable from the
  HTTP paths every adapter uses, so `make ci` was red on `main`. No rho-llm code changed.

### Changed

- CI workflow: `actions/checkout@v4` → `@v7`, `actions/setup-go@v5` → `@v6`
  (Node 24 runtime; clears the Node 20 deprecation warning on GitHub Actions).
- CLAUDE.md: added a fetch-first rule for releases — an agent's git state is a session-start
  snapshot, not a live view of the remote, and this repo takes releases from several people.

> **Note:** the code in this entry was tagged `v0.7.2`; this CHANGELOG section and the
> `docs/ARCHITECTURE.md` stamp followed one commit later and are tagged `v0.7.3`. `v0.7.2`
> was already cached by the Go module proxy, so the tag was left where it was rather than
> moved — a published tag is immutable.

## [0.7.1] - 2026-08-07

### Fixed

- Bump module Go version `1.26.4` → `1.26.5` so `govulncheck` (and CI via
  `go-version-file: go.mod`) pick up the stdlib fix for GO-2026-5856
  (`crypto/tls` Encrypted Client Hello privacy leak). Direct deps unchanged
  (`godotenv` already at latest stable `v1.5.1`).

## [0.7.0] - 2026-08-06

### Added

- **Multi-vendor BatchClient drivers** on the same public `Submit`/`Get`/`Results`/
  `Cancel` surface as OpenAI:
  - Anthropic Message Batches (`provider/anthropicbatch`, protocol `anthropic`) —
    completion items only; 24h max turnaround; results JSONL; `SupportsBatch` on
    anthropic/claude presets.
  - Gemini Batch (`provider/geminibatch`, protocol `gemini`) — completion and
    embedding items via inline batch jobs; 24h max turnaround; `SupportsBatch` on
    gemini/google presets. Live Gemini returns `BATCH_STATE_*` (also accepts
    documented `JOB_STATE_*` aliases).
- **Gemini modality driver** (`RegisterModalityDriver("gemini")`): embeddings
  (`embedContent`) and image generation (`generateContent` + IMAGE modality).
  Speech/transcription remain unsupported (fail closed). Registry entries for
  `gemini-embedding-001`/`gemini-embedding-2` and Gemini image models.
- **OpenAI Realtime public surface** (`realtime.go`): `OpenRealtimeSession` with a
  injectable `RealtimeDialer`/`RealtimeConn`, neutral event types, OpenAI wire
  encode/decode (`input_audio_buffer.*`, `response.*`, errors), and offline state
  machine tests. Distinct from one-shot Complete/Stream.
- Model registry refresh (2026-08-06) for current first-party catalogs:
  - **Anthropic:** `claude-fable-5`, `claude-opus-5`, `claude-sonnet-5`, plus the
    `claude-haiku-4-5` alias ID; defaults/aliases (`sonnet`/`opus`/`claude`/`fable`)
    point at the Claude 5 family.
  - **OpenAI:** GPT-5.6 Sol/Terra/Luna (`gpt-5.6-sol`, `gpt-5.6-terra`,
    `gpt-5.6-luna`) with Responses API + pricing; `gpt-image-2`; default `gpt` → Sol.
  - **xAI:** `grok-4.5`, stable `grok-4.20-0309-*` / multi-agent / `grok-build-0.1`
    slugs (beta IDs retained); default `xai`/`grok` → `grok-4.5`.
  - **Meta Model API:** first-class `meta` provider (`https://api.meta.ai/v1`) with
    `muse-spark-1.2`, `muse-spark-1.1`, and `muse-spark-1.2-contributor` (standard
    vs contributor pricing). Documented provider count 23 → 24.
  - **Mistral:** Pixtral vision models + `mistral-embed` / `mistral-large-latest` /
    `codestral-latest`; embeddings proven via openai_compat `/embeddings`.
  - **Chinese hosts:** expanded Z.ai/GLM, MiniMax, Moonshot/Kimi, DashScope/Qwen,
    DeepSeek V4 flash/pro; DashScope `text-embedding-v2/v3`; regional presets
    `dashscope-cn`/`qwen-cn` and `moonshot-cn`/`kimi-cn` for mainland bases.
    DeepSeek embeddings remain fail-closed (no first-party vector API).
- Optional live smokes (`live_smoke_new_test.go`, skipped under `-short` / missing
  keys) for Anthropic/Gemini/OpenAI batch, Gemini embeddings + image gen,
  Anthropic `WaitForBatch`, and OpenAI Realtime.

### Fixed

- gosec G104 in shared `DoHTTP` retry path: drain-and-discard now discards
  `resp.Body.Close()` errors explicitly (`_ = resp.Body.Close()`), restoring a
  green `make ci` security gate after the v0.6.0 transport addition.
- Default chat capability metadata no longer under-declares features the adapters
  and docs already advertise: OpenAI gains `CapabilityDocumentInput`; xAI/Meta and
  other cloud OpenAI-compatible providers gain vision, document, and structured
  output; Anthropic/Gemini advertise batch where drivers exist. Local
  `ollama`/`vllm`/`lmstudio` stay chat-focused so deployments must declare extras.
  `RegisterModel` with a zero capability set now applies the same defaults as
  package init (instead of registering a model that always fails closed at dispatch).

## [0.6.0] - 2026-08-06

### Added

- Deployment-scoped transport and capability authority through
  `Config.ProxyURL`, `Config.DisableProxy`, and `Config.ModelCapabilities`.
  Public deployments can use one reviewed proxy without consulting ambient
  proxy variables; private/local deployments can explicitly bypass them. Exact
  per-config capability metadata takes precedence over the global model registry
  and cannot authorize another request model.
- Provider-neutral `CapabilityTemperature` validation. An explicit
  `Request.Temperature` now requires both protocol encoding support and reviewed
  model/deployment support before transport; nil retains the provider default.
  The Gemini catalog now uses stable `gemini-3.1-flash-lite`, adds GA
  `gemini-3.6-flash` and `gemini-3.5-flash-lite`, and removes the shut-down
  Gemini 3.1 Flash-Lite preview from defaults, aliases, and discovery. The two
  July GA models reject explicit temperature through capability validation
  because Google deprecates and ignores their sampling parameters.
- Evidence-based media output integrity. `GenerateImages` verifies decoded PNG,
  JPEG, or WebP signatures before assigning a media type; URL-only responses fail
  because exact base64 bytes were requested. `SynthesizeSpeech` now returns `*SpeechResponse` with bytes and a
  verified media type, validating provider Content-Type plus bounded MP3, Ogg
  Opus, AAC, FLAC, or WAV signatures (raw L16 requires an exact header). New pure
  `ValidateEmbeddingRequest`, `ValidateImageRequest`, `ValidateSpeechRequest`, and
  `ValidateTranscriptionRequest` APIs allow startup-time adapter/profile review
  without network I/O. Transcription rejects BCP47 language tags that cannot be
  represented as exact lowercase ISO-639-1 on the OpenAI-compatible wire.
- Registered non-chat modality adapters. The provider-neutral root now exposes
  `ModalityClient`, `NewModalityClient`, request/response contracts, and pure
  validation only. OpenAI-compatible endpoint construction, JSON/multipart
  shapes, format mapping, response decoding, and media signature verification
  live in `provider/openaicompat`; the deleted root standalone network functions
  are not retained as compatibility wrappers. A modality client keeps one safe
  HTTP transport for its worker lifetime and uses shared retry, proxy,
  cancellation, bounded-body, and classified-error policy.
- Strict input representation integrity. Image/PDF content requires canonical
  RFC 4648 base64 and a signature matching its declared MIME type; transcription
  validates FLAC, MP4/M4A, MP3, Ogg, WAV, or WebM signatures before dispatch.
  Exact output geometry/count/format, voice, and language remain
  application-owned request values rather than rho defaults.
- Speech output requests accept canonical MIME values only. Provider response
  aliases such as `audio/opus` and `audio/x-wav` are normalized during payload
  verification, but cannot create an admitted-request/returned-schema mismatch.
- Typed, fail-closed model capability metadata covering chat, streaming, tools,
  structured output, vision, document input, embeddings, batch, image generation,
  speech synthesis, transcription, sampling temperature, and enforceable reasoning effort.
  `CapabilityReasoning` deliberately excludes intrinsic reasoning that cannot honor
  a requested `ThinkingLevel`. `RequireCapabilities` and
  `ValidateRequestCapabilities` reject unknown or undeclared model combinations
  before dispatch; `RegisterModel` is the single extension point for reviewed local
  Ollama, vLLM, and custom OpenAI-compatible deployments.

### Changed

- Batch persistence is now provider-neutral and fail-closed. Exact schema-v2
  `BatchHandle` values retain a neutral external ID, operation, lifecycle, and
  bounded versioned opaque adapter state; OpenAI endpoint and Files API IDs are
  adapter-owned. `Get`, `Results`, `Cancel`, and `WaitForBatch` consume the full
  handle. `BatchItem.ItemID` replaces provider-named `CustomID`, lifecycle states
  are closed and normalized, and unknown provider states fail. Callers must set
  `BatchOptions.MaxTurnaround`; root selects no default, while OpenAI admits and
  maps exactly 24 hours and enforces its 50,000-request,
  50,000-total-embedding-input, and 200 MB JSONL limits
  before upload. There is no decoder or alias for the previous handle or item grammar.
- Modality and OpenAI Batch HTTP operations now share the library retry policy,
  retry hooks, caller cancellation, and proxy-aware safe HTTP transport. Credential
  rotation and circuit breaking remain available only through the chat `Client`
  auth pool; each `ModalityClient` represents one exact deployment credential.
- Gemini tool-result routing accepts only the current invertible
  `call_<index>_<name>` synthetic-ID grammar. Older ambiguous `call_<name>` IDs
  are no longer reinterpreted as function names.

### Removed

- The compatibility-only `Backoff` wrapper. Callers configure and invoke
  `RetryPolicy.Delay` directly so retry semantics have one public authority.

### Fixed

- Anthropic extended-thinking requests no longer silently rewrite an explicit
  sampling temperature. Pure capability validation rejects values other than
  the protocol-required `1` before transport; an omitted temperature remains
  omitted, and the adapter only translates the application-owned request.
- OpenAI Responses function tools now use the required top-level
  `{type,name,description,parameters}` shape instead of the Chat Completions
  `{type,function:{...}}` shape. Responses structured output is emitted through
  `text.format`.
- `SafeHTTPClient` preserves standard `HTTP_PROXY`/`HTTPS_PROXY` discovery while
  retaining TLS and cross-origin credential stripping; all chat, non-chat, and
  batch transports honor the explicit per-config proxy policy.

## [0.5.0] - 2026-06-24

### Added

- Three first-class OpenAI-compatible providers (presets + native registry entries), so they
  resolve, route, and cost-estimate without a custom `BaseURL`:
  - **Z.ai / GLM** (`zai`, aliases `z-ai`/`glm`) → `https://api.z.ai/api/openai/v1`; default
    model `glm-5.2` (1M context, $1.40/$4.40 per 1M, cache-read $0.26, MIT open weights).
  - **MiniMax** (`minimax`) → `https://api.minimax.io/v1`; default model `MiniMax-M3`
    (1M context, multimodal; native pay-go **standard** tier $0.60/$2.40 per 1M, cache-read
    $0.12; the >512K long-context tier is 2× and not modeled; open weights).
  - **Moonshot / Kimi** (`moonshot`, alias `kimi`) → `https://api.moonshot.ai/v1`; default
    model `kimi-k2.7-code` (262K context, $0.95/$4.00 per 1M, cache-read $0.16, always-thinking
    multimodal coder, modified-MIT open weights).
  - Adversarial test coverage in `harden_newproviders_test.go` (fail-closed mis-case, alias
    cross-map agreement, case-exact reverse map, cache/sentinel cost math, discovery integrity,
    cross-host auth-strip, concurrent alias resolution under `-race`).
  - Bumps the documented provider count 20 → 23 (README, ARCHITECTURE, CLAUDE.md).
- **Asynchronous Batch API** — a new `BatchClient` interface (the bulk counterpart to `Client`)
  for offline, ~50%-cheaper request processing. `NewBatchClient(cfg)` resolves a per-protocol
  batch driver via a parallel registry (`RegisterBatchProvider`, mirroring the `Client` driver
  pattern). First implementation: **OpenAI** (`provider/openaibatch`), covering
  `/v1/chat/completions`, `/v1/responses`, and `/v1/embeddings`. Each batch line is built and
  parsed by the **same wire-translation code** as the synchronous adapters (new
  `BuildChatBatchLineBody`/`ParseChatBatchResultBody` on `openaicompat`, the responses
  equivalents on `openairesponses`, and the embeddings codecs on
  `openaicompat`) — a parity test asserts the batch line
  body is byte-identical to what `Complete` POSTs. The interface is provider-agnostic so
  Anthropic Message Batches and others can register later; transport differences (OpenAI's
  Files-API upload/download vs. Anthropic's inline submit) live inside each driver.
  - `BatchItem` (a `custom_id` + a chat `Request` **xor** an `EmbeddingRequest`), `BatchResult`,
    `BatchOptions`, `BatchStatus`, and a serializable, versioned `BatchHandle`
    (`schema_version`/`LoadBatchHandle`) so a batch can be submitted, persisted, and polled after
    a process restart. `Endpoint` on the handle drives result parsing on resume (the items are
    gone). `WaitForBatch` offers caller-controlled polling that honors `context` cancellation.
  - Cost: `CostInput.Batch` applies the 50% batch discount in `EstimateCost`, and
    `Usage.AddBatchResponse` folds batch results at batch pricing (token clamping shared with
    `AddResponse`).
  - Config: `ProviderPreset.SupportsBatch` gates batch availability (true for OpenAI only —
    most `openai_compat` resellers do not expose `/v1/batches`); `Config.MaxBatchDownloadBytes`
    (default 256 MB) caps result-file downloads, far above the 32 MB sync-response cap so real
    batches are not silently truncated.
  - Two further adversarial hardening rounds on the merged batch code closed the coverage holes
    the original suite left at 0% — `WaitForBatch` (poll-to-terminal, ctx-cancel mid-poll, Get-error
    propagation), `BatchStatus.Terminal`, `Usage.AddBatchResponse` (half-cost + `TokensNotReported`
    clamp), the full `/v1/responses` and embeddings **Submit** paths, the upload/create/download/cancel
    non-2xx paths, batch-level `errors.data[]` surfacing, the exact oversize-download boundary
    (cap vs cap+1), a newline-less final line, a >64 KB single line, cross-file `custom_id` dedup, and
    hostile-body decode errors. No defect found (the merged code was correct); the off-by-one
    download cap, the `Terminal` set, and the 50% discount were each proven red without their logic.
    Added `batch_harden_test.go`, `batch_harden2_test.go`, `provider/openaibatch/batch_harden_internal_test.go`.

### Changed

- Registry: refreshed the Fireworks block (snapshot 2026-06-22) with three new open-weight
  releases, tracked under the Fireworks host (matching the existing `kimi-k2p6` precedent),
  prices cross-checked against each provider + Together AI:
  - **Kimi K2.7 Code** (`…/kimi-k2p7-code`, Moonshot, released 2026-06-12, modified-MIT) —
    262K context, $0.95/$4.00 per 1M, thinking-capable multimodal coder.
  - **GLM-5.2** (`…/glm-5p2`, Z.ai, released 2026-06-13, MIT) — 1M context, $1.40/$4.40 per 1M,
    long-horizon reasoning/coding model.
  - **MiniMax M3** (`…/minimax-m3`, released 2026-06-01, open-weight) — 1M context, multimodal,
    $0.30/$1.20 per 1M (Fireworks host rate).

### Fixed

- **Per-request model aliases now resolve before the wire** (openai_compat adapter). `NewClient`
  resolved `cfg.Model` at construction, but a per-call `Request{Model: "glm"}` / `"minimax-m3"` /
  `"kimi"` override shipped the alias verbatim to the provider, which only recognizes the canonical
  ID — silently breaking the request. `buildRequest` now applies `ResolveModelAlias` (unknown IDs
  pass through untouched). Found by the break-rounds campaign; pinned by
  `TestPerRequestAliasResolvesToCanonicalWireModel`.
- **Auth-pool credential leak: per-key `apikey|baseurl` BaseURL secrets are now scrubbed from error
  text.** `redactErr`, `AuthProfile.MarkFailed`, and `AuthProfile.MarshalJSON` previously redacted
  only the APIKey, so userinfo (`user:pass@`) or a `?token=`/`?key=` secret embedded in a per-key
  BaseURL override leaked into logs, `LastError`, and serialized JSON whenever an upstream error
  echoed the URL (the BaseURL *field* was already redacted — this closes the matching gap in the
  free-text paths). New shared helper `redactProfileSecrets` keeps the host/path visible for
  debugging. Pinned by `TestAuthProfileDoesNotLeakBaseURLCredsInLastError` and
  `TestRedactErrStripsBaseURLCredentials`.

## [0.4.16] - 2026-06-18

### Changed

- **No local variable shadows an imported package** — the Gemini adapter named its request-URL
  local `url`, shadowing the imported `net/url` package; renamed to `endpoint` there and, for
  consistency, in the OpenAI-compatible / Responses adapters and the `cache_gemini` example.
  No behavior change.

## [0.4.15] - 2026-06-17

Hardening pass 14 — verified `EstimateCost` across all five token types (input/output/thinking/
cache-write/cache-read) and the `TokensNotReported (-1)` sentinel clamp; cost is correct,
finite, and never negative (no defect; regression coverage).

## [0.4.14] - 2026-06-17

Hardening pass 13 — verified `Conversation` JSON serialization round-trips losslessly for
hostile content (unicode, emoji, embedded control chars incl. NUL, every content kind); no
defect (regression coverage).

## [0.4.13] - 2026-06-17

Hardening pass 12 — verified the `apikey|baseurl` auth-pool split and `ResolveAuthHeader`/
`ResolveBaseURL` resolution handle edge inputs correctly (no defect; regression coverage). The
adversarial attack surface is exhausted on an evidence basis (the last two passes probed fresh
surfaces and found nothing).

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

- **`.mailmap`** added — unifies two `Rene Heinzl` commit identities for `git shortlog`/blame.

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
- **Gemini 3.1 Flash-Lite preview** — Pricing corrected from $0/$0 to $0.25/$1.50. Added `ThoughtSignature: true` (required for all Gemini 3.x models).
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

- **Module path migration** — Migrated from the former internal module path to `github.com/bds421/rho-llm`.

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
