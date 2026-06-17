# Hardening coverage log

Test command: `make ci` (fmt + vet + build + test-race + gosec + govulncheck).
Version: CHANGELOG `[X.Y.Z]` section + annotated `vX.Y.Z` tag + `docs/ARCHITECTURE.md` stamp.

## Already hardened (prior break-rounds campaigns 1 & 2, before this log)
- Resilience: circuit breaker (probe identity, single-owner, wrap), pool Close/rotate race, retry/backoff, cooldown clamps.
- Streaming: all 4 adapters' parseStream (truncation/EOF, tool-call flush, trailing garbage, stop reasons, empty-data lines).
- Persistence: FileStore atomicity/durability/traversal, MemoryStore, Conversation serialization, Clone (deep + circular guard).
- Handoff: NormalizeForProvider (thinking degrade, tool-id dedup/pairing, signature stripping incl. ThoughtSignature).
- Secrets: APIKey/LastError/BaseURL redaction in logs + MarshalJSON.
- Registry: RegisterModel/Alias validation + discovery sync, EstimateCost alias resolution.
- Request-building: nil tool schema/input coercion across adapters.

## Hardened by this log
- **Pass 1 (v0.4.2):** session cost/model provenance. Found & fixed a serious wrong-result
  defect — a custom `Client` returning an empty `Response.Model` made `EstimateCost("")`
  return `$0`, silently under-reporting cost. Fix: `SendMessages` attributes the turn to the
  model it ran against. Test: `TestSessionCostFallsBackToRequestModelWhenResponseModelEmpty`.

- **Pass 2 (v0.4.3):** `oauth.go` device flow. Found & fixed a serious hang — `PollDeviceToken`
  ignored `expires_in` and could poll forever against a misbehaving server with no ctx deadline
  (contradicting its own doc). Also clamped an overflow-prone poll interval and surfaced the
  RFC `error_description`. Tests: `TestPollDeviceTokenStopsAtExpiry`, `TestClampPollInterval`,
  `TestStartDeviceAuthSurfacesErrorDescription`.

- **Pass 3 (v0.4.4):** `capabilities.go` HTTP body integrity. Found & fixed silent data
  corruption — `doRaw` swallowed the body-read error, so a truncated response (connection drop /
  Content-Length mismatch) returned partial bytes as success (worst for `SynthesizeSpeech` raw
  audio). Test: `TestSynthesizeSpeechRejectsTruncatedBody`.

- **Pass 4 (v0.4.5):** `middleware.go` LoggingClient. Found & fixed a crash — `Complete`
  dereferenced a `(nil,nil)` inner response while logging metadata. Test:
  `TestLoggingClientDoesNotPanicOnNilResponse`.

- **Pass 5 (v0.4.6):** SSRF guard (`config.go CheckBaseURL`). Found & fixed a security bypass —
  a trailing-dot host (`localhost.`, `127.0.0.1.`) evaded the loopback/private checks yet
  resolves to loopback. Test: `TestCheckBaseURLBlocksTrailingDotBypass`.

- **Pass 6 (v0.4.7):** adapter `buildRequest` message integrity. No serious bug. Fixed a
  non-serious gap — the Anthropic adapter emitted an empty-content message as `"content":null`
  (Anthropic rejects it); now dropped (Gemini already skipped empties). Test:
  `TestAnthropicBuildRequestDropsEmptyContentMessage`.

- **Pass 7 (v0.4.8):** OpenAI-family `buildRequest` empty-message. No defect — both adapters
  already avoid a null-content message; confirming test added
  (`TestOpenAIFamilyBuildRequestDropsEmptyContentMessage`). buildRequest empty-message area now
  fully covered across all 4 adapters.

- **Pass 8 (v0.4.9):** structured output. No serious bug. Fixed a non-serious edge — a
  `json_schema` format with no schema emitted `"schema":null`; now degrades to json_object.
  Test: `TestOpenAICompatJSONSchemaWithoutSchemaDegrades`.

- **Pass 9 (v0.4.10):** `ValidateImageSource` direct coverage (no defect). Meaningful
  untested behavior assessed as exhausted; `untested_surface_remaining=false`. Remaining code
  (simple.go wrappers, discovery/factory) is trivial or transitively tested. Test:
  `TestValidateImageSourceRejectsMalformed`.

- **Pass 10 (v0.4.11):** gemini URL construction. Found a low-severity URL-injection gap (model
  not escaped in the path) — fixed with url.PathEscape. NOTE: finding this in a previously-
  untested path showed pass 9's "surface exhausted" was premature, so reverted
  untested_surface_remaining to true. Test: `TestGeminiEscapesModelInURL`.

- **Pass 11 (v0.4.12):** fixed-path adapter URLs. No defect — anthropic/openai_compat/
  openai_responses use fixed paths; a hostile model can't inject. URL surface now fully
  covered. Test: `TestFixedPathAdaptersIgnoreHostileModelInURL`.

- **Pass 12 (v0.4.13):** AuthPool `key|baseurl` split + auth/base resolution. No defect — edge
  keys split correctly. Surface now exhausted on an evidence basis (passes 11-12 probed fresh
  surfaces and found nothing). `untested_surface_remaining=false`. Test:
  `TestAuthPoolKeyBaseURLSplit`.

## Still untested / weak (candidate future passes)
- `capabilities.go` capability flags vs actual model support (thinking/image/tool on an unsupporting model).
- `buildRequest` with an empty model across adapters (sends `model:""` vs erroring early).
- openai_compat sending a ContentDocument (PDF) it can't support (vs openai_responses' clear error).
- Image/document validators: structurally correct but the image validator (`ValidateImageSource`) is untested.
- `middleware.go` logging, `discovery.go`, `factory.go` protocol resolution edges.
