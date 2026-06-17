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

## Still untested / weak (candidate future passes)
- `capabilities.go` capability flags vs actual model support (thinking/image/tool on an unsupporting model).
- `buildRequest` with an empty model across adapters (sends `model:""` vs erroring early).
- openai_compat sending a ContentDocument (PDF) it can't support (vs openai_responses' clear error).
- Image/document validators: structurally correct but the image validator (`ValidateImageSource`) is untested.
- `middleware.go` logging, `discovery.go`, `factory.go` protocol resolution edges.
