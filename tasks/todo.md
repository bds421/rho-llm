# rho-llm — Roadmap & Issue Tracker

> **Workflow:** when an item here is completed, **remove it from this file** and add a
> corresponding entry to `CHANGELOG.md` (under `## [Unreleased]`). This keeps `todo.md`
> a list of *open* work and `CHANGELOG.md` the record of *shipped* work. See `CLAUDE.md`.

Priority: 🔴 critical · 🟠 high · 🟡 medium · 🟢 low
Effort: **S** (hours) · **M** (≈1 day) · **L** (multi-day)

---

## Security findings (June 2026 architecture/security audit)

### F2 🟠 M — No SSRF protection; scheme-validation test is misleading
`BaseURL` and the per-key `"apikey|baseurl"` override accept any http(s) URL (incl.
`http://169.254.169.254/…`) with no allowlist / internal-IP filtering, and the API key
is sent there. `TestBaseURLSchemeValidation` implies non-http schemes are "rejected", but
there is no scheme validation — they only fail incidentally inside `net/http`.
- [ ] Document the `BaseURL` trust boundary explicitly (README + `Config` godoc)
- [ ] (Optional) opt-in allowlist / internal-IP denylist for hardened deployments
- [ ] Replace/rename the misleading test to assert what is actually enforced

### F4 🟡 S — TLS-downgrade redirect keeps the key
`sameHost` compares host+port but ignores scheme, so an `https → http` same-host redirect
does **not** strip auth headers — the key would go out in plaintext.
- [ ] Strip sensitive headers on scheme change too (or refuse `https → http` redirects)

### F5 🟡 S — Provider error bodies flow into logs
`APIError.Message` (truncated upstream body) is logged on failure; provider 4xx bodies can
echo request fragments. The "privacy-safe logging" claim covers content, not error bodies.
- [ ] Note this in the logging docs; consider a redaction/elision hook

---

## Feature roadmap — pi (`pi-mono/packages/ai`) parity

rho is a Go tribute to pi; these are capabilities pi has that rho lacks.

### P1 ✅ shipped — Conversation + Session + cross-provider handoff
Phases 1–2 done (see CHANGELOG `[Unreleased]` and ARCHITECTURE §13): serializable
`Conversation`, concurrency-safe `Session`, `SwitchProvider` handoff, `NormalizeForProvider`,
`ContentThinking` round-trip with same-provider signature replay. Only the deferred
persistence piece remains:

#### P1.3 🟢 S — Conversation persistence (`Store`) — deferred Phase 3
Conversations serialize via `json.Marshal` / `LoadConversation` today. A pluggable store was
intentionally deferred when the handoff core shipped.
- [ ] `Store` interface: `Save(ctx, id, *Conversation)` / `Load(ctx, id) (*Conversation, error)`
- [ ] In-memory default + a file/SQLite example

### P2 🟠 L — Image generation (output modality)
pi has `generateImages()`, `getImageModel()`, `getImageModels()`. rho has no image output.
- [ ] `GenerateImages` entry point + per-provider endpoints (OpenAI, Gemini)
- [ ] Image-output entries in the model registry

### P3 🟠 M — Tool results with image content
pi tool results can carry image blocks; rho's `ToolResultContent` is a plain string.
- [ ] Extend the `ContentPart` tool-result round-trip to carry images
- [ ] Adapter serialization (Anthropic / OpenAI / Gemini)

### P4 🟡 M — OAuth login flows
pi: `loginAnthropic()`, `loginOpenAICodex()`, `loginGitHubCopilot()`. rho is API-key only.
- [ ] OAuth helper package for Anthropic / Copilot / Codex
- [ ] Token storage + refresh wired into auth

### P5 🟡 M — Finer-grained streaming events
pi emits `text_start/delta/end`, `thinking_start/delta/end`, `toolcall_start/delta/end`.
rho's events are coarse (no start/end boundaries), making clean multi-block UI rendering harder.
- [ ] Add boundary event types (additive to `EventType`)
- [ ] Emit them from all four adapters

### P6 🟡 S — First-class faux/mock provider for consumers
pi ships `registerFauxProvider()`, `fauxAssistantMessage()`, etc. rho exposes no test double.
- [ ] Public in-memory mock `Client` + scripted-response helpers

### P7 🟢 M — Provider breadth (15 → pi's 30+)
Mostly auth/endpoint variants on protocols rho already speaks.
- [ ] Amazon Bedrock
- [ ] Google Vertex AI
- [ ] Azure OpenAI (Responses)
- [ ] Others: Cloudflare, Together AI, Fireworks, NVIDIA NIM, Vercel AI Gateway

### P8 🟡 M — Tool-call argument validation
pi validates tool-call arguments against the tool's schema (`validateToolCall()`, TypeBox).
rho passes raw JSON schema to the provider and returns whatever the model emits — callers get
no pre-execution validation.
- [ ] `ValidateToolCall(tool Tool, call ToolCall) error` against `Tool.InputSchema` (JSON Schema)
- [ ] Optional: surface validation failures as a typed error for the tool-loop

### P9 🟢 S — Typed model-discovery API
pi exposes `getModels()` / `getProviders()`. rho only has `GetModelInfo` + `ResolveModelAlias`,
so callers can't enumerate or filter the registry programmatically.
- [ ] `Models()` / `Providers()` enumerators over the registry (filter by capability: thinking, tools, vision, docs)

### P10 🟢 S — `CompleteSimple` / `StreamSimple` convenience wrappers
pi ships `completeSimple()`/`streamSimple()`. Minor ergonomics parity for the common
"just send a prompt" path.
- [ ] Thin wrappers taking a prompt string (+ optional system) instead of a full `Request`

### Backlog (neither pi nor rho has — evaluate demand)
- [ ] Embeddings API
- [ ] Structured output / JSON mode
- [ ] Audio in / out

---

## Code quality / docs (from the June 2026 architecture/security audit)

### Q1 🟡 M — De-duplicate the four provider adapters
The adapters share ~70% identical structure (build → marshal → POST → status check →
bounded read → SSE loop). The security-critical bits (bounded reads, header setting,
error construction) are re-implemented in each.
- [ ] Extract a shared `httptransport` helper (do-request + bounded-read + error-from-status)
- [ ] Keep wire-format translation per-adapter; centralize only the HTTP plumbing

### Q2 🟢 S — Fix the "zero external dependencies" README claim
`README.md` says "Zero external dependencies (stdlib only)", but `go.mod` requires
`joho/godotenv` (imported only by the 3 `examples/` files, yet still a module-level require,
so importers pull it transitively).
- [ ] Either correct the README line, or move `examples/` to its own module to make the claim true

---

## Where rho already leads pi — do not regress
- **PDF / document input** (native Anthropic/Gemini) — pi has no document type
- **Multi-key auth-pool rotation** with per-key cooldown
- **Circuit breaker** (3-state) + configurable retry/backoff + retry hooks
- **Per-model cost estimation** registry
