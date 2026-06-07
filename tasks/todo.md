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

### P1 🟠 L — Conversation `Context` + cross-provider handoff  *(pi's headline feature)*
pi exposes a serializable `Context` (history + usage) that can be saved and resumed on a
**different** provider mid-session. rho is stateless — callers hand-manage `[]Message`.
- [ ] Design a `Context`/`Session` type: messages + accumulated usage + provider-neutral state
- [ ] JSON (de)serialization round-trip
- [ ] Provider handoff API (switch provider/model while preserving history)

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

### Backlog (neither pi nor rho has — evaluate demand)
- [ ] Embeddings API
- [ ] Structured output / JSON mode
- [ ] Audio in / out

---

## Where rho already leads pi — do not regress
- **PDF / document input** (native Anthropic/Gemini) — pi has no document type
- **Multi-key auth-pool rotation** with per-key cooldown
- **Circuit breaker** (3-state) + configurable retry/backoff + retry hooks
- **Per-model cost estimation** registry
