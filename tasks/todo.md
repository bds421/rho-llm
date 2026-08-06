# rho-llm — Roadmap & Issue Tracker

> **Workflow:** when an item here is completed, **remove it from this file** and add a
> corresponding entry to `CHANGELOG.md` (under `## [Unreleased]`). This keeps `todo.md`
> a list of *open* work and `CHANGELOG.md` the record of *shipped* work. See `CLAUDE.md`.

Priority: 🔴 critical · 🟠 high · 🟡 medium · 🟢 low
Effort: **S** (hours) · **M** (≈1 day) · **L** (multi-day)

---

## Where rho already leads pi — do not regress
- **PDF / document input** (native Anthropic/Gemini) — pi has no document type
- **Multi-key auth-pool rotation** with per-key cooldown
- **Circuit breaker** (3-state) + configurable retry/backoff + retry hooks
- **Per-model cost estimation** registry

---

## Open work

_(Phase A–D multi-vendor batch/modality/catalog/realtime shipped in v0.7.0; see `CHANGELOG.md`.)_

### Follow-ups (optional)
- WebSocket dialer helper for OpenAI Realtime production use (live smoke uses a test-only dialer)
- Gemini Live / multi-vendor realtime beyond OpenAI reference
- Expand live smokes to Meta/Mistral/CN hosts when keys are available
