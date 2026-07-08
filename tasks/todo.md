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

- 🟡 **M** — Wire `CacheControl` through the `openai_compat` adapter for providers with
  explicit prompt-caching breakpoints (Mistral added this in pi `ai` 0.79.8). Today the
  neutral `ContentPart.CacheControl` / `Tool.CacheControl` fields are only honored by the
  Anthropic adapter (`provider/anthropic/anthropic.go`) — `openai_compat` silently drops
  them, so setting `CacheControl: true` against Mistral currently has no effect.
- 🟡 **M** — Add an extensible passthrough for provider-specific request-body extras (e.g.
  OpenRouter's `provider` routing object: fallbacks, provider preferences/order,
  quantization constraints — pi `ai` added full `OpenRouterRouting` support in 0.67.0).
  rho-llm has no `ExtraBody`/raw-passthrough mechanism today; needs a design that keeps
  `Message`/`ContentPart` as the neutral currency (per CLAUDE.md) while letting
  provider-specific request fields ride through `Config` or `Request` without polluting
  the neutral types.
