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
- 🟢 **S** — Add a *typed* OpenRouter routing convenience on top of `Request.SamplingParams`
  (shipped). The raw passthrough already carries the `provider` object, so this is now
  ergonomics — a `OpenRouterRouting` struct that marshals into that key — not a missing
  mechanism.

- 🔴 **S** — **Dated price time-bomb: re-stamp Gemini Flash on 2027-01-01.** `gemini-3.8-flash`,
  `gemini-3.7-flash` and `gemini-3.6-flash` carry INTRODUCTORY rates ($0.75/$3.75, cache $0.075)
  that **double to $1.50/$7.50 (cache $0.15) on 2027-01-01** per ai.google.dev pricing. `ModelInfo`
  holds one flat price, so from that date `EstimateCost` will understate Gemini Flash cost by 2x
  until these rows are updated.
- 🟡 **M** — **`EstimateCost` cannot express non-flat pricing**, which is now the norm rather than
  the exception. Three shapes are unmodeled today, each silently understating cost:
  (1) **whole-request tier cliffs** — OpenAI bills >272K-input requests at ~2x input/1.5x output
  *for the entire request*, and xAI does the same at >=200K ("billed at the higher rate for all
  tokens in the request"); (2) **time-of-day** — DeepSeek's peak rate is 2x off-peak;
  (3) **scheduled future prices** (see the Gemini item above). Consider a `Pricing` shape that can
  express a tier predicate + an effective-date, rather than more flat float fields.
- 🟢 **S** — **Confirm MiniMax-M3 pricing tier.** `platform.minimax.io/docs/price` and
  `.../api-reference/text-pricing` both 404 as of 2026-09-17, so the registry's 0.60/2.40/0.12 is
  carried forward unverified. A research pass suggested those are the **>512K** tier and that the
  standard (<=512K) tier is 0.30/1.20/0.06 — if true, sub-512K estimates are 2x too high. Verify
  at a primary source and correct.
- 🟢 **S** — **Ant Ling (`antling`/`ling`) has a preset but no registry entries**, so its models
  get no cost estimation. Ant Ling publishes prices in **CNY**, which conflicts with the registry's
  USD convention — needs an FX/currency decision before the rows can be added.
- 🟢 **S** — **Decide the fate of the legacy xAI IDs.** `grok-4.20-beta*`, `grok-4-1-fast-*`,
  `grok-4-fast-*`, `grok-4-0709`, `grok-code-fast-1`, `grok-3`, `grok-3-mini` are no longer in
  docs.x.ai's model table, but xAI has announced **no** retirement or rename for them, so they were
  kept rather than deleted on absence alone. Confirm against a live API call and drop the dead ones.

### Gaps vs. upstream pi `ai` (compared 2026-09-17 against `earendil-works/pi`, `packages/ai` @ 0.85.1)

- 🟡 **S** — **Per-model thinking-level map + clamping** (pi 0.72.0, `max` added 0.80.6). rho's
  flat `ThinkingLevel` will happily send a level a given model rejects; pi has had to fix this
  repeatedly per provider.
- 🟢 **S** — **Custom request headers** on `Config`/`Request` (pi 0.78.0) — unblocks gateways
  (Azure, Cloudflare, corporate proxies) without new adapters.
- ℹ️ **Re-scope the existing Mistral caching item**: pi 0.79.8 implements it as a request-level
  `prompt_cache_key` derived from the session ID — NOT per-content-block `CacheControl`
  breakpoints as that todo entry currently assumes. The mechanism differs, so the design does too.
- ℹ️ **Deliberate non-goals, not gaps**: pi ships Bedrock (SigV4), Vertex (GCP OAuth), Azure,
  Cloudflare and subscription/OAuth-backed providers. rho declined these on stdlib-only grounds
  (see the comment in `provider.go`); this is the largest structural divergence and should stay a
  documented non-goal unless that constraint changes.
