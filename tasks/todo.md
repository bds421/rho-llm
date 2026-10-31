# rho-llm — Roadmap & Issue Tracker

> **Workflow:** when an item here is completed, **remove it from this file** and add a
> corresponding entry to `CHANGELOG.md` (under `## [Unreleased]`). This keeps `todo.md`
> a list of *open* work and `CHANGELOG.md` the record of *shipped* work. See `CLAUDE.md`.

Priority: 🔴 critical · 🟠 high · 🟡 medium · 🟢 low
Effort: **S** (hours) · **M** (≈1 day) · **L** (multi-day)

---

## Current state (2026-10-31)

`v0.9.5` is the latest published release (Fable review 2026-10-07 items H1–H5, M1, M3,
M5–M8 and the low-severity hygiene batch — see `CHANGELOG.md` `[0.9.5]`). The review's
remaining items are listed below as open work.

---

## Open work

### 🟠 Finish the tutorial verification — **M**
`../rho-llm-tutorial` — 11 cloud modules pass live (01,02,03,04,05,07,09,10,11,12,18).
Still unrun, blocked on environment, not on code:

- **Needs Ollama** (08, 15, 20): host `renelinux2023@rene-pcoffice-2023` was busy on
  2026-09-18 (`llama-server` at 833% CPU, `qwen3.8:27b` holding 16 GB VRAM). Do **not**
  run against it while another test is in flight — Ollama would evict that model.
  Also missing two models: `ollama pull qwen3:4b` and `ollama pull mistral-small3.2:24b`
  (`deepseek-r1:14b` is present).
  ⚠️ `20_capability_test` is expensive + slow (120m timeout) — never run casually.
- **Needs cloud-ctl on :8085** (21, 22): service wasn't running locally.

### 🟡 `examples/failover` env-var mismatch — **S**
It reads `OPENAI_PRIMARY_KEY` / `OPENAI_BACKUP_KEY` / `AZURE_OPENAI_KEY`, but the
tutorial `.env` only defines `OPENAI_API_KEY`. Either add the aliases to the `.env` or
make the example fall back to `OPENAI_API_KEY`. Rotation itself is verified working
(bad key → 401 → profile disabled → rotated → success).

### 🟡 Examples use `panic(err)` — **S**
Shipped examples panic on error instead of the error handling the README teaches.
Fine for a demo; inconsistent with the docs.

### 🟢 `o3-pro` returns 404 for our key — **S**
Could be tier-gating rather than retirement — a 404 can't distinguish them. Left in the
registry deliberately. Re-check with a key that has o3 access before removing.

### 🟠 Registry follow-ups from the 2026-10-31 refresh — **S/M**
Left out of the refresh because the primary source could not confirm them:
- **Gemini `gemini-2.0-flash` / `gemini-3-pro-preview`:** ai.google.dev lists both as
  "Shut down", but `TestLiveModelsStillRegistered` records them as live on 2026-09-18.
  Re-check `GET /v1beta/models` with a key, then retire or keep.
- **Kimi K3 / K3.1 Pro** (platform.kimi.ai pricing tables did not render) — and the K2
  series retires end of December 2026 (`kimi-k2.7-code` is the moonshot default).
- **Meta `muse-spark-1.3`** (the docs' recommended model; no published price found).
- **Mistral Large 4** (API ID `mistral-large-4` vs `mistral-large-4-0` unconfirmed);
  `mistral-large-2512` and Medium 3.5 retire 2027-04-30.
- **Perplexity:** Sonar Chat Completions support ended 2026-09-27; requests are being
  reformulated onto the Agent API (`POST /v1/agent`). Prices unverified. Likely needs an
  adapter, not data.
- **OpenAI:** cache-read prices for `gpt-5.2`/`gpt-5.1`/`gpt-5`/`gpt-5-mini`/`gpt-5-nano`/`o3`;
  `gpt-image-2.5-{sunburst,flare}`; upcoming shutdowns `gpt-image-1.5`/`-1-mini` (Dec 1),
  `gpt-5*`/`o3`/`o3-pro` (Dec 11), `tts-1`/`gpt-4o-mini-tts` (Jan 6), `gpt-4.1*` (Jan 14),
  `whisper-1`/`gpt-4o-*transcribe` (Feb 26), `gpt-5.1`/`gpt-5.3-codex`/`gpt-5.4-nano` (Apr 1).
- **Gemini:** `gemini-3.7-flash`/`gemini-3.5-flash` deprecated 2026-10-08 (no shutdown
  date); `store=true` is the default for new paid Generate Content projects since
  2026-10-26 — decide whether the adapter should send `store:false`.
- **DeepInfra / Cohere / Groq previews:** DeepInfra DeepSeek-V4-Flash price/context,
  Cohere `command-a-plus-05-2026` (no public price), Groq `qwen/qwen3.8-27b` (preview).
- Long-context price tiers (Haiku 5.5 >100K, GPT >272K, Grok/Gemini >200K, MiniMax
  >512K, Qwen >256K) are not modeled; `EstimateCost` uses the short tier.

### From the 2026-10-07 Fable review (not in v0.9.5)
Review: `health-dashboard/docs/reviews/2026-10-07-fable-rho-llm-review.md`.
- 🟠 **M2 chat parity — M.** Per-attempt usage events for `Complete`/`Stream`
  (`OperationChat` through an `observeModality`-style wrapper) and a
  `NewFallbackClient` for chat reusing `tryEach`/`FailoverPolicy`.
- 🟡 **M4 registry staleness — S.** `ModelInfo.VerifiedAt` + a one-time slog note when
  pricing/capabilities are older than a threshold; hand lists
  (`geminiWithoutSamplingControls`, `anthropicWithoutSamplingControls`, the anthropic
  adapter's `adaptiveThinkingModels`) into registry metadata.
- 🟡 **M9 structured-output ergonomics — M.** JSON fence stripping, Gemini schema-dialect
  normalisation for `ResponseFormat`, typed error for a `max_tokens`-truncated structured
  response, `ErrUnsupportedParameter` instead of error-string sniffing, export the Gemini
  inline-audio limit, redact `BaseURL` credentials in constructor errors.
- 🟢 **Collapse the four adapter HTTP/SSE skeletons onto `transport.go` — L.**
- 🟢 One shared transport per process/BaseURL instead of one per adapter/profile; batch
  codec constructors allocate transports they never use.
- 🟢 Move `examples/` to a nested module so `go.mod` is literally stdlib-only (drop godotenv).
- 🟢 Tighten break tests that accept any error (`security_test.go:380,451,942,999`,
  `provider/gemini/transcription_break_test.go`); de-flake the `<= 900ms` wall-clock
  assertion in `fallback_modality_test.go`; HTTP-level 429/401/5xx tests for the Anthropic
  and Responses adapters; coverage for fallback embeddings/images/speech.
- 🟢 Anthropic in-stream `error` events are not surfaced as errors (only as a truncated
  stream); Responses now redacts/truncates them — align Anthropic.
- 🟢 GitLab mirror is stale.

### Follow-ups (optional, pre-existing)
- WebSocket dialer helper for OpenAI Realtime production use (live smoke uses a test-only dialer)
- Gemini Live / multi-vendor realtime beyond OpenAI reference
- Expand live smokes to Meta/Mistral/CN hosts when keys are available

---

## Where rho already leads pi — do not regress
- **PDF / document input** (native Anthropic/Gemini) — pi has no document type
- **Multi-key auth-pool rotation** with per-key cooldown
- **Circuit breaker** (3-state) + configurable retry/backoff + retry hooks
- **Per-model cost estimation** registry

---

## Environment notes (2026-09-18)

- **API keys** live in `../rho-llm-tutorial/.env` (gitignored, mode 600):
  `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` — all validated HTTP 200.
  The Gemini key was copied from `../rho-pdf/.env`; if rho-pdf rotates it, this breaks.
  Load with `set -a && . ./.env && set +a`.
- **These are paid APIs.** Free metadata endpoints (`GET /v1/models`) validate keys and
  audit model IDs at zero token cost — prefer them over inference calls.
- **Consumer check** (CLAUDE.md requires it before release):
  ```bash
  cd ../rho-llm-tutorial && go work init && go work use -r . && go work use ../rho-llm
  make build-all && make vet-all && make test-stress
  rm -f go.work go.work.sum      # don't leave it behind
  ```
  All 22 modules passed against this tree on 2026-09-18.
