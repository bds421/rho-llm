# rho-llm — Roadmap & Issue Tracker

> **Workflow:** when an item here is completed, **remove it from this file** and add a
> corresponding entry to `CHANGELOG.md` (under `## [Unreleased]`). This keeps `todo.md`
> a list of *open* work and `CHANGELOG.md` the record of *shipped* work. See `CLAUDE.md`.

Priority: 🔴 critical · 🟠 high · 🟡 medium · 🟢 low
Effort: **S** (hours) · **M** (≈1 day) · **L** (multi-day)

---

## Current state (2026-11-06)

`v0.13.0` is the latest release (v0.12.0: Anthropic server tools with web search,
`pause_turn` continuation, per-iteration usage and search pricing; v0.12.1: refuse
thinking disabled at effort xhigh/max on Haiku 5.5 and Opus 5; v0.13.0: Anthropic API
version 2026-10-21 by default, structured output, typed Effort, text documents,
partial usage on broken streams —
see `CHANGELOG.md`).

---

## Open work — consolidated backlog (2026-11-04)

Merges the pi-ai 1.5.0 survey (2026-11-03), the 2026-10-07 Fable review
(`health-dashboard/docs/reviews/2026-10-07-fable-rho-llm-review.md`), the registry
refresh follow-ups and our own items into one list. Source tags: **[pi]** pi has it,
**[review]** Fable review, **[registry]** model/price data, **[own]** found by us.
Duplicates were merged (long-context tiers, malformed tool JSON → M9).

pi is now published as `@earendil-works/pi-ai` (repo `earendil-works/pi`;
`@mariozechner/pi-ai` stopped at 0.73.1).

### 📅 Dated — act on these dates
- **2026-11-30** `claude-sonnet-4-5` retires → move to `retiredModels` (→ `claude-sonnet-5-5`). [registry]
- **2026-12-01** `gemini-2.5-pro`, `gemini-2.5-flash` shut down (Google routes to
  `gemini-3.8-flash` / `gemini-3.5-flash-lite`); `gpt-image-1.5`, `gpt-image-1-mini`. [registry]
- **2026-12-11** `gpt-5*`, `o3`, `o3-pro`. **Dec 2026** Kimi K2 series. **2027-01-06**
  `tts-1`, `gpt-4o-mini-tts`. **2027-01-14** `gpt-4.1*`. **2027-02-26** `whisper-1`,
  `gpt-4o-*transcribe`. **2027-04-01** `gpt-5.1`, `gpt-5.3-codex`, `gpt-5.4-nano`.
  **2027-04-30** `mistral-large-2512`, Mistral Medium 3.5. [registry]
- Each retirement also needs a sweep of `../rho-llm-tutorial` (tutorials 04/08 still use
  Sonnet 4.6 / Haiku 4.5 on purpose; 21 has gpt-4.1).

### 🟠 Next up (high value)
1. **Anthropic message threads — M** [pi]. Beta `message-threads-2026-08-12`: only new
   messages are sent per turn (big token savings in long sessions).
2. **Chat fallback + per-attempt usage — M** [review M2]. `OperationChat` usage events
   for `Complete`/`Stream` via an `observeModality`-style wrapper, and a
   `NewFallbackClient` for chat reusing `tryEach`/`FailoverPolicy`.
3. **Bedrock / Vertex note — S** [docs]. `2026-10-21` (the default since v0.13.0) works on
   Bedrock only for Claude 4.7+; rho-llm has no Bedrock client, but a gateway in front of
   one needs `Config.AnthropicVersion = AnthropicVersion20230601` for older models.

### Reasoning & conversation state
- 🟡 **xAI on the Responses API with encrypted reasoning replay — M** [pi].
- 🟡 **Mid-conversation effort/system/tool changes that keep the cache — L** [pi].
  In-place `{"role":"system"}` messages were tried in v0.12.2 and dropped: Haiku 5.5
  declined instructions placed there (live, 3 of 3) that it followed in `system`.
- 🟢 Deferred tool loading [pi].

### Caching & cost
- 🟡 **Long-context price tiers — S** [pi][registry]. Not modeled; `EstimateCost` uses
  the short tier (Haiku 5.5 >100K, GPT >272K, Grok/Gemini >200K, MiniMax >512K, Qwen >256K).
- 🟡 **Gemini `store` default — S** [registry]. `store=true` is the default for new paid
  Generate Content projects since 2026-10-26; decide whether to send `store:false`.
- 🟢 Provider-reported cost when the API returns it [pi].

### Tools & structured output
- 🟡 **Streamed tool-call argument deltas — M** [pi]. `input_json_delta` is buffered;
  tool calls are emitted once complete.
- 🟡 **Strict tool schemas — M** [pi]. `strict: prefer/require`, retry once non-strict.
- 🟡 **Structured-output ergonomics — M** [review M9][pi]. JSON fence stripping, Gemini
  schema-dialect normalisation for `ResponseFormat`, typed error for a
  `max_tokens`-truncated structured response, malformed tool JSON surfaced as an error
  (pi), `ErrUnsupportedParameter` instead of error-string sniffing, export the Gemini
  inline-audio limit, redact `BaseURL` credentials in constructor errors.

### Errors & observability
- 🟡 **Registry staleness — S** [review M4]. `ModelInfo.VerifiedAt` + a one-time slog note
  when metadata is older than a threshold; move the hand lists
  (`geminiWithoutSamplingControls`, `anthropicWithoutSamplingControls`,
  `adaptiveThinkingModels`) into registry metadata. pi generates its catalog from
  models.dev — worth comparing before building our own tooling.
- 🟢 Anthropic client-side refusal fallback (retry on another model with fallback
  credit) [pi]. Server-side `fallbacks` works since v0.12.0.
- 🟢 Anthropic server tools: typed web fetch results; a turn that mixes server and
  client tool calls (works through the normal tool loop, not modeled); which
  refusals are billed (`EstimateCost` counts every iteration) [own].
- 🟢 `durationMs` on responses, telemetry spans + User-Agent, raw provider stream-event
  hook [pi].

### Providers & registry data
- 🟡 **Perplexity Agent API adapter — M** [registry]. Sonar Chat Completions ended
  2026-09-27; requests are reformulated onto `POST /v1/agent`. Prices unverified.
- 🟢 Unconfirmed data: Kimi K3 / K3.1 Pro prices; Meta `muse-spark-1.3` price; OpenAI cache-read prices for
  `gpt-5.2`/`5.1`/`5`/`5-mini`/`5-nano`/`o3`; `gpt-image-2.5-{sunburst,flare}`; DeepInfra
  DeepSeek-V4-Flash; Cohere `command-a-plus-05-2026`; Groq `qwen/qwen3.8-27b`. [registry]
- 🟡 **Mistral Medium 3.5** — `mistral-medium-latest` now serves Medium 3.5 (262,144
  context; the row says 131,072 at $0.40/$2.00). Price not found on docs.mistral.ai
  (2026-11-10). Also unregistered but live: `magistral-*`, `voxtral-*`, `mistral-ocr-*`,
  `codestral-embed`, `ministral-3b-2512`. [registry]
- 🟢 **Claude Mythos 5.2** (`claude-mythos-5-2`, $8/$40) — verified-organizations only;
  not in our key's model list, not registered. [registry]
- 🟢 **Gemini new but not registered (2026-11-10):** `gemini-omni-1.1/1.2-flash` (video),
  `gemini-3.8-live*`, `gemini-3.8-flash(-lite)-tts`, `gemini-embedding-2-preview`,
  `deep-research-*`, `antigravity-preview-*`, `lyria-3.5`. Nano Banana limits still
  disagree with the docs (`GET /v1beta/models`: 3.1-flash-image and nano-banana-2.1
  input 65,536 vs registry 131,072; 3-pro-image 131,072 vs 65,536). [registry]
- 🟢 Gemini 3.5/3.7 Flash auto-route to 3.8 Flash (deprecated 2026-10-08, no shutdown
  date) — billed prices may differ from the registry. [registry]
- 🟢 Nano Banana limits: docs vs `GET /v1beta/models` disagree (e.g. `gemini-3-pro-image`
  65,536 vs 131,072 input); kept the docs values. [registry]
- 🟢 `o3-pro` returns 404 for our key — tier-gating or retirement? Moot after 2026-12-11. [own]
- 🟢 Bedrock / Vertex / Azure Foundry providers, classifier models [pi].
- ⚪ Probably out of scope: subscription OAuth flows [pi].
- **Not registered on purpose (2026-11-03):** `gemini-omni-*` (video, Interactions API
  only); `gemma-4-*` on the Gemini API (free tier only, data used for training,
  thinkingLevel only `high`/`minimal`); `gemini-*-latest` (moving targets); Live/TTS/
  native-audio models (no realtime Gemini adapter).

### Realtime
- 🟢 WebSocket dialer helper for OpenAI Realtime production use (live smoke uses a
  test-only dialer). [own]
- 🟢 Gemini Live / multi-vendor realtime beyond the OpenAI reference. [own]

### Code health & tests
- 🟢 **Collapse the four adapter HTTP/SSE skeletons onto `transport.go` — L** [review].
  Do before streamed tool deltas / message threads if those touch every adapter.
- 🟢 One shared transport per process/BaseURL; batch codec constructors allocate
  transports they never use. [review]
- 🟢 Move `examples/` to a nested module so `go.mod` is stdlib-only (drop godotenv). [review]
- 🟡 `examples/failover` reads `OPENAI_PRIMARY_KEY`/`OPENAI_BACKUP_KEY`/`AZURE_OPENAI_KEY`;
  fall back to `OPENAI_API_KEY` — S. Rotation itself verified working. [own]
- 🟡 Examples use `panic(err)` instead of the error handling the README teaches — S. [own]
- 🟢 Tighten break tests that accept any error (`security_test.go:380,451,942,999`,
  `provider/gemini/transcription_break_test.go`); de-flake the `<= 900ms` assertion in
  `fallback_modality_test.go`; HTTP-level 429/401/5xx tests for the Anthropic and
  Responses adapters; coverage for fallback embeddings/images/speech. [review]

### Verification & ops
- 🟠 **Finish the tutorial verification — M** [own]. Cloud modules pass live; still
  unrun, blocked on environment:
  - **Ollama** (08, 15, 20): host `renelinux2023@rene-pcoffice-2023` was busy on
    2026-09-18 — do **not** run while another test is in flight (Ollama would evict
    that model). Missing: `ollama pull qwen3:4b`, `ollama pull mistral-small3.2:24b`.
    ⚠️ `20_capability_test` is expensive + slow (120m timeout) — never run casually.
  - **cloud-ctl on :8085** (21, 22): service wasn't running locally.
- 🟢 Expand live smokes to Meta/Mistral/CN hosts when keys are available. [own]
- 🟢 Internal mirror is stale. [own]

---

## Where rho already leads pi — do not regress
- **PDF / document input** (native Anthropic/Gemini) — pi has no document type
- **Embeddings, Batch API, speech-to-text** — pi is chat-only
- **Multi-key auth-pool rotation** with per-key cooldown
- **Circuit breaker** (3-state) + configurable retry/backoff + retry hooks
- **Per-model cost estimation** registry

---

## Environment notes (2026-09-18)

- **API keys** live in `../rho-llm-tutorial/.env` (gitignored, mode 600):
  `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` — all validated HTTP 200.
  The Gemini key was copied from another local project's `.env`; if that key rotates, this breaks.
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
