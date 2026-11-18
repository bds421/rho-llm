# rho-llm — Roadmap & Issue Tracker

> **Workflow:** when an item here is completed, **remove it from this file** and add a
> corresponding entry to `CHANGELOG.md` (under `## [Unreleased]`), citing its id (e.g. `B3`).
> This keeps `todo.md` a list of *open* work and `CHANGELOG.md` the record of *shipped* work.
> See `CLAUDE.md`. Released versions and what they contain live in `CHANGELOG.md` only.

Priority: 🔴 critical · 🟠 high · 🟡 medium · 🟢 low
Effort: **S** (hours) · **M** (≈1 day) · **L** (multi-day)
Ids: **B** bug · **D** docs · **P** pi-derived feature · **F** other feature · **R** registry data · **C** code health

Sources: the 2026-11-12 full review of v0.17.0 (six lanes; every B item below has a failing
test in the review scratch copy unless marked *unverified*), the pi-ai 1.5.2 comparison
(2026-11-12; pi is `@earendil-works/pi-ai`, repo `earendil-works/pi`), the registry refreshes
of 2026-10-31 and 2026-11-10, and earlier reviews.

---

## 🔴 Bugs

None open (B1–B37 shipped in v0.17.1, B15b and B31b in v0.17.2).

---

## 📅 Dated — act on these dates (registry)
- **2026-11-30** `claude-sonnet-4-5` retires (→ `claude-sonnet-5-5`). Opus 4.8 fast mode removed.
- **2026-12-01** `gemini-2.5-pro` → routed to `gemini-3.8-flash`, `gemini-2.5-flash` → routed to
  `gemini-3.5-flash-lite` (both then get the Gemini 3 request rules: no prefilled model turn,
  strict thought signatures, `minimal` rejected on 3.8 Flash); `gpt-image-1.5`, `gpt-image-1-mini`;
  `gemini-4-argon` and `gemini-4-carbon` prices double ($4 / $20 / $0.20).
- **2026-12-11** `gpt-5`, `o3` (→ `gpt-6.1-sol`), `gpt-5-mini` (→ `gpt-6.1-terra`), `gpt-5-nano`
  (→ `gpt-6-luna`), `gpt-5-pro`, `o3-pro` (→ `gpt-6.1-sol` with `reasoning.mode: pro`), per
  developers.openai.com/api/docs/deprecations (2026-11-18). The `gpt5` alias points at
  `gpt-5.6-sol` today. `o3-pro` already left `GET /v1/models` (2026-11-18).
- **Late Dec 2026** Kimi K2 series (`kimi-k2.7-code`, still the `moonshot-cn`/`kimi-cn`
  default: confirm K3.1 Pro on api.moonshot.cn before switching).
- **2027-01-01** `gemini-3.8-flash`, `gemini-3.6-flash` prices double.
- **2027-01-06** `tts-1`, `gpt-4o-mini-tts`. **2027-01-14** dated `gpt-4.1*` snapshots.
- **2027-02-10** `gpt-image-2` (→ `gpt-image-2.6-{sunburst,flare}`; the `gpt-image-1` alias
  points at it). **2027-02-26** `whisper-1`, `gpt-4o-*transcribe`.
- **2027-03-15** `gemini-2.5-flash-image`. **2027-04-01** `gpt-5.1`, `gpt-5.3-codex`,
  `gpt-5.4-nano`. **2027-04-30** `mistral-large-2512`, Mistral Medium 3.5
  (`mistral-medium-latest`). **2027-05-07** `gemini-3.1-flash-lite`.
- **2028-05-14** `gemini-embedding-001` (→ `gemini-embedding-2`).
- Each retirement also needs a sweep of `../rho-llm-tutorial`.

---

## 🟠 Next up — features (minor releases)
- **P3 — Per-host compat flags as data, remainder — M.** The reasoning dialects and replay
  rules are data since P2 (`provider/openaicompat/compat.go`); still scattered: `max_tokens` vs
  `max_completion_tokens`, `developer` role, `store:false`, non-null assistant content, tool-ID
  format, usage field variants (DeepSeek `prompt_cache_hit_tokens`, Kimi top-level
  `cached_tokens`, `reasoning_tokens`), and a per-model override (pi `compat-schema.ts`).
  Also: OpenRouter `reasoning_details` (structured, signed reasoning) is not read or replayed;
  per-model effort value maps (Kimi K3 has no `medium`, GLM-5.3 only low/high/max).
- **F7 — Gemini Interactions follow-ups — S/M.** Long decode (`continuation_token`) on the
  `gemini_interactions` protocol; confirm whether `total_input_tokens` includes
  `total_cached_tokens` (treated like `promptTokenCount` today); the `failed` / `cancelled`
  payloads and in-band stream error shape (handled defensively, not provoked live); several
  thought steps in one turn keep only the last signature; refuse top-level sampling keys in
  `SamplingParams` as the `gemini` protocol does; structured output, `ThinkingBudget`,
  `previous_interaction_id`.
- **F1 — Chat fallback + per-attempt usage — M.** `OperationChat` usage events and a
  `NewFallbackClient` for chat reusing `tryEach`/`FailoverPolicy`.

## Reasoning, caching & conversation state
- 🟡 **P6 — Responses API reasoning replay — M** (`include: reasoning.encrypted_content`,
  item ids); also xAI on the Responses API, which is where xAI takes PDFs (`input_file` with
  `file_data`, live 2026-11-18); its Chat Completions refuses files, so `xai` has no document
  input until then.
- 🟡 **P7 — Prompt caching on OpenAI hosts — S/M.** `prompt_cache_key`, Responses
  `session-id` affinity header, retention; optional automatic cache markers.
- 🟡 **P8 — Mid-conversation effort/system/tool changes that keep the cache — L.** In-place
  `{"role":"system"}` was tried in v0.12.2 and dropped (Haiku 5.5 ignored it, 3 of 3 live).
  Since P1 this also matters for thinking: the Anthropic adapter hoists every `RoleSystem`
  message into the top-level `system`, which changes it and makes replayed thinking stale, so
  those blocks are dropped (`drop_block`) instead of replayed.
- 🟡 **R1 — Long-context price tiers — S.** Haiku 5.5 >100K, GPT >272K, Grok/Gemini >200K,
  MiniMax >512K, Qwen >256K, Muse Ember >200K; Azure keeps OpenAI's tier.
- 🟡 **R2 — OpenAI cache writes — S.** GPT-5.6+ bill writes at 1.25× input (30 min) and 2.5×
  (12h); rows have no write price.
- 🟡 **F2 — Gemini `store` default — S.** `store=true` default for new paid projects since
  2026-10-26; decide whether to send `store:false` on generateContent (the
  `gemini_interactions` adapter already sends `store:false` unless the caller opts in).
- 🟢 **P4b — Message threads, rest — S.** A key pool that rotates keys costs one replay per
  rotation (a thread belongs to its key); the fallback-model (`fallback_message`) stream path is
  untested live; the 4m30s reuse window is a margin over the documented 5 minutes.
- 🟢 Provider-reported cost; service-tier pricing [pi]. Deferred tool loading [pi].

## Tools & structured output
- 🟡 **P9 — Streamed tool-call argument deltas — M.** Buffered today; pi streams with a
  tolerant parser, JSON repair and `argumentsError`.
- 🟡 **P10 — Strict tool schemas — M.** `strict: prefer/require`, retry once non-strict and
  remember it per session.
- 🟡 **P11 — Tool-result images on OpenAI / Responses / Gemini — S/M** (Anthropic only today).
- 🟡 **F3 — Structured-output ergonomics — M.** JSON fence stripping, Gemini schema-dialect
  normalisation for `ResponseFormat`, typed error for a `max_tokens`-truncated structured
  response, malformed tool JSON as an error, `ErrUnsupportedParameter` instead of
  error-string sniffing, export the Gemini inline-audio limit.
- 🟢 **P12 — Full JSON-Schema tool validation with coercion** (pi `validation.ts`).

## Errors & observability
- 🟡 **P13 — Custom request headers per model/request, User-Agent / attribution — S.**
- 🟢 **P14 — `durationMs`, telemetry spans, raw provider stream-event hook** [pi].
- 🟢 **P15 — Context token estimate and `maxTokens` clamp to the window** [pi `estimate.ts`].
- 🟢 Typed server-side refusal fallback (`fallbacks:"default"`, beta
  `server-side-fallback-2026-07-01`, not on Haiku 5.5); client-side refusal fallback.
- 🟢 Anthropic server tools: typed web fetch results; mixed server + client tool turns; which
  refusals are billed.

## Registry & catalog
- 🟡 **R3 — Generated catalog — M/L.** pi builds its catalog from models.dev v2 on every
  build with per-provider override files. Go equivalent: `go generate` → `registry_gen.go` +
  overrides; move the hand lists (`adaptiveThinkingModels`, `*WithoutSamplingControls`) into
  metadata; `ModelInfo.VerifiedAt` + staleness note.
- 🟢 **R5 — DashScope workspace endpoints** (`https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com`,
  recommended since 2026-10-13) and the Qwen Token Plan CN provider.
- 🟢 **R6 — Unconfirmed data.** OpenAI cache-read prices for `gpt-5.2`/`5.1`/`5`/`5-mini`/
  `5-nano`/`o3`; `gpt-image-2.5-*` prices; DashScope `qwen3-max`, `qwen3.5-omni-light`,
  `text-embedding-v2` missing from the 2026-11-07 price page; Nemotron 3 Ultra context (1,048,576 per build.nvidia.com) and
  seven Fireworks rows with context 0; max output of Muse Spark/Wonder and Mistral Large 4;
  Cohere `command-a-03-2025` price; Nano Banana limits (docs vs `GET /v1beta/models`
  disagree); Gemini 3.5/3.7 Flash are routed to 3.6/3.8 Flash (billing may differ).
- 🟡 **R8 — Found 2026-11-18, not yet registered or acted on.**
  - Gemini: `gemini-decision-1` (Decisions API, `createDecision`; `models/{m}:createDecision`
    answers 404, path unknown), `antigravity-preview-*` (agent), `gemini-omni-1.2-flash`
    (`omni-1.1-flash` routed to it 2026-12-03). Source URLs in registry comments still say
    ai.google.dev; the docs moved to aistudio.google.com/docs (301).
  - OpenAI: Responses API changes for `gpt-6.2-*` (reasoning kept across an edited prefix,
    top-level `instructions` handling; migration guide unread), discounted background
    completion windows (beta), `gpt-6-cyber` / `gpt-6.2-astra-law` (restricted). 12-hour
    cache writes and >272K tiers are R1/R2.
  - Anthropic: computer use needs `computer_toolset_20260801` (`computer_20251124` is a 400);
    linked billing beta `user-profiles-billing-2026-10-01`; 5.2 thinking-prefix check is
    enforced for all accounts (P1 covers it via `drop_block`).
  - Others: DashScope `qwen4-max` (no price published), xAI `grok-imagine-image-2.0` /
    `grok-imagine-video-1.5(-lite)`, DeepSeek `deepseek-flash` built-in tools and file input
    (2026-11-13), Perplexity Agent API live check (`perplexity-agent` is docs-only: no key),
    citations / `search_results` / `usage.cost` not mapped, Sonar row prices unverified,
    Decisions API,
    OpenRouter `openai/gpt-6.2-astra-pro`.
- 🟢 **R7 — Live but unregistered.** Mistral `voxtral-*`, `mistral-ocr-*`, `codestral-embed`;
  Z.ai `glm-5.3-flashx`; `kimi-k3.1-pro-highspeed`; DeepInfra Kimi K3.1 Pro, DeepSeek V4 Pro;
  Qwen `qwen3.7-*`, `qwen3.8-omni-flash`; Claude Mythos 5.2 (verified orgs only).
- 🟢 Bedrock / Vertex / Azure Foundry providers (a gateway in front of Bedrock needs
  `AnthropicVersion20230601` for pre-4.7 models); classifier models [pi].
- ⚪ Out of scope: subscription OAuth flows [pi].
- **Not registered on purpose:** `gemini-omni-*` (video, Interactions API only),
  `gemini-3.8-live*`, `*-tts`, native-audio models (no realtime Gemini adapter); `gemma-4-*` on
  the Gemini API (free tier only, data used for training); `gemini-*-latest` (moving targets).

## Realtime
- 🟢 WebSocket dialer helper for OpenAI Realtime (live smoke uses a test-only dialer); Gemini
  Live / other vendors.

---

## 📝 Docs
- **D1 — `docs/ARCHITECTURE.md` drift — M.** `BetaFeatures` in the defaults block;
  `ProviderPreset.SupportsBatch` missing at L276, `provider.go:8` comment; add
  `Recover` to `BatchClient`; unexported `modelRegistry`/`modelAliases`/`defaultModels` +
  `retiredModels` (L39, 594-596); real type names (`anthropic.Client` …, L162, 255-258);
  `newSingleClient`, `MarkSuccessByName`, `MarkFailedByNameWithCooldown` (L111, 447, 451);
  "Probe re-arm (Unreleased)" → v0.4.1 (L544); logging wraps the pool, and the factory tree
  needs `openai_responses` (L289-301, 114-127, 262-270); package tree incomplete; "kit"
  (L529). Add sections: egress/SSRF, capabilities, OpenAI Responses, batch drivers. Rewrite
  version-tagged paragraphs in the present tense.
- **D2 — README leftovers — S.** Quick Start shows `import llm "github.com/bds421/rho-llm"`;
  Gemini padding logs at debug (L1417); retryable list incl. 504/529 (L716, 738);
  ThoughtSignature on all Gemini 3+ (L1384); retired-alias note placement (L1345);
  Anthropic/Gemini base URLs with paths (L43-44); package tree (L1471-1502).
- **D3 — README structure — M.** Table of contents; promote Modalities, Batch and Security to
  top-level sections; fold "Exponential Backoff" into Retry; drop "Before v0.9.5…" asides;
  document `WithTools`, `Session.StreamMessages`, `Conversation.ToRequest`/`Append`,
  `NewToolResultParts`, `NewWebSearchTool`, `BatchClient.Recover`/`RecoveryKey`/
  `ErrBatchNotFound`, `PooledClient.PoolStatus`, `PresetFor`, mock helpers,
  `Config.MaxBatchDownloadBytes`; generate the alias and default-model tables from the
  registry (missing today: Kimi, Meta, MiniMax, `gpt5.6-*`, `grok-build`). Keep the package
  tree only in ARCHITECTURE and the Config table only in README.
- **D5 — Release bookkeeping — S.** The `v0.7.6` tag exists only locally (never pushed).
- 🟠 **D7 — Tutorials for v0.19/v0.20 features, with thorough tests — M.** Update
  `../rho-llm-tutorial` (pinning the new release): new modules or sections, each with offline
  tests (httptest fakes, `-race`, hostile inputs) and gated live tests, for Gemini 4 long decode
  and thinking levels (`xhigh`, `minimal`→`low`, `CachedContent` refusal), the
  `gemini-interactions` preset, `Session` with `WithMessageThreads()` (delta turns, every
  fallback), the Perplexity Agent API preset, and the GPT-6.2 / Gemini 4 registry rows. Wire the
  new tests into the tutorial `Makefile` (`test-stress` or a new offline target) and fix D6 in
  the same pass.
- **D6 — Consumer repo.** `../rho-llm-tutorial/13_registry_deep` calls the removed
  `GetAvailableModels` (4 sites) — breaks when the tutorials move to v0.17.x.

---

## 🧹 Code health & tests
- 🟡 **C1 — Shared SSE reader and tool-argument decoder — M/L.** B2, B3, B5, B8 and B26 are
  each one bug copied across adapters; collapse the four HTTP/SSE skeletons onto
  `transport.go` (line handling, error-chunk hook, terminal-event rule). Do before P4/P9.
- 🟡 **C2 — Wire-level tests.** Normalize tests assert on the neutral transcript, never on
  what adapters send; pool tests share one scripted client across keys (why B10/B13 slipped).
  Add multi-key scripted pools and adapter round-trip tests. openai_compat coverage is 44.7%,
  openaibatch 33.9%.
- 🟡 **C3 — Live test matrix — M.** Same suites per provider gated on env keys, plus a
  handoff matrix (pi `cross-provider-handoff.test.ts`). Confirms B6, B7, B25, B28, P2.
- 🟢 **C8 — Documents outside user turns.** A PDF in an assistant message is refused by
  Anthropic, forwarded by Gemini, and silently dropped (not even validated) by openai_compat
  and Responses; pick one rule. `ContentPart.Title` goes out verbatim as the filename (no
  sanitising); untitled PDFs are numbered per message. Found by tutorial 26.
- 🟢 **C9 — Perplexity Agent built-in tools.** `SamplingParams["tools"]` is a reserved key, so
  Agent API tools such as `web_search` cannot be passed; needs a typed option. Found by
  tutorial 29.
- 🟢 **C7 — Shared Gemini model rules.** `minimal`/`xhigh`/Gemini 4 traits are duplicated in
  `provider/gemini/traits.go` and `provider/geminiinteractions`; move them to one internal
  package.
- 🟢 **C4 — Tests by concept in `provider/`.** `v095_*`, `*_break_*`, `batch_harden_*` files
  still exist there; the CLAUDE.md rule covers the root only.
- 🟢 **C5 — Library `go.mod` dependency-free.** Move `examples/` to a nested module (drops
  godotenv for consumers); examples use `panic(err)`; `examples/failover` env names.
- 🟢 **C6 — CI supply chain.** Pin actions by SHA; pin `gosec`/`govulncheck` versions; add
  `SECURITY.md` and Dependabot.
- 🟢 One shared transport per process/BaseURL; batch codec constructors allocate unused
  transports. Tighten break tests that accept any error; de-flake the `<= 900ms` assertion in
  `fallback_modality_test.go`.

---

## Verification environment
- API keys for live checks live in `../rho-llm-tutorial/.env` (gitignored): `GEMINI_API_KEY`,
  `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`. Load with `set -a && . ./.env && set +a`. These are
  paid APIs: prefer free metadata endpoints (`GET /v1/models`) and keep inference checks small.
- Consumer check before a release (CLAUDE.md): a `go.work` pointing the tutorial modules at
  this tree, then `make build-all && make vet-all && make test-stress`; remove `go.work` after.
- Ollama tutorials (08, 15, 20) need a local Ollama with `qwen3:4b` and `mistral-small3.2:24b`;
  `20_capability_test` is slow (120 min timeout). Tutorials 21/22 need cloud-ctl on :8085.
