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

## 🔴 Bugs — next patch release

### Tool calls
- 🟢 **B7b — Normalize edge cases — S.** Duplicate IDs inside one assistant message remap both
  results to the second call; a tool call with an empty ID keeps the call, drops its result
  and skips orphan repair (`normalize.go:77-99, 159`).
- **B9 — `ValidateToolCall` skips `required`/`enum` for Go-authored schemas — S.**
  Asserts `.([]any)`; `[]string` (the repo's own examples) is silently accepted
  (`validate.go:32, 56`).

### Key pool, retry, fallback
- **B10 — A 429 burst on one key opens the breaker for the whole pool — S/M.** Five
  concurrent 429s on key a (threshold 5) open the circuit although key b is healthy; a later
  success in Open doesn't close it (`pool.go:510-512, 636, 681-683`). Key-scoped errors
  (429, auth) must not count as endpoint failures while another key is healthy.
- **B11 — Fallback never fails over when the primary hits its own `Config.Timeout` — S.**
  `isProviderFailure` treats `Client.Timeout`'s error as caller cancellation because it
  matches `context.DeadlineExceeded` (`fallback_modality.go:130`). Decide on `ctx.Err()`.
- **B12 — One request-scoped 403 permanently disables every key — S.** No recovery for
  non-active keys (`pool.go:153-159`, `errors.go:195-201`). Long cooldown instead of a
  permanent flag; stop rotating once a second key returns the same 403 for the same request.
- **B13 — After waiting for the soonest cooldown, the retry uses the still-cooling key — S.**
  `backoffAfterFailedRotation` sleeps then continues "with same client" (`pool.go:406-420,
  554-556, 726-728`). Re-rotate after the sleep.
- **B14 — `DisableRetries` silently disables the breaker and key-health marking — S.**
  Early return before `recordFailure`/`markFailed`; `NewClient` bypasses the pool entirely
  (`pool.go:495-498, 674-678`; `factory.go:14-34`).
- **B15 — `MaxRetries` means three different things — S.** Chat pool: a cap (1 or 2 lowers
  attempts below 3); `DoHTTP`: attempts with a floor of 3; doc says "minimum 3"
  (`config.go:462`, `pool.go:424-440`, `transport.go:64-73`). Pick one meaning, fix README
  618/1242.
- 🟡 **B16 — `DoHTTP` doesn't retry Anthropic 529 — S.** `retryableHTTPStatus` omits it
  (`transport.go:160-164`); batch calls fail on the first overload.
- 🟡 **B17 — Stream idle watchdog counts consumer time — S.** Measures since the last `Read`,
  not time blocked in `Read`; a slow loop body trips a retryable timeout (`config.go:218-227`).
- 🟢 **B18 — `rc.Release()` in pool `Stream` not deferred** (*suspected*) (`pool.go:613-656`).

### Session & conversation
- **B19 — Breaking out of `Session.Stream` on `EventDone` drops the completed turn — S.**
  `session.go:224-226` returns before `commitTurn`; message, reply and usage vanish.
- **B20 — A text document (or a PDF for openai_responses) bricks the session after a
  handoff — S.** Not degraded in `NormalizeForProvider` (`normalize.go:129-131`); every later
  turn fails and history can't be edited. Degrade to a text part for non-Anthropic targets.
- **B21 — `SwitchProvider` keeps a `Model` set via `WithBaseRequest` — S.** The new client
  is asked for the old provider's model (`session.go:258-264`).
- 🟡 **B22 — Cost is $0 for snapshot-suffixed model IDs — S.** `pricingFor` is exact-match +
  alias; `gpt-5.4-2026-03-05`, `claude-opus-5-5-20260901` price 0; one unknown iteration
  model zeroes the response (`registry.go:747-783`). Strip the suffix via `versionSuffix`;
  fall back to the top-level model. `LoggingClient` prices by request model, `Session` by
  response model.
- 🟢 **B23 — Conversation round trip not exact; `LoadConversation` permissive — S.**
  `ToolInput` integers > 2^53 change (decode with `UseNumber`); `null`/`{}` load as empty;
  negative usage loads as-is; `ToRequest` shares the backing array (`conversation.go:158`);
  `WithConversation` keeps the caller's pointer (`session.go:65`).
- 🟢 **B24 — Same-provider unsigned thinking is replayed as visible text** (design; pinned by
  `TestNormalizeUnsignedThinkingDegradesEvenSameProvider`). Decide: drop instead.

### Gemini / batch wire details
- 🟡 **B25 — Gemini thought summaries never requested** (*unverified live*). pi sends
  `thinkingConfig.includeThoughts: true` (`google-generative-ai.ts:399`); rho sends only
  `thinkingLevel` (`gemini.go:209-212`), so `EventThinking` probably never fires.
- 🟡 **B26 — 256 KB SSE line cap kills long Responses streams and Anthropic batch
  downloads — S.** `response.completed` repeats the whole output on one line (fails at ~300
  KB); `anthropicbatch.parseResults` reuses the cap for JSONL and ignores
  `MaxBatchDownloadBytes` (`responses.go:190`, `anthropicbatch.go:220-229, 376-411`).
- 🟡 **B27 — Anthropic batch per-item errors lose their message — S.** Nested
  `{"type":"error","error":{type,message}}` read as flat (`anthropicbatch.go:363-371, 396`).
- 🟡 **B28 — Gemini batch results don't decode the documented shape** (*doc-verified, not
  live*). `inlinedResponses` is an object wrapping an array (`geminibatch.go:302-318`).
- 🟢 **B29 — Batch IDs go into URL paths unescaped — S.** Anthropic ID and Gemini name allow
  `/`, `..`, `?` (`anthropicbatch.go:190, 220, 236`; `geminibatch.go:221, 286`).
- 🟢 **B30 — Gemini tool results: `IsError` ignored, images dropped, no id on Gemini 3+.**
  Always `{"result": text}` (`gemini.go:408-417`).

### Smaller correctness items
- 🟢 **B31 — Anthropic wire edge cases** (*rejection unverified*): forced `tool_choice` with
  budget thinking (Haiku/Sonnet/Opus 4.5–4.6); a rich tool result of empty parts serializes
  `"content": null`; whitespace-only text blocks sent; images inside `is_error` results; tool
  schema root `anyOf`/`oneOf` passed through (`anthropic.go:427, 548-629, 681`).
- 🟢 **B32 — Secrets in non-JSON output.** `fmt.Sprintf("%+v", cfg)` and slog text handlers
  print `APIKey` (no `String`/`LogValue`); BaseURL userinfo username and query secrets reach
  returned transport errors (`config.go:710`, `types.go:1001`).
- 🟢 **B33 — Registry API edges.** `RegisterModel` accepts an ID an alias shadows (entry
  unreachable); whitespace IDs/aliases accepted; `GetDefaultModel` falls back to
  `claude-sonnet-5` for 10 presets and keeps a non-preset `"gpt"` key (`registry.go:418, 593,
  653`).
- 🟢 **B34 — `LoggingClient` logs `tokens_in=-1` on an abandoned stream** (`middleware.go:111-143`).
- 🟢 **B35 — Realtime surface is half-finished.** Doc mentions a "registered dialer" that
  doesn't exist; `EncodeRealtimeEvent` drops all but four fields (no usable
  `session.update`); concurrent `send` writes concurrently; `err == io.EOF`
  (`realtime.go:72, 132, 168-207`). Finish or label experimental.
- 🟢 **B36 — `BlockPrivateBaseURL` misses Azure's wire server** `168.63.129.16/32`
  (`privateaddr.go:18-49`).
- 🟢 **B37 — `examples/cache_gemini` puts the key in `?key=`** with a plain `http.Client`.

---

## 📅 Dated — act on these dates (registry)
- **2026-11-18** `MiniMax-M2.5` retires (its FAQ table already says retired) → `retiredModels`
  (→ `MiniMax-M3.1-Flash`).
- **2026-11-30** `claude-sonnet-4-5` retires (→ `claude-sonnet-5-5`). Opus 4.8 fast mode removed.
- **2026-12-01** `gemini-2.5-pro` → routed to `gemini-3.8-flash`, `gemini-2.5-flash` → routed to
  `gemini-3.5-flash-lite`; `gpt-image-1.5`, `gpt-image-1-mini`; `gemini-4-argon` price doubles
  ($4 / $20 / $0.20).
- **2026-12-11** `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5-pro`, `o3`, `o3-pro` (→ `gpt-5.6-sol`).
- **Late Dec 2026** Kimi K2 series (`kimi-k2.7-code`, still the `moonshot-cn`/`kimi-cn`
  default: confirm K3.1 Pro on api.moonshot.cn before switching).
- **2027-01-01** `gemini-3.8-flash`, `gemini-3.6-flash` prices double.
- **2027-01-06** `tts-1`, `gpt-4o-mini-tts`. **2027-01-14** dated `gpt-4.1*` snapshots.
- **2027-02-10** `gpt-image-2` (→ `gpt-image-2.6-{sunburst,flare}`; the `gpt-image-1` alias
  points at it). **2027-02-26** `whisper-1`, `gpt-4o-*transcribe`.
- **2027-03-15** `gemini-2.5-flash-image`. **2027-04-01** `gpt-5.1`, `gpt-5.3-codex`,
  `gpt-5.4-nano`. **2027-04-30** `mistral-large-2512`, Mistral Medium 3.5
  (`mistral-medium-latest`). **2027-05-07** `gemini-3.1-flash-lite`.
- Each retirement also needs a sweep of `../rho-llm-tutorial`.

---

## 🟠 Next up — features (minor releases)
- **P1 — Anthropic `thinking.block_binding` — S/M.** Send
  `{"prefix_mismatch_behavior":"drop_block"}` (beta `thinking-binding-controls-2026-08-01`)
  and parse `input_transformations`. Without it Fable 5.2 / Mythos 5.2 (every account) and
  Opus / Sonnet / Haiku 5.5, Fable 5.1 (accounts created after 2026-08-31) answer 400
  "…bound to a different conversation" when a session changes tools, system or history. The
  adapter itself edits history (hoists `RoleSystem`, drops empty parts). Not reproduced live.
- **P2 — Reasoning on OpenAI-compatible hosts — M.** `openai_compat` refuses any
  `ThinkingLevel`; DeepSeek, Kimi, Qwen, GLM, OpenRouter, Together, Groq, vLLM support it in
  host-specific dialects (pi: 11 `thinkingFormat`s, `openai-completions.ts:871-968`). Includes
  replaying `reasoning_content` on tool-call turns, which DeepSeek V4 / Kimi K3 require
  (*400 without it unverified*), and reading `reasoning_text` / OpenRouter `reasoning_details`.
- **P3 — Per-host compat flags as data — M.** Detected defaults from provider/BaseURL,
  overridable per model (pi `compat-schema.ts`): `max_tokens` vs `max_completion_tokens`,
  `developer` role, `store:false`, non-null assistant content, tool-ID format, usage field
  variants (DeepSeek `prompt_cache_hit_tokens`, Kimi top-level `cached_tokens`,
  `reasoning_tokens`). Replaces scattered `provider == …` checks.
- **P4 — Anthropic message threads — M.** Beta `message-threads-2026-08-12`: send only new
  messages; full request when the prefix changed or the server can't continue.
- **P5 — Overflow and retryability classifiers — S.** ~27 overflow patterns + exclusions
  (Anthropic 413, llama.cpp, MiniMax, DashScope, z.ai, xAI, Groq); quota/billing 429s
  non-retryable; honour `x-should-retry`; fail fast when `Retry-After` exceeds the cap
  (pi `overflow.ts`, `retry.ts`). Includes the OpenRouter "input + max_tokens" false positive.
- **F1 — Chat fallback + per-attempt usage — M.** `OperationChat` usage events and a
  `NewFallbackClient` for chat reusing `tryEach`/`FailoverPolicy`.

## Reasoning, caching & conversation state
- 🟡 **P6 — Responses API reasoning replay — M** (`include: reasoning.encrypted_content`,
  item ids); also xAI on the Responses API.
- 🟡 **P7 — Prompt caching on OpenAI hosts — S/M.** `prompt_cache_key`, Responses
  `session-id` affinity header, retention; optional automatic cache markers.
- 🟡 **P8 — Mid-conversation effort/system/tool changes that keep the cache — L.** In-place
  `{"role":"system"}` was tried in v0.12.2 and dropped (Haiku 5.5 ignored it, 3 of 3 live).
- 🟡 **R1 — Long-context price tiers — S.** Haiku 5.5 >100K, GPT >272K, Grok/Gemini >200K,
  MiniMax >512K, Qwen >256K, Muse Ember >200K; Azure keeps OpenAI's tier.
- 🟡 **R2 — OpenAI cache writes — S.** GPT-5.6+ bill writes at 1.25× input (30 min) and 2.5×
  (12h); rows have no write price.
- 🟡 **F2 — Gemini `store` default — S.** `store=true` default for new paid projects since
  2026-10-26; decide whether to send `store:false`.
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
- 🟡 **R4 — Perplexity Agent API adapter — M.** Sonar Chat Completions ended 2026-09-27;
  documented path is `POST /v1/sonar`, replacement `POST /v1/agent`; only published rate is
  `perplexity/sonar` $0.25 / $2.50, so our `sonar*` rows are stale.
- 🟢 **R5 — DashScope workspace endpoints** (`https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com`,
  recommended since 2026-10-13) and the Qwen Token Plan CN provider.
- 🟢 **R6 — Unconfirmed data.** OpenAI cache-read prices for `gpt-5.2`/`5.1`/`5`/`5-mini`/
  `5-nano`/`o3`; `gpt-image-2.5-*` prices; DashScope `qwen3-max`, `qwen3.5-omni-light`,
  `text-embedding-v2` missing from the 2026-11-07 price page; `gemini-4-argon` max output (row
  32,768, model page 65,536); Nemotron 3 Ultra context (1,048,576 per build.nvidia.com) and
  seven Fireworks rows with context 0; max output of Muse Spark/Wonder and Mistral Large 4;
  Cohere `command-a-03-2025` price; Nano Banana limits (docs vs `GET /v1beta/models`
  disagree); Gemini 3.5/3.7 Flash are routed to 3.6/3.8 Flash (billing may differ).
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
  Live / other vendors. See B35 first.

---

## 📝 Docs
- **D1 — `docs/ARCHITECTURE.md` drift — M.** Default model is `claude-sonnet-5-5` (L776,
  plus `MaxRetries: 10`, `BetaFeatures`); batch is OpenAI, Anthropic and Gemini (L878-880,
  926-929, `ProviderPreset.SupportsBatch` missing at L276, `provider.go:8` comment); add
  `Recover` to `BatchClient`; unexported `modelRegistry`/`modelAliases`/`defaultModels` +
  `retiredModels` (L39, 594-596); real type names (`anthropic.Client` …, L162, 255-258);
  `newSingleClient`, `MarkSuccessByName`, `MarkFailedByNameWithCooldown` (L111, 447, 451);
  "Probe re-arm (Unreleased)" → v0.4.1 (L544); logging wraps the pool, and the factory tree
  needs `openai_responses` (L289-301, 114-127, 262-270); package tree incomplete; "kit"
  (L529). Add sections: egress/SSRF, capabilities, OpenAI Responses, batch drivers. Rewrite
  version-tagged paragraphs in the present tense.
- **D2 — README — S.** `MaxRetries` (L618, 1242; after B15); Fable 5.2 in the thinking notes
  (L440, 446-447); an "Upgrading to v0.17.0" note (`GetAvailableModels` → `ModelsByProvider`
  filtered on `CapabilityChat`; `llm.Max*` → `Config`); Quick Start shows
  `import llm "github.com/bds421/rho-llm"`; Gemini padding logs at debug (L1417); retryable
  list incl. 504/529 (L716, 738); ThoughtSignature on all Gemini 3+ (L1384); retired-alias
  note placement and completeness (L1345); Anthropic/Gemini base URLs with paths (L43-44);
  pi link → `earendil-works/pi`, and describe the relationship accurately (shared chat
  core, diverged since); package tree (L1471-1502).
- **D3 — README structure — M.** Table of contents; promote Modalities, Batch and Security to
  top-level sections; fold "Exponential Backoff" into Retry; drop "Before v0.9.5…" asides;
  document `WithTools`, `Session.StreamMessages`, `Conversation.ToRequest`/`Append`,
  `NewToolResultParts`, `NewWebSearchTool`, `BatchClient.Recover`/`RecoveryKey`/
  `ErrBatchNotFound`, `PooledClient.PoolStatus`, `PresetFor`, mock helpers,
  `Config.MaxBatchDownloadBytes`; generate the alias and default-model tables from the
  registry (missing today: Kimi, Meta, MiniMax, `gpt5.6-*`, `grok-build`). Keep the package
  tree only in ARCHITECTURE and the Config table only in README.
- **D4 — `CLAUDE.md` — S.** Tutorial repo has 25 modules + root (not 22); Fable 5/5.1/5.2 in
  the thinking-replay line; pi link.
- **D5 — Release bookkeeping — S.** The `v0.7.6` tag exists only locally (never pushed).
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
