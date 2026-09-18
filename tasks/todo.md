# rho-llm — Roadmap & Issue Tracker

> **Workflow:** when an item here is completed, **remove it from this file** and add a
> corresponding entry to `CHANGELOG.md` (under `## [Unreleased]`). This keeps `todo.md`
> a list of *open* work and `CHANGELOG.md` the record of *shipped* work. See `CLAUDE.md`.

Priority: 🔴 critical · 🟠 high · 🟡 medium · 🟢 low
Effort: **S** (hours) · **M** (≈1 day) · **L** (multi-day)

---

## ⏸️ RESUME HERE — v0.7.6 is committed + tagged locally, NOT pushed

**State as of 2026-09-18:** working tree clean, `make ci` green, `v0.7.6` annotated
tag created on commit `f50cab7`. `origin/main` is still at `v0.7.5` (`35817fe`).

**Verification done (round 2):** every fix was reverted one at a time to confirm its
tests go red — all pinned. That audit caught the Gemini break-tests being vacuous
(they passed with the fix removed); they now assert the real property and fail across
8 sub-cases when reverted. Live re-test: 3 providers x 4 models, buffered + streaming
tool calls with full round-trips, parallel tool calls, `MaxTokens: 0`, thinking at
16000 and 2000, retired-model error, and an Anthropic->Gemini handoff — **0 failures**.

**Test-quality audit (round 3):** ran a mutation audit — flip an invariant in
production code, see whether any test goes red. A surviving mutant = an unpinned line.
Fixed three survivors (TLS 1.2 floor, redirect hop cap, Gemini `TokensNotReported`
sentinel); all three now killed, each verified by re-mutating. One survivor was left
deliberately: the `Timeout` branch in `applyConfigFloors` is redundant with
`NewSafeHTTPClient`'s own floor, so no test can observe it — documented in the code
rather than papered over with a fake test.

**Worth continuing:** only ~10 invariants were mutated out of 672 tests. Running a
real mutation tool (e.g. `go-mutesting`) over the whole package would likely find more
unpinned lines. The 10 assertion-free tests found by static scan were all checked and
are legitimate (nil-safety / `-race` tests where a panic is the failure).

### 🔴 Next action: decide whether to push — **S**
```bash
cd ~/Work/2026/bds421/rho/rho-llm
git fetch origin --tags                 # ALWAYS first — others release mid-session
git ls-remote --tags origin | tail -5   # confirm v0.7.6 is still free
git push origin main --follow-tags      # GitHub ONLY (GitLab mirror is stale)
```
Pre-push gate already passed: secret scan clean, README current, `make ci` green.
If the push is rejected, **never** `--force` — re-fetch and rebase.

### What v0.7.6 contains (5 bugs, all found by RUNNING code, not reading it)
1. **Gemini tool calls silently dropped** — Gemini returns `finishReason:"STOP"` even
   when requesting a function call, so the adapter reported `end_turn`. The documented
   agentic loop (`for resp.StopReason == "tool_use"`) never ran. Fixed in both
   `parseResponse` and `parseStream`; `MAX_TOKENS` preserved.
2. **Anthropic thinking budget > max_tokens → HTTP 400** — budget clamped only to the
   model's registry ceiling, not the request's `max_tokens`. Now clamped to both, with
   Anthropic's 1024 minimum respected (thinking disabled when both bounds can't hold).
3. **`Config.MaxTokens: 0` → HTTP 400** — the 8192 default lived only in
   `DefaultConfig()`, so struct-literal configs (what the README teaches) sent
   `max_tokens: 0`. Added `DefaultMaxTokens` floor.
4. **Same floor missing on batch/modality paths** — `NewBatchClient` emitted
   `"max_tokens": 0` in every entry. All three constructors now share
   `applyConfigFloors` (factory.go).
5. **7 retired Anthropic model IDs** — verified HTTP 404 against the live API.
   Removed + `RetiredModelReplacement` gives an actionable dispatch error.

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
