# CLAUDE.md

Guidance for working in the `rho-llm` repository.

## What this is

`github.com/bds421/rho-llm` — a unified, provider-agnostic LLM client for Go covering
**20 providers across 4 wire protocols** (`anthropic`, `gemini`, `openai_compat`,
`openai_responses`). Features: streaming (`iter.Seq2`), tool use, extended thinking,
image + PDF/document input, **serializable conversations with cross-provider handoff**,
multi-key auth-pool rotation, circuit breaking, configurable retry, and cost estimation.
A Go tribute to the TypeScript **pi** `ai` package
(`github.com/badlogic/pi-mono/tree/main/packages/ai`). See `docs/ARCHITECTURE.md` for the
full design and `tasks/todo.md` for the roadmap.

## Build & verify

- `make build` — compile (`go build ./...`)
- `make test` — unit tests (`-short`; skips integration tests that need API keys)
- `make test-race` — race detector
- `make check` — quick pre-commit (`vet build test`)
- `make ci` — full pipeline: `fmt-check vet build test-race security vulncheck`
  (also run on push/PR via `.github/workflows/ci.yml`)
- **Always `gofmt -w` your changed files** — `fmt-check` is the first CI gate and fails the
  whole pipeline on any unformatted file.

## Testing  (IMPORTANT)

- **"If nothing breaks, the tests aren't complete."** Adversarial / break-the-system tests must
  genuinely try to break the code — nil/empty/malformed/corrupt inputs, concurrency (under `-race`),
  failure paths, and hostile transcripts — not just the happy path. A break-it pass where everything
  passes means it wasn't hostile enough.
- **Prove each test catches its bug.** When you fix a defect, confirm the regression test fails
  *without* the fix (revert it, run, watch it go red, restore). A test that passes regardless is
  vacuous.
- **Verify backward compatibility against the consumer repo.** The sibling `../rho-llm-tutorial`
  (22 separate Go modules, each pinning a released rho-llm) is a real consumer. To test *unreleased*
  changes against it, point all its modules at this working tree with a temporary workspace
  (`cd ../rho-llm-tutorial && go work init && go work use -r . && go work use ../rho-llm` — `go.work`
  is gitignored there), then `make build-all` + `make vet-all` + `make test-stress` (offline, `-race`).
  `make test-capability` and tutorials like `15_multi_provider` need API keys / a live Ollama.
  Remove the `go.work` when done.

## Task / changelog workflow  (IMPORTANT)

`tasks/todo.md` tracks **open** work. `CHANGELOG.md` records **shipped** work. They are
complementary, never overlapping.

When you **complete** an item from `tasks/todo.md`:

1. **Remove** the item from `tasks/todo.md` — it is no longer open work.
2. **Add** a corresponding entry to `CHANGELOG.md` under `## [Unreleased]`, in the right
   Keep-a-Changelog section (`Added` / `Changed` / `Fixed` / `Security`). Describe what
   changed and cite the roadmap id (e.g. `P1`) or audit-finding id (e.g. `F1`) when it has one.

Never leave a completed item checked-off and lingering in `tasks/todo.md` — move it. A
finished task lives in the changelog, not the todo list.

## Releases & remotes

- **Two remotes.** `origin` → GitHub `github.com/bds421/rho-llm` (canonical/public). `gitlab` →
  `git@gitlab2024.bds421-cloud.com:bds421/rho/llm.git` (internal mirror on the bds421-cloud GitLab,
  alongside the team's other `bds421/rho/*` projects). Releases can go to either or both.
- ⚠️ **`bds421/rho/pdf` is a DIFFERENT project** (an AI PDF-extraction service, cloned at
  `../rho-pdf`) — never push rho-llm there or merge the two.
- **Committer identity.** This checkout has no global git user configured (git otherwise invents a
  machine-hostname identity — set author + committer explicitly per commit). GitLab commits use
  `Rene Heinzl <reneheinzl@macbookpro.lan>` (the identity the `bds421/rho/*` GitLab repos use); the
  GitHub history's canonical identity is `rene@bds421.com` (unified via `.mailmap`).
- **Before pushing to GitHub** (public, and the Go module source) ALWAYS check first: (1) **no
  leaked credentials** in tracked files / the diff — secret grep over `git ls-files`, only test
  fakes allowed, `.gitignore` covers `.env`/`.claude`; (2) **`README.md` is up to date**; and
  (3) **CI is green** — run `make ci` (incl. `gosec`/`govulncheck`, which aren't installed by
  default — `go install` them first) so the GitHub Actions check won't go red. Leaks and stale
  public docs are hard to retract. (GitLab is the internal mirror — this gate is for the public
  GitHub push.)
- **Cutting a release:**
  1. Move `## [Unreleased]` content into `## [X.Y.Z] - YYYY-MM-DD` in `CHANGELOG.md` (leave an empty
     `[Unreleased]`). Minor bump for features, patch for fixes.
  2. Bump the version stamp at the top of `docs/ARCHITECTURE.md`.
  3. Commit with a conventional subject ending in `(vX.Y.Z)`; create an **annotated** tag `vX.Y.Z`
     with a `vX.Y.Z — summary` message (matches existing tags).
  4. Push `main` + tags to the intended remote(s).

## Research & subagents  (IMPORTANT)

Treat subagent (`Agent` tool) and web-research output as **unverified leads, not facts** — do not
trust them blindly. Before relying on a specific claim (file paths, line numbers, type/field names,
API signatures, versions, bug/benchmark assertions), verify it against the primary source (the real
repo, code, or official docs). Cross-check load-bearing claims across independent sources — one
agent's confident assertion is not evidence — and label anything still unverified as such.

## Architecture map

Layered decorator stack over a stateless `Client` interface (`Complete`/`Stream`/`Provider`/`Model`/`Close`):

- **Entry:** `factory.go` (`NewClient`/`NewClientWithKeys`) → resolves protocol (`provider.go`) → looks up the registered adapter (`register.go`, driver pattern; adapters self-register in `init()` and are wired by `provider/all.go`).
- **Resilience decorators:** `middleware.go` (privacy-safe logging) wraps `pool.go` (`PooledClient`: multi-key `AuthPool` rotation + retry), which uses `circuitbreaker.go` (3-state, nil-safe) and `retrypolicy.go` (backoff + jitter + hooks).
- **Adapters** (`provider/*/`): translate the neutral `[]Message` to each wire format and normalize responses/stop-reasons back. All HTTP goes through `SafeHTTPClient` (`config.go`: TLS 1.2+, strips auth headers on cross-host redirect).
- **Conversation layer** (sits *above* the `Client`, never changes single-request behavior):
  - `conversation.go` — `Conversation` (serializable transcript + `Usage`, versioned JSON via `LoadConversation`).
  - `session.go` — `Session` (mutex-guarded driver; `Send`/`Stream` auto-append with provenance; `SwitchProvider` = provider handoff).
  - `normalize.go` — `NormalizeForProvider` (the single cross-provider translation pass).
- **Registry:** `registry.go` — model metadata, alias resolution, `EstimateCost`.

## Conventions

- Conventional-commit subjects (`feat:`, `fix:`, `docs:`, `refactor:`, `security:`).
- Structured errors via `APIError` + the `Is*()` classifiers; never put API keys in errors
  or logs (`Config` and `AuthProfile` redact `APIKey` in their `MarshalJSON`).
- Per-client tunables live on `Config`; read them via the `Effective*()` accessors, not the
  package-global `llm.Max*` vars (those are only the process-wide fallback).
- New providers register themselves in `init()` via `llm.RegisterProvider` and are wired
  through `provider/all.go` (the `database/sql` driver pattern).
- **Messages are the neutral currency.** Keep `Message`/`ContentPart` a concrete tagged struct
  (discriminator + `omitempty` fields) — never a bare interface — so it JSON-round-trips without
  custom marshalers. Provider-opaque artifacts ride as optional fields (`ThinkingSignature`,
  `ThoughtSignature`); assistant messages carry `Provider`/`Model`/`StopReason` provenance.
- **Cross-provider handoff is lossy by design** for reasoning (signatures don't transfer) but
  lossless for text/images/documents/tools. All handoff translation belongs in one place —
  `NormalizeForProvider` — not scattered across adapters. Only the Anthropic adapter replays
  thinking blocks (it captures the signature); other adapters skip thinking parts, and
  normalization degrades cross-provider thinking to text before it reaches them.
- **Serialized formats are versioned and language-agnostic.** Stamp a `schema_version`, key by
  explicit JSON tags, and never tie the on-disk shape to Go type/package names (so persisted
  conversations survive refactors).
- **Thinking replay requires thinking enabled.** Anthropic rejects replayed thinking blocks unless
  extended thinking is on for the request — keep `ThinkingLevel` set across turns (`WithBaseRequest`
  or `Config`), or historical thinking blocks are silently dropped (no error).
- **Mind the `TokensNotReported` (-1) sentinel.** A streamed turn may carry it when the provider
  omits usage; clamp token counts to `>= 0` before accumulating or estimating cost (negatives
  corrupt `Usage`).
