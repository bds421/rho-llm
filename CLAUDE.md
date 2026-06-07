# CLAUDE.md

Guidance for working in the `rho-llm` repository.

## What this is

`github.com/bds421/rho-llm` — a unified, provider-agnostic LLM client for Go covering
15 providers across 4 wire protocols (`anthropic`, `gemini`, `openai_compat`,
`openai_responses`), with streaming (`iter.Seq2`), tool use, extended thinking,
image/PDF input, multi-key auth-pool rotation, circuit breaking, configurable retry,
and cost estimation. A Go tribute to the TypeScript **pi** `ai` package
(`github.com/badlogic/pi-mono/tree/main/packages/ai`). See `docs/ARCHITECTURE.md` for the
full design.

## Build & verify

- `make build` — compile (`go build ./...`)
- `make test` — unit tests (`-short`; skips integration tests that need API keys)
- `make test-race` — race detector
- `make check` — quick pre-commit (`vet build test`)
- `make ci` — full pipeline: `fmt-check vet build test-race security vulncheck`
  (also run on push/PR via `.github/workflows/ci.yml`)
- **Always `gofmt -w` your changed files** — `fmt-check` is the first CI gate and fails the
  whole pipeline on any unformatted file.

## Task / changelog workflow  (IMPORTANT)

`tasks/todo.md` tracks **open** work. `CHANGELOG.md` records **shipped** work. They are
complementary, never overlapping.

When you **complete** an item from `tasks/todo.md`:

1. **Remove** the item from `tasks/todo.md` — it is no longer open work.
2. **Add** a corresponding entry to `CHANGELOG.md` under `## [Unreleased]`, in the right
   Keep-a-Changelog section (`Added` / `Changed` / `Fixed` / `Security`). Describe what
   changed and cite the audit-finding id (e.g. `F1`) or roadmap id (e.g. `P3`) when it has one.

Never leave a completed item checked-off and lingering in `tasks/todo.md` — move it. A
finished task lives in the changelog, not the todo list.

## Conventions

- Conventional-commit subjects (`feat:`, `fix:`, `docs:`, `refactor:`, `security:`).
- Structured errors via `APIError` + the `Is*()` classifiers; never put API keys in errors
  or logs (`Config` and `AuthProfile` redact `APIKey` in their `MarshalJSON`).
- Per-client tunables live on `Config`; read them via the `Effective*()` accessors, not the
  package-global `llm.Max*` vars (those are only the process-wide fallback).
- New providers register themselves in `init()` via `llm.RegisterProvider` and are wired
  through `provider/all.go` (the `database/sql` driver pattern).
