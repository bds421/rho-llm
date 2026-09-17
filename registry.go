package llm

import (
	"fmt"
	"slices"
	"strings"
	"sync"
)

// registryMu guards the registry maps (modelRegistry, modelAliases,
// availableModels, defaultModels). The built-in data is written at package
// init; RegisterModel/RegisterModelAlias extend it at runtime, so every
// reader takes the read lock.
var registryMu sync.RWMutex

// ModelInfo holds per-model metadata used to adapt API requests and UI.
type ModelInfo struct {
	ID                   string  // Full model identifier
	Provider             string  // anthropic, xai, gemini, openai, groq, etc.
	MaxTokens            int     // Model-specific output limit (0 = use config default)
	ContextWindow        int     // Max input tokens (0 = unknown)
	InputPricePer1M      float64 // USD per 1M input tokens (0 = unknown/free)
	OutputPricePer1M     float64 // USD per 1M output tokens (0 = unknown/free)
	CacheWritePricePer1M float64 // Anthropic: price per 1M cache creation tokens (0 = not applicable)
	CacheReadPricePer1M  float64 // Anthropic/Gemini: price per 1M cached input tokens (0 = not applicable)
	SupportsThinking     bool    // Anthropic extended thinking
	ThoughtSignature     bool    // Gemini 3 models require thought_signature in function call responses
	Thinking             bool    // Model uses internal chain-of-thought reasoning (e.g. qwen3, deepseek-r1) — consumes output tokens invisibly
	ResponsesAPI         bool    // Model requires OpenAI Responses API (/v1/responses) for reasoning effort control (GPT-5 family)
	NoToolSupport        bool    // Model does not support tool/function calling (e.g. deepseek-r1, gemma)
	Label                string  // Short display name
}

// modelRegistry maps model ID to its metadata. Built-in entries are written
// at init; RegisterModel extends/overrides at runtime under registryMu.
var modelRegistry = map[string]ModelInfo{
	// Anthropic — from platform.claude.com/docs models overview + pricing (2026-09-17).
	// Every Claude model ID is a pinned snapshot; from the 4.6 generation on the IDs are dateless
	// (claude-opus-5 is itself the snapshot, not an alias for a dated version).
	// Cache pricing: 5m write = 1.25x input, read = 0.1x input — EXCEPT Fable 5.1 / Mythos 5.1,
	// whose cache reads are 0.025x input ($0.25/MTok), not 0.1x.
	// RETIRED and removed (requests now fail on the first-party API, per
	// platform.claude.com/docs/en/about-claude/model-deprecations): claude-opus-4-1(-20250805)
	// and claude-opus-4-0/claude-opus-4-20250514 (retired 2026-08-05 / 2026-06-15),
	// claude-sonnet-4-0/claude-sonnet-4-20250514 (2026-06-15), claude-3-haiku-20240307 (2026-04-20).
	"claude-fable-5-1":           {ID: "claude-fable-5-1", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 1000000, InputPricePer1M: 10.00, OutputPricePer1M: 50.00, CacheWritePricePer1M: 12.50, CacheReadPricePer1M: 0.25, SupportsThinking: true, Label: "Fable 5.1"},
	"claude-fable-5":             {ID: "claude-fable-5", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 1000000, InputPricePer1M: 10.00, OutputPricePer1M: 50.00, CacheWritePricePer1M: 12.50, CacheReadPricePer1M: 1.00, SupportsThinking: true, Label: "Fable 5"},
	"claude-opus-5":              {ID: "claude-opus-5", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 1000000, InputPricePer1M: 5.00, OutputPricePer1M: 25.00, CacheWritePricePer1M: 6.25, CacheReadPricePer1M: 0.50, SupportsThinking: true, Label: "Opus 5"},
	"claude-opus-4-8":            {ID: "claude-opus-4-8", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 1000000, InputPricePer1M: 5.00, OutputPricePer1M: 25.00, CacheWritePricePer1M: 6.25, CacheReadPricePer1M: 0.50, SupportsThinking: true, Label: "Opus 4.8"},
	"claude-opus-4-7":            {ID: "claude-opus-4-7", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 1000000, InputPricePer1M: 5.00, OutputPricePer1M: 25.00, CacheWritePricePer1M: 6.25, CacheReadPricePer1M: 0.50, SupportsThinking: true, Label: "Opus 4.7"},
	"claude-opus-4-6":            {ID: "claude-opus-4-6", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 1000000, InputPricePer1M: 5.00, OutputPricePer1M: 25.00, CacheWritePricePer1M: 6.25, CacheReadPricePer1M: 0.50, SupportsThinking: true, Label: "Opus 4.6"},
	"claude-opus-4-5":            {ID: "claude-opus-4-5", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 200000, InputPricePer1M: 5.00, OutputPricePer1M: 25.00, CacheWritePricePer1M: 6.25, CacheReadPricePer1M: 0.50, SupportsThinking: true, Label: "Opus 4.5"},
	"claude-opus-4-5-20251101":   {ID: "claude-opus-4-5-20251101", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 200000, InputPricePer1M: 5.00, OutputPricePer1M: 25.00, CacheWritePricePer1M: 6.25, CacheReadPricePer1M: 0.50, SupportsThinking: true, Label: "Opus 4.5 (Nov)"},
	"claude-sonnet-5":            {ID: "claude-sonnet-5", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 1000000, InputPricePer1M: 2.00, OutputPricePer1M: 10.00, CacheWritePricePer1M: 2.50, CacheReadPricePer1M: 0.20, SupportsThinking: true, Label: "Sonnet 5"},
	"claude-sonnet-4-6":          {ID: "claude-sonnet-4-6", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 1000000, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, CacheWritePricePer1M: 3.75, CacheReadPricePer1M: 0.30, SupportsThinking: true, Label: "Sonnet 4.6"},
	"claude-sonnet-4-5":          {ID: "claude-sonnet-4-5", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 200000, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, CacheWritePricePer1M: 3.75, CacheReadPricePer1M: 0.30, SupportsThinking: true, Label: "Sonnet 4.5"},
	"claude-sonnet-4-5-20250929": {ID: "claude-sonnet-4-5-20250929", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 200000, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, CacheWritePricePer1M: 3.75, CacheReadPricePer1M: 0.30, SupportsThinking: true, Label: "Sonnet 4.5 (Sep)"},
	"claude-haiku-4-5":           {ID: "claude-haiku-4-5", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 200000, InputPricePer1M: 1.00, OutputPricePer1M: 5.00, CacheWritePricePer1M: 1.25, CacheReadPricePer1M: 0.10, SupportsThinking: true, Label: "Haiku 4.5"},
	"claude-haiku-4-5-20251001":  {ID: "claude-haiku-4-5-20251001", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 200000, InputPricePer1M: 1.00, OutputPricePer1M: 5.00, CacheWritePricePer1M: 1.25, CacheReadPricePer1M: 0.10, SupportsThinking: true, Label: "Haiku 4.5 (Oct)"},

	// xAI / Grok — from docs.x.ai/developers/models (2026-09-17; docs moved from docs.x.ai/docs/*).
	// Prices here are the <200K-prompt tier. xAI bills a CLIFF, not a marginal tier: "Requests
	// whose prompt reaches the listed token threshold are billed at the higher rate for all
	// tokens in the request" (>=200K doubles input/cached/output). EstimateCost models only the
	// standard tier, so a >=200K-prompt request costs 2x what it reports — see the note on
	// EstimateCost. ID NOTE: the docs now list the 4.20 family WITHOUT the "-beta" infix
	// (grok-4.20-0309-reasoning, not grok-4.20-beta-0309-reasoning); both spellings are kept —
	// the new ones because they are what the docs publish, the old ones because xAI has
	// announced no retirement/rename for them and they may still resolve.
	"grok-4.6":                          {ID: "grok-4.6", Provider: "xai", ContextWindow: 500000, InputPricePer1M: 2.00, OutputPricePer1M: 6.00, CacheReadPricePer1M: 0.50, Thinking: true, Label: "Grok 4.6"},
	"grok-4.5":                          {ID: "grok-4.5", Provider: "xai", ContextWindow: 500000, InputPricePer1M: 2.00, OutputPricePer1M: 6.00, CacheReadPricePer1M: 0.30, Thinking: true, Label: "Grok 4.5"},
	"grok-4.3":                          {ID: "grok-4.3", Provider: "xai", ContextWindow: 1000000, InputPricePer1M: 1.25, OutputPricePer1M: 2.50, CacheReadPricePer1M: 0.20, Thinking: true, Label: "Grok 4.3"},
	"grok-4.20-0309-reasoning":          {ID: "grok-4.20-0309-reasoning", Provider: "xai", ContextWindow: 1000000, InputPricePer1M: 1.25, OutputPricePer1M: 2.50, CacheReadPricePer1M: 0.20, Thinking: true, Label: "Grok 4.20 R"},
	"grok-4.20-0309-non-reasoning":      {ID: "grok-4.20-0309-non-reasoning", Provider: "xai", ContextWindow: 1000000, InputPricePer1M: 1.25, OutputPricePer1M: 2.50, CacheReadPricePer1M: 0.20, Label: "Grok 4.20"},
	"grok-4.20-multi-agent-0309":        {ID: "grok-4.20-multi-agent-0309", Provider: "xai", ContextWindow: 1000000, InputPricePer1M: 1.25, OutputPricePer1M: 2.50, CacheReadPricePer1M: 0.20, Thinking: true, Label: "Grok 4.20 Agent"},
	"grok-build-0.1":                    {ID: "grok-build-0.1", Provider: "xai", ContextWindow: 256000, InputPricePer1M: 1.00, OutputPricePer1M: 2.00, CacheReadPricePer1M: 0.20, Thinking: true, Label: "Grok Build"},
	"grok-4.20-beta":                    {ID: "grok-4.20-beta", Provider: "xai", ContextWindow: 2000000, InputPricePer1M: 2.00, OutputPricePer1M: 6.00, Thinking: true, Label: "Grok 4.20 Beta (legacy ID)"},
	"grok-4.20-beta-0309-reasoning":     {ID: "grok-4.20-beta-0309-reasoning", Provider: "xai", ContextWindow: 2000000, InputPricePer1M: 2.00, OutputPricePer1M: 6.00, Thinking: true, Label: "Grok 4.20 R (legacy ID)"},
	"grok-4.20-beta-0309-non-reasoning": {ID: "grok-4.20-beta-0309-non-reasoning", Provider: "xai", ContextWindow: 2000000, InputPricePer1M: 2.00, OutputPricePer1M: 6.00, Label: "Grok 4.20 (legacy ID)"},
	"grok-4.20-multi-agent-beta-0309":   {ID: "grok-4.20-multi-agent-beta-0309", Provider: "xai", ContextWindow: 2000000, InputPricePer1M: 2.00, OutputPricePer1M: 6.00, Thinking: true, Label: "Grok 4.20 Agent (legacy ID)"},
	"grok-4-1-fast-reasoning":           {ID: "grok-4-1-fast-reasoning", Provider: "xai", ContextWindow: 2000000, InputPricePer1M: 0.20, OutputPricePer1M: 0.50, Thinking: true, Label: "Grok 4.1 R (legacy ID)"},
	"grok-4-1-fast-non-reasoning":       {ID: "grok-4-1-fast-non-reasoning", Provider: "xai", ContextWindow: 2000000, InputPricePer1M: 0.20, OutputPricePer1M: 0.50, Label: "Grok 4.1 (legacy ID)"},
	"grok-4-fast-reasoning":             {ID: "grok-4-fast-reasoning", Provider: "xai", ContextWindow: 2000000, InputPricePer1M: 0.20, OutputPricePer1M: 0.50, Thinking: true, Label: "Grok 4 R (legacy ID)"},
	"grok-4-fast-non-reasoning":         {ID: "grok-4-fast-non-reasoning", Provider: "xai", ContextWindow: 2000000, InputPricePer1M: 0.20, OutputPricePer1M: 0.50, Label: "Grok 4 (legacy ID)"},
	"grok-4-0709":                       {ID: "grok-4-0709", Provider: "xai", ContextWindow: 256000, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, Thinking: true, Label: "Grok 4 (Jul, legacy ID)"},
	"grok-code-fast-1":                  {ID: "grok-code-fast-1", Provider: "xai", ContextWindow: 256000, InputPricePer1M: 0.20, OutputPricePer1M: 1.50, Thinking: true, Label: "Grok Code (legacy ID)"},
	"grok-3":                            {ID: "grok-3", Provider: "xai", ContextWindow: 131072, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, Label: "Grok 3 (legacy ID)"},
	"grok-3-mini":                       {ID: "grok-3-mini", Provider: "xai", ContextWindow: 131072, InputPricePer1M: 0.30, OutputPricePer1M: 0.50, Thinking: true, Label: "Grok 3 Mini (legacy ID)"},

	// Gemini — from ai.google.dev/gemini-api/docs/pricing (2026-09-17).
	// SCHEDULED PRICE INCREASE: gemini-3.8/3.7/3.6-flash carry INTRODUCTORY rates through
	// 2026-12-31 ($0.75/$3.75, cache $0.075) that DOUBLE on 2027-01-01 to $1.50/$7.50
	// (cache $0.15), per the pricing page. ModelInfo holds one flat price, so these rows will
	// UNDERSTATE cost from 2027-01-01 — they must be re-stamped then. (Tracked in tasks/todo.md.)
	// gemini-3.1-pro-preview is tiered by prompt size (<=200K vs >200K): the >200K tier is
	// $4.00/$18.00 — the standard tier is modeled here.
	"gemini-3.8-flash":              {ID: "gemini-3.8-flash", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 0.75, OutputPricePer1M: 3.75, CacheReadPricePer1M: 0.075, ThoughtSignature: true, Label: "Gemini 3.8 Flash (intro price to 2026-12-31)"},
	"gemini-3.7-flash":              {ID: "gemini-3.7-flash", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 0.75, OutputPricePer1M: 3.75, CacheReadPricePer1M: 0.075, ThoughtSignature: true, Label: "Gemini 3.7 Flash (intro price to 2026-12-31)"},
	"gemini-3.6-flash":              {ID: "gemini-3.6-flash", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 0.75, OutputPricePer1M: 3.75, CacheReadPricePer1M: 0.075, ThoughtSignature: true, Label: "Gemini 3.6 Flash (intro price to 2026-12-31)"},
	"gemini-3.5-flash":              {ID: "gemini-3.5-flash", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 1.50, OutputPricePer1M: 9.00, CacheReadPricePer1M: 0.15, ThoughtSignature: true, Label: "Gemini 3.5 Flash"},
	"gemini-3.5-flash-lite":         {ID: "gemini-3.5-flash-lite", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 0.30, OutputPricePer1M: 2.50, ThoughtSignature: true, Label: "Gemini 3.5 Flash Lite"},
	"gemini-3.1-pro-preview":        {ID: "gemini-3.1-pro-preview", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 2.00, OutputPricePer1M: 12.00, CacheReadPricePer1M: 0.20, ThoughtSignature: true, Label: "Gemini 3.1 Pro"},
	"gemini-3.1-flash-lite-preview": {ID: "gemini-3.1-flash-lite-preview", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 0.25, OutputPricePer1M: 1.50, ThoughtSignature: true, Label: "Gemini 3.1 Flash Lite"},
	"gemini-3-pro-preview":          {ID: "gemini-3-pro-preview", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 2.00, OutputPricePer1M: 12.00, ThoughtSignature: true, Label: "Gemini 3 Pro"},
	"gemini-3-flash-preview":        {ID: "gemini-3-flash-preview", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 0.50, OutputPricePer1M: 3.00, ThoughtSignature: true, Label: "Gemini 3 Flash"},
	"gemini-2.5-pro":                {ID: "gemini-2.5-pro", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 1.25, OutputPricePer1M: 10.00, Thinking: true, Label: "Gemini 2.5 Pro"},
	"gemini-2.5-flash":              {ID: "gemini-2.5-flash", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 0.30, OutputPricePer1M: 2.50, Thinking: true, Label: "Gemini 2.5 Flash"},
	"gemini-2.5-flash-lite":         {ID: "gemini-2.5-flash-lite", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 0.10, OutputPricePer1M: 0.40, Thinking: true, Label: "Flash Lite"},
	"gemini-2.0-flash":              {ID: "gemini-2.0-flash", Provider: "gemini", MaxTokens: 8192, ContextWindow: 1048576, InputPricePer1M: 0.10, OutputPricePer1M: 0.40, Label: "Gemini 2.0 Flash"},

	// OpenAI — GPT-6 / GPT-5.6 families (2026-09-17, from developers.openai.com/api/docs/pricing;
	// the old platform.openai.com/docs/* URLs now 301 to developers.openai.com/api/docs/*).
	// LONG-CONTEXT CLIFF: requests over 272K input tokens are billed at a higher rate for the
	// WHOLE request (gpt-6-astra 20/2/75, sol 8/0.80/30, terra 4/0.40/18, luna 0.40/0.04/1.80).
	// Prices below are the standard (<=272K) tier — EstimateCost does not model the cliff.
	"gpt-6-astra":   {ID: "gpt-6-astra", Provider: "openai", MaxTokens: 128000, ContextWindow: 1050000, InputPricePer1M: 10.00, OutputPricePer1M: 50.00, CacheReadPricePer1M: 1.00, Thinking: true, ResponsesAPI: true, Label: "GPT-6 Astra"},
	"gpt-5.6-sol":   {ID: "gpt-5.6-sol", Provider: "openai", MaxTokens: 128000, ContextWindow: 1050000, InputPricePer1M: 4.00, OutputPricePer1M: 20.00, CacheReadPricePer1M: 0.40, Thinking: true, ResponsesAPI: true, Label: "GPT-5.6 Sol"},
	"gpt-5.6-terra": {ID: "gpt-5.6-terra", Provider: "openai", MaxTokens: 128000, ContextWindow: 1050000, InputPricePer1M: 2.00, OutputPricePer1M: 12.00, CacheReadPricePer1M: 0.20, Thinking: true, ResponsesAPI: true, Label: "GPT-5.6 Terra"},
	"gpt-5.6-luna":  {ID: "gpt-5.6-luna", Provider: "openai", MaxTokens: 128000, ContextWindow: 1050000, InputPricePer1M: 0.20, OutputPricePer1M: 1.20, CacheReadPricePer1M: 0.02, Thinking: true, ResponsesAPI: true, Label: "GPT-5.6 Luna"},

	// OpenAI — GPT-5.x family (2026-06-03)
	// Reasoning models use ResponsesAPI: true — reasoning effort is controlled via /v1/responses, not Chat Completions.
	// Non-reasoning "chat" variants use Chat Completions normally.
	"gpt-5.5-pro":         {ID: "gpt-5.5-pro", Provider: "openai", MaxTokens: 128000, ContextWindow: 1048576, InputPricePer1M: 30.00, OutputPricePer1M: 180.00, Thinking: true, ResponsesAPI: true, Label: "GPT-5.5 Pro"},
	"gpt-5.5":             {ID: "gpt-5.5", Provider: "openai", MaxTokens: 128000, ContextWindow: 1048576, InputPricePer1M: 5.00, OutputPricePer1M: 30.00, CacheReadPricePer1M: 0.50, Thinking: true, ResponsesAPI: true, Label: "GPT-5.5"},
	"gpt-5.4-pro":         {ID: "gpt-5.4-pro", Provider: "openai", MaxTokens: 128000, ContextWindow: 1048576, InputPricePer1M: 30.00, OutputPricePer1M: 180.00, Thinking: true, ResponsesAPI: true, Label: "GPT-5.4 Pro"},
	"gpt-5.4":             {ID: "gpt-5.4", Provider: "openai", MaxTokens: 128000, ContextWindow: 1048576, InputPricePer1M: 2.50, OutputPricePer1M: 15.00, CacheReadPricePer1M: 0.25, Thinking: true, ResponsesAPI: true, Label: "GPT-5.4"},
	"gpt-5.4-mini":        {ID: "gpt-5.4-mini", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 0.75, OutputPricePer1M: 4.50, Thinking: true, ResponsesAPI: true, Label: "GPT-5.4 Mini"},
	"gpt-5.4-nano":        {ID: "gpt-5.4-nano", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 0.20, OutputPricePer1M: 1.25, Thinking: true, ResponsesAPI: true, Label: "GPT-5.4 Nano"},
	"gpt-5.3-chat-latest": {ID: "gpt-5.3-chat-latest", Provider: "openai", MaxTokens: 16384, ContextWindow: 128000, InputPricePer1M: 1.75, OutputPricePer1M: 14.00, Label: "GPT-5.3 Chat"},
	"gpt-5.3-codex":       {ID: "gpt-5.3-codex", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 1.75, OutputPricePer1M: 14.00, Thinking: true, ResponsesAPI: true, Label: "GPT-5.3 Codex"},
	"gpt-5.2":             {ID: "gpt-5.2", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 1.75, OutputPricePer1M: 14.00, Thinking: true, ResponsesAPI: true, Label: "GPT-5.2"},
	"gpt-5.2-pro":         {ID: "gpt-5.2-pro", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 21.00, OutputPricePer1M: 168.00, Thinking: true, ResponsesAPI: true, Label: "GPT-5.2 Pro"},
	"gpt-5.2-chat-latest": {ID: "gpt-5.2-chat-latest", Provider: "openai", MaxTokens: 16384, ContextWindow: 128000, InputPricePer1M: 1.75, OutputPricePer1M: 14.00, Label: "GPT-5.2 Chat"},
	"gpt-5.1":             {ID: "gpt-5.1", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 1.25, OutputPricePer1M: 10.00, Thinking: true, ResponsesAPI: true, Label: "GPT-5.1"},
	"gpt-5.1-chat-latest": {ID: "gpt-5.1-chat-latest", Provider: "openai", MaxTokens: 16384, ContextWindow: 128000, InputPricePer1M: 1.25, OutputPricePer1M: 10.00, Label: "GPT-5.1 Chat"},
	"gpt-5":               {ID: "gpt-5", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 1.25, OutputPricePer1M: 10.00, Thinking: true, ResponsesAPI: true, Label: "GPT-5"},
	"gpt-5-pro":           {ID: "gpt-5-pro", Provider: "openai", MaxTokens: 272000, ContextWindow: 400000, InputPricePer1M: 15.00, OutputPricePer1M: 120.00, Thinking: true, ResponsesAPI: true, Label: "GPT-5 Pro"},
	"gpt-5-chat-latest":   {ID: "gpt-5-chat-latest", Provider: "openai", MaxTokens: 16384, ContextWindow: 128000, InputPricePer1M: 1.25, OutputPricePer1M: 10.00, Label: "GPT-5 Chat"},
	"gpt-5-codex":         {ID: "gpt-5-codex", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 1.25, OutputPricePer1M: 10.00, Thinking: true, ResponsesAPI: true, Label: "GPT-5 Codex"},
	"gpt-5-mini":          {ID: "gpt-5-mini", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 0.25, OutputPricePer1M: 2.00, Thinking: true, ResponsesAPI: true, Label: "GPT-5 Mini"},
	"gpt-5-nano":          {ID: "gpt-5-nano", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 0.05, OutputPricePer1M: 0.40, Thinking: true, ResponsesAPI: true, Label: "GPT-5 Nano"},

	// OpenAI — GPT-4.1 family (non-reasoning, 1M context)
	"gpt-4.1":      {ID: "gpt-4.1", Provider: "openai", MaxTokens: 32768, ContextWindow: 1048576, InputPricePer1M: 2.00, OutputPricePer1M: 8.00, Label: "GPT-4.1"},
	"gpt-4.1-mini": {ID: "gpt-4.1-mini", Provider: "openai", MaxTokens: 32768, ContextWindow: 1048576, InputPricePer1M: 0.40, OutputPricePer1M: 1.60, Label: "GPT-4.1 Mini"},
	"gpt-4.1-nano": {ID: "gpt-4.1-nano", Provider: "openai", MaxTokens: 32768, ContextWindow: 1048576, InputPricePer1M: 0.10, OutputPricePer1M: 0.40, Label: "GPT-4.1 Nano"},

	// OpenAI — O-series reasoning models
	"o3":      {ID: "o3", Provider: "openai", MaxTokens: 100000, ContextWindow: 200000, InputPricePer1M: 2.00, OutputPricePer1M: 8.00, Thinking: true, Label: "O3"},
	"o3-mini": {ID: "o3-mini", Provider: "openai", MaxTokens: 100000, ContextWindow: 200000, InputPricePer1M: 1.10, OutputPricePer1M: 4.40, Thinking: true, Label: "O3 Mini"},
	"o3-pro":  {ID: "o3-pro", Provider: "openai", MaxTokens: 100000, ContextWindow: 200000, InputPricePer1M: 20.00, OutputPricePer1M: 80.00, Thinking: true, Label: "O3 Pro"},
	"o4-mini": {ID: "o4-mini", Provider: "openai", MaxTokens: 100000, ContextWindow: 200000, InputPricePer1M: 1.10, OutputPricePer1M: 4.40, Thinking: true, Label: "O4 Mini"},

	// Groq — cloud inference (2026-09-17, from console.groq.com/docs/models).
	// PRICE FIX: gpt-oss-120b was carried at $3.00/$8.00 and gpt-oss-20b at $0.30/$0.80 — both
	// were ~20x/13x too high vs Groq's published rates, and 120b is the `groq` default, so cost
	// estimates for it were badly overstated. Corrected to the documented $0.15/$0.60 and
	// $0.075/$0.30. Groq's llama-3.x/llama-4 models were removed in the 2026-07-08 pass
	// (deprecated, shut down 2026-07-17 / 2026-08-16) and are still gone from the catalog.
	// qwen/qwen3.6-27b is REMOVED: no longer in Groq's catalog; superseded by qwen/qwen3.8-27b,
	// which Groq lists as PREVIEW (not production) — added, labelled as such.
	// deepseek-r1-distill-* are REMOVED: no longer listed in Groq's model docs.
	"openai/gpt-oss-120b": {ID: "openai/gpt-oss-120b", Provider: "groq", MaxTokens: 65536, ContextWindow: 131072, InputPricePer1M: 0.15, OutputPricePer1M: 0.60, Label: "GPT-OSS 120B"},
	"openai/gpt-oss-20b":  {ID: "openai/gpt-oss-20b", Provider: "groq", MaxTokens: 65536, ContextWindow: 131072, InputPricePer1M: 0.075, OutputPricePer1M: 0.30, Label: "GPT-OSS 20B"},
	"qwen/qwen3.8-27b":    {ID: "qwen/qwen3.8-27b", Provider: "groq", ContextWindow: 131072, InputPricePer1M: 0.80, OutputPricePer1M: 4.00, Thinking: true, Label: "Qwen3.8 27B (preview)"},

	// Mistral — cloud API (2026-09-17, from docs.mistral.ai models_overview + mistral.ai/pricing/api).
	// BREAKING CATALOG CHANGE: Mistral renamed its whole lineup (mistral-large-3-25-12 style) and
	// retired the old IDs — ALL 11 previously registered IDs (mistral-large-2512,
	// mistral-medium-latest, mistral-small-2603/2506, codestral-2508, ministral-8b/14b-2512)
	// are gone from the catalog, and the entire Magistral and Devstral lines are retired.
	// Context/max-output are not published on the pricing page and were NOT confirmed — left 0
	// (0 = unknown, falls back to the config default) rather than carried over from the old IDs.
	// Mistral now also hosts Z.ai's GLM as first-party IDs (z-ai-glm-5-3 / z-ai-glm-5-2).
	"mistral-large-3-25-12":    {ID: "mistral-large-3-25-12", Provider: "mistral", InputPricePer1M: 0.50, OutputPricePer1M: 1.50, Label: "Mistral Large 3"},
	"mistral-medium-3-5-26-04": {ID: "mistral-medium-3-5-26-04", Provider: "mistral", InputPricePer1M: 1.50, OutputPricePer1M: 7.50, Label: "Mistral Medium 3.5"},
	"mistral-small-4-0-26-03":  {ID: "mistral-small-4-0-26-03", Provider: "mistral", InputPricePer1M: 0.15, OutputPricePer1M: 0.60, Label: "Mistral Small 4"},
	"ministral-3-14b-25-12":    {ID: "ministral-3-14b-25-12", Provider: "mistral", InputPricePer1M: 0.20, OutputPricePer1M: 0.20, Label: "Ministral 3 14B"},
	"ministral-3-8b-25-12":     {ID: "ministral-3-8b-25-12", Provider: "mistral", InputPricePer1M: 0.15, OutputPricePer1M: 0.15, Label: "Ministral 3 8B"},
	"ministral-3-3b-25-12":     {ID: "ministral-3-3b-25-12", Provider: "mistral", InputPricePer1M: 0.10, OutputPricePer1M: 0.10, Label: "Ministral 3 3B"},
	"codestral-25-08":          {ID: "codestral-25-08", Provider: "mistral", InputPricePer1M: 0.30, OutputPricePer1M: 0.90, Label: "Codestral"},
	"z-ai-glm-5-3":             {ID: "z-ai-glm-5-3", Provider: "mistral", InputPricePer1M: 1.40, OutputPricePer1M: 4.40, Thinking: true, Label: "GLM-5.3 (Mistral-hosted)"},
	"z-ai-glm-5-2":             {ID: "z-ai-glm-5-2", Provider: "mistral", InputPricePer1M: 1.40, OutputPricePer1M: 4.40, Thinking: true, Label: "GLM-5.2 (Mistral-hosted)"},

	// DeepSeek — cloud API (2026-09-17, from api-docs.deepseek.com/quick_start/pricing).
	// CORRECTNESS NOTE: deepseek-v4-flash is RETIRED. DeepSeek still ACCEPTS the old ID but
	// silently routes it to deepseek-flash at Flash pricing — so the previous row's prices
	// ($0.14/$0.28) no longer matched what the request actually costs. Kept as an alias row so
	// existing callers still get a (correct) estimate, pointing at the real prices.
	// PEAK/OFF-PEAK: DeepSeek bills 2x during peak (01:00-04:00 and 06:00-10:00 UTC, Mon-Fri);
	// off-peak is 50% of peak. ModelInfo has no time-of-day concept, so the OFF-PEAK (lower)
	// rate is modeled here — a peak-hours request costs 2x what EstimateCost reports.
	"deepseek-flash":    {ID: "deepseek-flash", Provider: "deepseek", MaxTokens: 384000, ContextWindow: 1000000, InputPricePer1M: 0.15, OutputPricePer1M: 0.60, CacheReadPricePer1M: 0.003, Thinking: true, Label: "DeepSeek Flash (off-peak price)"},
	"deepseek-v4-pro":   {ID: "deepseek-v4-pro", Provider: "deepseek", MaxTokens: 384000, ContextWindow: 1000000, InputPricePer1M: 0.66, OutputPricePer1M: 1.98, CacheReadPricePer1M: 0.022, Thinking: true, Label: "DeepSeek V4 Pro (off-peak price)"},
	"deepseek-v4-flash": {ID: "deepseek-v4-flash", Provider: "deepseek", MaxTokens: 384000, ContextWindow: 1000000, InputPricePer1M: 0.15, OutputPricePer1M: 0.60, CacheReadPricePer1M: 0.003, Thinking: true, Label: "DeepSeek Flash (retired ID, routes to deepseek-flash)"},

	// Cohere — OpenAI-compatible API (2026-04-04; command-a-plus-05-2026 confirmed live via
	// docs.cohere.com/docs/models on 2026-07-08, but cohere.com/pricing does not list a price
	// for it yet — left at 0 rather than guessed).
	"command-a-plus-05-2026": {ID: "command-a-plus-05-2026", Provider: "cohere", ContextWindow: 128000, Thinking: true, Label: "Command A+ (price unconfirmed)"},
	"command-a-03-2025":      {ID: "command-a-03-2025", Provider: "cohere", MaxTokens: 4096, ContextWindow: 256000, InputPricePer1M: 2.50, OutputPricePer1M: 10.00, Thinking: true, Label: "Command A"},

	// Z.ai / GLM — OpenAI-compatible API at api.z.ai (2026-09-17, docs.z.ai/guides/overview/pricing).
	// Open weights, but licences differ per model: GLM-5.3 ships under a custom "GLM-5.3 License"
	// (NOT MIT); GLM-5.3-Flash is MIT. Context windows are not published on the pricing page —
	// glm-5.2's 1M is carried over from the earlier snapshot; 5.3/5.3-flash left 0 (unknown).
	"glm-5.3":       {ID: "glm-5.3", Provider: "zai", InputPricePer1M: 1.40, OutputPricePer1M: 4.40, CacheReadPricePer1M: 0.26, Thinking: true, Label: "GLM-5.3"},
	"glm-5.3-flash": {ID: "glm-5.3-flash", Provider: "zai", InputPricePer1M: 0.15, OutputPricePer1M: 0.50, CacheReadPricePer1M: 0.03, Thinking: true, Label: "GLM-5.3 Flash"},
	"glm-5.2":       {ID: "glm-5.2", Provider: "zai", MaxTokens: 131072, ContextWindow: 1048576, InputPricePer1M: 1.40, OutputPricePer1M: 4.40, CacheReadPricePer1M: 0.26, Thinking: true, Label: "GLM-5.2"},

	// MiniMax — OpenAI-compatible API at api.minimax.io (2026-06-22; prices NOT re-confirmed on
	// 2026-09-17 — platform.minimax.io/docs/price and .../api-reference/text-pricing both 404,
	// so these are carried forward unchanged from the 2026-06-22 snapshot).
	// A research pass claimed the standard (<=512K) tier is 0.30/1.20/0.06 and that the values
	// below are the >512K tier — that would make sub-512K estimates 2x too high. It could NOT be
	// confirmed at a primary source, so nothing was changed. Re-verify when the docs are reachable.
	// Per the 2026-06-22 snapshot these are the STANDARD tier (<=512K); >512K is exactly 2x.
	"MiniMax-M3": {ID: "MiniMax-M3", Provider: "minimax", ContextWindow: 1048576, InputPricePer1M: 0.60, OutputPricePer1M: 2.40, CacheReadPricePer1M: 0.12, Label: "MiniMax M3"},

	// Moonshot / Kimi — OpenAI-compatible API (2026-09-17, from platform.kimi.ai/docs/pricing;
	// platform.moonshot.ai/docs/* now 301s to platform.kimi.ai/docs/*). Open weights.
	// CacheReadPricePer1M is the published cache-hit input rate.
	"kimi-k3":                  {ID: "kimi-k3", Provider: "moonshot", ContextWindow: 1048576, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, CacheReadPricePer1M: 0.30, Thinking: true, Label: "Kimi K3"},
	"kimi-k2.7-code":           {ID: "kimi-k2.7-code", Provider: "moonshot", MaxTokens: 262144, ContextWindow: 262144, InputPricePer1M: 0.95, OutputPricePer1M: 4.00, CacheReadPricePer1M: 0.19, Thinking: true, Label: "Kimi K2.7 Code"},
	"kimi-k2.7-code-highspeed": {ID: "kimi-k2.7-code-highspeed", Provider: "moonshot", MaxTokens: 262144, ContextWindow: 262144, InputPricePer1M: 1.90, OutputPricePer1M: 8.00, CacheReadPricePer1M: 0.38, Thinking: true, Label: "Kimi K2.7 Code Highspeed"},
	"kimi-k2.6":                {ID: "kimi-k2.6", Provider: "moonshot", ContextWindow: 262144, InputPricePer1M: 0.95, OutputPricePer1M: 4.00, CacheReadPricePer1M: 0.16, Label: "Kimi K2.6"},

	// Perplexity — Sonar API (from docs.perplexity.ai/getting-started/pricing, 2026-06-08).
	// Representative subset; the registry only affects cost estimation + discovery — any model ID can be passed directly.
	"sonar":               {ID: "sonar", Provider: "perplexity", ContextWindow: 128000, InputPricePer1M: 1.00, OutputPricePer1M: 1.00, Label: "Sonar"},
	"sonar-pro":           {ID: "sonar-pro", Provider: "perplexity", ContextWindow: 200000, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, Label: "Sonar Pro"},
	"sonar-reasoning-pro": {ID: "sonar-reasoning-pro", Provider: "perplexity", ContextWindow: 128000, InputPricePer1M: 2.00, OutputPricePer1M: 8.00, Thinking: true, Label: "Sonar Reasoning Pro"},
	"sonar-deep-research": {ID: "sonar-deep-research", Provider: "perplexity", ContextWindow: 128000, InputPricePer1M: 2.00, OutputPricePer1M: 8.00, Thinking: true, Label: "Sonar Deep Research"},

	// Fireworks — serverless (from docs.fireworks.ai/serverless/pricing, 2026-06-22).
	// Open-weight families (GLM/Z.ai, MiniMax, Kimi) tracked under the host you'd call; prices cross-checked vs the provider + Together.
	"accounts/fireworks/models/fireworks/kimi-k2p7-code":    {ID: "accounts/fireworks/models/fireworks/kimi-k2p7-code", Provider: "fireworks", ContextWindow: 262144, InputPricePer1M: 0.95, OutputPricePer1M: 4.00, Thinking: true, Label: "Kimi K2.7 Code"},
	"accounts/fireworks/models/fireworks/kimi-k2p6":         {ID: "accounts/fireworks/models/fireworks/kimi-k2p6", Provider: "fireworks", ContextWindow: 262144, InputPricePer1M: 0.95, OutputPricePer1M: 4.00, Label: "Kimi K2.6"},
	"accounts/fireworks/models/fireworks/glm-5p2":           {ID: "accounts/fireworks/models/fireworks/glm-5p2", Provider: "fireworks", ContextWindow: 1048576, InputPricePer1M: 1.40, OutputPricePer1M: 4.40, Thinking: true, Label: "GLM-5.2"},
	"accounts/fireworks/models/fireworks/minimax-m3":        {ID: "accounts/fireworks/models/fireworks/minimax-m3", Provider: "fireworks", ContextWindow: 1048576, InputPricePer1M: 0.30, OutputPricePer1M: 1.20, Label: "MiniMax M3"},
	"accounts/fireworks/models/fireworks/deepseek-v4-pro":   {ID: "accounts/fireworks/models/fireworks/deepseek-v4-pro", Provider: "fireworks", ContextWindow: 163840, InputPricePer1M: 1.74, OutputPricePer1M: 3.48, Thinking: true, Label: "DeepSeek V4 Pro"},
	"accounts/fireworks/models/fireworks/deepseek-v4-flash": {ID: "accounts/fireworks/models/fireworks/deepseek-v4-flash", Provider: "fireworks", ContextWindow: 163840, InputPricePer1M: 0.14, OutputPricePer1M: 0.28, Thinking: true, Label: "DeepSeek V4 Flash"},
	"accounts/fireworks/models/fireworks/qwen3p6-plus":      {ID: "accounts/fireworks/models/fireworks/qwen3p6-plus", Provider: "fireworks", ContextWindow: 262144, InputPricePer1M: 0.50, OutputPricePer1M: 3.00, Thinking: true, Label: "Qwen 3.6 Plus"},

	// Together AI — serverless (from together.ai/pricing, 2026-06-08; IDs as the API exposes them).
	"deepseek-ai/DeepSeek-V3":                 {ID: "deepseek-ai/DeepSeek-V3", Provider: "together", ContextWindow: 131072, InputPricePer1M: 2.10, OutputPricePer1M: 4.40, Thinking: true, Label: "DeepSeek (Together)"},
	"meta-llama/Llama-3.3-70B-Instruct-Turbo": {ID: "meta-llama/Llama-3.3-70B-Instruct-Turbo", Provider: "together", ContextWindow: 131072, InputPricePer1M: 1.04, OutputPricePer1M: 1.04, Label: "Llama 3.3 70B (Together)"},

	// DeepInfra — serverless (from deepinfra.com/pricing, 2026-06-08; distinct IDs to avoid registry key collisions with other hosts).
	"Qwen/Qwen3-32B":                        {ID: "Qwen/Qwen3-32B", Provider: "deepinfra", ContextWindow: 40960, InputPricePer1M: 0.08, OutputPricePer1M: 0.28, Thinking: true, Label: "Qwen3 32B (DeepInfra)"},
	"meta-llama/Meta-Llama-3.1-8B-Instruct": {ID: "meta-llama/Meta-Llama-3.1-8B-Instruct", Provider: "deepinfra", ContextWindow: 131072, InputPricePer1M: 0.02, OutputPricePer1M: 0.05, Label: "Llama 3.1 8B (DeepInfra)"},
	"deepseek-ai/DeepSeek-V4-Flash":         {ID: "deepseek-ai/DeepSeek-V4-Flash", Provider: "deepinfra", ContextWindow: 163840, InputPricePer1M: 0.10, OutputPricePer1M: 0.20, Thinking: true, Label: "DeepSeek V4 Flash (DeepInfra)"},

	// NVIDIA NIM — hosted catalog at build.nvidia.com is free for development (rate-limited; no public per-token price), so prices are left 0. IDs as integrate.api.nvidia.com exposes them.
	"nvidia/llama-3.1-nemotron-70b-instruct": {ID: "nvidia/llama-3.1-nemotron-70b-instruct", Provider: "nvidia", ContextWindow: 131072, Label: "Llama 3.1 Nemotron 70B (NIM, free dev tier)"},
	"meta/llama-3.1-8b-instruct":             {ID: "meta/llama-3.1-8b-instruct", Provider: "nvidia", ContextWindow: 131072, Label: "Llama 3.1 8B (NIM, free dev tier)"},

	// DashScope (Alibaba) — OpenAI-compatible API (2026-04-03). qwen3.7-max/qwen3.7-plus confirmed
	// as real model IDs (superseding qwen3.6-plus as flagship) via alibabacloud.com/help/en/model-studio/models
	// on 2026-07-08, but that pricing page is JS-rendered and no price could be confirmed — left
	// at 0 rather than guessed; qwen3.6-plus remains the default until 3.7 pricing is confirmed.
	"qwen3.7-max":        {ID: "qwen3.7-max", Provider: "dashscope", ContextWindow: 1000000, Thinking: true, Label: "Qwen 3.7 Max (price unconfirmed)"},
	"qwen3.7-plus":       {ID: "qwen3.7-plus", Provider: "dashscope", ContextWindow: 1000000, Thinking: true, Label: "Qwen 3.7 Plus (price unconfirmed)"},
	"qwen3.6-plus":       {ID: "qwen3.6-plus", Provider: "dashscope", MaxTokens: 65536, ContextWindow: 1000000, InputPricePer1M: 0.29, OutputPricePer1M: 1.74, Thinking: true, Label: "Qwen 3.6 Plus"},
	"qwen3.5-omni-plus":  {ID: "qwen3.5-omni-plus", Provider: "dashscope", MaxTokens: 16384, ContextWindow: 256000, Thinking: true, Label: "Qwen 3.5 Omni Plus"},
	"qwen3.5-omni-flash": {ID: "qwen3.5-omni-flash", Provider: "dashscope", MaxTokens: 16384, ContextWindow: 256000, Thinking: true, Label: "Qwen 3.5 Omni Flash"},
	"qwen3.5-omni-light": {ID: "qwen3.5-omni-light", Provider: "dashscope", MaxTokens: 16384, ContextWindow: 256000, Thinking: true, Label: "Qwen 3.5 Omni Light"},

	// Ollama — popular local models (no pricing, context varies by quantization)
	"deepseek-r1:14b":      {ID: "deepseek-r1:14b", Provider: "ollama", Thinking: true, NoToolSupport: true, Label: "DeepSeek R1 14B"},
	"mistral-small3.2:24b": {ID: "mistral-small3.2:24b", Provider: "ollama", Label: "Mistral Small 3.2 24B"},
	"qwen3-coder:30b":      {ID: "qwen3-coder:30b", Provider: "ollama", Label: "Qwen3 Coder 30B"},
	"qwen3.5:latest":       {ID: "qwen3.5:latest", Provider: "ollama", Thinking: true, Label: "Qwen3.5 9B"},
	"qwen3.5:27b":          {ID: "qwen3.5:27b", Provider: "ollama", Thinking: true, Label: "Qwen3.5 27B"},
	"qwen3:8b":             {ID: "qwen3:8b", Provider: "ollama", Thinking: true, Label: "Qwen3 8B"},
	"qwen3:4b":             {ID: "qwen3:4b", Provider: "ollama", Thinking: true, Label: "Qwen3 4B"},
	"gemma4:31b":           {ID: "gemma4:31b", Provider: "ollama", Thinking: true, Label: "Gemma4 31B"},
	"gemma4:26b":           {ID: "gemma4:26b", Provider: "ollama", Thinking: true, Label: "Gemma4 26B MoE"},
	"gemma4:12b":           {ID: "gemma4:12b", Provider: "ollama", Thinking: true, Label: "Gemma4 12B"},
	"gemma4:e2b":           {ID: "gemma4:e2b", Provider: "ollama", Thinking: true, Label: "Gemma4 E2B"},
	"gemma4:e4b":           {ID: "gemma4:e4b", Provider: "ollama", Thinking: true, Label: "Gemma4 E4B"},
	"gemma3:12b":           {ID: "gemma3:12b", Provider: "ollama", NoToolSupport: true, Label: "Gemma3 12B"},
	"gemma3:4b":            {ID: "gemma3:4b", Provider: "ollama", NoToolSupport: true, Label: "Gemma3 4B"},
	"gemma2:2b":            {ID: "gemma2:2b", Provider: "ollama", NoToolSupport: true, Label: "Gemma2 2B"},
}

var defaultModels = map[string]string{
	"anthropic": "claude-opus-5",
	"claude":    "claude-opus-5",
	"xai":       "grok-4.6",
	"grok":      "grok-4.6",
	"gemini":    "gemini-3.8-flash",
	"google":    "gemini-3.8-flash",
	"openai":    "gpt-5.6-sol",
	"gpt":       "gpt-5.6-sol",
	"groq":      "openai/gpt-oss-120b",
	"mistral":   "mistral-small-4-0-26-03",
	"deepseek":  "deepseek-flash",
	"cohere":    "command-a-03-2025",
	"zai":       "glm-5.3",
	"z-ai":      "glm-5.3",
	"glm":       "glm-5.3",
	"minimax":   "MiniMax-M3",
	"moonshot":  "kimi-k3",
	"kimi":      "kimi-k3",
	"dashscope": "qwen3.6-plus",
	"qwen":      "qwen3.6-plus",
	"ollama":    "qwen3:8b",
}

var availableModels = map[string][]string{
	"anthropic": {
		"claude-fable-5-1",
		"claude-fable-5",
		"claude-opus-5",
		"claude-opus-4-8",
		"claude-opus-4-7",
		"claude-opus-4-6",
		"claude-opus-4-5",
		"claude-sonnet-5",
		"claude-sonnet-4-6",
		"claude-sonnet-4-5",
		"claude-haiku-4-5",
	},
	"xai": {
		"grok-4.6",
		"grok-4.5",
		"grok-4.3",
		"grok-4.20-0309-reasoning",
		"grok-4.20-0309-non-reasoning",
		"grok-4.20-multi-agent-0309",
		"grok-build-0.1",
	},
	"gemini": {
		"gemini-3.8-flash",
		"gemini-3.7-flash",
		"gemini-3.6-flash",
		"gemini-3.5-flash",
		"gemini-3.5-flash-lite",
		"gemini-3.1-pro-preview",
		"gemini-3.1-flash-lite-preview",
		"gemini-3-pro-preview",
		"gemini-3-flash-preview",
		"gemini-2.5-pro",
		"gemini-2.5-flash",
		"gemini-2.5-flash-lite",
		"gemini-2.0-flash",
	},
	"openai": {
		"gpt-6-astra",
		"gpt-5.6-sol",
		"gpt-5.6-terra",
		"gpt-5.6-luna",
		"gpt-5.5-pro",
		"gpt-5.5",
		"gpt-5.4-pro",
		"gpt-5.4",
		"gpt-5.4-mini",
		"gpt-5.4-nano",
		"gpt-5.3-chat-latest",
		"gpt-5.3-codex",
		"gpt-5.2-pro",
		"gpt-5.2",
		"gpt-5.2-chat-latest",
		"gpt-5.1",
		"gpt-5.1-chat-latest",
		"gpt-5-pro",
		"gpt-5",
		"gpt-5-codex",
		"gpt-5-chat-latest",
		"gpt-5-mini",
		"gpt-5-nano",
		"gpt-4.1",
		"gpt-4.1-mini",
		"gpt-4.1-nano",
		"o3-pro",
		"o3",
		"o3-mini",
		"o4-mini",
	},
	"groq": {
		"openai/gpt-oss-120b",
		"openai/gpt-oss-20b",
		"qwen/qwen3.8-27b",
	},
	"mistral": {
		"mistral-large-3-25-12",
		"mistral-medium-3-5-26-04",
		"mistral-small-4-0-26-03",
		"ministral-3-14b-25-12",
		"ministral-3-8b-25-12",
		"ministral-3-3b-25-12",
		"codestral-25-08",
		"z-ai-glm-5-3",
		"z-ai-glm-5-2",
	},
	"deepseek": {
		"deepseek-flash",
		"deepseek-v4-pro",
	},
	"cohere": {
		"command-a-plus-05-2026",
		"command-a-03-2025",
	},
	"zai": {
		"glm-5.3",
		"glm-5.3-flash",
		"glm-5.2",
	},
	"minimax": {
		"MiniMax-M3",
	},
	"moonshot": {
		"kimi-k3",
		"kimi-k2.7-code",
		"kimi-k2.7-code-highspeed",
		"kimi-k2.6",
	},
	"dashscope": {
		"qwen3.7-max",
		"qwen3.7-plus",
		"qwen3.6-plus",
		"qwen3.5-omni-plus",
		"qwen3.5-omni-flash",
		"qwen3.5-omni-light",
	},
	"ollama": {
		"deepseek-r1:14b",
		"mistral-small3.2:24b",
		"qwen3-coder:30b",
		"qwen3.5:latest",
		"qwen3.5:27b",
		"qwen3:8b",
		"qwen3:4b",
		"gemma4:31b",
		"gemma4:26b",
		"gemma4:e4b",
		"gemma3:12b",
		"gemma3:4b",
		"gemma2:2b",
	},
}

var modelAliases = map[string]string{
	// Anthropic aliases. Bare family names track the CURRENT generation (Claude 5), per
	// platform.claude.com/docs models overview (2026-09-17).
	"opus":     "claude-opus-5",
	"sonnet":   "claude-sonnet-5",
	"haiku":    "claude-haiku-4-5",
	"claude":   "claude-opus-5",
	"fable":    "claude-fable-5-1",
	"opus5":    "claude-opus-5",
	"opus-5":   "claude-opus-5",
	"sonnet5":  "claude-sonnet-5",
	"sonnet-5": "claude-sonnet-5",
	"opus4.8":  "claude-opus-4-8",
	"opus-4-8": "claude-opus-4-8",

	// xAI/Grok aliases. Bare/short names now point at the current docs IDs (grok-4.6 flagship,
	// 4.20 family without the "-beta" infix); the legacy spellings remain reachable by full ID.
	"grok":               "grok-4.6",
	"grok4.6":            "grok-4.6",
	"grok-4-6":           "grok-4.6",
	"grok4.5":            "grok-4.5",
	"grok-4-5":           "grok-4.5",
	"grok4.3":            "grok-4.3",
	"grok-4-3":           "grok-4.3",
	"grok4.2":            "grok-4.20-0309-non-reasoning",
	"grok4.20":           "grok-4.20-0309-non-reasoning",
	"grok4":              "grok-4.6",
	"grok-4":             "grok-4.6",
	"grok-reasoning":     "grok-4.20-0309-reasoning",
	"grok-4-reasoning":   "grok-4.20-0309-reasoning",
	"grok-multi-agent":   "grok-4.20-multi-agent-0309",
	"grok-build":         "grok-build-0.1",
	"grok4.1":            "grok-4-1-fast-non-reasoning",
	"grok-4-1":           "grok-4-1-fast-non-reasoning",
	"grok-4-1-reasoning": "grok-4-1-fast-reasoning",
	"grok-code":          "grok-code-fast-1",
	"grok-mini":          "grok-3-mini",

	// Groq aliases. llama/llama-70b/llama-8b/llama4/llama-4-scout are REMOVED: Groq deprecated
	// every Llama model it hosted (console.groq.com/docs/deprecations) with no Llama
	// replacement — gpt-oss/qwen3.8-27b are Groq's own recommended replacements.
	"groq":        "openai/gpt-oss-120b",
	"gpt-oss":     "openai/gpt-oss-120b",
	"gpt-oss-20b": "openai/gpt-oss-20b",
	"qwen3.8-27b": "qwen/qwen3.8-27b",

	// Mistral aliases. magistral/devstral are REMOVED with no replacement — Mistral retired
	// both lines entirely (docs.mistral.ai models_overview, checked 2026-09-17).
	"mistral-large":  "mistral-large-3-25-12",
	"mistral-medium": "mistral-medium-3-5-26-04",
	"mistral-small":  "mistral-small-4-0-26-03",
	"codestral":      "codestral-25-08",
	"ministral":      "ministral-3-8b-25-12",

	// DeepSeek aliases
	"deepseek-cloud":  "deepseek-flash",
	"deepseek-v4":     "deepseek-flash",
	"deepseek-v4-pro": "deepseek-v4-pro",

	// Cohere aliases
	"cohere":         "command-a-03-2025",
	"command-a":      "command-a-03-2025",
	"command-a-plus": "command-a-plus-05-2026",

	// Z.ai / GLM aliases
	"glm":       "glm-5.3",
	"glm5.3":    "glm-5.3",
	"glm-5-3":   "glm-5.3",
	"glm-flash": "glm-5.3-flash",
	"glm5.2":    "glm-5.2",
	"glm-5-2":   "glm-5.2",

	// MiniMax aliases
	"minimax":    "MiniMax-M3",
	"minimax-m3": "MiniMax-M3",

	// Moonshot / Kimi aliases
	"kimi":      "kimi-k3",
	"kimi-k3":   "kimi-k3",
	"k3":        "kimi-k3",
	"kimi-k2.7": "kimi-k2.7-code",
	"k2.7":      "kimi-k2.7-code",
	"kimi-k2.6": "kimi-k2.6",

	// DashScope aliases
	"qwen-cloud": "qwen3.6-plus",
	"qwen-omni":  "qwen3.5-omni-plus",

	// Ollama aliases
	"deepseek":      "deepseek-r1:14b",
	"mistral-local": "mistral-small3.2:24b",
	"qwen-local":    "qwen3:8b",
	"qwen-code":     "qwen3-coder:30b",
	"gemma":         "gemma4:e4b",
	"gemma4":        "gemma4:e4b",
	"gemma3":        "gemma3:12b",

	// OpenAI aliases
	"gpt":         "gpt-5.6-sol",
	"gpt6":        "gpt-6-astra",
	"gpt-6":       "gpt-6-astra",
	"astra":       "gpt-6-astra",
	"gpt5.6":      "gpt-5.6-sol",
	"sol":         "gpt-5.6-sol",
	"terra":       "gpt-5.6-terra",
	"luna":        "gpt-5.6-luna",
	"gpt5.5":      "gpt-5.5",
	"gpt5.5-pro":  "gpt-5.5-pro",
	"gpt5.4":      "gpt-5.4",
	"gpt5.3":      "gpt-5.3-chat-latest",
	"gpt5":        "gpt-5.6-sol",
	"gpt5.2":      "gpt-5.2",
	"gpt5.1":      "gpt-5.1",
	"gpt5-mini":   "gpt-5-mini",
	"gpt5-nano":   "gpt-5-nano",
	"gpt4.1":      "gpt-4.1",
	"gpt5.4-mini": "gpt-5.4-mini",
	"gpt5.4-nano": "gpt-5.4-nano",
	"gpt-instant": "gpt-5.3-chat-latest",

	// Gemini aliases
	"gemini":     "gemini-3.8-flash",
	"gemini3.8":  "gemini-3.8-flash",
	"gemini3.7":  "gemini-3.7-flash",
	"gemini3.6":  "gemini-3.6-flash",
	"gemini3.5":  "gemini-3.5-flash",
	"gemini-pro": "gemini-3.1-pro-preview",
	"gemini3.1":  "gemini-3.1-pro-preview",
	"gemini3":    "gemini-3.1-pro-preview",
	"gemini-3":   "gemini-3-pro-preview",
	"flash":      "gemini-3.8-flash",
	"flash-lite": "gemini-3.5-flash-lite",
}

// RegisterModel adds (or overrides) a model's metadata at runtime, so unlisted
// or newly released models get cost estimation, capability flags, and
// discovery without waiting for a library release — and stale built-in pricing
// can be corrected in place. info.ID and info.Provider are required.
// Safe for concurrent use.
func RegisterModel(info ModelInfo) error {
	if strings.TrimSpace(info.ID) == "" {
		return fmt.Errorf("llm: RegisterModel: ID is required (non-blank)")
	}
	if strings.TrimSpace(info.Provider) == "" {
		return fmt.Errorf("llm: RegisterModel: Provider is required (non-blank)")
	}
	registryMu.Lock()
	defer registryMu.Unlock()
	prev, existed := modelRegistry[info.ID]
	modelRegistry[info.ID] = info

	// Keep availableModels (the discovery list) in sync with modelRegistry,
	// independent of whether the model already existed. Two cases the old
	// `if !existed` guard got wrong: (1) correcting a built-in that is in
	// modelRegistry but absent from availableModels — it must now surface in
	// discovery; (2) an override that moves a model to a different provider —
	// it must leave the old provider's list.
	if existed && prev.Provider != info.Provider {
		availableModels[prev.Provider] = slices.DeleteFunc(
			availableModels[prev.Provider],
			func(id string) bool { return id == info.ID },
		)
	}
	if !slices.Contains(availableModels[info.Provider], info.ID) {
		availableModels[info.Provider] = append(availableModels[info.Provider], info.ID)
	}
	return nil
}

// RegisterModelAlias adds (or re-points) a model alias at runtime. The target
// model must already be registered (built-in or via RegisterModel).
// Safe for concurrent use.
func RegisterModelAlias(alias, modelID string) error {
	if alias == "" || modelID == "" {
		return fmt.Errorf("llm: RegisterModelAlias: alias and model ID are required")
	}
	registryMu.Lock()
	defer registryMu.Unlock()
	if _, ok := modelRegistry[modelID]; !ok {
		return fmt.Errorf("llm: RegisterModelAlias: unknown model %q (register it first)", modelID)
	}
	// An alias key that collides with a real model ID would silently shadow that
	// model: ResolveModelAlias checks aliases first, so every request for the real
	// model would be redirected to the alias target. Reject it.
	if _, ok := modelRegistry[alias]; ok {
		return fmt.Errorf("llm: RegisterModelAlias: alias %q is already a registered model ID (would shadow it)", alias)
	}
	modelAliases[alias] = modelID
	return nil
}

// GetModelInfo returns the ModelInfo for a model ID, or false if not found.
func GetModelInfo(model string) (ModelInfo, bool) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	info, ok := modelRegistry[model]
	return info, ok
}

// GetDefaultModel returns the default model for a provider.
func GetDefaultModel(provider string) string {
	registryMu.RLock()
	defer registryMu.RUnlock()
	if model, ok := defaultModels[provider]; ok {
		return model
	}
	return defaultModels["anthropic"]
}

// GetAvailableModels returns the ordered model list for a provider.
// Returns nil if the provider has no models registered.
// Returns a copy to prevent callers from mutating global registry state.
func GetAvailableModels(provider string) []string {
	registryMu.RLock()
	defer registryMu.RUnlock()
	src := availableModels[provider]
	if src == nil {
		return nil
	}
	dst := make([]string, len(src))
	copy(dst, src)
	return dst
}

// ResolveModelAlias resolves a model alias to its full name.
// Returns the input unchanged if not an alias.
func ResolveModelAlias(model string) string {
	registryMu.RLock()
	defer registryMu.RUnlock()
	if full, ok := modelAliases[model]; ok {
		return full
	}
	return model
}

// ProviderForModel detects the provider from a model name.
// Returns empty string if the model is not recognized.
func ProviderForModel(model string) string {
	registryMu.RLock()
	defer registryMu.RUnlock()
	if info, ok := modelRegistry[model]; ok {
		return info.Provider
	}
	return ""
}

// CostInput holds all token counts needed for accurate cost estimation.
type CostInput struct {
	Model             string
	InputTokens       int
	OutputTokens      int
	ThinkingTokens    int  // Gemini: separate from output; Anthropic: 0 (bundled in OutputTokens)
	CacheCreateTokens int  // Anthropic: tokens written to cache
	CacheReadTokens   int  // Anthropic/Gemini: tokens read from cache
	Batch             bool // request was processed via the Batch API (billed at 50% of the sync rate)
}

// EstimateCost returns the estimated cost in USD for a request/response.
// Accepts a CostInput with all token types for accurate cache-aware pricing.
// Returns 0 if the model is not in the registry or has no pricing data —
// register unlisted models via RegisterModel to get real estimates.
func EstimateCost(input CostInput) float64 {
	registryMu.RLock()
	// Resolve an alias to its target under the same read lock (ResolveModelAlias
	// takes the lock too, and RWMutex is not re-entrant — so inline it).
	model := input.Model
	if full, isAlias := modelAliases[model]; isAlias {
		model = full
	}
	info, ok := modelRegistry[model]
	registryMu.RUnlock()
	if !ok {
		return 0
	}

	// Clamp negatives to 0 (sentinels like TokensNotReported = -1)
	clamp := func(n int) int {
		if n < 0 {
			return 0
		}
		return n
	}

	inputTok := clamp(input.InputTokens)
	outputTok := clamp(input.OutputTokens)
	thinkingTok := clamp(input.ThinkingTokens)
	cacheCreateTok := clamp(input.CacheCreateTokens)
	cacheReadTok := clamp(input.CacheReadTokens)

	inputCost := float64(inputTok) * info.InputPricePer1M / 1_000_000
	outputCost := float64(outputTok+thinkingTok) * info.OutputPricePer1M / 1_000_000

	var cacheCost float64
	if info.CacheWritePricePer1M > 0 {
		cacheCost += float64(cacheCreateTok) * info.CacheWritePricePer1M / 1_000_000
	}
	if info.CacheReadPricePer1M > 0 {
		cacheCost += float64(cacheReadTok) * info.CacheReadPricePer1M / 1_000_000
	}

	total := inputCost + outputCost + cacheCost

	// Batch APIs (OpenAI, Anthropic) bill at half the synchronous rate. Apply the
	// discount once to the final total so it composes uniformly with cache pricing.
	if input.Batch {
		total *= 0.5
	}

	return total
}
