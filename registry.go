package llm

import (
	"fmt"
	"math"
	"strings"
	"sync"
)

// registryMu guards the registry maps (modelRegistry, modelAliases,
// defaultModels). The built-in data is written at package init;
// RegisterModel/RegisterModelAlias extend it at runtime, so every reader
// takes the read lock.
var registryMu sync.RWMutex

// ModelInfo holds per-model metadata used to adapt API requests and UI.
type ModelInfo struct {
	ID                     string        // Full model identifier
	Provider               string        // anthropic, xai, gemini, openai, groq, etc.
	MaxTokens              int           // Model-specific output limit (0 = use config default)
	ContextWindow          int           // Max input tokens (0 = unknown)
	InputPricePer1M        float64       // USD per 1M input tokens (0 = unknown/free)
	OutputPricePer1M       float64       // USD per 1M output tokens (0 = unknown/free)
	CacheWritePricePer1M   float64       // Anthropic: price per 1M cache creation tokens (0 = not applicable: cache writes cost nothing extra)
	CacheWrite1hPricePer1M float64       // Anthropic: price per 1M 1-hour cache writes (0 = unknown: billed at the documented 2× InputPricePer1M, 0 if that is unknown too)
	CacheReadPricePer1M    float64       // price per 1M cache-read input tokens (0 = unknown: cache reads are billed at InputPricePer1M)
	AudioInputPricePer1M   float64       // USD per 1M audio input tokens (0 = unknown: audio tokens are not priced)
	AudioPricePerMinute    float64       // USD per minute of input audio for duration-billed speech-to-text (0 = unknown/not duration-billed)
	WebSearchPricePer1K    float64       // USD per 1,000 server-side web searches (0 = unknown: searches are not priced)
	SupportsThinking       bool          // Adapter/model supports explicit thinking-level control
	ThoughtSignature       bool          // Gemini 3 models require thought_signature in function call responses
	Thinking               bool          // Model uses internal chain-of-thought reasoning (e.g. qwen3, deepseek-r1) — consumes output tokens invisibly
	ResponsesAPI           bool          // Model requires OpenAI Responses API (/v1/responses) for reasoning effort control (GPT-5 family)
	NoToolSupport          bool          // Model does not support tool/function calling (e.g. deepseek-r1, gemma)
	Capabilities           CapabilitySet // Reviewed model operations and request features.
	Label                  string        // Short display name
}

// retiredModels maps IDs the provider has sunset to the recommended
// replacement. Registry lookup is fail-closed, so a removed entry would
// otherwise surface as a generic "no reviewed capability metadata" error and
// hide the real cause. Verified against each provider's live models endpoint
// (2026-09-18): every ID here returns HTTP 404 from the provider. The
// 2026-10-31 additions come from each provider's deprecation page (shutdown
// date passed) or a live probe (Fireworks 404 "Model not found", NVIDIA 410
// "end of life").
var retiredModels = map[string]string{
	// Moonshot retired these on every platform in August 2026; calls return 404
	// (platform.kimi.ai/docs/platform-changelog, checked 2026-11-10).
	"kimi-k2.5":        "kimi-k3.1-pro",
	"moonshot-v1-128k": "kimi-k3.1-pro",
	// DeepInfra deprecated it on 2026-07-16 (models/list "replaced_by").
	"meta-llama/Meta-Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
	// xAI answers these with grok-4.3 / grok-build-0.1 (response "model", checked
	// 2026-11-10) and bills that model, so the old ID's price would be wrong.
	"grok-3":                                    "grok-4.3",
	"grok-3-mini":                               "grok-4.3",
	"grok-4-0709":                               "grok-4.3",
	"grok-4-fast-reasoning":                     "grok-4.3",
	"grok-4-fast-non-reasoning":                 "grok-4.3",
	"grok-4-1-fast-reasoning":                   "grok-4.3",
	"grok-4-1-fast-non-reasoning":               "grok-4.3",
	"grok-code-fast-1":                          "grok-build-0.1",
	"claude-3-haiku-20240307":                   "claude-haiku-4-5",
	"claude-opus-4-0":                           "claude-opus-4-5",
	"claude-opus-4-1":                           "claude-opus-4-5",
	"claude-opus-4-1-20250805":                  "claude-opus-4-5",
	"claude-opus-4-20250514":                    "claude-opus-4-5",
	"claude-sonnet-4-0":                         "claude-sonnet-4-6",
	"claude-sonnet-4-20250514":                  "claude-sonnet-4-6",
	"gpt-5.3-chat-latest":                       "gpt-5.6-sol",
	"gpt-5.2-chat-latest":                       "gpt-5.6-sol",
	"gpt-5.1-chat-latest":                       "gpt-5.6-sol",
	"gpt-5-chat-latest":                         "gpt-5.6-sol",
	"gpt-5-codex":                               "gpt-5.6-sol",
	"gpt-4.1-nano":                              "gpt-5.6-luna",
	"o4-mini":                                   "gpt-5.6-terra",
	"o3-mini":                                   "gpt-5.6-sol",
	"gpt-image-1":                               "gpt-image-2",
	"gemini-2.0-flash":                          "gemini-3.6-flash",
	"gemini-3-pro-preview":                      "gemini-3.1-pro-preview",
	"mistral-small-2506":                        "mistral-small-2603",
	"magistral-small-2509":                      "mistral-small-2603",
	"magistral-medium-2509":                     "mistral-medium-latest",
	"devstral-2512":                             "mistral-medium-latest",
	"devstral-small-2-2512":                     "mistral-medium-latest",
	"pixtral-large-2411":                        "mistral-medium-latest",
	"pixtral-12b-2409":                          "ministral-14b-2512",
	"deepseek-chat":                             "deepseek-flash",
	"deepseek-reasoner":                         "deepseek-flash",
	"deepseek-r1-distill-llama-70b":             "openai/gpt-oss-120b",
	"deepseek-r1-distill-qwen-32b":              "openai/gpt-oss-120b",
	"meta-llama/llama-4-scout-17b-16e-instruct": "openai/gpt-oss-120b",
	"accounts/fireworks/models/fireworks/kimi-k2p7-code":    "accounts/fireworks/models/kimi-k3",
	"accounts/fireworks/models/fireworks/kimi-k2p6":         "accounts/fireworks/models/kimi-k3",
	"accounts/fireworks/models/fireworks/glm-5p2":           "accounts/fireworks/models/glm-5p3",
	"accounts/fireworks/models/fireworks/minimax-m3":        "accounts/fireworks/models/minimax-m3",
	"accounts/fireworks/models/fireworks/deepseek-v4-pro":   "accounts/fireworks/models/deepseek-v4p1-flash",
	"accounts/fireworks/models/fireworks/deepseek-v4-flash": "accounts/fireworks/models/deepseek-v4p1-flash",
	"accounts/fireworks/models/fireworks/qwen3p6-plus":      "accounts/fireworks/models/qwen3p8-max",
	"deepseek-ai/DeepSeek-V3":                               "deepseek-ai/DeepSeek-V4.1-Flash",
	"meta-llama/Llama-3.3-70B-Instruct-Turbo":               "deepseek-ai/DeepSeek-V4.1-Flash",
	"meta/llama-3.1-8b-instruct":                            "nvidia/llama-3.1-nemotron-70b-instruct",
}

// RetiredModelReplacement reports whether model has been retired by its
// provider and, if so, the registered ID that replaces it.
func RetiredModelReplacement(model string) (string, bool) {
	replacement, ok := retiredModels[ResolveModelAlias(model)]
	return replacement, ok
}

// modelRegistry maps model ID to its metadata. Built-in entries are written
// at init; RegisterModel extends/overrides at runtime under registryMu.
var modelRegistry = map[string]ModelInfo{
	// Anthropic — from platform.claude.com/docs/en/about-claude/pricing (2026-10-31)
	// Cache pricing: write = 1.25× input; read is usually 0.1× input but the
	// page sets it per model (Fable 5.2 $0.20, Fable 5.1 $0.25, Opus 5.5 $0.20,
	// Sonnet 5.5 $0.10). Fable 5.2 rechecked 2026-11-10.
	// 1-hour cache writes: 2× input (same page, 2026-11-06).
	// Web search: $10 per 1,000 searches, on top of tokens (same page, 2026-11-06).
	// Sampling: API version 2026-10-21 (DefaultAnthropicVersion) rejects
	// temperature, top_p and top_k on every model; on a pinned 2023-06-01 only
	// the 4.6 and 4.5 rows accept them (anthropicWithoutSamplingControls).
	// Structured output (output_config.format) and documents: every row.
	// Claude 5 family (no minor segment in major-version IDs).
	"claude-fable-5-2":  {ID: "claude-fable-5-2", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 1000000, InputPricePer1M: 8.00, OutputPricePer1M: 40.00, CacheWritePricePer1M: 10.00, CacheWrite1hPricePer1M: 16.00, CacheReadPricePer1M: 0.20, WebSearchPricePer1K: 10.00, SupportsThinking: true, Label: "Fable 5.2"},
	"claude-fable-5-1":  {ID: "claude-fable-5-1", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 1000000, InputPricePer1M: 10.00, OutputPricePer1M: 50.00, CacheWritePricePer1M: 12.50, CacheWrite1hPricePer1M: 20.00, CacheReadPricePer1M: 0.25, WebSearchPricePer1K: 10.00, SupportsThinking: true, Label: "Fable 5.1"},
	"claude-opus-5-5":   {ID: "claude-opus-5-5", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 1000000, InputPricePer1M: 4.00, OutputPricePer1M: 20.00, CacheWritePricePer1M: 5.00, CacheWrite1hPricePer1M: 8.00, CacheReadPricePer1M: 0.20, WebSearchPricePer1K: 10.00, SupportsThinking: true, Label: "Opus 5.5"},
	"claude-sonnet-5-5": {ID: "claude-sonnet-5-5", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 1000000, InputPricePer1M: 2.00, OutputPricePer1M: 10.00, CacheWritePricePer1M: 2.50, CacheWrite1hPricePer1M: 4.00, CacheReadPricePer1M: 0.10, WebSearchPricePer1K: 10.00, SupportsThinking: true, Label: "Sonnet 5.5"},
	// Haiku 5.5 prompts over 100K tokens cost 5× (not modeled; short-prompt rates).
	"claude-haiku-5-5":           {ID: "claude-haiku-5-5", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 1000000, InputPricePer1M: 0.10, OutputPricePer1M: 0.50, CacheWritePricePer1M: 0.125, CacheWrite1hPricePer1M: 0.20, CacheReadPricePer1M: 0.01, WebSearchPricePer1K: 10.00, SupportsThinking: true, Label: "Haiku 5.5"},
	"claude-fable-5":             {ID: "claude-fable-5", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 1000000, InputPricePer1M: 10.00, OutputPricePer1M: 50.00, CacheWritePricePer1M: 12.50, CacheWrite1hPricePer1M: 20.00, CacheReadPricePer1M: 1.00, WebSearchPricePer1K: 10.00, SupportsThinking: true, Label: "Fable 5"},
	"claude-opus-5":              {ID: "claude-opus-5", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 1000000, InputPricePer1M: 5.00, OutputPricePer1M: 25.00, CacheWritePricePer1M: 6.25, CacheWrite1hPricePer1M: 10.00, CacheReadPricePer1M: 0.50, WebSearchPricePer1K: 10.00, SupportsThinking: true, Label: "Opus 5"},
	"claude-sonnet-5":            {ID: "claude-sonnet-5", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 1000000, InputPricePer1M: 2.00, OutputPricePer1M: 10.00, CacheWritePricePer1M: 2.50, CacheWrite1hPricePer1M: 4.00, CacheReadPricePer1M: 0.20, WebSearchPricePer1K: 10.00, SupportsThinking: true, Label: "Sonnet 5"},
	"claude-opus-4-8":            {ID: "claude-opus-4-8", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 1000000, InputPricePer1M: 5.00, OutputPricePer1M: 25.00, CacheWritePricePer1M: 6.25, CacheWrite1hPricePer1M: 10.00, CacheReadPricePer1M: 0.50, WebSearchPricePer1K: 10.00, SupportsThinking: true, Label: "Opus 4.8"},
	"claude-opus-4-7":            {ID: "claude-opus-4-7", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 1000000, InputPricePer1M: 5.00, OutputPricePer1M: 25.00, CacheWritePricePer1M: 6.25, CacheWrite1hPricePer1M: 10.00, CacheReadPricePer1M: 0.50, WebSearchPricePer1K: 10.00, SupportsThinking: true, Label: "Opus 4.7"},
	"claude-opus-4-6":            {ID: "claude-opus-4-6", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 1000000, InputPricePer1M: 5.00, OutputPricePer1M: 25.00, CacheWritePricePer1M: 6.25, CacheWrite1hPricePer1M: 10.00, CacheReadPricePer1M: 0.50, WebSearchPricePer1K: 10.00, SupportsThinking: true, Label: "Opus 4.6"},
	"claude-opus-4-5":            {ID: "claude-opus-4-5", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 200000, InputPricePer1M: 5.00, OutputPricePer1M: 25.00, CacheWritePricePer1M: 6.25, CacheWrite1hPricePer1M: 10.00, CacheReadPricePer1M: 0.50, WebSearchPricePer1K: 10.00, SupportsThinking: true, Label: "Opus 4.5"},
	"claude-opus-4-5-20251101":   {ID: "claude-opus-4-5-20251101", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 200000, InputPricePer1M: 5.00, OutputPricePer1M: 25.00, CacheWritePricePer1M: 6.25, CacheWrite1hPricePer1M: 10.00, CacheReadPricePer1M: 0.50, WebSearchPricePer1K: 10.00, SupportsThinking: true, Label: "Opus 4.5 (Nov)"},
	"claude-sonnet-4-6":          {ID: "claude-sonnet-4-6", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 1000000, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, CacheWritePricePer1M: 3.75, CacheWrite1hPricePer1M: 6.00, CacheReadPricePer1M: 0.30, WebSearchPricePer1K: 10.00, SupportsThinking: true, Label: "Sonnet 4.6"},
	"claude-sonnet-4-5":          {ID: "claude-sonnet-4-5", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 200000, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, CacheWritePricePer1M: 3.75, CacheWrite1hPricePer1M: 6.00, CacheReadPricePer1M: 0.30, WebSearchPricePer1K: 10.00, SupportsThinking: true, Label: "Sonnet 4.5"},
	"claude-sonnet-4-5-20250929": {ID: "claude-sonnet-4-5-20250929", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 200000, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, CacheWritePricePer1M: 3.75, CacheWrite1hPricePer1M: 6.00, CacheReadPricePer1M: 0.30, WebSearchPricePer1K: 10.00, SupportsThinking: true, Label: "Sonnet 4.5 (Sep)"},
	"claude-haiku-4-5-20251001":  {ID: "claude-haiku-4-5-20251001", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 200000, InputPricePer1M: 1.00, OutputPricePer1M: 5.00, CacheWritePricePer1M: 1.25, CacheWrite1hPricePer1M: 2.00, CacheReadPricePer1M: 0.10, WebSearchPricePer1K: 10.00, SupportsThinking: true, Label: "Haiku 4.5"},
	"claude-haiku-4-5":           {ID: "claude-haiku-4-5", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 200000, InputPricePer1M: 1.00, OutputPricePer1M: 5.00, CacheWritePricePer1M: 1.25, CacheWrite1hPricePer1M: 2.00, CacheReadPricePer1M: 0.10, WebSearchPricePer1K: 10.00, SupportsThinking: true, Label: "Haiku 4.5"},

	// xAI / Grok — from docs.x.ai/developers/models (2026-10-31), prices rechecked
	// against GET /v1/language-models on 2026-11-10. The 4.20 beta slugs are
	// aliases of the 0309 IDs (modelAliases); grok-3/4/4.1-fast and
	// grok-code-fast-1 are retired (retiredModels).
	// Short-context rates; long-context (≥200k prompt) is 2× and not modeled here.
	"grok-4.7":                     {ID: "grok-4.7", Provider: "xai", ContextWindow: 500000, InputPricePer1M: 2.00, OutputPricePer1M: 6.00, CacheReadPricePer1M: 0.50, Thinking: true, Label: "Grok 4.7"},
	"grok-4.6":                     {ID: "grok-4.6", Provider: "xai", ContextWindow: 500000, InputPricePer1M: 2.00, OutputPricePer1M: 6.00, CacheReadPricePer1M: 0.50, Thinking: true, Label: "Grok 4.6"},
	"grok-4.5":                     {ID: "grok-4.5", Provider: "xai", ContextWindow: 500000, InputPricePer1M: 2.00, OutputPricePer1M: 6.00, CacheReadPricePer1M: 0.30, Thinking: true, Label: "Grok 4.5"},
	"grok-4.3":                     {ID: "grok-4.3", Provider: "xai", ContextWindow: 1000000, InputPricePer1M: 1.25, OutputPricePer1M: 2.50, CacheReadPricePer1M: 0.20, Thinking: true, Label: "Grok 4.3"},
	"grok-4.20-0309-reasoning":     {ID: "grok-4.20-0309-reasoning", Provider: "xai", ContextWindow: 1000000, InputPricePer1M: 1.25, OutputPricePer1M: 2.50, CacheReadPricePer1M: 0.20, Thinking: true, Label: "Grok 4.20 R"},
	"grok-4.20-0309-non-reasoning": {ID: "grok-4.20-0309-non-reasoning", Provider: "xai", ContextWindow: 1000000, InputPricePer1M: 1.25, OutputPricePer1M: 2.50, CacheReadPricePer1M: 0.20, Label: "Grok 4.20"},
	"grok-4.20-multi-agent-0309":   {ID: "grok-4.20-multi-agent-0309", Provider: "xai", ContextWindow: 1000000, InputPricePer1M: 1.25, OutputPricePer1M: 2.50, CacheReadPricePer1M: 0.20, Thinking: true, Label: "Grok 4.20 Agent"},
	"grok-build-0.1":               {ID: "grok-build-0.1", Provider: "xai", ContextWindow: 256000, InputPricePer1M: 1.00, OutputPricePer1M: 2.00, CacheReadPricePer1M: 0.20, Thinking: true, Label: "Grok Build 0.1"},

	// Gemini — from ai.google.dev/gemini-api/docs/pricing and /models (2026-10-31)
	// Prices are standard tier (<=200K context). Long-context tier (>200K) roughly doubles.
	// Introductory rates: gemini-4-argon doubles from 2026-12-01; gemini-3.8-flash and
	// gemini-3.6-flash double from 2027-01-01.
	// AudioInputPricePer1M from ai.google.dev/gemini-api/docs/pricing (fetched 2026-10-07,
	// page "last updated 2026-10-07"): set only where the page lists the audio rate
	// (an explicit "(audio)" price, or one price for all input modalities).
	"gemini-4-argon":                     {ID: "gemini-4-argon", Provider: "gemini", MaxTokens: 32768, ContextWindow: 1048576, InputPricePer1M: 2.00, OutputPricePer1M: 10.00, CacheReadPricePer1M: 0.10, SupportsThinking: true, ThoughtSignature: true, Label: "Gemini 4 Argon"},
	"gemini-3.8-flash":                   {ID: "gemini-3.8-flash", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 0.75, OutputPricePer1M: 3.75, CacheReadPricePer1M: 0.075, SupportsThinking: true, ThoughtSignature: true, Label: "Gemini 3.8 Flash"},
	"gemini-3.7-flash":                   {ID: "gemini-3.7-flash", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 0.75, OutputPricePer1M: 3.75, CacheReadPricePer1M: 0.075, SupportsThinking: true, ThoughtSignature: true, Label: "Gemini 3.7 Flash"},
	"gemini-3.6-flash":                   {ID: "gemini-3.6-flash", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 0.75, OutputPricePer1M: 3.75, CacheReadPricePer1M: 0.075, SupportsThinking: true, ThoughtSignature: true, Label: "Gemini 3.6 Flash"},
	"gemini-3.5-flash":                   {ID: "gemini-3.5-flash", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 1.50, OutputPricePer1M: 9.00, CacheReadPricePer1M: 0.15, SupportsThinking: true, ThoughtSignature: true, AudioInputPricePer1M: 1.50, Label: "Gemini 3.5 Flash"},
	"gemini-3.5-flash-lite":              {ID: "gemini-3.5-flash-lite", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 0.30, OutputPricePer1M: 2.50, CacheReadPricePer1M: 0.03, SupportsThinking: true, ThoughtSignature: true, AudioInputPricePer1M: 0.30, Label: "Gemini 3.5 Flash Lite"},
	"gemini-3.1-pro-preview":             {ID: "gemini-3.1-pro-preview", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 2.00, OutputPricePer1M: 12.00, CacheReadPricePer1M: 0.20, SupportsThinking: true, ThoughtSignature: true, Label: "Gemini 3.1 Pro"},
	"gemini-3.1-pro-preview-customtools": {ID: "gemini-3.1-pro-preview-customtools", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 2.00, OutputPricePer1M: 12.00, CacheReadPricePer1M: 0.20, SupportsThinking: true, ThoughtSignature: true, Label: "Gemini 3.1 Pro (custom tools)"},
	"gemini-3.1-flash-lite":              {ID: "gemini-3.1-flash-lite", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 0.25, OutputPricePer1M: 1.50, CacheReadPricePer1M: 0.025, SupportsThinking: true, ThoughtSignature: true, AudioInputPricePer1M: 0.50, Label: "Gemini 3.1 Flash Lite"},
	"gemini-3-flash-preview":             {ID: "gemini-3-flash-preview", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 0.50, OutputPricePer1M: 3.00, CacheReadPricePer1M: 0.05, SupportsThinking: true, ThoughtSignature: true, AudioInputPricePer1M: 1.00, Label: "Gemini 3 Flash"},
	"gemini-2.5-pro":                     {ID: "gemini-2.5-pro", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 1.25, OutputPricePer1M: 10.00, CacheReadPricePer1M: 0.125, Thinking: true, Label: "Gemini 2.5 Pro"},
	"gemini-2.5-flash":                   {ID: "gemini-2.5-flash", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 0.30, OutputPricePer1M: 2.50, CacheReadPricePer1M: 0.03, Thinking: true, AudioInputPricePer1M: 1.00, Label: "Gemini 2.5 Flash"},
	"gemini-2.5-flash-lite":              {ID: "gemini-2.5-flash-lite", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 0.10, OutputPricePer1M: 0.40, CacheReadPricePer1M: 0.01, Thinking: true, AudioInputPricePer1M: 0.30, Label: "Flash Lite"},
	// Gemini non-chat modality models (explicit capabilities; not chat defaults).
	"gemini-embedding-001":        {ID: "gemini-embedding-001", Provider: "gemini", ContextWindow: 2048, InputPricePer1M: 0.15, Capabilities: CapabilitySet(CapabilityEmbeddings | CapabilityBatch), Label: "Gemini Embedding"},
	"gemini-embedding-2":          {ID: "gemini-embedding-2", Provider: "gemini", ContextWindow: 8192, InputPricePer1M: 0.20, Capabilities: CapabilitySet(CapabilityEmbeddings | CapabilityBatch), Label: "Gemini Embedding 2"},
	"gemini-2.5-flash-image":      {ID: "gemini-2.5-flash-image", Provider: "gemini", ContextWindow: 32768, Capabilities: CapabilitySet(CapabilityImageGeneration | CapabilityBatch), Label: "Gemini 2.5 Flash Image"},
	"gemini-3-pro-image":          {ID: "gemini-3-pro-image", Provider: "gemini", ContextWindow: 65536, Capabilities: CapabilitySet(CapabilityImageGeneration | CapabilityBatch), Label: "Gemini 3 Pro Image"},
	"gemini-3.1-flash-image":      {ID: "gemini-3.1-flash-image", Provider: "gemini", ContextWindow: 131072, Capabilities: CapabilitySet(CapabilityImageGeneration | CapabilityBatch), Label: "Gemini 3.1 Flash Image (Nano Banana 2)"},
	"gemini-3.1-flash-lite-image": {ID: "gemini-3.1-flash-lite-image", Provider: "gemini", ContextWindow: 65536, Capabilities: CapabilitySet(CapabilityImageGeneration | CapabilityBatch), Label: "Gemini 3.1 Flash Lite Image (Nano Banana 2 Lite)"},
	"gemini-nano-banana-2.1":      {ID: "gemini-nano-banana-2.1", Provider: "gemini", ContextWindow: 131072, Capabilities: CapabilitySet(CapabilityImageGeneration | CapabilityBatch), Label: "Nano Banana 2.1"},
	"gemini-3-pro-image-preview":  {ID: "gemini-3-pro-image-preview", Provider: "gemini", ContextWindow: 65536, Capabilities: CapabilitySet(CapabilityImageGeneration | CapabilityBatch), Label: "Gemini 3 Pro Image"},
	// Pricing (ai.google.dev/gemini-api/docs/pricing, 2026-10-07): $2.00/1M audio input,
	// $12.00/1M text output. Audio-only input, so no text input price.
	// Dedicated speech-to-text (ai.google.dev/gemini-api/docs/models/gemini-3.5-transcribe,
	// 2026-10-06): audio-only generateContent with audioTranscriptionConfig, 98,304
	// input / 32,768 output tokens, no Batch API, no thinking. Live-verified.
	"gemini-3.5-transcribe": {ID: "gemini-3.5-transcribe", Provider: "gemini", MaxTokens: 32768, ContextWindow: 98304, Capabilities: CapabilitySet(CapabilityTranscription), OutputPricePer1M: 12.00, AudioInputPricePer1M: 2.00, Label: "Gemini 3.5 Transcribe"},

	// OpenAI — GPT-6 / GPT-5.x families (developers.openai.com/api/docs/pricing, 2026-10-31)
	// Reasoning models use ResponsesAPI: true — reasoning effort is controlled via /v1/responses, not Chat Completions.
	// Non-reasoning "chat" variants use Chat Completions normally.
	// GPT-6 family: 1.05M context, 922K max input, 128K output. Tool calling
	// requires the Responses API ("Chat Completions supports function calling
	// only with reasoning_effort set to none"). Prompts over 272K cost 2× input,
	// 1.5× output (not modeled).
	"gpt-6-astra":   {ID: "gpt-6-astra", Provider: "openai", MaxTokens: 128000, ContextWindow: 922000, InputPricePer1M: 10.00, OutputPricePer1M: 50.00, CacheReadPricePer1M: 1.00, Thinking: true, ResponsesAPI: true, Label: "GPT-6 Astra"},
	"gpt-6.1-sol":   {ID: "gpt-6.1-sol", Provider: "openai", MaxTokens: 128000, ContextWindow: 922000, InputPricePer1M: 2.00, OutputPricePer1M: 10.00, CacheReadPricePer1M: 0.10, Thinking: true, ResponsesAPI: true, Label: "GPT-6.1 Sol"},
	"gpt-6.1-terra": {ID: "gpt-6.1-terra", Provider: "openai", MaxTokens: 128000, ContextWindow: 922000, InputPricePer1M: 1.00, OutputPricePer1M: 5.00, CacheReadPricePer1M: 0.05, Thinking: true, ResponsesAPI: true, Label: "GPT-6.1 Terra"},
	"gpt-6-sol":     {ID: "gpt-6-sol", Provider: "openai", MaxTokens: 128000, ContextWindow: 922000, InputPricePer1M: 2.00, OutputPricePer1M: 10.00, CacheReadPricePer1M: 0.20, Thinking: true, ResponsesAPI: true, Label: "GPT-6 Sol"},
	"gpt-6-luna":    {ID: "gpt-6-luna", Provider: "openai", MaxTokens: 128000, ContextWindow: 922000, InputPricePer1M: 0.10, OutputPricePer1M: 0.50, CacheReadPricePer1M: 0.01, Thinking: true, ResponsesAPI: true, Label: "GPT-6 Luna"},
	// GPT-5.6 family (Sol/Terra/Luna). Alias gpt-5.6 routes to Sol on the wire.
	// Sol is at promotional pricing "at least through November 21, 2026".
	"gpt-5.6-sol":   {ID: "gpt-5.6-sol", Provider: "openai", MaxTokens: 128000, ContextWindow: 1050000, InputPricePer1M: 4.00, OutputPricePer1M: 20.00, CacheReadPricePer1M: 0.40, Thinking: true, ResponsesAPI: true, Label: "GPT-5.6 Sol"},
	"gpt-5.6-terra": {ID: "gpt-5.6-terra", Provider: "openai", MaxTokens: 128000, ContextWindow: 1050000, InputPricePer1M: 2.00, OutputPricePer1M: 12.00, CacheReadPricePer1M: 0.20, Thinking: true, ResponsesAPI: true, Label: "GPT-5.6 Terra"},
	"gpt-5.6-luna":  {ID: "gpt-5.6-luna", Provider: "openai", MaxTokens: 128000, ContextWindow: 1050000, InputPricePer1M: 0.20, OutputPricePer1M: 1.20, CacheReadPricePer1M: 0.02, Thinking: true, ResponsesAPI: true, Label: "GPT-5.6 Luna"},
	"gpt-5.5-pro":   {ID: "gpt-5.5-pro", Provider: "openai", MaxTokens: 128000, ContextWindow: 1048576, InputPricePer1M: 30.00, OutputPricePer1M: 180.00, Thinking: true, ResponsesAPI: true, Label: "GPT-5.5 Pro"},
	"gpt-5.5":       {ID: "gpt-5.5", Provider: "openai", MaxTokens: 128000, ContextWindow: 1048576, InputPricePer1M: 5.00, OutputPricePer1M: 30.00, CacheReadPricePer1M: 0.50, Thinking: true, ResponsesAPI: true, Label: "GPT-5.5"},
	"gpt-5.4-pro":   {ID: "gpt-5.4-pro", Provider: "openai", MaxTokens: 128000, ContextWindow: 1048576, InputPricePer1M: 30.00, OutputPricePer1M: 180.00, Thinking: true, ResponsesAPI: true, Label: "GPT-5.4 Pro"},
	"gpt-5.4":       {ID: "gpt-5.4", Provider: "openai", MaxTokens: 128000, ContextWindow: 1048576, InputPricePer1M: 2.50, OutputPricePer1M: 15.00, CacheReadPricePer1M: 0.25, Thinking: true, ResponsesAPI: true, Label: "GPT-5.4"},
	"gpt-5.4-mini":  {ID: "gpt-5.4-mini", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 0.75, OutputPricePer1M: 4.50, CacheReadPricePer1M: 0.075, Thinking: true, ResponsesAPI: true, Label: "GPT-5.4 Mini"},
	"gpt-5.4-nano":  {ID: "gpt-5.4-nano", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 0.20, OutputPricePer1M: 1.25, CacheReadPricePer1M: 0.02, Thinking: true, ResponsesAPI: true, Label: "GPT-5.4 Nano"},
	"gpt-5.3-codex": {ID: "gpt-5.3-codex", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 1.75, OutputPricePer1M: 14.00, CacheReadPricePer1M: 0.175, Thinking: true, ResponsesAPI: true, Label: "GPT-5.3 Codex"},
	"gpt-5.2":       {ID: "gpt-5.2", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 1.75, OutputPricePer1M: 14.00, Thinking: true, ResponsesAPI: true, Label: "GPT-5.2"},
	"gpt-5.2-pro":   {ID: "gpt-5.2-pro", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 21.00, OutputPricePer1M: 168.00, Thinking: true, ResponsesAPI: true, Label: "GPT-5.2 Pro"},
	"gpt-5.1":       {ID: "gpt-5.1", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 1.25, OutputPricePer1M: 10.00, Thinking: true, ResponsesAPI: true, Label: "GPT-5.1"},
	"gpt-5":         {ID: "gpt-5", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 1.25, OutputPricePer1M: 10.00, Thinking: true, ResponsesAPI: true, Label: "GPT-5"},
	"gpt-5-pro":     {ID: "gpt-5-pro", Provider: "openai", MaxTokens: 272000, ContextWindow: 400000, InputPricePer1M: 15.00, OutputPricePer1M: 120.00, Thinking: true, ResponsesAPI: true, Label: "GPT-5 Pro"},
	"gpt-5-mini":    {ID: "gpt-5-mini", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 0.25, OutputPricePer1M: 2.00, Thinking: true, ResponsesAPI: true, Label: "GPT-5 Mini"},
	"gpt-5-nano":    {ID: "gpt-5-nano", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 0.05, OutputPricePer1M: 0.40, Thinking: true, ResponsesAPI: true, Label: "GPT-5 Nano"},

	// OpenAI — GPT-4.1 family (non-reasoning, 1M context)
	"gpt-4.1":      {ID: "gpt-4.1", Provider: "openai", MaxTokens: 32768, ContextWindow: 1048576, InputPricePer1M: 2.00, OutputPricePer1M: 8.00, Label: "GPT-4.1"},
	"gpt-4.1-mini": {ID: "gpt-4.1-mini", Provider: "openai", MaxTokens: 32768, ContextWindow: 1048576, InputPricePer1M: 0.40, OutputPricePer1M: 1.60, Label: "GPT-4.1 Mini"},

	// OpenAI — O-series reasoning models
	"o3":     {ID: "o3", Provider: "openai", MaxTokens: 100000, ContextWindow: 200000, InputPricePer1M: 2.00, OutputPricePer1M: 8.00, Thinking: true, Label: "O3"},
	"o3-pro": {ID: "o3-pro", Provider: "openai", MaxTokens: 100000, ContextWindow: 200000, InputPricePer1M: 20.00, OutputPricePer1M: 80.00, Thinking: true, Label: "O3 Pro"},

	// Groq — cloud inference (console.groq.com/docs/models, 2026-11-10).
	// The Llama rows are enterprise-only since 2026-08-16 (docs/deprecations: free and
	// developer tiers use openai/gpt-oss-*); their prices are the last published ones.
	"llama-3.3-70b-versatile": {ID: "llama-3.3-70b-versatile", Provider: "groq", MaxTokens: 32768, ContextWindow: 131072, InputPricePer1M: 0.59, OutputPricePer1M: 0.79, Label: "Llama 3.3 70B"},
	"llama-3.1-8b-instant":    {ID: "llama-3.1-8b-instant", Provider: "groq", MaxTokens: 131072, ContextWindow: 131072, InputPricePer1M: 0.05, OutputPricePer1M: 0.08, Label: "Llama 3.1 8B"},
	"openai/gpt-oss-120b":     {ID: "openai/gpt-oss-120b", Provider: "groq", MaxTokens: 65536, ContextWindow: 131072, InputPricePer1M: 0.15, OutputPricePer1M: 0.60, Label: "GPT-OSS 120B"},
	"qwen/qwen3.8-27b":        {ID: "qwen/qwen3.8-27b", Provider: "groq", MaxTokens: 16384, ContextWindow: 131072, InputPricePer1M: 0.80, OutputPricePer1M: 4.00, Label: "Qwen 3.8 27B (preview)"},
	"openai/gpt-oss-20b":      {ID: "openai/gpt-oss-20b", Provider: "groq", MaxTokens: 65536, ContextWindow: 131072, InputPricePer1M: 0.075, OutputPricePer1M: 0.30, Label: "GPT-OSS 20B"},

	// Mistral — cloud API (docs.mistral.ai/models/<card>, 2026-11-10). Large 3 and Medium 3.5
	// are deprecated (retire 2027-04-30); mistral-medium-latest now points to Medium 3.5.
	// Mistral Large 4 — docs.mistral.ai/models/mistral-large-4-0 (2026-11-10).
	// Max output is not published; the context window is used, as for Large 3.
	"mistral-large-4-0":     {ID: "mistral-large-4-0", Provider: "mistral", MaxTokens: 1048576, ContextWindow: 1048576, InputPricePer1M: 0.70, OutputPricePer1M: 2.10, CacheReadPricePer1M: 0.07, Thinking: true, Label: "Mistral Large 4"},
	"mistral-large-2512":    {ID: "mistral-large-2512", Provider: "mistral", MaxTokens: 262144, ContextWindow: 262144, InputPricePer1M: 0.50, OutputPricePer1M: 1.50, Label: "Mistral Large"},
	"mistral-medium-latest": {ID: "mistral-medium-latest", Provider: "mistral", MaxTokens: 262144, ContextWindow: 262144, InputPricePer1M: 1.50, OutputPricePer1M: 7.50, Thinking: true, Label: "Mistral Medium 3.5"},
	"mistral-small-2603":    {ID: "mistral-small-2603", Provider: "mistral", MaxTokens: 262144, ContextWindow: 262144, InputPricePer1M: 0.15, OutputPricePer1M: 0.60, Thinking: true, Label: "Mistral Small 4"},
	"codestral-2508":        {ID: "codestral-2508", Provider: "mistral", MaxTokens: 131072, ContextWindow: 131072, InputPricePer1M: 0.30, OutputPricePer1M: 0.90, Label: "Codestral"},
	"ministral-3b-2512":     {ID: "ministral-3b-2512", Provider: "mistral", MaxTokens: 262144, ContextWindow: 262144, InputPricePer1M: 0.10, OutputPricePer1M: 0.10, Label: "Ministral 3B"},
	"ministral-8b-2512":     {ID: "ministral-8b-2512", Provider: "mistral", MaxTokens: 262144, ContextWindow: 262144, InputPricePer1M: 0.15, OutputPricePer1M: 0.15, Label: "Ministral 8B"},
	"ministral-14b-2512":    {ID: "ministral-14b-2512", Provider: "mistral", MaxTokens: 262144, ContextWindow: 262144, InputPricePer1M: 0.20, OutputPricePer1M: 0.20, Label: "Ministral 14B"},
	"mistral-embed":         {ID: "mistral-embed", Provider: "mistral", ContextWindow: 8192, InputPricePer1M: 0.10, Capabilities: CapabilitySet(CapabilityEmbeddings), Label: "Mistral Embed"},
	"mistral-large-latest":  {ID: "mistral-large-latest", Provider: "mistral", MaxTokens: 262144, ContextWindow: 262144, InputPricePer1M: 0.50, OutputPricePer1M: 1.50, Label: "Mistral Large Latest"},
	"codestral-latest":      {ID: "codestral-latest", Provider: "mistral", MaxTokens: 131072, ContextWindow: 131072, InputPricePer1M: 0.30, OutputPricePer1M: 0.90, Label: "Codestral Latest"},

	// DeepSeek — api-docs.deepseek.com/quick_start/pricing (2026-10-31). No first-party /embeddings.
	// Peak-hour rates (off-peak is 50%). deepseek-v4-flash is a legacy name routed to V4.1 Flash.
	"deepseek-flash":    {ID: "deepseek-flash", Provider: "deepseek", MaxTokens: 384000, ContextWindow: 1000000, InputPricePer1M: 0.30, OutputPricePer1M: 1.20, CacheReadPricePer1M: 0.006, Thinking: true, Label: "DeepSeek V4.1 Flash"},
	"deepseek-v4-flash": {ID: "deepseek-v4-flash", Provider: "deepseek", MaxTokens: 384000, ContextWindow: 1000000, InputPricePer1M: 0.30, OutputPricePer1M: 1.20, CacheReadPricePer1M: 0.006, Thinking: true, Label: "DeepSeek V4 Flash (legacy → V4.1)"},
	"deepseek-v4-pro":   {ID: "deepseek-v4-pro", Provider: "deepseek", MaxTokens: 384000, ContextWindow: 1000000, InputPricePer1M: 1.32, OutputPricePer1M: 3.96, CacheReadPricePer1M: 0.044, Thinking: true, Label: "DeepSeek V4 Pro"},

	// Cohere — OpenAI-compatible API (docs.cohere.com/docs/models + cohere.com/pricing, 2026-11-10)
	"command-a-plus-05-2026": {ID: "command-a-plus-05-2026", Provider: "cohere", MaxTokens: 64000, ContextWindow: 128000, InputPricePer1M: 2.50, OutputPricePer1M: 10.00, Thinking: true, Label: "Command A+"},
	"command-a-03-2025":      {ID: "command-a-03-2025", Provider: "cohere", MaxTokens: 8192, ContextWindow: 256000, InputPricePer1M: 2.50, OutputPricePer1M: 10.00, Label: "Command A"},

	// Z.ai / GLM — OpenAI-compatible API at api.z.ai/api/paas/v4 (docs.z.ai pricing, 2026-10-31). Open weights; license per model card.
	"glm-5.3":       {ID: "glm-5.3", Provider: "zai", MaxTokens: 131072, ContextWindow: 1048576, InputPricePer1M: 1.40, OutputPricePer1M: 4.40, CacheReadPricePer1M: 0.26, Thinking: true, Label: "GLM-5.3"},
	"glm-5.3-flash": {ID: "glm-5.3-flash", Provider: "zai", MaxTokens: 131072, ContextWindow: 1048576, InputPricePer1M: 0.15, OutputPricePer1M: 0.50, CacheReadPricePer1M: 0.03, Thinking: true, Label: "GLM-5.3 Flash"},
	"glm-5.2":       {ID: "glm-5.2", Provider: "zai", MaxTokens: 131072, ContextWindow: 1048576, InputPricePer1M: 1.40, OutputPricePer1M: 4.40, CacheReadPricePer1M: 0.26, Thinking: true, Label: "GLM-5.2"},
	"glm-4.7":       {ID: "glm-4.7", Provider: "zai", MaxTokens: 131072, ContextWindow: 200000, InputPricePer1M: 0.60, OutputPricePer1M: 2.20, CacheReadPricePer1M: 0.11, Thinking: true, Label: "GLM-4.7"},
	"glm-4.6":       {ID: "glm-4.6", Provider: "zai", MaxTokens: 131072, ContextWindow: 200000, InputPricePer1M: 0.60, OutputPricePer1M: 2.20, CacheReadPricePer1M: 0.11, Label: "GLM-4.6"},

	// MiniMax — OpenAI-compatible API at api.minimax.io (platform.minimax.io pay-as-you-go, 2026-10-31).
	// Open weights. Multimodal. MiniMax-M3 rates are the <=512K-input tier; above 512K they double.
	"MiniMax-M3":         {ID: "MiniMax-M3", Provider: "minimax", ContextWindow: 1048576, InputPricePer1M: 0.30, OutputPricePer1M: 1.20, CacheReadPricePer1M: 0.06, Label: "MiniMax M3"},
	"MiniMax-M3.1-Flash": {ID: "MiniMax-M3.1-Flash", Provider: "minimax", ContextWindow: 1000000, InputPricePer1M: 0.15, OutputPricePer1M: 0.60, CacheReadPricePer1M: 0.015, Thinking: true, Label: "MiniMax M3.1 Flash"},
	"MiniMax-M2.5":       {ID: "MiniMax-M2.5", Provider: "minimax", ContextWindow: 1000000, InputPricePer1M: 0.30, OutputPricePer1M: 1.20, CacheReadPricePer1M: 0.06, Label: "MiniMax M2.5"},

	// Moonshot / Kimi — OpenAI-compatible API at api.moonshot.ai (platform.kimi.ai/docs/pricing/chat, 2026-11-10).
	// K3 models: thinking always on, 1M context, max_completion_tokens default 131072 (up to
	// 1048576), sampling parameters fixed. Cache writes are billed (5m = 1x input, 1h = 2x).
	// The K2 series stopped taking new users on 2026-11-01 and retires late December 2026.
	"kimi-k3.1-pro":  {ID: "kimi-k3.1-pro", Provider: "moonshot", MaxTokens: 131072, ContextWindow: 1048576, InputPricePer1M: 2.00, OutputPricePer1M: 10.00, CacheWritePricePer1M: 2.00, CacheWrite1hPricePer1M: 4.00, CacheReadPricePer1M: 0.20, Thinking: true, Label: "Kimi K3.1 Pro"},
	"kimi-k3":        {ID: "kimi-k3", Provider: "moonshot", MaxTokens: 131072, ContextWindow: 1048576, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, CacheWritePricePer1M: 3.00, CacheWrite1hPricePer1M: 6.00, CacheReadPricePer1M: 0.30, Thinking: true, Label: "Kimi K3"},
	"kimi-k2.7-code": {ID: "kimi-k2.7-code", Provider: "moonshot", MaxTokens: 262144, ContextWindow: 262144, InputPricePer1M: 0.95, OutputPricePer1M: 4.00, CacheReadPricePer1M: 0.19, Thinking: true, Label: "Kimi K2.7 Code"},

	// Perplexity — Sonar API (from docs.perplexity.ai/getting-started/pricing, 2026-06-08).
	// Representative subset; the registry only affects cost estimation + discovery — any model ID can be passed directly.
	"sonar":               {ID: "sonar", Provider: "perplexity", ContextWindow: 128000, InputPricePer1M: 1.00, OutputPricePer1M: 1.00, Label: "Sonar"},
	"sonar-pro":           {ID: "sonar-pro", Provider: "perplexity", ContextWindow: 200000, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, Label: "Sonar Pro"},
	"sonar-reasoning-pro": {ID: "sonar-reasoning-pro", Provider: "perplexity", ContextWindow: 128000, InputPricePer1M: 2.00, OutputPricePer1M: 8.00, Thinking: true, Label: "Sonar Reasoning Pro"},
	"sonar-deep-research": {ID: "sonar-deep-research", Provider: "perplexity", ContextWindow: 128000, InputPricePer1M: 2.00, OutputPricePer1M: 8.00, Thinking: true, Label: "Sonar Deep Research"},

	// Fireworks — serverless (from docs.fireworks.ai/serverless/pricing, 2026-11-10; IDs live-probed).
	// Open-weight families (GLM/Z.ai, MiniMax, Kimi) tracked under the host you'd call; prices cross-checked vs the provider + Together.
	// Context windows are not published per host, so they are left unknown (0).
	"accounts/fireworks/models/kimi-k3p1-pro":       {ID: "accounts/fireworks/models/kimi-k3p1-pro", Provider: "fireworks", InputPricePer1M: 2.00, OutputPricePer1M: 10.00, CacheReadPricePer1M: 0.20, Thinking: true, Label: "Kimi K3.1 Pro"},
	"accounts/fireworks/models/kimi-k3":             {ID: "accounts/fireworks/models/kimi-k3", Provider: "fireworks", InputPricePer1M: 3.00, OutputPricePer1M: 15.00, CacheReadPricePer1M: 0.30, Thinking: true, Label: "Kimi K3"},
	"accounts/fireworks/models/glm-5p3":             {ID: "accounts/fireworks/models/glm-5p3", Provider: "fireworks", InputPricePer1M: 1.40, OutputPricePer1M: 4.40, CacheReadPricePer1M: 0.26, Thinking: true, Label: "GLM-5.3"},
	"accounts/fireworks/models/glm-5p3-flash":       {ID: "accounts/fireworks/models/glm-5p3-flash", Provider: "fireworks", InputPricePer1M: 0.15, OutputPricePer1M: 0.50, CacheReadPricePer1M: 0.03, Thinking: true, Label: "GLM-5.3 Flash"},
	"accounts/fireworks/models/minimax-m3":          {ID: "accounts/fireworks/models/minimax-m3", Provider: "fireworks", InputPricePer1M: 0.30, OutputPricePer1M: 1.20, CacheReadPricePer1M: 0.06, Label: "MiniMax M3"},
	"accounts/fireworks/models/deepseek-v4p1-flash": {ID: "accounts/fireworks/models/deepseek-v4p1-flash", Provider: "fireworks", InputPricePer1M: 0.30, OutputPricePer1M: 1.20, CacheReadPricePer1M: 0.006, Thinking: true, Label: "DeepSeek V4.1 Flash"},
	"accounts/fireworks/models/qwen3p8-max":         {ID: "accounts/fireworks/models/qwen3p8-max", Provider: "fireworks", InputPricePer1M: 2.00, OutputPricePer1M: 6.00, CacheReadPricePer1M: 0.25, Thinking: true, Label: "Qwen 3.8 Max"},

	// Together AI — serverless (together.ai/pricing + docs.together.ai/docs/serverless-models, 2026-11-10).
	"moonshotai/Kimi-K3.1-Pro":        {ID: "moonshotai/Kimi-K3.1-Pro", Provider: "together", ContextWindow: 1048576, InputPricePer1M: 2.00, OutputPricePer1M: 10.00, CacheReadPricePer1M: 0.20, Thinking: true, Label: "Kimi K3.1 Pro (Together)"},
	"moonshotai/Kimi-K3":              {ID: "moonshotai/Kimi-K3", Provider: "together", ContextWindow: 1048576, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, CacheReadPricePer1M: 0.30, Thinking: true, Label: "Kimi K3 (Together)"},
	"zai-org/GLM-5.3":                 {ID: "zai-org/GLM-5.3", Provider: "together", ContextWindow: 1048575, InputPricePer1M: 1.40, OutputPricePer1M: 4.40, CacheReadPricePer1M: 0.26, Thinking: true, Label: "GLM-5.3 (Together)"},
	"zai-org/GLM-5.3-Flash":           {ID: "zai-org/GLM-5.3-Flash", Provider: "together", ContextWindow: 1048575, InputPricePer1M: 0.15, OutputPricePer1M: 0.50, CacheReadPricePer1M: 0.03, Thinking: true, Label: "GLM-5.3 Flash (Together)"},
	"deepseek-ai/DeepSeek-V4.1-Flash": {ID: "deepseek-ai/DeepSeek-V4.1-Flash", Provider: "together", ContextWindow: 1000000, InputPricePer1M: 0.30, OutputPricePer1M: 1.20, CacheReadPricePer1M: 0.006, Thinking: true, Label: "DeepSeek V4.1 Flash (Together)"},
	"Qwen/Qwen3.8-Flash":              {ID: "Qwen/Qwen3.8-Flash", Provider: "together", ContextWindow: 1000000, InputPricePer1M: 0.15, OutputPricePer1M: 0.47, Thinking: true, Label: "Qwen 3.8 Flash (Together)"},

	// DeepInfra — serverless (from deepinfra.com/pricing + api.deepinfra.com/models/list, 2026-11-10; distinct IDs to avoid registry key collisions with other hosts).
	"Qwen/Qwen3-32B": {ID: "Qwen/Qwen3-32B", Provider: "deepinfra", ContextWindow: 40960, InputPricePer1M: 0.08, OutputPricePer1M: 0.28, Thinking: true, Label: "Qwen3 32B (DeepInfra)"},
	"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": {ID: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", Provider: "deepinfra", ContextWindow: 131072, InputPricePer1M: 0.02, OutputPricePer1M: 0.04, Label: "Llama 3.1 8B Turbo (DeepInfra)"},
	"deepseek-ai/DeepSeek-V4-Flash":               {ID: "deepseek-ai/DeepSeek-V4-Flash", Provider: "deepinfra", ContextWindow: 1048576, InputPricePer1M: 0.09, OutputPricePer1M: 0.18, CacheReadPricePer1M: 0.018, Thinking: true, Label: "DeepSeek V4 Flash (DeepInfra)"},

	// Meta Model API — OpenAI-compatible at api.meta.ai (dev.meta.ai/docs/pricing-rate-limits, 2026-11-10). Muse Spark closed-weights; Llama remains on hosts like Groq/Together.
	// Standard tier keeps prompts/completions out of training; contributor tier is discounted and may train.
	// Muse Wonder is in preview; Muse Ember bills prompts over 200K input tokens at 3x (not modeled).
	"muse-wonder-1.0":            {ID: "muse-wonder-1.0", Provider: "meta", ContextWindow: 1048576, InputPricePer1M: 4.00, OutputPricePer1M: 20.00, CacheReadPricePer1M: 0.20, Thinking: true, Label: "Muse Wonder 1.0"},
	"muse-spark-1.3":             {ID: "muse-spark-1.3", Provider: "meta", MaxTokens: 128000, ContextWindow: 1048576, InputPricePer1M: 1.25, OutputPricePer1M: 4.25, CacheReadPricePer1M: 0.15, Thinking: true, Label: "Muse Spark 1.3"},
	"muse-ember-1.0":             {ID: "muse-ember-1.0", Provider: "meta", MaxTokens: 524288, ContextWindow: 1048576, InputPricePer1M: 0.10, OutputPricePer1M: 0.50, CacheReadPricePer1M: 0.004, Thinking: true, Label: "Muse Ember 1.0"},
	"muse-spark-1.2":             {ID: "muse-spark-1.2", Provider: "meta", MaxTokens: 128000, ContextWindow: 1048576, InputPricePer1M: 1.25, OutputPricePer1M: 4.25, CacheReadPricePer1M: 0.15, Thinking: true, Label: "Muse Spark 1.2"},
	"muse-spark-1.1":             {ID: "muse-spark-1.1", Provider: "meta", MaxTokens: 128000, ContextWindow: 1048576, InputPricePer1M: 1.25, OutputPricePer1M: 4.25, CacheReadPricePer1M: 0.15, Thinking: true, Label: "Muse Spark 1.1"},
	"muse-spark-1.3-contributor": {ID: "muse-spark-1.3-contributor", Provider: "meta", MaxTokens: 128000, ContextWindow: 1048576, InputPricePer1M: 0.10, OutputPricePer1M: 0.20, CacheReadPricePer1M: 0.002, Thinking: true, Label: "Muse Spark 1.3 (Contributor)"},
	"muse-spark-1.2-contributor": {ID: "muse-spark-1.2-contributor", Provider: "meta", MaxTokens: 128000, ContextWindow: 1048576, InputPricePer1M: 0.10, OutputPricePer1M: 0.20, CacheReadPricePer1M: 0.002, Thinking: true, Label: "Muse Spark 1.2 (Contributor)"},

	// NVIDIA NIM — hosted catalog at build.nvidia.com is free for development (rate-limited; no public per-token price), so prices are left 0. IDs as integrate.api.nvidia.com exposes them.
	"nvidia/nemotron-3-ultra-550b-a55b":      {ID: "nvidia/nemotron-3-ultra-550b-a55b", Provider: "nvidia", Label: "Nemotron 3 Ultra (NIM, free dev tier)"},
	"nvidia/llama-3.1-nemotron-70b-instruct": {ID: "nvidia/llama-3.1-nemotron-70b-instruct", Provider: "nvidia", ContextWindow: 131072, Label: "Llama 3.1 Nemotron 70B (NIM, free dev tier)"},

	// DashScope (Alibaba) — Model Studio international (Singapore) pricing (2026-10-31). Intl default; override BaseURL for CN.
	// Rates are the shortest input tier (qwen3.6: <=256K; above that they rise 2–4×).
	"qwen3.8-max":        {ID: "qwen3.8-max", Provider: "dashscope", ContextWindow: 1000000, InputPricePer1M: 2.00, OutputPricePer1M: 6.00, Thinking: true, Label: "Qwen 3.8 Max"},
	"qwen3.8-flash":      {ID: "qwen3.8-flash", Provider: "dashscope", ContextWindow: 1000000, InputPricePer1M: 0.15, OutputPricePer1M: 0.47, Thinking: true, Label: "Qwen 3.8 Flash"},
	"qwen3.6-plus":       {ID: "qwen3.6-plus", Provider: "dashscope", MaxTokens: 65536, ContextWindow: 1000000, InputPricePer1M: 0.50, OutputPricePer1M: 3.00, Thinking: true, Label: "Qwen 3.6 Plus"},
	"qwen3.6-flash":      {ID: "qwen3.6-flash", Provider: "dashscope", MaxTokens: 32768, ContextWindow: 1000000, InputPricePer1M: 0.25, OutputPricePer1M: 1.50, Thinking: true, Label: "Qwen 3.6 Flash"},
	"qwen3-max":          {ID: "qwen3-max", Provider: "dashscope", MaxTokens: 65536, ContextWindow: 262144, InputPricePer1M: 1.20, OutputPricePer1M: 6.00, Thinking: true, Label: "Qwen 3 Max"},
	"qwen3.5-omni-plus":  {ID: "qwen3.5-omni-plus", Provider: "dashscope", MaxTokens: 16384, ContextWindow: 256000, Thinking: true, Label: "Qwen 3.5 Omni Plus"},
	"qwen3.5-omni-flash": {ID: "qwen3.5-omni-flash", Provider: "dashscope", MaxTokens: 16384, ContextWindow: 256000, Thinking: true, Label: "Qwen 3.5 Omni Flash"},
	"qwen3.5-omni-light": {ID: "qwen3.5-omni-light", Provider: "dashscope", MaxTokens: 16384, ContextWindow: 256000, Thinking: true, Label: "Qwen 3.5 Omni Light"},
	"text-embedding-v3":  {ID: "text-embedding-v3", Provider: "dashscope", ContextWindow: 8192, InputPricePer1M: 0.07, Capabilities: CapabilitySet(CapabilityEmbeddings), Label: "Qwen Text Embedding V3"},
	"text-embedding-v2":  {ID: "text-embedding-v2", Provider: "dashscope", ContextWindow: 2048, InputPricePer1M: 0.07, Capabilities: CapabilitySet(CapabilityEmbeddings), Label: "Qwen Text Embedding V2"},
	// Same embedding IDs resolve for dashscope-cn via provider family mapping when using BaseURL override.

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
	"gemma4:e4b":           {ID: "gemma4:e4b", Provider: "ollama", Thinking: true, Label: "Gemma4 E4B"},
	"gemma3:12b":           {ID: "gemma3:12b", Provider: "ollama", NoToolSupport: true, Label: "Gemma3 12B"},
	"gemma3:4b":            {ID: "gemma3:4b", Provider: "ollama", NoToolSupport: true, Label: "Gemma3 4B"},
	"gemma2:2b":            {ID: "gemma2:2b", Provider: "ollama", NoToolSupport: true, Label: "Gemma2 2B"},

	// OpenAI-compatible non-chat models. These are separate model identities so
	// callers cannot infer modality support from a chat provider preset.
	"text-embedding-3-small": {ID: "text-embedding-3-small", Provider: "openai", Capabilities: CapabilitySet(CapabilityEmbeddings | CapabilityBatch), Label: "Text Embedding 3 Small"},
	"text-embedding-3-large": {ID: "text-embedding-3-large", Provider: "openai", Capabilities: CapabilitySet(CapabilityEmbeddings | CapabilityBatch), Label: "Text Embedding 3 Large"},
	"gpt-image-2":            {ID: "gpt-image-2", Provider: "openai", Capabilities: CapabilitySet(CapabilityImageGeneration), Label: "GPT Image 2"},
	// Listed by GET /v1/models on 2026-11-10 (2.6 created 2026-10-28).
	"gpt-image-2.6-flare":    {ID: "gpt-image-2.6-flare", Provider: "openai", Capabilities: CapabilitySet(CapabilityImageGeneration), Label: "GPT Image 2.6 Flare"},
	"gpt-image-2.6-sunburst": {ID: "gpt-image-2.6-sunburst", Provider: "openai", Capabilities: CapabilitySet(CapabilityImageGeneration), Label: "GPT Image 2.6 Sunburst"},
	"gpt-image-2.5-flare":    {ID: "gpt-image-2.5-flare", Provider: "openai", Capabilities: CapabilitySet(CapabilityImageGeneration), Label: "GPT Image 2.5 Flare"},
	"gpt-image-2.5-sunburst": {ID: "gpt-image-2.5-sunburst", Provider: "openai", Capabilities: CapabilitySet(CapabilityImageGeneration), Label: "GPT Image 2.5 Sunburst"},
	"tts-1":                  {ID: "tts-1", Provider: "openai", Capabilities: CapabilitySet(CapabilitySpeechSynthesis), Label: "TTS 1"},
	"whisper-1":              {ID: "whisper-1", Provider: "openai", Capabilities: CapabilitySet(CapabilityTranscription), AudioPricePerMinute: 0.006, Label: "Whisper 1"},
	// Transcription models from developers.openai.com/api/docs/guides/speech-to-text (2026-10-06).
	// Pricing (fetched 2026-10-07): developers.openai.com/api/docs/models/<id> lists
	// gpt-4o-transcribe "Audio tokens" $2.50 in / $10.00 out and gpt-4o-mini-transcribe
	// $1.25 / $5.00 per 1M; gpt-transcribe "Transcription audio duration" $0.0045/min;
	// developers.openai.com/api/docs/pricing lists Whisper at $0.006/min. Text prompt
	// tokens have no listed price for these models, so InputPricePer1M stays 0.
	"gpt-transcribe":         {ID: "gpt-transcribe", Provider: "openai", Capabilities: CapabilitySet(CapabilityTranscription), AudioPricePerMinute: 0.0045, Label: "GPT Transcribe"},
	"gpt-4o-transcribe":      {ID: "gpt-4o-transcribe", Provider: "openai", Capabilities: CapabilitySet(CapabilityTranscription), OutputPricePer1M: 10.00, AudioInputPricePer1M: 2.50, Label: "GPT-4o Transcribe"},
	"gpt-4o-mini-transcribe": {ID: "gpt-4o-mini-transcribe", Provider: "openai", Capabilities: CapabilitySet(CapabilityTranscription), OutputPricePer1M: 5.00, AudioInputPricePer1M: 1.25, Label: "GPT-4o mini Transcribe"},
	// xAI speech-to-text (docs.x.ai/developers/model-capabilities/audio/speech-to-text, 2026-10-06):
	// POST /v1/stt, not OpenAI-compatible; keyterm vocabulary.
	// Pricing: docs.x.ai/developers/models (fetched 2026-10-07) "STT (Batch) $0.10 / hour";
	// /v1/stt is the batch endpoint and returns the billed "duration" in seconds.
	"grok-voice-transcribe-2.0": {ID: "grok-voice-transcribe-2.0", Provider: "xai", Capabilities: CapabilitySet(CapabilityTranscription), AudioPricePerMinute: 0.10 / 60, Label: "Grok Voice Transcribe 2.0"},
}

func init() {
	// Existing curated entries are chat-generation models. Preserve their known
	// negative tool metadata and add only conservative provider-family features;
	// deployment-specific local/compatible models must declare their own set via
	// RegisterModel.
	for id, info := range modelRegistry {
		if info.Capabilities != 0 {
			continue
		}
		info.Capabilities = defaultChatCapabilities(info)
		modelRegistry[id] = info
	}
}

var defaultModels = map[string]string{
	"anthropic":    "claude-sonnet-5-5",
	"claude":       "claude-sonnet-5-5",
	"xai":          "grok-4.7",
	"grok":         "grok-4.7",
	"gemini":       "gemini-3.5-flash-lite",
	"google":       "gemini-3.5-flash-lite",
	"openai":       "gpt-6.1-sol",
	"gpt":          "gpt-6.1-sol",
	"meta":         "muse-spark-1.3",
	"groq":         "openai/gpt-oss-120b",
	"mistral":      "mistral-small-2603",
	"deepseek":     "deepseek-flash",
	"cohere":       "command-a-03-2025",
	"zai":          "glm-5.3",
	"z-ai":         "glm-5.3",
	"glm":          "glm-5.3",
	"minimax":      "MiniMax-M3",
	"moonshot":     "kimi-k3.1-pro",
	"kimi":         "kimi-k3.1-pro",
	"moonshot-cn":  "kimi-k2.7-code",
	"kimi-cn":      "kimi-k2.7-code",
	"dashscope":    "qwen3.6-plus",
	"qwen":         "qwen3.6-plus",
	"dashscope-cn": "qwen3.6-plus",
	"qwen-cn":      "qwen3.6-plus",
	"ollama":       "qwen3:8b",
}

var modelAliases = map[string]string{
	// Anthropic aliases
	"opus":   "claude-opus-5-5",
	"sonnet": "claude-sonnet-5-5",
	"fable":  "claude-fable-5-2",
	"haiku":  "claude-haiku-5-5",
	"claude": "claude-sonnet-5-5",

	// xAI/Grok aliases
	"grok":     "grok-4.7",
	"grok4.7":  "grok-4.7",
	"grok-4-7": "grok-4.7",
	"grok4.6":  "grok-4.6",
	"grok-4-6": "grok-4.6",
	"grok4.5":  "grok-4.5",
	"grok-4-5": "grok-4.5",
	"grok4.3":  "grok-4.3",
	"grok-4-3": "grok-4.3",
	"grok4.2":  "grok-4.20-0309-reasoning",
	"grok4.20": "grok-4.20-0309-reasoning",
	"grok4":    "grok-4.7",
	"grok-4":   "grok-4.7",
	// The 4.1/fast/mini models are retired; these follow xAI's own redirects.
	"grok4.1":            "grok-4.3",
	"grok-4-1":           "grok-4.3",
	"grok-reasoning":     "grok-4.3",
	"grok-4-reasoning":   "grok-4.3",
	"grok-4-1-reasoning": "grok-4.3",
	"grok-code":          "grok-build-0.1",
	"grok-mini":          "grok-4.3",
	"grok-build":         "grok-build-0.1",
	// Former beta slugs: xAI serves them as the 0309 models.
	"grok-4.20-beta":                    "grok-4.20-0309-reasoning",
	"grok-4.20-beta-0309-reasoning":     "grok-4.20-0309-reasoning",
	"grok-4.20-beta-0309-non-reasoning": "grok-4.20-0309-non-reasoning",
	"grok-4.20-multi-agent-beta-0309":   "grok-4.20-multi-agent-0309",

	// Groq aliases
	"groq":          "openai/gpt-oss-120b",
	"llama":         "llama-3.3-70b-versatile",
	"llama-70b":     "llama-3.3-70b-versatile",
	"llama-8b":      "llama-3.1-8b-instant",
	"llama4":        "meta-llama/llama-4-scout-17b-16e-instruct",
	"llama-4-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
	"gpt-oss":       "openai/gpt-oss-120b",
	"gpt-oss-20b":   "openai/gpt-oss-20b",

	// Mistral aliases
	"mistral-large":   "mistral-large-4-0", // Large 3 (2512) deprecated 2026-10-31
	"mistral-large-4": "mistral-large-4-0",
	"mistral-medium":  "mistral-medium-latest",
	"mistral-small":   "mistral-small-2603",
	"magistral":       "magistral-medium-2509",
	"codestral":       "codestral-2508",
	"devstral":        "devstral-2512",
	"devstral-small":  "devstral-small-2-2512",
	"ministral":       "ministral-8b-2512",
	"pixtral":         "pixtral-large-2411",
	"pixtral-large":   "pixtral-large-2411",

	// DeepSeek aliases
	"deepseek-cloud":    "deepseek-flash",
	"deepseek-v4":       "deepseek-flash",
	"deepseek-reasoner": "deepseek-reasoner",

	// Cohere aliases
	"cohere":    "command-a-03-2025",
	"command-a": "command-a-03-2025",

	// Z.ai / GLM aliases
	"glm":     "glm-5.3",
	"glm5.3":  "glm-5.3",
	"glm-5-3": "glm-5.3",
	"glm5.2":  "glm-5.2",
	"glm-5-2": "glm-5.2",
	"glm4.7":  "glm-4.7",
	"glm-4-7": "glm-4.7",

	// MiniMax aliases
	"minimax":      "MiniMax-M3",
	"minimax-m3":   "MiniMax-M3",
	"minimax-m2.5": "MiniMax-M2.5",

	// Moonshot / Kimi aliases
	"kimi":      "kimi-k3.1-pro",
	"kimi-k3.1": "kimi-k3.1-pro",
	"kimi-k2.5": "kimi-k2.5",
	"kimi-k2.7": "kimi-k2.7-code",
	"k2.7":      "kimi-k2.7-code",

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
	"gpt":          "gpt-6.1-sol",
	"gpt6":         "gpt-6-sol",
	"gpt-6":        "gpt-6-sol",
	"gpt6.1":       "gpt-6.1-sol",
	"gpt-6.1":      "gpt-6.1-sol",
	"gpt6-luna":    "gpt-6-luna",
	"gpt6-astra":   "gpt-6-astra",
	"gpt5.6":       "gpt-5.6-sol",
	"gpt-5.6":      "gpt-5.6-sol",
	"gpt5.6-sol":   "gpt-5.6-sol",
	"gpt5.6-terra": "gpt-5.6-terra",
	"gpt5.6-luna":  "gpt-5.6-luna",
	"gpt5.5":       "gpt-5.5",
	"gpt5.5-pro":   "gpt-5.5-pro",
	"gpt5.4":       "gpt-5.4",
	"gpt5.3":       "gpt-5.3-chat-latest",
	"gpt5":         "gpt-5.6-sol",
	"gpt5.2":       "gpt-5.2",
	"gpt5.1":       "gpt-5.1",
	"gpt5-mini":    "gpt-5-mini",
	"gpt5-nano":    "gpt-5-nano",
	"gpt4.1":       "gpt-4.1",
	"gpt5.4-mini":  "gpt-5.4-mini",
	"gpt5.4-nano":  "gpt-5.4-nano",
	"gpt-instant":  "gpt-5.3-chat-latest",

	// Meta aliases
	"muse":         "muse-spark-1.3",
	"muse-spark":   "muse-spark-1.3",
	"muse-spark-1": "muse-spark-1.3",

	// Gemini aliases
	"gemini":         "gemini-3.5-flash-lite",
	"gemini3.6":      "gemini-3.6-flash",
	"gemini3.5":      "gemini-3.5-flash",
	"gemini3.5-lite": "gemini-3.5-flash-lite",
	"gemini-pro":     "gemini-3.1-pro-preview",
	"gemini3.1":      "gemini-3.1-pro-preview",
	"gemini3.1-lite": "gemini-3.1-flash-lite",
	"gemini3":        "gemini-3.1-pro-preview",
	"gemini-3":       "gemini-3.1-pro-preview",
	"flash":          "gemini-3.8-flash",
	"flash-lite":     "gemini-3.5-flash-lite",
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
	if info.Capabilities&^allCapabilities != 0 {
		return fmt.Errorf("llm: RegisterModel: capabilities contain unknown bits")
	}
	// Match package init: a zero capability set means "apply conservative chat
	// defaults for this provider family" rather than "reviewed as capability-less"
	// (which would fail closed on every Complete). Explicit modality-only models
	// must set Capabilities (e.g. CapabilityEmbeddings) themselves.
	if info.Capabilities == 0 {
		info.Capabilities = defaultChatCapabilities(info)
	}
	registryMu.Lock()
	defer registryMu.Unlock()
	modelRegistry[info.ID] = info
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
	return "claude-sonnet-5"
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
//
// Cache-token contract (the same for every provider, matching llm.Response):
// InputTokens, CacheReadTokens and CacheCreateTokens are DISJOINT. InputTokens
// is the uncached input only; the full prompt is
// InputTokens + CacheReadTokens + CacheCreateTokens. Each token is therefore
// priced exactly once: InputTokens at InputPricePer1M, CacheReadTokens at
// CacheReadPricePer1M and CacheCreateTokens at CacheWritePricePer1M. Do NOT
// pass a provider's raw total prompt count (Gemini promptTokenCount, OpenAI
// prompt_tokens / input_tokens — all of which include cached tokens) as
// InputTokens alongside CacheReadTokens: that charges the cached tokens twice.
type CostInput struct {
	Model             string
	InputTokens       int // uncached input tokens only (excludes CacheReadTokens and CacheCreateTokens)
	OutputTokens      int
	ThinkingTokens    int // Gemini, openai_responses: separate from output; Anthropic: 0 (bundled in OutputTokens)
	CacheCreateTokens int // tokens written to cache (Anthropic); disjoint from InputTokens
	CacheReadTokens   int // tokens read from cache (Anthropic/Gemini/OpenAI-style); disjoint from InputTokens
	// CacheCreate1hTokens is the share of CacheCreateTokens written with the
	// 1-hour TTL, priced at CacheWrite1hPricePer1M (the rest at
	// CacheWritePricePer1M). Capped at CacheCreateTokens.
	CacheCreate1hTokens int
	Batch               bool // request was processed via the Batch API (billed at 50% of the sync rate)
	// AudioInputTokens is the audio share of InputTokens. It is priced at
	// AudioInputPricePer1M (0 when unknown — never at the text rate) and the
	// rest of InputTokens at InputPricePer1M.
	AudioInputTokens int
	// AudioSeconds is input audio duration for duration-billed speech-to-text,
	// priced at AudioPricePerMinute. Negative or non-finite values count as 0.
	AudioSeconds float64
	// WebSearchRequests is the number of billed web searches
	// (Response.ServerToolUse.WebSearchRequests), priced at
	// WebSearchPricePer1K.
	WebSearchRequests int
	// Iterations, when set, replaces the token fields above: each iteration is
	// priced at its own model (Response.Iterations).
	Iterations []UsageIteration
}

// EstimateCost returns the estimated cost in USD for a request/response.
// Accepts a CostInput with all token types for accurate cache-aware pricing.
// Returns 0 if the model is not in the registry or has no pricing data —
// register unlisted models via RegisterModel to get real estimates.
//
// When Iterations is set (Anthropic usage.iterations), the tokens are priced
// per iteration at that iteration's model — a fallback attempt at the
// fallback model's prices — and the top-level token fields are ignored; an
// iteration whose model has no registry entry adds 0. WebSearchRequests are
// priced at the model's WebSearchPricePer1K, without the batch discount
// (searches cost the same in a batch).
//
// The result is always finite and >= 0, and monotonic in every count (more
// tokens or seconds never cost less; AudioInputTokens is a share of
// InputTokens, so raise both together). Negative counts count as 0, a
// negative/NaN/Inf registered price counts as unknown (0), and a total that
// would overflow saturates at math.MaxFloat64.
func EstimateCost(input CostInput) float64 {
	info, ok := pricingFor(input.Model)
	// Searches are billed per request at the top-level model's rate, or at
	// the first priced iteration model's when the top-level model is
	// unregistered.
	searchInfo, searchPriced := info, ok
	var total float64
	if len(input.Iterations) > 0 {
		for _, it := range input.Iterations {
			model := it.Model
			if model == "" {
				model = input.Model
			}
			itInfo, known := pricingFor(model)
			if !known {
				continue
			}
			if !searchPriced {
				searchInfo, searchPriced = itInfo, true
			}
			total += tokenCost(itInfo, CostInput{
				InputTokens:         it.InputTokens,
				OutputTokens:        it.OutputTokens,
				CacheCreateTokens:   it.CacheCreationTokens,
				CacheReadTokens:     it.CacheReadTokens,
				CacheCreate1hTokens: it.CacheCreation1hTokens,
				Batch:               input.Batch,
			})
		}
	} else if ok {
		total = tokenCost(info, input)
	}
	if searchPriced && input.WebSearchRequests > 0 {
		total += float64(input.WebSearchRequests) * sanitizePrice(searchInfo.WebSearchPricePer1K) / 1000
	}
	return saturateCost(total)
}

// pricingFor looks up a model's registry entry, resolving an alias under the
// same read lock (ResolveModelAlias takes the lock too, and RWMutex is not
// re-entrant — so inline it). A model ID the registry does not know is priced
// as the registered model it is a snapshot of: providers answer with dated IDs
// ("gpt-5.4-2026-03-05", "claude-opus-5-5-20260901"), and an exact-match lookup
// priced those at $0 (B22).
func pricingFor(model string) (ModelInfo, bool) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	if info, ok := lookupLocked(model); ok {
		return info, true
	}
	for i := len(model) - 1; i > 0; i-- {
		if model[i] != '-' || !versionSuffix.MatchString(model[i:]) {
			continue
		}
		if info, ok := lookupLocked(model[:i]); ok {
			return info, true
		}
	}
	return ModelInfo{}, false
}

// lookupLocked resolves an alias and looks up the entry. The caller holds
// registryMu.
func lookupLocked(model string) (ModelInfo, bool) {
	if full, isAlias := modelAliases[model]; isAlias {
		model = full
	}
	info, ok := modelRegistry[model]
	return info, ok
}

// tokenCost prices the token and audio fields of input at info's prices.
func tokenCost(info ModelInfo, input CostInput) float64 {
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

	audioTok := clamp(input.AudioInputTokens)
	if audioTok > inputTok {
		audioTok = inputTok
	}
	audioSeconds := input.AudioSeconds
	if math.IsNaN(audioSeconds) || math.IsInf(audioSeconds, 0) || audioSeconds < 0 {
		audioSeconds = 0
	}

	// Price each dimension in float64: token counts are converted before any
	// arithmetic, so no int sum can wrap (OutputTokens+ThinkingTokens at
	// math.MaxInt used to overflow to a negative cost). Every term is a finite,
	// non-negative count times a sanitized price, and IEEE rounding is
	// monotonic, so the total is monotonic in each input.
	perToken := func(tokens int, pricePer1M float64) float64 {
		return float64(tokens) * sanitizePrice(pricePer1M) / 1_000_000
	}
	inputCost := perToken(inputTok-audioTok, info.InputPricePer1M)
	inputCost += perToken(audioTok, info.AudioInputPricePer1M)
	inputCost += audioSeconds / 60 * sanitizePrice(info.AudioPricePerMinute)
	outputCost := perToken(outputTok, info.OutputPricePer1M) + perToken(thinkingTok, info.OutputPricePer1M)

	// A zero cache-write price means "not applicable" (no write surcharge);
	// sanitizePrice already maps zero, negative and non-finite prices to 0.
	// A zero cache-read price means "unknown": CacheReadTokens are excluded
	// from InputTokens by contract, so pricing them at 0 would make cached
	// input free — bill them at the plain input rate instead (what the provider
	// charges when there is no cache discount, and never more than that).
	cacheReadPrice := sanitizePrice(info.CacheReadPricePer1M)
	if cacheReadPrice == 0 {
		cacheReadPrice = info.InputPricePer1M
	}
	// The 1-hour share of the cache writes is priced at the 1-hour rate. With
	// that rate unregistered it is the documented 2x input price (Anthropic
	// pricing: "1-hour cache write: 2x base input price"), or 0 when the input
	// price is unknown too.
	write1hTok := clamp(input.CacheCreate1hTokens)
	if write1hTok > cacheCreateTok {
		write1hTok = cacheCreateTok
	}
	write1hPrice := sanitizePrice(info.CacheWrite1hPricePer1M)
	if write1hPrice == 0 {
		write1hPrice = 2 * sanitizePrice(info.InputPricePer1M)
	}
	cacheCost := perToken(cacheCreateTok-write1hTok, info.CacheWritePricePer1M) +
		perToken(write1hTok, write1hPrice) +
		perToken(cacheReadTok, cacheReadPrice)

	total := inputCost + outputCost + cacheCost

	// Batch APIs (OpenAI, Anthropic) bill at half the synchronous rate. Apply the
	// discount once to the final total so it composes uniformly with cache pricing.
	if input.Batch {
		total *= 0.5
	}

	return total
}

// sanitizePrice treats a negative, NaN or infinite registered price as unknown
// (0): RegisterModel accepts any float64, and such a price would otherwise
// produce a negative, NaN or Inf cost.
func sanitizePrice(p float64) float64 {
	if p > 0 && !math.IsInf(p, 1) {
		return p
	}
	return 0
}

// saturateCost clamps a cost into [0, math.MaxFloat64]: an overflow to +Inf
// (absurd prices or durations) saturates rather than leaking Inf, and NaN or a
// negative can never escape.
func saturateCost(c float64) float64 {
	switch {
	case math.IsNaN(c) || c <= 0:
		return 0
	case c > math.MaxFloat64:
		return math.MaxFloat64
	}
	return c
}

// addCost adds two costs, saturating at math.MaxFloat64 instead of reaching
// +Inf. Used to accumulate running totals.
func addCost(a, b float64) float64 {
	return saturateCost(a + b)
}
