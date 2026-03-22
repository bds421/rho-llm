package llm

// ModelInfo holds per-model metadata used to adapt API requests and UI.
type ModelInfo struct {
	ID               string  // Full model identifier
	Provider         string  // anthropic, xai, gemini, openai, groq, etc.
	MaxTokens        int     // Model-specific output limit (0 = use config default)
	ContextWindow    int     // Max input tokens (0 = unknown)
	InputPricePer1M  float64 // USD per 1M input tokens (0 = unknown/free)
	OutputPricePer1M float64 // USD per 1M output tokens (0 = unknown/free)
	CacheWritePricePer1M float64 // Anthropic: price per 1M cache creation tokens (0 = not applicable)
	CacheReadPricePer1M  float64 // Anthropic/Gemini: price per 1M cached input tokens (0 = not applicable)
	SupportsThinking bool    // Anthropic extended thinking
	ThoughtSignature bool    // Gemini 3 models require thought_signature in function call responses
	Thinking               bool // Model uses internal chain-of-thought reasoning (e.g. qwen3, deepseek-r1) — consumes output tokens invisibly
	SupportsReasoningEffort bool // OpenAI o-series: supports reasoning parameter with effort control (none/low/medium/high)
	NoToolSupport          bool  // Model does not support tool/function calling (e.g. deepseek-r1, gemma)
	Label            string  // Short display name
}

// modelRegistry maps model ID to its metadata.
// Immutable after init — no mutex needed.
var modelRegistry = map[string]ModelInfo{
	// Anthropic — from platform.claude.com/docs (2026-03-18)
	// Short aliases (claude-opus-4-5) resolve server-side to dated versions (claude-opus-4-5-20251101)
	// Anthropic — from platform.claude.com/docs (2026-03-18)
	// Cache pricing: write = 1.25× input, read = 0.1× input (per Anthropic docs)
	"claude-opus-4-6":            {ID: "claude-opus-4-6", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 1000000, InputPricePer1M: 5.00, OutputPricePer1M: 25.00, CacheWritePricePer1M: 6.25, CacheReadPricePer1M: 0.50, SupportsThinking: true, Label: "Opus 4.6"},
	"claude-opus-4-5":            {ID: "claude-opus-4-5", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 200000, InputPricePer1M: 5.00, OutputPricePer1M: 25.00, CacheWritePricePer1M: 6.25, CacheReadPricePer1M: 0.50, SupportsThinking: true, Label: "Opus 4.5"},
	"claude-opus-4-5-20251101":   {ID: "claude-opus-4-5-20251101", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 200000, InputPricePer1M: 5.00, OutputPricePer1M: 25.00, CacheWritePricePer1M: 6.25, CacheReadPricePer1M: 0.50, SupportsThinking: true, Label: "Opus 4.5 (Nov)"},
	"claude-opus-4-1":            {ID: "claude-opus-4-1", Provider: "anthropic", MaxTokens: 32000, ContextWindow: 200000, InputPricePer1M: 15.00, OutputPricePer1M: 75.00, CacheWritePricePer1M: 18.75, CacheReadPricePer1M: 1.50, SupportsThinking: true, Label: "Opus 4.1"},
	"claude-opus-4-1-20250805":   {ID: "claude-opus-4-1-20250805", Provider: "anthropic", MaxTokens: 32000, ContextWindow: 200000, InputPricePer1M: 15.00, OutputPricePer1M: 75.00, CacheWritePricePer1M: 18.75, CacheReadPricePer1M: 1.50, SupportsThinking: true, Label: "Opus 4.1 (Aug)"},
	"claude-opus-4-0":            {ID: "claude-opus-4-0", Provider: "anthropic", MaxTokens: 32000, ContextWindow: 200000, InputPricePer1M: 15.00, OutputPricePer1M: 75.00, CacheWritePricePer1M: 18.75, CacheReadPricePer1M: 1.50, SupportsThinking: true, Label: "Opus 4.0"},
	"claude-opus-4-20250514":     {ID: "claude-opus-4-20250514", Provider: "anthropic", MaxTokens: 32000, ContextWindow: 200000, InputPricePer1M: 15.00, OutputPricePer1M: 75.00, CacheWritePricePer1M: 18.75, CacheReadPricePer1M: 1.50, SupportsThinking: true, Label: "Opus 4.0 (May)"},
	"claude-sonnet-4-6":          {ID: "claude-sonnet-4-6", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 1000000, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, CacheWritePricePer1M: 3.75, CacheReadPricePer1M: 0.30, SupportsThinking: true, Label: "Sonnet 4.6"},
	"claude-sonnet-4-5":          {ID: "claude-sonnet-4-5", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 200000, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, CacheWritePricePer1M: 3.75, CacheReadPricePer1M: 0.30, SupportsThinking: true, Label: "Sonnet 4.5"},
	"claude-sonnet-4-5-20250929": {ID: "claude-sonnet-4-5-20250929", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 200000, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, CacheWritePricePer1M: 3.75, CacheReadPricePer1M: 0.30, SupportsThinking: true, Label: "Sonnet 4.5 (Sep)"},
	"claude-sonnet-4-0":          {ID: "claude-sonnet-4-0", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 200000, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, CacheWritePricePer1M: 3.75, CacheReadPricePer1M: 0.30, SupportsThinking: true, Label: "Sonnet 4.0"},
	"claude-sonnet-4-20250514":   {ID: "claude-sonnet-4-20250514", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 200000, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, CacheWritePricePer1M: 3.75, CacheReadPricePer1M: 0.30, SupportsThinking: true, Label: "Sonnet 4.0 (May)"},
	"claude-haiku-4-5-20251001":  {ID: "claude-haiku-4-5-20251001", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 200000, InputPricePer1M: 1.00, OutputPricePer1M: 5.00, CacheWritePricePer1M: 1.25, CacheReadPricePer1M: 0.10, SupportsThinking: true, Label: "Haiku 4.5"},
	"claude-3-haiku-20240307":    {ID: "claude-3-haiku-20240307", Provider: "anthropic", MaxTokens: 4096, ContextWindow: 200000, InputPricePer1M: 0.25, OutputPricePer1M: 1.25, CacheWritePricePer1M: 0.3125, CacheReadPricePer1M: 0.025, SupportsThinking: false, Label: "Haiku 3 (legacy)"},

	// xAI / Grok — from docs.x.ai/developers/models (2026-03-18)
	"grok-4.20-beta":                      {ID: "grok-4.20-beta", Provider: "xai", ContextWindow: 2000000, InputPricePer1M: 2.00, OutputPricePer1M: 6.00, Thinking: true, Label: "Grok 4.20 Beta"},
	"grok-4.20-beta-0309-reasoning":       {ID: "grok-4.20-beta-0309-reasoning", Provider: "xai", ContextWindow: 2000000, InputPricePer1M: 2.00, OutputPricePer1M: 6.00, Thinking: true, Label: "Grok 4.20 R"},
	"grok-4.20-beta-0309-non-reasoning":   {ID: "grok-4.20-beta-0309-non-reasoning", Provider: "xai", ContextWindow: 2000000, InputPricePer1M: 2.00, OutputPricePer1M: 6.00, Label: "Grok 4.20"},
	"grok-4.20-multi-agent-beta-0309":     {ID: "grok-4.20-multi-agent-beta-0309", Provider: "xai", ContextWindow: 2000000, InputPricePer1M: 2.00, OutputPricePer1M: 6.00, Thinking: true, Label: "Grok 4.20 Agent"},
	"grok-4-1-fast-reasoning":             {ID: "grok-4-1-fast-reasoning", Provider: "xai", ContextWindow: 2000000, InputPricePer1M: 0.20, OutputPricePer1M: 0.50, Thinking: true, Label: "Grok 4.1 R"},
	"grok-4-1-fast-non-reasoning":         {ID: "grok-4-1-fast-non-reasoning", Provider: "xai", ContextWindow: 2000000, InputPricePer1M: 0.20, OutputPricePer1M: 0.50, Label: "Grok 4.1"},
	"grok-4-fast-reasoning":               {ID: "grok-4-fast-reasoning", Provider: "xai", ContextWindow: 2000000, InputPricePer1M: 0.20, OutputPricePer1M: 0.50, Thinking: true, Label: "Grok 4 R"},
	"grok-4-fast-non-reasoning":           {ID: "grok-4-fast-non-reasoning", Provider: "xai", ContextWindow: 2000000, InputPricePer1M: 0.20, OutputPricePer1M: 0.50, Label: "Grok 4"},
	"grok-4-0709":                         {ID: "grok-4-0709", Provider: "xai", ContextWindow: 256000, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, Thinking: true, Label: "Grok 4 (Jul)"},
	"grok-code-fast-1":                    {ID: "grok-code-fast-1", Provider: "xai", ContextWindow: 256000, InputPricePer1M: 0.20, OutputPricePer1M: 1.50, Thinking: true, Label: "Grok Code"},
	"grok-3":                              {ID: "grok-3", Provider: "xai", ContextWindow: 131072, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, Label: "Grok 3"},
	"grok-3-mini":                         {ID: "grok-3-mini", Provider: "xai", ContextWindow: 131072, InputPricePer1M: 0.30, OutputPricePer1M: 0.50, Thinking: true, Label: "Grok 3 Mini"},

	// Gemini — from GET https://generativelanguage.googleapis.com/v1beta/models (2026-03-05)
	"gemini-3.1-pro-preview":        {ID: "gemini-3.1-pro-preview", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 1.25, OutputPricePer1M: 10.00, ThoughtSignature: true, Label: "Gemini 3.1 Pro"},
	"gemini-3.1-flash-lite-preview": {ID: "gemini-3.1-flash-lite-preview", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 0.25, OutputPricePer1M: 1.50, ThoughtSignature: true, Label: "Gemini 3.1 Flash Lite"},
	"gemini-3-pro-preview":          {ID: "gemini-3-pro-preview", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 1.25, OutputPricePer1M: 10.00, ThoughtSignature: true, Label: "Gemini 3 Pro"},
	"gemini-3-flash-preview":        {ID: "gemini-3-flash-preview", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 0.15, OutputPricePer1M: 0.60, ThoughtSignature: true, Label: "Gemini 3 Flash"},
	"gemini-2.5-pro":                {ID: "gemini-2.5-pro", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 1.25, OutputPricePer1M: 10.00, Thinking: true, Label: "Gemini 2.5 Pro"},
	"gemini-2.5-flash":              {ID: "gemini-2.5-flash", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 0.15, OutputPricePer1M: 0.60, Thinking: true, Label: "Gemini 2.5 Flash"},
	"gemini-2.5-flash-lite":         {ID: "gemini-2.5-flash-lite", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 0.0, OutputPricePer1M: 0.0, Thinking: true, Label: "Flash Lite"},
	"gemini-2.0-flash":              {ID: "gemini-2.0-flash", Provider: "gemini", MaxTokens: 8192, ContextWindow: 1048576, InputPricePer1M: 0.10, OutputPricePer1M: 0.40, Label: "Gemini 2.0 Flash"},

	// OpenAI — GPT-5.x family (2026-03-05)
	"gpt-5.4-pro":         {ID: "gpt-5.4-pro", Provider: "openai", MaxTokens: 128000, ContextWindow: 1048576, InputPricePer1M: 30.00, OutputPricePer1M: 180.00, Thinking: true, Label: "GPT-5.4 Pro"},
	"gpt-5.4":             {ID: "gpt-5.4", Provider: "openai", MaxTokens: 128000, ContextWindow: 1048576, InputPricePer1M: 2.50, OutputPricePer1M: 15.00, Thinking: true, Label: "GPT-5.4"},
	"gpt-5.4-mini":        {ID: "gpt-5.4-mini", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 0.75, OutputPricePer1M: 4.50, Thinking: true, Label: "GPT-5.4 Mini"},
	"gpt-5.4-nano":        {ID: "gpt-5.4-nano", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 0.20, OutputPricePer1M: 1.25, Thinking: true, Label: "GPT-5.4 Nano"},
	"gpt-5.3-chat-latest": {ID: "gpt-5.3-chat-latest", Provider: "openai", MaxTokens: 16384, ContextWindow: 128000, InputPricePer1M: 1.75, OutputPricePer1M: 14.00, Label: "GPT-5.3 Chat"},
	"gpt-5.3-codex":       {ID: "gpt-5.3-codex", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 1.75, OutputPricePer1M: 14.00, Thinking: true, Label: "GPT-5.3 Codex"},
	"gpt-5.2":             {ID: "gpt-5.2", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 1.75, OutputPricePer1M: 14.00, Thinking: true, Label: "GPT-5.2"},
	"gpt-5.2-pro":         {ID: "gpt-5.2-pro", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 21.00, OutputPricePer1M: 168.00, Thinking: true, Label: "GPT-5.2 Pro"},
	"gpt-5.2-chat-latest": {ID: "gpt-5.2-chat-latest", Provider: "openai", MaxTokens: 16384, ContextWindow: 128000, InputPricePer1M: 1.75, OutputPricePer1M: 14.00, Label: "GPT-5.2 Chat"},
	"gpt-5.1":             {ID: "gpt-5.1", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 1.25, OutputPricePer1M: 10.00, Thinking: true, Label: "GPT-5.1"},
	"gpt-5.1-chat-latest": {ID: "gpt-5.1-chat-latest", Provider: "openai", MaxTokens: 16384, ContextWindow: 128000, InputPricePer1M: 1.25, OutputPricePer1M: 10.00, Label: "GPT-5.1 Chat"},
	"gpt-5":               {ID: "gpt-5", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 1.25, OutputPricePer1M: 10.00, Thinking: true, Label: "GPT-5"},
	"gpt-5-pro":           {ID: "gpt-5-pro", Provider: "openai", MaxTokens: 272000, ContextWindow: 400000, InputPricePer1M: 15.00, OutputPricePer1M: 120.00, Thinking: true, Label: "GPT-5 Pro"},
	"gpt-5-chat-latest":   {ID: "gpt-5-chat-latest", Provider: "openai", MaxTokens: 16384, ContextWindow: 128000, InputPricePer1M: 1.25, OutputPricePer1M: 10.00, Label: "GPT-5 Chat"},
	"gpt-5-codex":         {ID: "gpt-5-codex", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 1.25, OutputPricePer1M: 10.00, Thinking: true, Label: "GPT-5 Codex"},
	"gpt-5-mini":          {ID: "gpt-5-mini", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 0.25, OutputPricePer1M: 2.00, Thinking: true, Label: "GPT-5 Mini"},
	"gpt-5-nano":          {ID: "gpt-5-nano", Provider: "openai", MaxTokens: 128000, ContextWindow: 400000, InputPricePer1M: 0.05, OutputPricePer1M: 0.40, Thinking: true, Label: "GPT-5 Nano"},

	// OpenAI — GPT-4.1 family (non-reasoning, 1M context)
	"gpt-4.1":      {ID: "gpt-4.1", Provider: "openai", MaxTokens: 32768, ContextWindow: 1048576, InputPricePer1M: 2.00, OutputPricePer1M: 8.00, Label: "GPT-4.1"},
	"gpt-4.1-mini": {ID: "gpt-4.1-mini", Provider: "openai", MaxTokens: 32768, ContextWindow: 1048576, InputPricePer1M: 0.40, OutputPricePer1M: 1.60, Label: "GPT-4.1 Mini"},
	"gpt-4.1-nano": {ID: "gpt-4.1-nano", Provider: "openai", MaxTokens: 32768, ContextWindow: 1048576, InputPricePer1M: 0.10, OutputPricePer1M: 0.40, Label: "GPT-4.1 Nano"},

	// OpenAI — O-series reasoning models
	"o3":      {ID: "o3", Provider: "openai", MaxTokens: 100000, ContextWindow: 200000, InputPricePer1M: 2.00, OutputPricePer1M: 8.00, Thinking: true, SupportsReasoningEffort: true, Label: "O3"},
	"o3-mini": {ID: "o3-mini", Provider: "openai", MaxTokens: 100000, ContextWindow: 200000, InputPricePer1M: 1.10, OutputPricePer1M: 4.40, Thinking: true, SupportsReasoningEffort: true, Label: "O3 Mini"},
	"o4-mini": {ID: "o4-mini", Provider: "openai", MaxTokens: 100000, ContextWindow: 200000, InputPricePer1M: 1.10, OutputPricePer1M: 4.40, Thinking: true, SupportsReasoningEffort: true, Label: "O4 Mini"},

	// Groq — cloud inference (2026-02-21)
	"llama-3.3-70b-versatile":       {ID: "llama-3.3-70b-versatile", Provider: "groq", MaxTokens: 32768, ContextWindow: 128000, InputPricePer1M: 0.59, OutputPricePer1M: 0.79, Label: "Llama 3.3 70B"},
	"llama-3.1-8b-instant":          {ID: "llama-3.1-8b-instant", Provider: "groq", MaxTokens: 8192, ContextWindow: 128000, InputPricePer1M: 0.05, OutputPricePer1M: 0.08, Label: "Llama 3.1 8B"},
	"openai/gpt-oss-120b":           {ID: "openai/gpt-oss-120b", Provider: "groq", MaxTokens: 16384, ContextWindow: 128000, InputPricePer1M: 3.00, OutputPricePer1M: 8.00, Label: "GPT-OSS 120B"},
	"openai/gpt-oss-20b":            {ID: "openai/gpt-oss-20b", Provider: "groq", MaxTokens: 16384, ContextWindow: 128000, InputPricePer1M: 0.30, OutputPricePer1M: 0.80, Label: "GPT-OSS 20B"},
	"deepseek-r1-distill-llama-70b": {ID: "deepseek-r1-distill-llama-70b", Provider: "groq", MaxTokens: 16384, ContextWindow: 128000, InputPricePer1M: 0.75, OutputPricePer1M: 0.99, Thinking: true, Label: "DeepSeek R1 70B"},
	"deepseek-r1-distill-qwen-32b":  {ID: "deepseek-r1-distill-qwen-32b", Provider: "groq", MaxTokens: 16384, ContextWindow: 128000, InputPricePer1M: 0.69, OutputPricePer1M: 0.69, Thinking: true, Label: "DeepSeek R1 32B"},

	// Mistral — cloud API (2026-03-18)
	"mistral-large-2512":    {ID: "mistral-large-2512", Provider: "mistral", MaxTokens: 262144, ContextWindow: 262144, InputPricePer1M: 0.50, OutputPricePer1M: 1.50, Label: "Mistral Large"},
	"mistral-medium-latest": {ID: "mistral-medium-latest", Provider: "mistral", MaxTokens: 131072, ContextWindow: 131072, InputPricePer1M: 0.40, OutputPricePer1M: 2.00, Label: "Mistral Medium"},
	"mistral-small-2603":    {ID: "mistral-small-2603", Provider: "mistral", MaxTokens: 262144, ContextWindow: 262144, InputPricePer1M: 0.15, OutputPricePer1M: 0.60, Thinking: true, Label: "Mistral Small 4"},
	"mistral-small-2506":    {ID: "mistral-small-2506", Provider: "mistral", MaxTokens: 131072, ContextWindow: 131072, InputPricePer1M: 0.10, OutputPricePer1M: 0.30, Label: "Mistral Small 3"},
	"magistral-medium-2509": {ID: "magistral-medium-2509", Provider: "mistral", MaxTokens: 65536, ContextWindow: 131072, InputPricePer1M: 2.00, OutputPricePer1M: 5.00, Thinking: true, Label: "Magistral Medium"},
	"magistral-small-2509":  {ID: "magistral-small-2509", Provider: "mistral", MaxTokens: 8192, ContextWindow: 131072, InputPricePer1M: 0.50, OutputPricePer1M: 1.50, Thinking: true, Label: "Magistral Small"},
	"codestral-2508":        {ID: "codestral-2508", Provider: "mistral", MaxTokens: 262144, ContextWindow: 262144, InputPricePer1M: 0.30, OutputPricePer1M: 0.90, Label: "Codestral"},
	"devstral-2512":         {ID: "devstral-2512", Provider: "mistral", MaxTokens: 262144, ContextWindow: 262144, InputPricePer1M: 0.40, OutputPricePer1M: 2.00, Label: "Devstral"},
	"ministral-8b-2512":     {ID: "ministral-8b-2512", Provider: "mistral", MaxTokens: 262144, ContextWindow: 262144, InputPricePer1M: 0.15, OutputPricePer1M: 0.15, Label: "Ministral 8B"},
	"ministral-14b-2512":    {ID: "ministral-14b-2512", Provider: "mistral", MaxTokens: 262144, ContextWindow: 262144, InputPricePer1M: 0.20, OutputPricePer1M: 0.20, Label: "Ministral 14B"},

	// Ollama — popular local models (no pricing, context varies by quantization)
	"deepseek-r1:14b":      {ID: "deepseek-r1:14b", Provider: "ollama", Thinking: true, NoToolSupport: true, Label: "DeepSeek R1 14B"},
	"mistral-small3.2:24b": {ID: "mistral-small3.2:24b", Provider: "ollama", Label: "Mistral Small 3.2 24B"},
	"qwen3-coder:30b":      {ID: "qwen3-coder:30b", Provider: "ollama", Label: "Qwen3 Coder 30B"},
	"qwen3:8b":             {ID: "qwen3:8b", Provider: "ollama", Thinking: true, Label: "Qwen3 8B"},
	"qwen3:4b":             {ID: "qwen3:4b", Provider: "ollama", Thinking: true, Label: "Qwen3 4B"},
	"gemma3:12b":           {ID: "gemma3:12b", Provider: "ollama", NoToolSupport: true, Label: "Gemma3 12B"},
	"gemma3:4b":            {ID: "gemma3:4b", Provider: "ollama", NoToolSupport: true, Label: "Gemma3 4B"},
	"gemma2:2b":            {ID: "gemma2:2b", Provider: "ollama", NoToolSupport: true, Label: "Gemma2 2B"},
}

var defaultModels = map[string]string{
	"anthropic": "claude-sonnet-4-6",
	"claude":    "claude-sonnet-4-6",
	"xai":       "grok-4.20-beta",
	"grok":      "grok-4.20-beta",
	"gemini":    "gemini-3.1-flash-lite-preview",
	"google":    "gemini-3.1-flash-lite-preview",
	"openai":    "gpt-5.4",
	"gpt":       "gpt-5.4",
	"groq":      "llama-3.3-70b-versatile",
	"mistral":   "mistral-small-2603",
	"ollama":    "qwen3:8b",
}

var availableModels = map[string][]string{
	"anthropic": {
		"claude-opus-4-6",
		"claude-opus-4-5",
		"claude-opus-4-1",
		"claude-opus-4-0",
		"claude-sonnet-4-6",
		"claude-sonnet-4-5",
		"claude-sonnet-4-0",
		"claude-haiku-4-5-20251001",
		"claude-3-haiku-20240307",
	},
	"xai": {
		"grok-4.20-beta",
		"grok-4.20-beta-0309-reasoning",
		"grok-4.20-beta-0309-non-reasoning",
		"grok-4.20-multi-agent-beta-0309",
		"grok-4-1-fast-reasoning",
		"grok-4-1-fast-non-reasoning",
		"grok-4-fast-reasoning",
		"grok-4-fast-non-reasoning",
		"grok-4-0709",
		"grok-code-fast-1",
		"grok-3",
		"grok-3-mini",
	},
	"gemini": {
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
		"o3",
		"o3-mini",
		"o4-mini",
	},
	"groq": {
		"llama-3.3-70b-versatile",
		"llama-3.1-8b-instant",
		"openai/gpt-oss-120b",
		"openai/gpt-oss-20b",
		"deepseek-r1-distill-llama-70b",
		"deepseek-r1-distill-qwen-32b",
	},
	"mistral": {
		"mistral-large-2512",
		"mistral-medium-latest",
		"mistral-small-2603",
		"mistral-small-2506",
		"magistral-medium-2509",
		"magistral-small-2509",
		"codestral-2508",
		"devstral-2512",
		"ministral-8b-2512",
		"ministral-14b-2512",
	},
	"ollama": {
		"deepseek-r1:14b",
		"mistral-small3.2:24b",
		"qwen3-coder:30b",
		"qwen3:8b",
		"qwen3:4b",
		"gemma3:12b",
		"gemma3:4b",
		"gemma2:2b",
	},
}

var modelAliases = map[string]string{
	// Anthropic aliases
	"opus":   "claude-opus-4-6",
	"sonnet": "claude-sonnet-4-6",
	"haiku":  "claude-haiku-4-5-20251001",
	"claude": "claude-sonnet-4-6",

	// xAI/Grok aliases
	"grok":               "grok-4.20-beta",
	"grok4.2":            "grok-4.20-beta",
	"grok4.20":           "grok-4.20-beta",
	"grok4":              "grok-4.20-beta",
	"grok-4":             "grok-4.20-beta",
	"grok4.1":            "grok-4-1-fast-non-reasoning",
	"grok-4-1":           "grok-4-1-fast-non-reasoning",
	"grok-reasoning":     "grok-4-fast-reasoning",
	"grok-4-reasoning":   "grok-4-fast-reasoning",
	"grok-4-1-reasoning": "grok-4-1-fast-reasoning",
	"grok-code":          "grok-code-fast-1",
	"grok-mini":          "grok-3-mini",

	// Groq aliases
	"groq":        "llama-3.3-70b-versatile",
	"llama":       "llama-3.3-70b-versatile",
	"llama-70b":   "llama-3.3-70b-versatile",
	"llama-8b":    "llama-3.1-8b-instant",
	"gpt-oss":     "openai/gpt-oss-120b",
	"gpt-oss-20b": "openai/gpt-oss-20b",

	// Mistral aliases
	"mistral-large":  "mistral-large-2512",
	"mistral-medium": "mistral-medium-latest",
	"mistral-small":  "mistral-small-2603",
	"magistral":      "magistral-medium-2509",
	"codestral":      "codestral-2508",
	"devstral":       "devstral-2512",
	"ministral":      "ministral-8b-2512",

	// Ollama aliases
	"deepseek":      "deepseek-r1:14b",
	"mistral-local": "mistral-small3.2:24b",
	"qwen":          "qwen3:8b",
	"qwen-code":     "qwen3-coder:30b",
	"gemma":         "gemma3:12b",

	// OpenAI aliases
	"gpt":         "gpt-5.4",
	"gpt5.4":      "gpt-5.4",
	"gpt5.3":      "gpt-5.3-chat-latest",
	"gpt5":        "gpt-5.4",
	"gpt5.2":      "gpt-5.2",
	"gpt5.1":      "gpt-5.1",
	"gpt5-mini":   "gpt-5-mini",
	"gpt5-nano":   "gpt-5-nano",
	"gpt4.1":      "gpt-4.1",
	"gpt5.4-mini":  "gpt-5.4-mini",
	"gpt5.4-nano":  "gpt-5.4-nano",
	"gpt-instant":  "gpt-5.3-chat-latest",

	// Gemini aliases
	"gemini":     "gemini-3.1-flash-lite-preview",
	"gemini-pro": "gemini-3.1-pro-preview",
	"gemini3.1":  "gemini-3.1-pro-preview",
	"gemini3":    "gemini-3.1-pro-preview",
	"gemini-3":   "gemini-3-pro-preview",
	"flash":      "gemini-2.5-flash",
	"flash-lite": "gemini-3.1-flash-lite-preview",
}

// GetModelInfo returns the ModelInfo for a model ID, or false if not found.
func GetModelInfo(model string) (ModelInfo, bool) {
	info, ok := modelRegistry[model]
	return info, ok
}

// GetDefaultModel returns the default model for a provider.
func GetDefaultModel(provider string) string {
	if model, ok := defaultModels[provider]; ok {
		return model
	}
	return "claude-sonnet-4-6"
}

// GetAvailableModels returns the ordered model list for a provider.
// Returns nil if the provider has no models registered.
// Returns a copy to prevent callers from mutating global registry state.
func GetAvailableModels(provider string) []string {
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
	if full, ok := modelAliases[model]; ok {
		return full
	}
	return model
}

// ProviderForModel detects the provider from a model name.
// Returns empty string if the model is not recognized.
func ProviderForModel(model string) string {
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
	ThinkingTokens    int // Gemini: separate from output; Anthropic: 0 (bundled in OutputTokens)
	CacheCreateTokens int // Anthropic: tokens written to cache
	CacheReadTokens   int // Anthropic/Gemini: tokens read from cache
}

// EstimateCost returns the estimated cost in USD for a request/response.
// Accepts a CostInput with all token types for accurate cache-aware pricing.
// Returns 0 if the model is not in the registry or has no pricing data.
func EstimateCost(input CostInput) float64 {
	info, ok := modelRegistry[input.Model]
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

	return inputCost + outputCost + cacheCost
}
