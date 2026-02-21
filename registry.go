package llm

// ModelInfo holds per-model metadata used to adapt API requests and UI.
type ModelInfo struct {
	ID               string  // Full model identifier
	Provider         string  // anthropic, xai, gemini, openai, groq, etc.
	MaxTokens        int     // Model-specific output limit (0 = use config default)
	ContextWindow    int     // Max input tokens (0 = unknown)
	InputPricePer1M  float64 // USD per 1M input tokens (0 = unknown/free)
	OutputPricePer1M float64 // USD per 1M output tokens (0 = unknown/free)
	SupportsThinking bool    // Anthropic extended thinking
	ThoughtSignature bool    // Gemini 3 models require thought_signature in function call responses
	Thinking         bool    // Model uses internal chain-of-thought reasoning (e.g. qwen3, deepseek-r1) — consumes output tokens invisibly
	NoToolSupport    bool    // Model does not support tool/function calling (e.g. deepseek-r1, gemma)
	Label            string  // Short display name
}

// modelRegistry maps model ID to its metadata.
// Immutable after init — no mutex needed.
var modelRegistry = map[string]ModelInfo{
	// Anthropic — from GET /v1/models (9 models, 2026-02-19)
	// Short aliases (claude-opus-4-5) resolve server-side to dated versions (claude-opus-4-5-20251101)
	"claude-opus-4-6":            {ID: "claude-opus-4-6", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 200000, InputPricePer1M: 15.00, OutputPricePer1M: 75.00, SupportsThinking: true, Label: "Opus 4.6"},
	"claude-opus-4-5":            {ID: "claude-opus-4-5", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 200000, InputPricePer1M: 15.00, OutputPricePer1M: 75.00, SupportsThinking: true, Label: "Opus 4.5"},
	"claude-opus-4-5-20251101":   {ID: "claude-opus-4-5-20251101", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 200000, InputPricePer1M: 15.00, OutputPricePer1M: 75.00, SupportsThinking: true, Label: "Opus 4.5 (Nov)"},
	"claude-opus-4-1":            {ID: "claude-opus-4-1", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 200000, InputPricePer1M: 15.00, OutputPricePer1M: 75.00, SupportsThinking: true, Label: "Opus 4.1"},
	"claude-opus-4-1-20250805":   {ID: "claude-opus-4-1-20250805", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 200000, InputPricePer1M: 15.00, OutputPricePer1M: 75.00, SupportsThinking: true, Label: "Opus 4.1 (Aug)"},
	"claude-opus-4-0":            {ID: "claude-opus-4-0", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 200000, InputPricePer1M: 15.00, OutputPricePer1M: 75.00, SupportsThinking: true, Label: "Opus 4.0"},
	"claude-opus-4-20250514":     {ID: "claude-opus-4-20250514", Provider: "anthropic", MaxTokens: 128000, ContextWindow: 200000, InputPricePer1M: 15.00, OutputPricePer1M: 75.00, SupportsThinking: true, Label: "Opus 4.0 (May)"},
	"claude-sonnet-4-6":          {ID: "claude-sonnet-4-6", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 200000, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, SupportsThinking: true, Label: "Sonnet 4.6"},
	"claude-sonnet-4-5":          {ID: "claude-sonnet-4-5", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 200000, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, SupportsThinking: true, Label: "Sonnet 4.5"},
	"claude-sonnet-4-5-20250929": {ID: "claude-sonnet-4-5-20250929", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 200000, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, SupportsThinking: true, Label: "Sonnet 4.5 (Sep)"},
	"claude-sonnet-4-0":          {ID: "claude-sonnet-4-0", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 200000, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, SupportsThinking: true, Label: "Sonnet 4.0"},
	"claude-sonnet-4-20250514":   {ID: "claude-sonnet-4-20250514", Provider: "anthropic", MaxTokens: 64000, ContextWindow: 200000, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, SupportsThinking: true, Label: "Sonnet 4.0 (May)"},
	"claude-haiku-4-5-20251001":  {ID: "claude-haiku-4-5-20251001", Provider: "anthropic", MaxTokens: 8192, ContextWindow: 200000, InputPricePer1M: 0.80, OutputPricePer1M: 4.00, SupportsThinking: false, Label: "Haiku 4.5"},
	"claude-3-haiku-20240307":    {ID: "claude-3-haiku-20240307", Provider: "anthropic", MaxTokens: 4096, ContextWindow: 200000, InputPricePer1M: 0.25, OutputPricePer1M: 1.25, SupportsThinking: false, Label: "Haiku 3 (legacy)"},

	// xAI / Grok — from GET https://api.x.ai/v1/models (2026-02-19)
	"grok-4-1-fast-reasoning":     {ID: "grok-4-1-fast-reasoning", Provider: "xai", ContextWindow: 131072, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, Label: "Grok 4.1 R"},
	"grok-4-1-fast-non-reasoning": {ID: "grok-4-1-fast-non-reasoning", Provider: "xai", ContextWindow: 131072, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, Label: "Grok 4.1"},
	"grok-4-fast-reasoning":       {ID: "grok-4-fast-reasoning", Provider: "xai", ContextWindow: 131072, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, Label: "Grok 4 R"},
	"grok-4-fast-non-reasoning":   {ID: "grok-4-fast-non-reasoning", Provider: "xai", ContextWindow: 131072, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, Label: "Grok 4"},
	"grok-4-0709":                 {ID: "grok-4-0709", Provider: "xai", ContextWindow: 131072, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, Label: "Grok 4 (Jul)"},
	"grok-code-fast-1":            {ID: "grok-code-fast-1", Provider: "xai", ContextWindow: 131072, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, Label: "Grok Code"},
	"grok-3":                      {ID: "grok-3", Provider: "xai", ContextWindow: 131072, InputPricePer1M: 3.00, OutputPricePer1M: 15.00, Label: "Grok 3"},
	"grok-3-mini":                 {ID: "grok-3-mini", Provider: "xai", ContextWindow: 131072, InputPricePer1M: 0.30, OutputPricePer1M: 0.50, Label: "Grok 3 Mini"},

	// Gemini — from GET https://generativelanguage.googleapis.com/v1beta/models (2026-02-19)
	"gemini-3.1-pro-preview": {ID: "gemini-3.1-pro-preview", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 1.25, OutputPricePer1M: 10.00, ThoughtSignature: true, Label: "Gemini 3.1 Pro"},
	"gemini-3-pro-preview":   {ID: "gemini-3-pro-preview", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 1.25, OutputPricePer1M: 10.00, ThoughtSignature: true, Label: "Gemini 3 Pro"},
	"gemini-3-flash-preview": {ID: "gemini-3-flash-preview", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 0.15, OutputPricePer1M: 0.60, ThoughtSignature: true, Label: "Gemini 3 Flash"},
	"gemini-2.5-pro":         {ID: "gemini-2.5-pro", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 1.25, OutputPricePer1M: 10.00, Label: "Gemini 2.5 Pro"},
	"gemini-2.5-flash":       {ID: "gemini-2.5-flash", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 0.15, OutputPricePer1M: 0.60, Label: "Gemini 2.5 Flash"},
	"gemini-2.5-flash-lite":  {ID: "gemini-2.5-flash-lite", Provider: "gemini", MaxTokens: 65536, ContextWindow: 1048576, InputPricePer1M: 0.0, OutputPricePer1M: 0.0, Label: "Flash Lite"},
	"gemini-2.0-flash":       {ID: "gemini-2.0-flash", Provider: "gemini", MaxTokens: 8192, ContextWindow: 1048576, InputPricePer1M: 0.10, OutputPricePer1M: 0.40, Label: "Gemini 2.0 Flash"},

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
	"xai":       "grok-4-fast-non-reasoning",
	"grok":      "grok-4-fast-non-reasoning",
	"gemini":    "gemini-2.5-flash-lite",
	"google":    "gemini-2.5-flash-lite",
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
		"grok-4-1-fast-reasoning",
		"grok-4-1-fast-non-reasoning",
		"grok-4-fast-reasoning",
		"grok-4-fast-non-reasoning",
		"grok-code-fast-1",
		"grok-3",
		"grok-3-mini",
	},
	"gemini": {
		"gemini-3.1-pro-preview",
		"gemini-3-pro-preview",
		"gemini-3-flash-preview",
		"gemini-2.5-pro",
		"gemini-2.5-flash",
		"gemini-2.5-flash-lite",
		"gemini-2.0-flash",
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
	"grok":               "grok-4-fast-non-reasoning",
	"grok4":              "grok-4-fast-non-reasoning",
	"grok-4":             "grok-4-fast-non-reasoning",
	"grok4.1":            "grok-4-1-fast-non-reasoning",
	"grok-4.1":           "grok-4-1-fast-non-reasoning",
	"grok-reasoning":     "grok-4-fast-reasoning",
	"grok-4-reasoning":   "grok-4-fast-reasoning",
	"grok-4.1-reasoning": "grok-4-1-fast-reasoning",
	"grok-code":          "grok-code-fast-1",
	"grok-mini":          "grok-3-mini",

	// Ollama aliases
	"deepseek":  "deepseek-r1:14b",
	"mistral":   "mistral-small3.2:24b",
	"qwen":      "qwen3:8b",
	"qwen-code": "qwen3-coder:30b",
	"gemma":     "gemma3:12b",

	// Gemini aliases
	"gemini":     "gemini-2.5-flash-lite",
	"gemini-pro": "gemini-3-pro-preview",
	"gemini3":    "gemini-3-pro-preview",
	"gemini-3":   "gemini-3-pro-preview",
	"flash":      "gemini-2.5-flash",
	"flash-lite": "gemini-2.5-flash-lite",
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

// EstimateCost returns the estimated cost in USD for a request/response.
// Returns 0 if the model is not in the registry or has no pricing data.
func EstimateCost(model string, inputTokens, outputTokens int) float64 {
	info, ok := modelRegistry[model]
	if !ok {
		return 0
	}
	if inputTokens < 0 {
		inputTokens = 0
	}
	if outputTokens < 0 {
		outputTokens = 0
	}
	inputCost := float64(inputTokens) * info.InputPricePer1M / 1_000_000
	outputCost := float64(outputTokens) * info.OutputPricePer1M / 1_000_000
	return inputCost + outputCost
}
