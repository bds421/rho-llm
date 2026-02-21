package llm

// ProviderPreset holds the default endpoint and auth configuration for a provider.
type ProviderPreset struct {
	BaseURL    string // Default API base URL
	AuthHeader string // Auth header prefix ("Bearer", "", etc.)
	Protocol   string // Wire protocol: "anthropic", "gemini", "openai_compat"
}

// presets maps provider names to their default configuration.
// Immutable after init — no mutex needed.
var presets = map[string]ProviderPreset{
	// Native protocols
	"anthropic": {BaseURL: "https://api.anthropic.com/v1", Protocol: "anthropic"},
	"claude":    {BaseURL: "https://api.anthropic.com/v1", Protocol: "anthropic"},
	"gemini":    {BaseURL: "https://generativelanguage.googleapis.com/v1beta/models", Protocol: "gemini"},
	"google":    {BaseURL: "https://generativelanguage.googleapis.com/v1beta/models", Protocol: "gemini"},

	// OpenAI-compatible: cloud providers
	"openai":     {BaseURL: "https://api.openai.com/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"xai":        {BaseURL: "https://api.x.ai/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"grok":       {BaseURL: "https://api.x.ai/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"groq":       {BaseURL: "https://api.groq.com/openai/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"cerebras":   {BaseURL: "https://api.cerebras.ai/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"mistral":    {BaseURL: "https://api.mistral.ai/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"openrouter": {BaseURL: "https://openrouter.ai/api/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},

	// OpenAI-compatible: local providers (no auth)
	"ollama":   {BaseURL: "http://localhost:11434/v1", AuthHeader: "", Protocol: "openai_compat"},
	"vllm":     {BaseURL: "http://localhost:8000/v1", AuthHeader: "", Protocol: "openai_compat"},
	"lmstudio": {BaseURL: "http://localhost:1234/v1", AuthHeader: "", Protocol: "openai_compat"},
}

// noAuthProviders lists providers that do not require API keys.
var noAuthProviders = map[string]bool{
	"ollama":   true,
	"vllm":     true,
	"lmstudio": true,
}

// IsNoAuthProvider returns true if the provider does not require an API key.
func IsNoAuthProvider(provider string) bool {
	return noAuthProviders[provider]
}

// PresetFor returns the preset for a provider, or a zero value if not found.
func PresetFor(provider string) (ProviderPreset, bool) {
	p, ok := presets[provider]
	return p, ok
}

// ResolveProtocol determines the wire protocol for a Config.
// Priority: explicit BaseURL with no matching preset -> openai_compat,
// then preset lookup, then fallback to openai_compat.
func ResolveProtocol(cfg Config) string {
	if preset, ok := presets[cfg.Provider]; ok {
		return preset.Protocol
	}

	// Unknown provider with a custom BaseURL -> assume OpenAI-compatible
	if cfg.BaseURL != "" {
		return "openai_compat"
	}

	// Fallback
	return "openai_compat"
}

// ResolveBaseURL returns the effective base URL for a Config.
// Config.BaseURL takes precedence over the provider preset.
func ResolveBaseURL(cfg Config) string {
	if cfg.BaseURL != "" {
		return cfg.BaseURL
	}
	if preset, ok := presets[cfg.Provider]; ok {
		return preset.BaseURL
	}
	return ""
}

// ResolveAuthHeader returns the effective auth header for a Config.
// Config.AuthHeader takes precedence over the provider preset.
func ResolveAuthHeader(cfg Config) string {
	if cfg.AuthHeader != "" {
		return cfg.AuthHeader
	}
	if preset, ok := presets[cfg.Provider]; ok {
		return preset.AuthHeader
	}
	return "Bearer"
}
