package llm

import "time"

// Config holds LLM client configuration. Self-contained replacement for
// temporal-agent's config.LLMConfig, designed for direct use by all services.
type Config struct {
	// Provider name: "anthropic", "openai", "xai", "gemini", "groq",
	// "cerebras", "mistral", "openrouter", "ollama", "vllm", "lmstudio", etc.
	Provider string `json:"provider"`

	// Model identifier (e.g., "claude-sonnet-4-6", "grok-4-fast-non-reasoning").
	Model string `json:"model"`

	// API key for authentication. Empty is valid for local providers (Ollama, vLLM, LM Studio).
	APIKey string `json:"api_key"`

	// Maximum output tokens.
	MaxTokens int `json:"max_tokens"`

	// Sampling temperature.
	Temperature float64 `json:"temperature"`

	// Extended thinking level: ThinkingLow, ThinkingMedium, ThinkingHigh (zero value = none).
	ThinkingLevel ThinkingLevel `json:"thinking_level"`

	// HTTP request timeout. For streaming, use context cancellation instead.
	Timeout time.Duration `json:"timeout"`

	// BaseURL overrides the provider's default endpoint.
	// Example: "http://my-proxy:8080/v1" for a custom OpenAI-compatible server.
	BaseURL string `json:"base_url,omitempty"`

	// AuthHeader overrides the authorization header format.
	// Only applies to OpenAI-compatible providers (openai, xai, groq, etc.).
	// Native adapters (anthropic, gemini) have fixed auth schemes and ignore this.
	// Default: "Bearer" (sends "Authorization: Bearer <key>").
	// Set to "" to skip auth entirely (e.g., local Ollama).
	AuthHeader string `json:"auth_header,omitempty"`

	// ProviderName overrides Client.Provider() return value.
	// Useful when routing through a proxy but wanting to identify the upstream.
	ProviderName string `json:"provider_name,omitempty"`

	// LogRequests enables metadata logging of API requests/responses.
	// Logs provider, model, message count, token usage, and errors.
	// Does NOT log message content (privacy safe).
	LogRequests bool `json:"log_requests,omitempty"`
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() Config {
	return Config{
		Provider:      "anthropic",
		Model:         "claude-sonnet-4-6",
		MaxTokens:     8192,
		Temperature:   1.0,
		ThinkingLevel: ThinkingNone,
		Timeout:       120 * time.Second,
		AuthHeader:    "Bearer",
	}
}
