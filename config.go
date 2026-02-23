package llm

import (
	"crypto/tls"
	"encoding/json"
	"net/http"
	"net/url"
	"time"
)

const (
	// MaxErrorBodyBytes caps the number of bytes read from an error response body.
	// Prevents OOM from malicious or broken endpoints returning enormous error messages.
	MaxErrorBodyBytes = 1 << 20 // 1 MB

	// MaxSSELineBytes caps the per-line buffer for SSE stream parsing.
	// Limits per-stream memory usage from malicious endpoints.
	MaxSSELineBytes = 256 * 1024 // 256 KB

	// MaxResponseBodyBytes caps the number of bytes read from a success response
	// body before JSON decoding. Prevents OOM from malicious endpoints returning
	// enormous JSON payloads. 32 MB is generous for any real LLM response.
	MaxResponseBodyBytes = 32 << 20 // 32 MB

	// MaxToolInputBytes caps accumulated tool input JSON during streaming.
	// Prevents OOM from malicious endpoints sending thousands of input_json_delta
	// or argument chunks that accumulate without bound.
	MaxToolInputBytes = 1 << 20 // 1 MB
)

// sensitiveHeaders are stripped on cross-domain redirects to prevent key leakage.
var sensitiveHeaders = []string{
	"Authorization",
	"x-api-key",
	"x-goog-api-key",
}

// SafeHTTPClient returns an http.Client with the given timeout that strips
// sensitive authentication headers on cross-domain redirects. Without this,
// a redirect to a different host leaks API keys (especially custom headers
// like x-api-key that Go's stdlib doesn't recognize as auth headers).
func SafeHTTPClient(timeout time.Duration) *http.Client {
	return &http.Client{
		Timeout: timeout,
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{MinVersion: tls.VersionTLS12},
		},
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 10 {
				return http.ErrUseLastResponse
			}
			if len(via) > 0 {
				prev := via[len(via)-1]
				if !sameHost(prev.URL, req.URL) {
					for _, h := range sensitiveHeaders {
						req.Header.Del(h)
					}
				}
			}
			return nil
		},
	}
}

// sameHost returns true if two URLs have the same host (including port).
func sameHost(a, b *url.URL) bool {
	return a.Host == b.Host
}

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

// MarshalJSON implements json.Marshaler. Redacts APIKey to prevent accidental
// secret leakage when Config is serialized for logging or debugging.
// Use the APIKey field directly when you need the actual value.
func (c Config) MarshalJSON() ([]byte, error) {
	type configAlias Config // break recursion
	tmp := configAlias(c)
	if tmp.APIKey != "" {
		tmp.APIKey = "REDACTED"
	}
	return json.Marshal(tmp)
}
