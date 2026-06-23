package llm

import (
	"crypto/tls"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// Default safety limits. Used when the corresponding Config field is zero.
const (
	DefaultMaxErrorBodyBytes    = 1 << 20    // 1 MB — caps error response body reads
	DefaultMaxSSELineBytes      = 256 * 1024 // 256 KB — caps per-line SSE buffer
	DefaultMaxResponseBodyBytes = 32 << 20   // 32 MB — caps success response body reads
	// DefaultMaxBatchDownloadBytes caps batch result-file downloads. Far larger than
	// the sync-response cap: a completed batch's output JSONL can be hundreds of MB,
	// and reusing the 32 MB cap would silently truncate real batches.
	DefaultMaxBatchDownloadBytes = 256 << 20 // 256 MB — caps batch result-file downloads
	DefaultMaxToolInputBytes     = 1 << 20   // 1 MB — caps accumulated tool input JSON
	DefaultMaxErrorMessageLen    = 4096      // bytes — caps stored error message length
	DefaultAnthropicVersion      = "2023-06-01"
)

// Backward-compatible package-level vars — used as the process-wide fallback when
// the corresponding Config field is zero (see the Effective* accessors). Values
// come from the Default* constants. Set these for global control, or set the
// matching Config fields for per-client control (which takes precedence).
//
// These vars are read without synchronization: set them once at program start,
// BEFORE creating clients or issuing requests. Mutating them while requests
// are in flight is a data race.
var (
	MaxErrorBodyBytes     int64 = DefaultMaxErrorBodyBytes
	MaxSSELineBytes       int   = DefaultMaxSSELineBytes
	MaxResponseBodyBytes  int64 = DefaultMaxResponseBodyBytes
	MaxBatchDownloadBytes int64 = DefaultMaxBatchDownloadBytes
	MaxToolInputBytes     int   = DefaultMaxToolInputBytes
)

// sensitiveHeaders are stripped on cross-origin redirects (scheme or host change)
// to prevent key leakage — including an https→http same-host downgrade, which
// would otherwise send the key in plaintext.
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
				if !sameOrigin(prev.URL, req.URL) {
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
// sameOrigin reports whether two URLs share scheme AND host (an https→http
// downgrade on the same host is NOT same-origin, so auth headers are stripped).
func sameOrigin(a, b *url.URL) bool {
	return a.Scheme == b.Scheme && a.Host == b.Host
}

// CheckBaseURL enforces Config.BlockPrivateBaseURL: when set, it rejects a BaseURL
// whose host is a loopback, private, link-local, or unspecified IP (e.g. the cloud
// metadata endpoint 169.254.169.254, or 127.0.0.1), or the literal "localhost".
// This is opt-in SSRF hardening for deployments that only talk to public provider
// endpoints; it is incompatible with local providers. Best-effort: it checks IP
// literals and "localhost" only — hostname DNS resolution and DNS-rebinding are
// out of scope. It is a no-op when BlockPrivateBaseURL is false or BaseURL is empty.
func CheckBaseURL(cfg Config) error {
	if !cfg.BlockPrivateBaseURL || cfg.BaseURL == "" {
		return nil
	}
	u, err := url.Parse(cfg.BaseURL)
	if err != nil {
		return fmt.Errorf("invalid base URL %q: %w", cfg.BaseURL, err)
	}
	// Normalize the host: a trailing dot ("localhost.", "127.0.0.1.") makes an
	// absolute FQDN that resolvers still map to loopback, yet it evades both the
	// literal "localhost" comparison and net.ParseIP — a guard bypass.
	host := strings.TrimSuffix(u.Hostname(), ".")
	if strings.EqualFold(host, "localhost") {
		return fmt.Errorf("base URL host %q is not allowed (BlockPrivateBaseURL)", host)
	}
	if ip := net.ParseIP(host); ip != nil {
		if ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() || ip.IsUnspecified() {
			return fmt.Errorf("base URL host %q is a private/loopback/link-local address (BlockPrivateBaseURL)", host)
		}
	}
	return nil
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

	// Sampling temperature. nil = omit from wire (provider default).
	Temperature *float64 `json:"temperature,omitempty"`

	// Extended thinking level: ThinkingLow, ThinkingMedium, ThinkingHigh (zero value = none).
	ThinkingLevel ThinkingLevel `json:"thinking_level"`

	// HTTP request timeout. For streaming, use context cancellation instead.
	Timeout time.Duration `json:"timeout"`

	// BaseURL overrides the provider's default endpoint.
	// Example: "http://my-proxy:8080/v1" for a custom OpenAI-compatible server.
	//
	// TRUST BOUNDARY: BaseURL (and the per-key "apikey|baseurl" override) is a
	// developer-supplied, trusted value — the client sends the API key to it. Do
	// NOT populate it from untrusted input without validation: any reachable
	// http(s) URL (including internal hosts like the cloud metadata endpoint
	// 169.254.169.254) would receive the key. Non-http(s) schemes are rejected by
	// net/http at request time, not by this library. For hardened deployments
	// that only talk to public endpoints, set BlockPrivateBaseURL.
	BaseURL string `json:"base_url,omitempty"`

	// BlockPrivateBaseURL is opt-in SSRF hardening: when true, client construction
	// rejects a BaseURL whose host is a loopback, private, link-local, or
	// unspecified IP, or "localhost". Incompatible with local providers
	// (ollama/vllm/lmstudio, which use localhost). Best-effort — it checks IP
	// literals and "localhost"; hostname DNS resolution and DNS-rebinding are out
	// of scope. See CheckBaseURL.
	BlockPrivateBaseURL bool `json:"block_private_base_url,omitempty"`

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

	// RetryPolicy configures backoff behavior. Nil uses DefaultRetryPolicy.
	RetryPolicy *RetryPolicy `json:"retry_policy,omitempty"`

	// CircuitThreshold is the number of consecutive failures before the circuit opens.
	// Zero (default) disables the circuit breaker.
	CircuitThreshold int `json:"circuit_threshold,omitempty"`

	// CircuitCooldown is how long the circuit stays open before allowing a probe.
	// Defaults to 30s when CircuitThreshold > 0.
	CircuitCooldown time.Duration `json:"circuit_cooldown,omitempty"`

	// CooldownRateLimit overrides the cooldown for 429 errors. Zero uses DefaultCooldownRateLimit.
	CooldownRateLimit time.Duration `json:"cooldown_rate_limit,omitempty"`

	// CooldownOverload overrides the cooldown for 503 errors. Zero uses DefaultCooldownOverload.
	CooldownOverload time.Duration `json:"cooldown_overload,omitempty"`

	// CooldownDefault overrides the cooldown for other transient errors. Zero uses DefaultCooldownDefault.
	CooldownDefault time.Duration `json:"cooldown_default,omitempty"`

	// RetryHook receives retry lifecycle events for observability. Not serialized.
	RetryHook RetryHook `json:"-"`

	// MaxRetries caps the number of retry/rotation iterations. Zero uses the
	// default (DefaultMaxRetries). Minimum effective value is 3 (for single-key
	// resilience against transient errors).
	MaxRetries int `json:"max_retries,omitempty"`

	// BetaFeatures lists provider-specific beta feature flags.
	// For Anthropic, these are joined with "," and sent as the "anthropic-beta" header.
	// Other providers ignore this field.
	// Default: []string{"interleaved-thinking-2025-05-14"} (set by DefaultConfig).
	BetaFeatures []string `json:"beta_features,omitempty"`

	// AnthropicVersion overrides the Anthropic API version header.
	// Default: DefaultAnthropicVersion ("2023-06-01").
	AnthropicVersion string `json:"anthropic_version,omitempty"`

	// Safety limits — zero values use the corresponding Default* constants.
	MaxErrorBodyBytes     int `json:"max_error_body_bytes,omitempty"`
	MaxSSELineBytes       int `json:"max_sse_line_bytes,omitempty"`
	MaxResponseBodyBytes  int `json:"max_response_body_bytes,omitempty"`
	MaxBatchDownloadBytes int `json:"max_batch_download_bytes,omitempty"`
	MaxToolInputBytes     int `json:"max_tool_input_bytes,omitempty"`
	MaxErrorMessageLen    int `json:"max_error_message_len,omitempty"`
}

// DefaultTimeout is applied when Config.Timeout is zero (the time.Duration zero value).
// Prevents unbounded HTTP clients when callers construct Config manually without
// calling DefaultConfig().
const DefaultTimeout = 120 * time.Second

// DefaultMaxRetries is the default cap on retry/rotation iterations in PooledClient.
// Prevents pathological retry storms with large key pools.
const DefaultMaxRetries = 10

const (
	// DefaultCooldownRateLimit is the cooldown applied to profiles after a 429 rate-limit error.
	DefaultCooldownRateLimit = 60 * time.Second

	// DefaultCooldownOverload is the cooldown applied to profiles after a 503 overloaded error.
	DefaultCooldownOverload = 30 * time.Second

	// DefaultCooldownDefault is the cooldown applied to profiles after other transient errors.
	DefaultCooldownDefault = 10 * time.Second

	// DefaultCircuitThreshold is the number of consecutive failures before opening the circuit.
	DefaultCircuitThreshold = 5

	// DefaultCircuitCooldown is how long the circuit stays open before allowing a probe request.
	DefaultCircuitCooldown = 30 * time.Second
)

// DefaultConfig returns a Config with sensible defaults.
//
// BetaFeatures defaults to the Anthropic interleaved-thinking flag even though
// it is provider-specific: every other provider ignores the field, and losing
// the flag would silently disable interleaved thinking for Anthropic users.
// Set BetaFeatures explicitly (or nil) to opt out.
func DefaultConfig() Config {
	return Config{
		Provider:         "anthropic",
		Model:            "claude-sonnet-4-6",
		MaxTokens:        8192,
		ThinkingLevel:    ThinkingNone,
		Timeout:          DefaultTimeout,
		AuthHeader:       "Bearer",
		CircuitThreshold: DefaultCircuitThreshold,
		CircuitCooldown:  DefaultCircuitCooldown,
		MaxRetries:       DefaultMaxRetries,
		BetaFeatures:     []string{"interleaved-thinking-2025-05-14"},
	}
}

// EffectiveMaxErrorBodyBytes returns the configured or default limit.
func (c Config) EffectiveMaxErrorBodyBytes() int64 {
	if c.MaxErrorBodyBytes > 0 {
		return int64(c.MaxErrorBodyBytes)
	}
	return MaxErrorBodyBytes
}

// EffectiveMaxSSELineBytes returns the configured or default limit.
func (c Config) EffectiveMaxSSELineBytes() int {
	if c.MaxSSELineBytes > 0 {
		return c.MaxSSELineBytes
	}
	return MaxSSELineBytes
}

// EffectiveMaxResponseBodyBytes returns the configured or default limit.
func (c Config) EffectiveMaxResponseBodyBytes() int64 {
	if c.MaxResponseBodyBytes > 0 {
		return int64(c.MaxResponseBodyBytes)
	}
	return MaxResponseBodyBytes
}

// EffectiveMaxBatchDownloadBytes returns the configured or default cap for batch
// result-file downloads. Defaults far above the 32 MB sync-response cap because a
// completed batch's output file can legitimately be hundreds of MB. The result is
// always positive: a non-positive override (or a negative process-wide var) would make
// the bounded download read zero bytes and silently truncate results, so it falls back
// to the default rather than disabling the cap.
func (c Config) EffectiveMaxBatchDownloadBytes() int64 {
	if c.MaxBatchDownloadBytes > 0 {
		return int64(c.MaxBatchDownloadBytes)
	}
	if MaxBatchDownloadBytes > 0 {
		return MaxBatchDownloadBytes
	}
	return DefaultMaxBatchDownloadBytes
}

// EffectiveMaxToolInputBytes returns the configured or default limit.
func (c Config) EffectiveMaxToolInputBytes() int {
	if c.MaxToolInputBytes > 0 {
		return c.MaxToolInputBytes
	}
	return MaxToolInputBytes
}

// EffectiveMaxErrorMessageLen returns the configured or default limit.
func (c Config) EffectiveMaxErrorMessageLen() int {
	if c.MaxErrorMessageLen > 0 {
		return c.MaxErrorMessageLen
	}
	return DefaultMaxErrorMessageLen
}

// EffectiveAnthropicVersion returns the configured or default API version.
func (c Config) EffectiveAnthropicVersion() string {
	if c.AnthropicVersion != "" {
		return c.AnthropicVersion
	}
	return DefaultAnthropicVersion
}

// redactURLCredentials scrubs a credential embedded in a URL — userinfo
// (user:password@) and common secret query parameters — while keeping the
// host/path visible for debugging. A URL it can't parse, or one with no
// credential, is returned unchanged (nothing safe to strip / preserve exact
// formatting). Shared by Config and AuthProfile marshaling.
func redactURLCredentials(raw string) string {
	if raw == "" {
		return raw
	}
	u, err := url.Parse(raw)
	if err != nil {
		return raw
	}
	changed := false
	if u.User != nil {
		u.User = url.User("REDACTED")
		changed = true
	}
	q := u.Query()
	qChanged := false
	for k := range q {
		switch strings.ToLower(k) {
		case "key", "api_key", "apikey", "token", "access_token", "auth":
			q.Set(k, "REDACTED")
			qChanged = true
		}
	}
	if qChanged {
		u.RawQuery = q.Encode()
		changed = true
	}
	if !changed {
		return raw
	}
	return u.String()
}

// MarshalJSON implements json.Marshaler. Redacts APIKey, and any credential
// embedded in BaseURL, to prevent accidental secret leakage when Config is
// serialized for logging or debugging. Use the fields directly for real values.
func (c Config) MarshalJSON() ([]byte, error) {
	type configAlias Config // break recursion
	tmp := configAlias(c)
	if tmp.APIKey != "" {
		tmp.APIKey = "REDACTED"
	}
	tmp.BaseURL = redactURLCredentials(tmp.BaseURL)
	return json.Marshal(tmp)
}
