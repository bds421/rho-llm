package llm

import (
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync/atomic"
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
	client, _ := NewSafeHTTPClient(Config{Timeout: timeout})
	return client
}

// NewSafeHTTPClient returns a redirect-safe client using the exact proxy
// policy declared by cfg. ProxyURL takes precedence over ambient proxy
// variables; DisableProxy deliberately bypasses them. The two controls are
// mutually exclusive so callers cannot silently weaken an explicit boundary.
func NewSafeHTTPClient(cfg Config) (*http.Client, error) {
	proxy, err := proxyPolicy(cfg)
	if err != nil {
		return nil, err
	}
	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	return &http.Client{
		Timeout: timeout,
		Transport: &http.Transport{
			Proxy:           proxy,
			TLSClientConfig: &tls.Config{MinVersion: tls.VersionTLS12},
			// Phase bounds that also protect streaming requests, whose client
			// (NewStreamingHTTPClient) has no overall Timeout. Each is capped
			// by the configured Timeout so a non-stream call is unchanged.
			DialContext:           (&net.Dialer{Timeout: min(timeout, 30*time.Second), KeepAlive: 30 * time.Second}).DialContext,
			TLSHandshakeTimeout:   min(timeout, 10*time.Second),
			ResponseHeaderTimeout: timeout,
			IdleConnTimeout:       90 * time.Second,
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
	}, nil
}

// NewStreamingHTTPClient derives the client a streaming request should use
// from base (normally the adapter's NewSafeHTTPClient result). http.Client's
// Timeout covers reading the whole body, so on the shared client a stream
// that legitimately runs longer than Config.Timeout — a ThinkingHigh turn, a
// long generation — was cut off mid-response with nothing to retry. The
// streaming client instead:
//
//   - shares base's transport (connection pool, proxy policy, TLS floor),
//     whose dial, TLS-handshake and ResponseHeaderTimeout bounds still apply
//     up to the first response byte;
//   - has no overall Timeout, so total stream duration is bounded only by the
//     request context;
//   - fails a stream that goes silent: when no body bytes arrive for
//     Config.Timeout (DefaultTimeout when zero), the body is closed and the
//     read returns a timeout net.Error. Gaps are measured per read, so a
//     stream is only ever cut later than the old whole-body Timeout would
//     have cut it, never earlier.
//
// Redirect handling is base's. base must not be nil.
func NewStreamingHTTPClient(base *http.Client, cfg Config) *http.Client {
	idle := cfg.Timeout
	if idle <= 0 {
		idle = DefaultTimeout
	}
	stream := *base
	stream.Timeout = 0
	inner := base.Transport
	if inner == nil {
		inner = http.DefaultTransport
	}
	stream.Transport = &idleTimeoutTransport{base: inner, idle: idle}
	return &stream
}

// idleTimeoutTransport wraps every response body in an idle watchdog.
type idleTimeoutTransport struct {
	base http.RoundTripper
	idle time.Duration
}

func (t *idleTimeoutTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.base.RoundTrip(req)
	if err != nil || resp == nil || resp.Body == nil {
		return resp, err
	}
	body := &idleTimeoutBody{rc: resp.Body, idle: t.idle}
	body.timer = time.AfterFunc(t.idle, body.expire)
	resp.Body = body
	return resp, nil
}

// CloseIdleConnections forwards to the shared transport so an adapter's
// Close still drains the pool.
func (t *idleTimeoutTransport) CloseIdleConnections() {
	if closer, ok := t.base.(interface{ CloseIdleConnections() }); ok {
		closer.CloseIdleConnections()
	}
}

type idleTimeoutBody struct {
	rc      io.ReadCloser
	idle    time.Duration
	timer   *time.Timer
	expired atomic.Bool
}

func (b *idleTimeoutBody) expire() {
	b.expired.Store(true)
	_ = b.rc.Close() // unblocks a pending Read
}

func (b *idleTimeoutBody) Read(p []byte) (int, error) {
	n, err := b.rc.Read(p)
	if b.expired.Load() {
		return n, &StreamIdleTimeoutError{Idle: b.idle}
	}
	if n > 0 {
		b.timer.Reset(b.idle)
	}
	return n, err
}

func (b *idleTimeoutBody) Close() error {
	b.timer.Stop()
	return b.rc.Close()
}

// StreamIdleTimeoutError reports a streaming response that sent no bytes for
// longer than Config.Timeout. It is a net.Error with Timeout() true, so a
// stall before the first event is retryable like any transport timeout.
type StreamIdleTimeoutError struct {
	Idle time.Duration
}

func (e *StreamIdleTimeoutError) Error() string {
	return fmt.Sprintf("llm: stream idle timeout: no data for %v", e.Idle)
}

// Timeout reports true (net.Error).
func (e *StreamIdleTimeoutError) Timeout() bool { return true }

// Temporary reports true (net.Error; deprecated upstream but part of the interface).
func (e *StreamIdleTimeoutError) Temporary() bool { return true }

func proxyPolicy(cfg Config) (func(*http.Request) (*url.URL, error), error) {
	proxyURL := strings.TrimSpace(cfg.ProxyURL)
	if cfg.DisableProxy && proxyURL != "" {
		return nil, fmt.Errorf("llm: ProxyURL and DisableProxy are mutually exclusive")
	}
	if cfg.DisableProxy {
		return nil, nil
	}
	if proxyURL == "" {
		return http.ProxyFromEnvironment, nil
	}
	parsed, err := url.Parse(proxyURL)
	if err != nil || parsed.Hostname() == "" || parsed.User != nil ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") ||
		(parsed.Path != "" && parsed.Path != "/") || parsed.RawQuery != "" ||
		parsed.Fragment != "" {
		return nil, fmt.Errorf("llm: ProxyURL is invalid")
	}
	return http.ProxyURL(parsed), nil
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

	// Model identifier (e.g., "claude-sonnet-5", "grok-4.5").
	Model string `json:"model"`

	// ModelCapabilities is an exact, deployment-scoped reviewed capability
	// declaration for Model. When non-zero it takes precedence over process-global
	// registry metadata and is never applied to a different request model.
	ModelCapabilities CapabilitySet `json:"model_capabilities,omitempty"`

	// API key for authentication. Empty is valid for local providers (Ollama, vLLM, LM Studio).
	APIKey string `json:"api_key"`

	// Maximum output tokens.
	MaxTokens int `json:"max_tokens"`

	// Sampling temperature. nil = omit from wire (provider default).
	Temperature *float64 `json:"temperature,omitempty"`

	// Extended thinking level: ThinkingLow, ThinkingMedium, ThinkingHigh (zero value = none).
	ThinkingLevel ThinkingLevel `json:"thinking_level"`

	// Timeout bounds a non-streaming HTTP request end to end (zero uses
	// DefaultTimeout). Streaming requests are not cut off after Timeout in
	// total: it bounds the wait for response headers and each silent gap
	// between body bytes instead (see NewStreamingHTTPClient); bound a
	// stream's total duration with the request context.
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

	// ProxyURL is an explicit HTTP(S) forward proxy for this config. It is
	// intended for reviewed public-provider egress and overrides HTTP_PROXY,
	// HTTPS_PROXY, and NO_PROXY. Credentials and path/query fragments are
	// rejected so this non-secret transport setting is safe to report.
	ProxyURL string `json:"proxy_url,omitempty"`

	// DisableProxy explicitly bypasses both ProxyURL and ambient proxy
	// variables. Use it for reviewed local/private model endpoints.
	DisableProxy bool `json:"disable_proxy,omitempty"`

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
	// It governs DoHTTP and every single-key chat retry. A provider
	// Retry-After hint (capped at MaxRetryAfter) stretches a backoff that
	// would be shorter. Multi-key pools that find every key in cooldown wait
	// for the soonest key instead (CooldownRateLimit/CooldownOverload/...).
	RetryPolicy *RetryPolicy `json:"retry_policy,omitempty"`

	// CircuitThreshold is the number of consecutive failures before the circuit opens.
	// Zero disables the circuit breaker. DefaultConfig sets
	// DefaultCircuitThreshold (5); a struct-literal Config leaves it 0, i.e.
	// no breaker — set it explicitly to get one.
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

	// UsageHook, when set, receives one UsageEvent per provider attempt of a
	// modality operation (TranscribeAudio, GenerateEmbeddings, GenerateImages,
	// SynthesizeSpeech) on clients built by NewModalityClient or
	// NewFallbackModalityClient — failed attempts included, requests rejected
	// by validation before dispatch excluded. Panics are recovered; the hook
	// never changes a call's result. Chat calls report usage on Response
	// instead. Not serialized.
	UsageHook UsageHook `json:"-"`

	// UsageHookCtx is UsageHook with the caller's context, for attributing
	// events to the request that caused them (e.g. a user ID in a context
	// value) on a client shared by many callers. Same events, same delivery
	// rules; when both hooks are set each receives every event, UsageHook
	// first. Not serialized.
	UsageHookCtx ContextUsageHook `json:"-"`

	// MaxRetries caps the number of retry/rotation iterations. Zero uses the
	// default: DefaultMaxRetries as the cap for pooled chat clients (which
	// make max(healthy keys, 3) attempts, so a single key gets 3), and
	// DefaultHTTPMaxAttempts (3) for the modality/batch transport (DoHTTP).
	// Minimum effective value is 3 (for single-key resilience against
	// transient errors).
	MaxRetries int `json:"max_retries,omitempty"`

	// RetryBudget bounds the total wall-clock time one call may spend across
	// its retry sequence (attempts plus backoff sleeps). Before each backoff
	// the client checks whether sleeping and trying again would still fit;
	// when it would not, the call returns the last error instead of sleeping.
	// An attempt already in flight is never cut short by the budget — use the
	// request context for a hard deadline. Zero (the default) or negative
	// means no budget: the sequence is bounded only by MaxRetries, the
	// backoff policy and ctx. Applies to pooled chat clients (Complete and
	// pre-data Stream retries) and to DoHTTP (modality and batch calls).
	RetryBudget time.Duration `json:"retry_budget,omitempty"`

	// DisableRetries forces exactly one provider transport attempt. It is for
	// callers whose durable outer execution authority owns retry, idempotency,
	// cancellation, and spend accounting. It applies to pooled chat clients and
	// the common non-chat/batch HTTP transport.
	DisableRetries bool `json:"disable_retries,omitempty"`

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

// DefaultMaxTokens is applied when Config.MaxTokens is zero (the int zero value).
// Several providers reject a request with max_tokens 0 outright (Anthropic
// returns HTTP 400 "max_tokens cannot be 0"), so callers that build a Config
// as a struct literal — without calling DefaultConfig() — would otherwise send
// an unusable request.
const DefaultMaxTokens = 8192

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
		Provider: "anthropic",
		// Track the registry's Anthropic default rather than hardcoding an ID —
		// a literal here silently drifts from defaultModels every time the
		// flagship moves (it had been left on claude-sonnet-4-6).
		Model:            GetDefaultModel("anthropic"),
		MaxTokens:        DefaultMaxTokens,
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
// redactProfileSecrets scrubs an auth profile's APIKey AND any credentials
// embedded in its (possibly per-key "apikey|baseurl") BaseURL out of a free-text
// string before it is logged or serialized. The BaseURL host/path stay visible
// for debugging; userinfo and secret query params (token/key/…) are stripped —
// both as a verbatim-URL replacement and as bare substrings, in case the error
// echoed only the credential rather than the whole URL.
func redactProfileSecrets(s, apiKey, baseURL string) string {
	s = redactSecret(s, apiKey)
	if baseURL == "" {
		return s
	}
	if red := redactURLCredentials(baseURL); red != baseURL {
		s = strings.ReplaceAll(s, baseURL, red)
		if u, err := url.Parse(baseURL); err == nil {
			if u.User != nil {
				if pw, ok := u.User.Password(); ok {
					s = redactSecret(s, pw)
				}
			}
			for k, vs := range u.Query() {
				switch strings.ToLower(k) {
				case "key", "api_key", "apikey", "token", "access_token", "auth":
					for _, v := range vs {
						s = redactSecret(s, v)
					}
				}
			}
		}
	}
	return s
}

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
