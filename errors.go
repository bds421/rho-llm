package llm

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"regexp"
	"strings"
	"syscall"
	"time"
)

// APIError represents a structured error from an LLM provider API.
// All adapters produce this type instead of plain fmt.Errorf, enabling
// callers to use errors.As() for reliable error classification.
type APIError struct {
	StatusCode int    // HTTP status code (429, 503, 401, etc.)
	Message    string // Response body or error description
	Provider   string // Provider that returned the error
	Retryable  bool   // Whether the request should be retried
	// RetryAfter is the provider's Retry-After hint (seconds, HTTP-date, or
	// OpenAI's retry-after-ms), clamped to MaxRetryAfter; 0 when absent. The
	// built-in retry loops use it as a floor for the next backoff.
	RetryAfter time.Duration

	// keyExhausted: this credential cannot serve requests for longer than a
	// retry waits (quota or billing exhausted, or a Retry-After beyond
	// MaxRetryAfter). See IsQuotaExhausted.
	keyExhausted bool
}

func (e *APIError) Error() string {
	return fmt.Sprintf("%s API error (status %d): %s", e.Provider, e.StatusCode, e.Message)
}

// NewRateLimitError creates an error for HTTP 429 responses.
func NewRateLimitError(provider, message string) *APIError {
	return &APIError{
		StatusCode: 429,
		Message:    message,
		Provider:   provider,
		Retryable:  true,
	}
}

// NewOverloadedError creates an error for HTTP 503 responses.
func NewOverloadedError(provider, message string) *APIError {
	return &APIError{
		StatusCode: 503,
		Message:    message,
		Provider:   provider,
		Retryable:  true,
	}
}

// NewAuthError creates an error for HTTP 401/403 responses.
func NewAuthError(provider, message string, statusCode int) *APIError {
	return &APIError{
		StatusCode: statusCode,
		Message:    message,
		Provider:   provider,
		Retryable:  false,
	}
}

// NewContextLengthError creates an error for context window exceeded (400 + pattern match).
func NewContextLengthError(provider, message string) *APIError {
	return &APIError{
		StatusCode: 400,
		Message:    message,
		Provider:   provider,
		Retryable:  false,
	}
}

// truncateErrorBody caps s at maxLen, appending a truncation marker if cut.
// A non-positive maxLen falls back to DefaultMaxErrorMessageLen.
func truncateErrorBody(s string, maxLen int) string {
	if maxLen <= 0 {
		maxLen = DefaultMaxErrorMessageLen
	}
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "... [truncated]"
}

// NewAPIErrorFromStatus constructs the appropriate APIError from an HTTP status code
// and body, truncating the stored message to DefaultMaxErrorMessageLen.
func NewAPIErrorFromStatus(provider string, status int, body string) *APIError {
	return NewAPIErrorFromStatusWithLimit(provider, status, body, DefaultMaxErrorMessageLen)
}

// NewAPIErrorFromStatusWithLimit is like NewAPIErrorFromStatus but truncates the stored
// message to maxMsgLen. Adapters pass Config.EffectiveMaxErrorMessageLen() so the limit
// is honored per client. A non-positive maxMsgLen falls back to DefaultMaxErrorMessageLen.
func NewAPIErrorFromStatusWithLimit(provider string, status int, body string, maxMsgLen int) *APIError {
	body = truncateErrorBody(body, maxMsgLen)
	switch {
	case status == 429:
		rl := NewRateLimitError(provider, body)
		if isQuotaMessage(body) {
			// Quota or billing exhausted: it resets in hours, not seconds,
			// so retrying the same key only burns the retry budget (P5).
			rl.Retryable = false
			rl.keyExhausted = true
		}
		return rl
	case status == 503 || status == 529:
		// 529 is Anthropic's "overloaded_error".
		overloaded := NewOverloadedError(provider, body)
		overloaded.StatusCode = status
		return overloaded
	case status == 401 || status == 403:
		return NewAuthError(provider, body, status)
	case status == 413 || (status >= 400 && status < 500 && status != 429 && isContextLengthMessage(body)):
		// 413 is a request too large for the endpoint whatever the body says
		// (Anthropic request_too_large, Cerebras without a body); other 4xx
		// overflow errors are recognised by their text (P5).
		cl := NewContextLengthError(provider, body)
		cl.StatusCode = status
		return cl
	default:
		return &APIError{
			StatusCode: status,
			Message:    body,
			Provider:   provider,
			Retryable:  status == 500 || status == 502 || status == 504 || status == 408,
		}
	}
}

// IsRateLimited reports whether err is a rate limit (429) error.
func IsRateLimited(err error) bool {
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr.StatusCode == 429
	}
	return false
}

// IsOverloaded reports whether err is a server overloaded error (503, or
// Anthropic's 529).
func IsOverloaded(err error) bool {
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr.StatusCode == 503 || apiErr.StatusCode == 529
	}
	return false
}

// IsRetryable reports whether err should trigger a retry.
// Treats both API-level transient errors (429, 502, 503) and
// pure network dial/timeout errors (e.g. proxy unreachable) as retryable.
//
// Three classes are explicitly NOT retryable even though they surface as
// net.Error through net/http's *url.Error wrapper:
//   - context.Canceled: the caller aborted — retrying with a dead context
//     cannot succeed, and counting it as a provider failure would poison
//     pool/breaker health. (context.DeadlineExceeded stays retryable: the
//     HTTP client's own timeout surfaces as it.)
//   - TLS certificate verification failures: an endpoint-identity /
//     configuration problem that no retry or key rotation can fix.
//   - ErrPrivateAddress: BlockPrivateBaseURL refused the address; a retry
//     dials the same address.
func IsRetryable(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) {
		return false
	}
	if errors.Is(err, ErrPrivateAddress) {
		return false
	}
	var certErr *tls.CertificateVerificationError
	if errors.As(err, &certErr) {
		return false
	}
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr.Retryable
	}

	// Type-based: properly-typed stdlib network errors
	var netErr net.Error
	if errors.As(err, &netErr) {
		return true
	}
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}
	if errors.Is(err, syscall.ECONNRESET) || errors.Is(err, syscall.ECONNREFUSED) {
		return true
	}

	// String fallback: errors where type info was lost (fmt.Errorf without %w).
	// Patterns are narrowed to network-transport phrases to avoid false positives
	// on API-level messages that happen to contain generic words like "request failed".
	errMsg := strings.ToLower(err.Error())
	return strings.Contains(errMsg, "connection refused") ||
		strings.Contains(errMsg, "no such host") ||
		strings.HasSuffix(errMsg, "timeout") || strings.Contains(errMsg, "timeout ") || strings.Contains(errMsg, "timed out") ||
		strings.HasSuffix(errMsg, "eof") || strings.Contains(errMsg, "eof:") ||
		strings.Contains(errMsg, "connection reset") ||
		strings.Contains(errMsg, "broken pipe")
}

// IsAuthError reports whether err is an authentication/authorization error (401/403).
func IsAuthError(err error) bool {
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr.StatusCode == 401 || apiErr.StatusCode == 403
	}
	return false
}

// IsContextLength reports whether err says the request exceeds the model's
// context window or the endpoint's request size: an HTTP 413, or another
// client error (or a stream error without a status) whose text is a known
// overflow message. A 429 never counts, whatever its text.
func IsContextLength(err error) bool {
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		switch {
		case apiErr.StatusCode == 413:
			return true
		case apiErr.StatusCode == 429:
			return false
		case apiErr.StatusCode == 0 || (apiErr.StatusCode >= 400 && apiErr.StatusCode < 500):
			return isContextLengthMessage(apiErr.Message)
		}
	}
	return false
}

// IsQuotaExhausted reports whether err says the credential cannot serve
// requests for longer than a retry would wait: a 429 whose body names an
// exhausted quota or budget (OpenAI insufficient_quota, Z.ai code 1308, …), or
// a response whose Retry-After exceeds MaxRetryAfter. Such an error is not
// retryable on the same key; a key pool rests the key and tries another.
func IsQuotaExhausted(err error) bool {
	var apiErr *APIError
	return errors.As(err, &apiErr) && apiErr.keyExhausted
}

// contextLengthPhrases are overflow messages, lower-cased (from pi-ai's
// overflow.ts and the providers' docs). Each is specific enough that an
// unrelated 4xx does not contain it.
var contextLengthPhrases = []string{
	"context length", "context_length", "maximum context", "context window",
	"prompt is too long", "prompt too long", "prompt exceeds max length",
	"input too long", "input is too long", "request too large", "request_too_large",
	"token count exceeds", "input token count",
	"maximum prompt length is",               // xAI
	"reduce the length of the messages",      // Groq
	"maximum allowed input length",           // OpenRouter
	"longer than the model's context length", // Together
	"exceeds the available context size",     // llama.cpp
	"greater than the context length",        // LM Studio
	"context window exceeds limit",           // MiniMax
	"exceeded model token limit",             // Kimi
	"range of input length should be",        // DashScope / Qwen
	"model_context_window_exceeded",          // Z.ai
	// Not "too many tokens" or "token limit exceeded": rate limits use them
	// too ("per token limit exceeded").
}

// notContextLengthPhrases mark rate limiting worded like an overflow, e.g.
// Bedrock's "ThrottlingException: Too many tokens, please wait".
var notContextLengthPhrases = []string{"rate limit", "rate_limit", "too many requests", "throttl", "please wait"}

// isContextLengthMessage reports whether an error body is an overflow message.
func isContextLengthMessage(body string) bool {
	lower := strings.ToLower(body)
	for _, p := range notContextLengthPhrases {
		if strings.Contains(lower, p) {
			return false
		}
	}
	for _, p := range contextLengthPhrases {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

// quotaPhrases mark a 429 that is an exhausted quota or budget, not a
// throttle (from pi-ai's retry.ts). "billing" alone is left out: Gemini's
// ordinary per-minute 429 says "check your plan and billing details".
var quotaPhrases = []string{
	"insufficient_quota", "quota exceeded", "out of budget", "monthly usage limit reached",
	"usage_limit_exceeded", "credit balance is too low",
}

func isQuotaMessage(body string) bool {
	lower := strings.ToLower(body)
	for _, p := range quotaPhrases {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return zaiPlanLimit.MatchString(body)
}

// zaiPlanLimit is Z.ai error code 1308: the plan's usage limit, sent as 429.
var zaiPlanLimit = regexp.MustCompile(`"code"\s*:\s*"?1308(?:[^0-9]|$)`)
