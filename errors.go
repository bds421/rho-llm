package llm

import (
	"errors"
	"fmt"
	"io"
	"net"
	"strings"
	"syscall"
)

// APIError represents a structured error from an LLM provider API.
// All adapters produce this type instead of plain fmt.Errorf, enabling
// callers to use errors.As() for reliable error classification.
type APIError struct {
	StatusCode int    // HTTP status code (429, 503, 401, etc.)
	Message    string // Response body or error description
	Provider   string // Provider that returned the error
	Retryable  bool   // Whether the request should be retried
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

// maxErrorMessageLen caps the length of error messages stored in APIError.Message.
// Without this, a malicious or broken endpoint could return a multi-GB error body
// that propagates through error chains and log systems.
const maxErrorMessageLen = 4096

// truncateErrorBody caps s at maxErrorMessageLen, appending a truncation marker if cut.
func truncateErrorBody(s string) string {
	if len(s) <= maxErrorMessageLen {
		return s
	}
	return s[:maxErrorMessageLen] + "... [truncated]"
}

// NewAPIErrorFromStatus constructs the appropriate APIError from an HTTP status code and body.
func NewAPIErrorFromStatus(provider string, status int, body string) *APIError {
	body = truncateErrorBody(body)
	switch {
	case status == 429:
		return NewRateLimitError(provider, body)
	case status == 503:
		return NewOverloadedError(provider, body)
	case status == 401 || status == 403:
		return NewAuthError(provider, body, status)
	case status == 400 && isContextLengthMessage(body):
		return NewContextLengthError(provider, body)
	default:
		return &APIError{
			StatusCode: status,
			Message:    body,
			Provider:   provider,
			Retryable:  status == 500 || status == 502 || status == 408,
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

// IsOverloaded reports whether err is a server overloaded (503) error.
func IsOverloaded(err error) bool {
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr.StatusCode == 503
	}
	return false
}

// IsRetryable reports whether err should trigger a retry.
// Treats both API-level transient errors (429, 502, 503) and
// pure network dial/timeout errors (e.g. proxy unreachable) as retryable.
func IsRetryable(err error) bool {
	if err == nil {
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

	// String fallback: errors where type info was lost (fmt.Errorf without %w)
	errMsg := strings.ToLower(err.Error())
	return strings.Contains(errMsg, "request failed") ||
		strings.Contains(errMsg, "connection refused") ||
		strings.Contains(errMsg, "no such host") ||
		strings.Contains(errMsg, "timeout") ||
		strings.Contains(errMsg, "eof") ||
		strings.Contains(errMsg, "connection reset")
}

// IsAuthError reports whether err is an authentication/authorization error (401/403).
func IsAuthError(err error) bool {
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr.StatusCode == 401 || apiErr.StatusCode == 403
	}
	return false
}

// IsContextLength reports whether err is a context window exceeded error.
func IsContextLength(err error) bool {
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr.StatusCode == 400 && isContextLengthMessage(apiErr.Message)
	}
	return false
}

// isContextLengthMessage checks if a 400 error body indicates context length exceeded.
func isContextLengthMessage(body string) bool {
	lower := strings.ToLower(body)
	return strings.Contains(lower, "context length") ||
		strings.Contains(lower, "context_length") ||
		strings.Contains(lower, "token limit") ||
		strings.Contains(lower, "too many tokens") ||
		strings.Contains(lower, "maximum context") ||
		strings.Contains(lower, "prompt is too long") ||
		strings.Contains(lower, "input too long") ||
		strings.Contains(lower, "request too large")
}
