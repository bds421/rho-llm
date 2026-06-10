package llm_test

// Regression tests for the v0.4.0 resilience-correctness fixes (architecture
// review findings R-H1/R-M3 classification, R-L1 factory bypass). These are
// break-the-system tests: each fails without its fix.

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"net/url"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// R-H1: context.Canceled — bare or wrapped the way net/http wraps it — must
// never classify as retryable. Retrying with a dead context cannot succeed,
// and the pool would poison key/breaker health treating it as one.
func TestIsRetryableContextCanceled(t *testing.T) {
	cases := []struct {
		name string
		err  error
	}{
		{"bare", context.Canceled},
		{"http-wrapped", &url.Error{Op: "Post", URL: "https://api.example.com/v1", Err: context.Canceled}},
	}
	for _, tc := range cases {
		if llm.IsRetryable(tc.err) {
			t.Errorf("%s: context cancellation classified retryable", tc.name)
		}
	}

	// The request's own HTTP timeout surfaces as context.DeadlineExceeded and
	// MUST stay retryable — only explicit cancellation is exempt.
	timeout := &url.Error{Op: "Post", URL: "https://api.example.com/v1", Err: context.DeadlineExceeded}
	if !llm.IsRetryable(timeout) {
		t.Error("HTTP client timeout (DeadlineExceeded) must remain retryable")
	}
}

// R-M3: a failed TLS certificate verification is a configuration /
// endpoint-identity problem. Retrying (or rotating keys) cannot fix it, so it
// must not burn the retry budget or put keys in cooldown.
func TestIsRetryableTLSCertificateError(t *testing.T) {
	certErr := &tls.CertificateVerificationError{Err: x509.UnknownAuthorityError{}}
	if llm.IsRetryable(certErr) {
		t.Error("bare certificate verification error classified retryable")
	}
	wrapped := &url.Error{Op: "Post", URL: "https://api.example.com/v1", Err: certErr}
	if llm.IsRetryable(wrapped) {
		t.Error("http-wrapped certificate verification error classified retryable")
	}
}

// R-L1: NewClientWithKeys with a nil/empty key slice must not silently bypass
// the resilience stack — it falls back to cfg.APIKey and still goes through
// PooledClient, exactly like NewClient.
func TestNewClientWithKeysEmptySliceStillPooled(t *testing.T) {
	cfg := llm.DefaultConfig()
	cfg.Provider = "ollama" // keyless provider: empty APIKey is valid
	cfg.Model = "llama3.3"

	for _, keys := range [][]string{nil, {}} {
		client, err := llm.NewClientWithKeys(cfg, keys)
		if err != nil {
			t.Fatalf("NewClientWithKeys(keys=%v): %v", keys, err)
		}
		if _, ok := client.(*llm.PooledClient); !ok {
			t.Fatalf("NewClientWithKeys(keys=%v) returned %T — resilience stack bypassed (want *llm.PooledClient)", keys, client)
		}
		client.Close()
	}
}
