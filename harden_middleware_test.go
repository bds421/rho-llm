package llm_test

// Hardening pass 4 — LoggingClient nil-response safety (middleware.go).

import (
	"context"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// A misbehaving inner Client that returns (nil, nil) must not crash the
// LoggingClient decorator: it checks err but then dereferences resp.InputTokens.
// (nilClient is defined in breaksystem2_test.go and returns (nil, nil).)
func TestLoggingClientDoesNotPanicOnNilResponse(t *testing.T) {
	lc := llm.WithLogging(nilClient{})
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("LoggingClient.Complete panicked on a (nil,nil) inner response: %v", r)
		}
	}()
	resp, err := lc.Complete(context.Background(), llm.Request{Model: "m"})
	// Transparent pass-through of the (nil,nil) is fine; the contract is just
	// "do not panic" — the caller (e.g. Session) handles the nil response.
	if resp != nil {
		t.Fatalf("expected nil response passed through, got %+v", resp)
	}
	_ = err
}
