package llm_test

// S1 break tests: a backoff (Retry-After or policy delay) that would sleep
// into the caller's ctx deadline must not be started. Sleeping turned a 429
// carrying Retry-After into a bare context.DeadlineExceeded, losing the hint
// the caller needs to schedule its own retry.

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
)

func retryAfter2s(_ int32, w http.ResponseWriter) {
	w.Header().Set("Retry-After", "2")
	w.WriteHeader(http.StatusTooManyRequests)
	_, _ = w.Write([]byte(`{"error":{"message":"slow down","type":"rate_limit_error"}}`))
}

func TestDoHTTPRetryAfterBeyondDeadlineReturns429(t *testing.T) {
	srv, hits := hitServer(t, retryAfter2s)
	cfg := llm.Config{DisableProxy: true, RetryPolicy: fastPolicy}
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	start := time.Now()
	resp, err := llm.DoHTTP(ctx, cfg, nil, func(ctx context.Context) (*http.Request, error) {
		return http.NewRequestWithContext(ctx, http.MethodPost, srv.URL, strings.NewReader("{}"))
	})
	if err != nil {
		t.Fatalf("want the 429 response, got err %v (DeadlineExceeded=%v)", err, errors.Is(err, context.DeadlineExceeded))
	}
	defer resp.Body.Close()
	if elapsed := time.Since(start); elapsed > 250*time.Millisecond {
		t.Fatalf("returned after %v: slept toward the deadline", elapsed)
	}
	if resp.StatusCode != http.StatusTooManyRequests || hits.Load() != 1 {
		t.Fatalf("status=%d hits=%d, want 429 / 1", resp.StatusCode, hits.Load())
	}
	var apiErr *llm.APIError
	if !errors.As(llm.ErrorFromResponse("p", resp, cfg), &apiErr) || apiErr.RetryAfter != 2*time.Second {
		t.Fatalf("RetryAfter lost: %+v", apiErr)
	}
}

// A policy backoff (no Retry-After) past the deadline also stops early, for a
// transport-level 503 as well.
func TestDoHTTPPolicyBackoffBeyondDeadlineStops(t *testing.T) {
	srv, hits := hitServer(t, func(_ int32, w http.ResponseWriter) { w.WriteHeader(http.StatusServiceUnavailable) })
	cfg := llm.Config{DisableProxy: true, RetryPolicy: &llm.RetryPolicy{BaseDelay: time.Second, MaxDelay: time.Second, Factor: 1}}
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	resp, err := llm.DoHTTP(ctx, cfg, nil, func(ctx context.Context) (*http.Request, error) {
		return http.NewRequestWithContext(ctx, http.MethodPost, srv.URL, strings.NewReader("{}"))
	})
	if err != nil || resp.StatusCode != http.StatusServiceUnavailable || hits.Load() != 1 {
		t.Fatalf("resp=%v err=%v hits=%d", resp, err, hits.Load())
	}
	_ = resp.Body.Close()
}

// A backoff that fits before the deadline still retries.
func TestDoHTTPBackoffWithinDeadlineStillRetries(t *testing.T) {
	srv, hits := hitServer(t, func(hit int32, w http.ResponseWriter) {
		if hit == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	resp, err := llm.DoHTTP(ctx, llm.Config{DisableProxy: true, RetryPolicy: fastPolicy}, nil, func(ctx context.Context) (*http.Request, error) {
		return http.NewRequestWithContext(ctx, http.MethodPost, srv.URL, strings.NewReader("{}"))
	})
	if err != nil || resp.StatusCode != http.StatusOK || hits.Load() != 2 {
		t.Fatalf("resp=%v err=%v hits=%d", resp, err, hits.Load())
	}
	_ = resp.Body.Close()
}

// Single-key chat Complete over a real adapter and HTTP server.
func TestChatCompleteRetryAfterBeyondDeadlineReturns429(t *testing.T) {
	srv, hits := hitServer(t, retryAfter2s)
	c, err := llm.NewClient(llm.Config{Provider: "openai", Model: "gpt-4.1", APIKey: "k", BaseURL: srv.URL, RetryPolicy: fastPolicy})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err = c.Complete(ctx, llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "x")}})
	if elapsed := time.Since(start); elapsed > 250*time.Millisecond {
		t.Fatalf("returned after %v: slept toward the deadline (err=%v)", elapsed, err)
	}
	if errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("got DeadlineExceeded, want the 429: %v", err)
	}
	var apiErr *llm.APIError
	if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusTooManyRequests || apiErr.RetryAfter != 2*time.Second {
		t.Fatalf("want 429 APIError with RetryAfter 2s, got %v (%+v)", err, apiErr)
	}
	if hits.Load() != 1 {
		t.Fatalf("hits=%d, want 1", hits.Load())
	}
}

// Same for Stream (pre-data failure).
func TestChatStreamRetryAfterBeyondDeadlineReturns429(t *testing.T) {
	srv, _ := hitServer(t, retryAfter2s)
	c, err := llm.NewClient(llm.Config{Provider: "openai", Model: "gpt-4.1", APIKey: "k", BaseURL: srv.URL, RetryPolicy: fastPolicy})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	var streamErr error
	for _, err := range c.Stream(ctx, llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "x")}}) {
		if err != nil {
			streamErr = err
		}
	}
	var apiErr *llm.APIError
	if !errors.As(streamErr, &apiErr) || apiErr.RetryAfter != 2*time.Second {
		t.Fatalf("want 429 APIError with RetryAfter, got %v", streamErr)
	}
}
