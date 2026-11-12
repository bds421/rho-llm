package llm_test

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider"
)

// TestIsRetryableNetworkErrors verifies network error detection.
func TestIsRetryableNetworkErrors(t *testing.T) {
	tests := []struct {
		errMsg   string
		expected bool
	}{
		{"connection refused", true},
		{"no such host", true},
		{"timeout exceeded", true},
		{"unexpected eof", true},
		{"connection reset by peer", true},
		{"request failed: dial tcp", false}, // string-only path; in production, %w wrapping triggers typed net.Error check
		{"some random error", false},
		{"permission denied", false},
	}

	for _, tc := range tests {
		t.Run(tc.errMsg, func(t *testing.T) {
			err := fmt.Errorf("%s", tc.errMsg)
			got := llm.IsRetryable(err)
			if got != tc.expected {
				t.Errorf("IsRetryable(%q) = %v, want %v", tc.errMsg, got, tc.expected)
			}
		})
	}
}

// TestIsRetryableTypedErrors verifies type-based network error detection in IsRetryable.
func TestIsRetryableTypedErrors(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		expected bool
	}{
		{"net.DNSError", &net.DNSError{Err: "no such host", Name: "example.com"}, true},
		{"io.EOF", io.EOF, true},
		{"wrapped io.EOF", fmt.Errorf("stream failed: %w", io.EOF), true},
		{"io.ErrUnexpectedEOF", io.ErrUnexpectedEOF, true},
		{"wrapped syscall.ECONNRESET", fmt.Errorf("write: %w", syscall.ECONNRESET), true},
		{"wrapped syscall.ECONNREFUSED", fmt.Errorf("dial: %w", syscall.ECONNREFUSED), true},
		{"non-retryable plain error", errors.New("permission denied"), false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := llm.IsRetryable(tc.err)
			if got != tc.expected {
				t.Errorf("IsRetryable(%v) = %v, want %v", tc.err, got, tc.expected)
			}
		})
	}
}

// TestConfigMaxRetriesDefault verifies DefaultConfig sets MaxRetries.
func TestConfigMaxRetriesDefault(t *testing.T) {
	cfg := llm.DefaultConfig()
	if cfg.MaxRetries != llm.DefaultMaxRetries {
		t.Errorf("DefaultConfig().MaxRetries = %d, want %d", cfg.MaxRetries, llm.DefaultMaxRetries)
	}
}

// TestIsRetryableStringFallback verifies the narrowed string-matching patterns.
func TestIsRetryableStringFallback(t *testing.T) {
	tests := []struct {
		msg      string
		expected bool
	}{
		// Should match
		{"connection refused", true},
		{"dial tcp: no such host", true},
		{"context deadline exceeded: timeout", true},
		{"read tcp: connection reset", true},
		{"write: broken pipe", true},
		{"unexpected eof", true}, // HasSuffix "eof" — network-level condition
		{"something eof", true},  // HasSuffix "eof"
		// Should NOT match (narrowed in this fix)
		{"your payment request failed", false},
		{"request failed: invalid model", false},
	}

	for _, tc := range tests {
		err := errors.New(tc.msg)
		got := llm.IsRetryable(err)
		if got != tc.expected {
			t.Errorf("IsRetryable(%q) = %v, want %v", tc.msg, got, tc.expected)
		}
	}
}

// TestMaxRetryAttemptsCapped verifies the retry cap prevents excessive iterations.
func TestMaxRetryAttemptsCapped(t *testing.T) {
	// Create a pool with 20 keys (all will fail), verify we don't do 20 retries
	keys := make([]string, 20)
	for i := range keys {
		keys[i] = fmt.Sprintf("key-%d", i)
	}

	pc, err := llm.NewPooledClient(llm.DefaultConfig(), keys, func(profile llm.AuthProfile) (llm.Client, error) {
		return &threadSafeMockClient{
			provider: "test",
			model:    "test-model",
			err:      llm.NewRateLimitError("test", "rate limited"),
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, err = pc.Complete(ctx, llm.Request{})
	if err == nil {
		t.Fatal("expected error when all keys fail")
	}
	// The test passes if it completes within 5s — without the cap,
	// 20 retries × backoff would take much longer.
}

func TestRetryEventTypeString(t *testing.T) {
	tests := []struct {
		event llm.RetryEventType
		want  string
	}{
		{llm.RetryAttemptFailed, "attempt_failed"},
		{llm.RetryRotating, "rotating"},
		{llm.RetryBackingOff, "backing_off"},
		{llm.RetryCircuitOpen, "circuit_open"},
		{llm.RetryExhausted, "exhausted"},
		{llm.RetryEventType(99), "RetryEventType(99)"},
	}
	for _, test := range tests {
		if got := test.event.String(); got != test.want {
			t.Errorf("%d.String() = %q, want %q", test.event, got, test.want)
		}
	}
}

// TestRetryHookReceivesEvents verifies the RetryHook fires on retry lifecycle events.
func TestRetryHookReceivesEvents(t *testing.T) {
	var mu sync.Mutex
	var events []llm.RetryEventType

	cfg := llm.DefaultConfig()
	cfg.CircuitThreshold = 0 // disable circuit for this test
	// Short cooldowns so retries exhaust quickly
	cfg.CooldownRateLimit = 10 * time.Millisecond
	cfg.CooldownOverload = 10 * time.Millisecond
	cfg.CooldownDefault = 10 * time.Millisecond
	cfg.RetryPolicy = &llm.RetryPolicy{
		BaseDelay: 5 * time.Millisecond,
		MaxDelay:  50 * time.Millisecond,
		Factor:    2.0,
		Jitter:    0,
	}
	cfg.RetryHook = func(evt llm.RetryEvent) {
		mu.Lock()
		events = append(events, evt.Type)
		mu.Unlock()
	}

	pc, err := llm.NewPooledClient(cfg, []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &threadSafeMockClient{
			provider: "test",
			model:    "m",
			err:      llm.NewRateLimitError("test", "rate limited"),
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient: %v", err)
	}
	defer pc.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, _ = pc.Complete(ctx, llm.Request{})

	mu.Lock()
	defer mu.Unlock()

	// We should have received at least AttemptFailed and Exhausted events
	hasAttemptFailed := false
	hasExhausted := false
	for _, evt := range events {
		if evt == llm.RetryAttemptFailed {
			hasAttemptFailed = true
		}
		if evt == llm.RetryExhausted {
			hasExhausted = true
		}
	}
	if !hasAttemptFailed {
		t.Error("RetryHook never received RetryAttemptFailed event")
	}
	if !hasExhausted {
		t.Error("RetryHook never received RetryExhausted event")
	}
}

// TestRetryPolicyFromConfig verifies that a custom RetryPolicy on Config is used.
func TestRetryPolicyFromConfig(t *testing.T) {
	customPolicy := &llm.RetryPolicy{
		BaseDelay: 10 * time.Millisecond,
		MaxDelay:  100 * time.Millisecond,
		Factor:    2.0,
		Jitter:    0,
	}

	cfg := llm.DefaultConfig()
	cfg.RetryPolicy = customPolicy
	cfg.CircuitThreshold = 0 // disable circuit breaker for this test
	// Short pool cooldowns so they don't dominate the backoff
	cfg.CooldownOverload = 10 * time.Millisecond
	cfg.CooldownDefault = 10 * time.Millisecond
	cfg.CooldownRateLimit = 10 * time.Millisecond

	pc, err := llm.NewPooledClient(cfg, []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &threadSafeMockClient{
			provider: "test",
			model:    "m",
			err:      llm.NewOverloadedError("test", "overloaded"),
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient: %v", err)
	}
	defer pc.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	start := time.Now()
	_, _ = pc.Complete(ctx, llm.Request{})
	elapsed := time.Since(start)

	// With 10ms base, 100ms max, and 10ms pool cooldowns, the backoffs
	// should be much shorter than the default 1s/30s. Under 2s means
	// the custom policy was used (default would take ~7s minimum).
	if elapsed > 2*time.Second {
		t.Errorf("elapsed = %v, expected < 2s with fast retry policy", elapsed)
	}
}

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

func TestRetryPolicyExponentialGrowth(t *testing.T) {
	p := llm.DefaultRetryPolicy

	// Each attempt should roughly double from the previous.
	// Average over many samples to smooth jitter.
	for attempt := 0; attempt < 5; attempt++ {
		var total time.Duration
		const samples = 200
		for i := 0; i < samples; i++ {
			total += p.Delay(attempt)
		}
		avg := total / time.Duration(samples)

		expected := p.BaseDelay
		for j := 0; j < attempt; j++ {
			expected = time.Duration(float64(expected) * p.Factor)
			if expected > p.MaxDelay {
				expected = p.MaxDelay
				break
			}
		}

		low := time.Duration(float64(expected) * 0.60)
		high := time.Duration(float64(expected) * 1.40)
		if high > p.MaxDelay {
			high = p.MaxDelay
		}

		if avg < low || avg > high {
			t.Errorf("attempt %d: avg=%v, expected ~%v (range %v-%v)", attempt, avg, expected, low, high)
		}
	}
}

func TestRetryPolicyMaxCap(t *testing.T) {
	p := llm.RetryPolicy{
		BaseDelay: 1 * time.Second,
		MaxDelay:  5 * time.Second,
		Factor:    2.0,
		Jitter:    0.25,
	}

	for i := 0; i < 100; i++ {
		d := p.Delay(20)
		if d > 5*time.Second {
			t.Errorf("Delay(20) = %v, exceeds max 5s", d)
		}
		if d <= 0 {
			t.Errorf("Delay(20) = %v, should be positive", d)
		}
	}
}

func TestRetryPolicyNegativeAttempt(t *testing.T) {
	p := llm.DefaultRetryPolicy
	d := p.Delay(-5)
	if d <= 0 || d > p.MaxDelay {
		t.Errorf("Delay(-5) = %v, expected positive value <= %v", d, p.MaxDelay)
	}
}

func TestRetryPolicyZeroJitter(t *testing.T) {
	p := llm.RetryPolicy{
		BaseDelay: 100 * time.Millisecond,
		MaxDelay:  10 * time.Second,
		Factor:    2.0,
		Jitter:    0, // no jitter — deterministic
	}

	tests := []struct {
		attempt int
		want    time.Duration
	}{
		{0, 100 * time.Millisecond},
		{1, 200 * time.Millisecond},
		{2, 400 * time.Millisecond},
		{3, 800 * time.Millisecond},
	}

	for _, tt := range tests {
		got := p.Delay(tt.attempt)
		if got != tt.want {
			t.Errorf("Delay(%d) = %v, want %v", tt.attempt, got, tt.want)
		}
	}
}

func TestRetryPolicyCustomFactor(t *testing.T) {
	p := llm.RetryPolicy{
		BaseDelay: 100 * time.Millisecond,
		MaxDelay:  10 * time.Second,
		Factor:    3.0, // triple each time
		Jitter:    0,
	}

	tests := []struct {
		attempt int
		want    time.Duration
	}{
		{0, 100 * time.Millisecond},
		{1, 300 * time.Millisecond},
		{2, 900 * time.Millisecond},
		{3, 2700 * time.Millisecond},
	}

	for _, tt := range tests {
		got := p.Delay(tt.attempt)
		if got != tt.want {
			t.Errorf("Delay(%d) = %v, want %v", tt.attempt, got, tt.want)
		}
	}
}

func TestRetryPolicyBaseExceedsMax(t *testing.T) {
	p := llm.RetryPolicy{
		BaseDelay: 10 * time.Second,
		MaxDelay:  5 * time.Second,
		Factor:    2.0,
		Jitter:    0.25,
	}

	for i := 0; i < 20; i++ {
		d := p.Delay(0)
		if d > 5*time.Second {
			t.Errorf("Delay(0) = %v, exceeds max 5s", d)
		}
		if d <= 0 {
			t.Errorf("Delay(0) = %v, should be positive", d)
		}
	}
}

func TestRetryPolicyVeryHighAttempt(t *testing.T) {
	p := llm.RetryPolicy{
		BaseDelay: 1 * time.Second,
		MaxDelay:  5 * time.Second,
		Factor:    2.0,
		Jitter:    0.25,
	}

	for i := 0; i < 10; i++ {
		d := p.Delay(1000)
		if d > 5*time.Second {
			t.Errorf("Delay(1000) = %v, exceeds max 5s", d)
		}
		if d <= 0 {
			t.Errorf("Delay(1000) = %v, should be positive", d)
		}
	}
}

func TestRetryPolicyJitterDistribution(t *testing.T) {
	p := llm.RetryPolicy{
		BaseDelay: 1 * time.Second,
		MaxDelay:  30 * time.Second,
		Factor:    2.0,
		Jitter:    0.25,
	}

	// Attempt 0: base = 1s, jitter ±25% → range [750ms, 1250ms]
	var minSeen, maxSeen time.Duration
	const samples = 1000
	for i := 0; i < samples; i++ {
		d := p.Delay(0)
		if minSeen == 0 || d < minSeen {
			minSeen = d
		}
		if d > maxSeen {
			maxSeen = d
		}
	}

	if minSeen >= 1*time.Second {
		t.Errorf("jitter never went below base: min=%v", minSeen)
	}
	if maxSeen <= 1*time.Second {
		t.Errorf("jitter never went above base: max=%v", maxSeen)
	}
}

func TestDefaultRetryPolicyValues(t *testing.T) {
	p := llm.DefaultRetryPolicy
	if p.BaseDelay != 1*time.Second {
		t.Errorf("BaseDelay = %v, want 1s", p.BaseDelay)
	}
	if p.MaxDelay != 30*time.Second {
		t.Errorf("MaxDelay = %v, want 30s", p.MaxDelay)
	}
	if p.Factor != 2.0 {
		t.Errorf("Factor = %v, want 2.0", p.Factor)
	}
	if p.Jitter != 0.25 {
		t.Errorf("Jitter = %v, want 0.25", p.Jitter)
	}
}

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

var fastPolicy = &llm.RetryPolicy{BaseDelay: time.Millisecond, MaxDelay: 2 * time.Millisecond, Factor: 2}

func hitServer(t *testing.T, handler func(hit int32, w http.ResponseWriter)) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handler(hits.Add(1), w)
	}))
	t.Cleanup(srv.Close)
	return srv, &hits
}

func doGet(t *testing.T, cfg llm.Config, url string, opts llm.HTTPCallOptions) (*http.Response, error) {
	t.Helper()
	resp, err := llm.DoHTTPWithOptions(context.Background(), cfg, nil, func(ctx context.Context) (*http.Request, error) {
		return http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader("{}"))
	}, opts)
	if resp != nil {
		t.Cleanup(func() { _ = resp.Body.Close() })
	}
	return resp, err
}

// H1: a struct-literal Config (MaxRetries 0) used to get 10 DoHTTP attempts —
// the ~2m22s modality stall. The default is now 3, like a single-key chat client.
func TestDoHTTPDefaultAttemptsIsThree(t *testing.T) {
	srv, hits := hitServer(t, func(_ int32, w http.ResponseWriter) { w.WriteHeader(http.StatusServiceUnavailable) })
	resp, err := doGet(t, llm.Config{DisableProxy: true, RetryPolicy: fastPolicy}, srv.URL, llm.HTTPCallOptions{})
	if err != nil || resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("want the final 503 response, got resp=%v err=%v", resp, err)
	}
	if got := hits.Load(); got != 3 || llm.DefaultHTTPMaxAttempts != 3 {
		t.Fatalf("hits = %d, want 3", got)
	}
}

// An explicit MaxRetries is still honoured (DefaultConfig sets 10).
func TestDoHTTPExplicitMaxRetriesHonoured(t *testing.T) {
	srv, hits := hitServer(t, func(_ int32, w http.ResponseWriter) { w.WriteHeader(http.StatusBadGateway) })
	_, _ = doGet(t, llm.Config{DisableProxy: true, RetryPolicy: fastPolicy, MaxRetries: 5}, srv.URL, llm.HTTPCallOptions{})
	if got := hits.Load(); got != 5 {
		t.Fatalf("hits = %d, want 5", got)
	}
}

// Retry-After (delta-seconds) is a floor for the backoff even when the policy
// would retry after a millisecond.
func TestDoHTTPHonoursRetryAfterSeconds(t *testing.T) {
	srv, hits := hitServer(t, func(hit int32, w http.ResponseWriter) {
		if hit == 1 {
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		w.WriteHeader(http.StatusOK)
	})
	var backoff time.Duration
	cfg := llm.Config{DisableProxy: true, RetryPolicy: fastPolicy, RetryHook: func(e llm.RetryEvent) {
		if e.Type == llm.RetryBackingOff {
			backoff = e.Backoff
		}
	}}
	start := time.Now()
	resp, err := doGet(t, cfg, srv.URL, llm.HTTPCallOptions{})
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("resp=%v err=%v", resp, err)
	}
	if elapsed := time.Since(start); elapsed < 950*time.Millisecond {
		t.Fatalf("retried after %v, want >= 1s (Retry-After ignored)", elapsed)
	}
	if backoff != time.Second || hits.Load() != 2 {
		t.Fatalf("backoff=%v hits=%d, want 1s / 2", backoff, hits.Load())
	}
}

// Retry-After as an HTTP-date is honoured too.
func TestDoHTTPHonoursRetryAfterHTTPDate(t *testing.T) {
	srv, _ := hitServer(t, func(hit int32, w http.ResponseWriter) {
		if hit == 1 {
			w.Header().Set("Retry-After", time.Now().Add(3*time.Second).UTC().Format(http.TimeFormat))
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	})
	// Abort the sleep early by cancelling once the backoff is chosen: only
	// the chosen backoff matters here. (A ctx deadline shorter than the
	// backoff would skip the sleep altogether — see TestDoHTTPPolicyBackoffBeyondDeadlineStops.)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var backoff time.Duration
	cfg := llm.Config{DisableProxy: true, RetryPolicy: fastPolicy, RetryHook: func(e llm.RetryEvent) {
		if e.Type == llm.RetryBackingOff {
			backoff = e.Backoff
			cancel()
		}
	}}
	_, err := llm.DoHTTP(ctx, cfg, nil, func(ctx context.Context) (*http.Request, error) {
		return http.NewRequestWithContext(ctx, http.MethodGet, srv.URL, nil)
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("want the ctx to cut the Retry-After sleep, got %v", err)
	}
	// HTTP-date has 1 s resolution: 3 s ahead lands in (2s, 3s].
	if backoff <= 1500*time.Millisecond || backoff > 3*time.Second {
		t.Fatalf("backoff = %v, want ≈2-3s from the HTTP-date", backoff)
	}
}

// A hostile Retry-After of an hour is clamped, and a RetryBudget smaller than
// the clamped wait stops the sequence instead of sleeping.
func TestDoHTTPRetryAfterHugeIsCappedAndBudgetStopsSleep(t *testing.T) {
	srv, hits := hitServer(t, func(_ int32, w http.ResponseWriter) {
		w.Header().Set("Retry-After", "3600")
		w.WriteHeader(http.StatusTooManyRequests)
	})
	var exhausted atomic.Int32
	cfg := llm.Config{DisableProxy: true, RetryPolicy: fastPolicy, RetryBudget: 2 * time.Second,
		RetryHook: func(e llm.RetryEvent) {
			if e.Type == llm.RetryExhausted {
				exhausted.Add(1)
			}
		}}
	start := time.Now()
	resp, err := doGet(t, cfg, srv.URL, llm.HTTPCallOptions{})
	if err != nil || resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("want the 429 response back, got resp=%v err=%v", resp, err)
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Fatalf("slept %v although the budget could not fit the Retry-After wait", elapsed)
	}
	if hits.Load() != 1 || exhausted.Load() != 1 {
		t.Fatalf("hits=%d exhausted=%d, want 1/1", hits.Load(), exhausted.Load())
	}
}

// RetryBudget bounds the total wall-clock across attempts.
func TestDoHTTPRetryBudgetExhaustion(t *testing.T) {
	srv, hits := hitServer(t, func(_ int32, w http.ResponseWriter) { w.WriteHeader(http.StatusServiceUnavailable) })
	cfg := llm.Config{
		DisableProxy: true, MaxRetries: 10, RetryBudget: 500 * time.Millisecond,
		RetryPolicy: &llm.RetryPolicy{BaseDelay: 200 * time.Millisecond, MaxDelay: 200 * time.Millisecond, Factor: 1},
	}
	start := time.Now()
	resp, err := doGet(t, cfg, srv.URL, llm.HTTPCallOptions{})
	elapsed := time.Since(start)
	if err != nil || resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("resp=%v err=%v", resp, err)
	}
	// t=0, 200ms, 400ms; a fourth would start at 600ms > 500ms budget.
	if got := hits.Load(); got != 3 {
		t.Fatalf("hits = %d, want 3 within a 500ms budget", got)
	}
	if elapsed > 550*time.Millisecond {
		t.Fatalf("elapsed %v overran the 500ms budget", elapsed)
	}
}

// H5: a create that the provider may have committed (500/502/504/408) is not
// resent; 429/503 (rejected before processing) still are.
func TestDoHTTPNonIdempotentStatusMatrix(t *testing.T) {
	cases := []struct {
		status   int
		wantHits int32
	}{
		{http.StatusInternalServerError, 1},
		{http.StatusBadGateway, 1},
		{http.StatusGatewayTimeout, 1},
		{http.StatusRequestTimeout, 1},
		{http.StatusTooManyRequests, 3},
		{http.StatusServiceUnavailable, 3},
	}
	for _, tc := range cases {
		srv, hits := hitServer(t, func(_ int32, w http.ResponseWriter) { w.WriteHeader(tc.status) })
		resp, err := doGet(t, llm.Config{DisableProxy: true, RetryPolicy: fastPolicy}, srv.URL, llm.HTTPCallOptions{NonIdempotent: true})
		if err != nil || resp.StatusCode != tc.status {
			t.Fatalf("status %d: resp=%v err=%v", tc.status, resp, err)
		}
		if got := hits.Load(); got != tc.wantHits {
			t.Errorf("status %d: hits = %d, want %d", tc.status, got, tc.wantHits)
		}
		// The idempotent default keeps retrying every retryable status.
		srv2, hits2 := hitServer(t, func(_ int32, w http.ResponseWriter) { w.WriteHeader(tc.status) })
		_, _ = doGet(t, llm.Config{DisableProxy: true, RetryPolicy: fastPolicy}, srv2.URL, llm.HTTPCallOptions{})
		if got := hits2.Load(); got != 3 {
			t.Errorf("status %d idempotent: hits = %d, want 3", tc.status, got)
		}
	}
}

// H5: a connection that dies after the request was sent may have committed
// the create — not resent. A dial failure (nothing sent) is.
func TestDoHTTPNonIdempotentTransportErrors(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		conn, _, err := w.(http.Hijacker).Hijack()
		if err == nil {
			_ = conn.Close() // request fully read, then the connection drops
		}
	}))
	defer srv.Close()
	_, err := doGet(t, llm.Config{DisableProxy: true, RetryPolicy: fastPolicy}, srv.URL, llm.HTTPCallOptions{NonIdempotent: true})
	if err == nil || hits.Load() != 1 {
		t.Fatalf("post-send drop: err=%v hits=%d, want an error after exactly 1 hit", err, hits.Load())
	}

	// Dial failure: a port nobody listens on.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	_ = ln.Close()
	var attempts atomic.Int32
	cfg := llm.Config{DisableProxy: true, RetryPolicy: fastPolicy, RetryHook: func(e llm.RetryEvent) {
		if e.Type == llm.RetryAttemptFailed || e.Type == llm.RetryExhausted {
			attempts.Add(1)
		}
	}}
	_, err = doGet(t, cfg, "http://"+addr, llm.HTTPCallOptions{NonIdempotent: true})
	if err == nil || attempts.Load() != 3 {
		t.Fatalf("dial failure: err=%v attempts=%d, want 3 attempts (safe to resend)", err, attempts.Load())
	}
}

// Retry-After rides the APIError built from a real HTTP response.
func TestErrorFromResponseCarriesRetryAfter(t *testing.T) {
	resp := &http.Response{StatusCode: 429, Header: http.Header{"Retry-After": []string{"7"}}, Body: http.NoBody}
	var apiErr *llm.APIError
	if !errors.As(llm.ErrorFromResponse("p", resp, llm.Config{}), &apiErr) || apiErr.RetryAfter != 7*time.Second {
		t.Fatalf("RetryAfter not parsed: %+v", apiErr)
	}
	resp.Header.Set("Retry-After", "999999")
	if !errors.As(llm.ErrorFromResponse("p", resp, llm.Config{}), &apiErr) || apiErr.RetryAfter != llm.MaxRetryAfter {
		t.Fatalf("RetryAfter not capped: %v", apiErr.RetryAfter)
	}
}

// H5 end to end: the openaicompat creates (image generation, speech
// synthesis) and the Gemini image call are not resent after a 500, while
// idempotent embeddings still retry it.
func TestModalityCreatesNotRetriedOn500(t *testing.T) {
	type call struct {
		name   string
		cfg    func(url string) llm.Config
		invoke func(llm.ModalityClient) error
		want   int32
	}
	openai := func(model string) func(string) llm.Config {
		return func(url string) llm.Config {
			c := cfgFor(url)
			c.Model, c.DisableProxy, c.RetryPolicy = model, true, fastPolicy
			return c
		}
	}
	_ = llm.RegisterModel(llm.ModelInfo{ID: "gemini-2.5-flash-image", Provider: "gemini",
		Capabilities: llm.Capabilities(llm.CapabilityImageGeneration)})
	calls := []call{
		{"openai images", openai("gpt-image-2"), func(c llm.ModalityClient) error {
			_, err := c.GenerateImages(context.Background(), llm.ImageRequest{Model: "gpt-image-2", Prompt: "x"})
			return err
		}, 1},
		{"openai speech", openai("tts-1"), func(c llm.ModalityClient) error {
			_, err := c.SynthesizeSpeech(context.Background(), llm.SpeechRequest{Model: "tts-1", Input: "hi", Voice: "alloy", MediaType: "audio/mpeg"})
			return err
		}, 1},
		{"openai embeddings (idempotent)", openai("text-embedding-3-small"), func(c llm.ModalityClient) error {
			_, err := c.GenerateEmbeddings(context.Background(), llm.EmbeddingRequest{Model: "text-embedding-3-small", Input: []string{"x"}})
			return err
		}, 3},
		{"gemini images", func(url string) llm.Config {
			return llm.Config{Provider: "gemini", Model: "gemini-2.5-flash-image", APIKey: "k",
				BaseURL: url, DisableProxy: true, RetryPolicy: fastPolicy}
		}, func(c llm.ModalityClient) error {
			_, err := c.GenerateImages(context.Background(), llm.ImageRequest{Model: "gemini-2.5-flash-image", Prompt: "x", N: 1})
			return err
		}, 1},
	}
	for _, tc := range calls {
		srv, hits := hitServer(t, func(_ int32, w http.ResponseWriter) { w.WriteHeader(http.StatusInternalServerError) })
		client, err := llm.NewModalityClient(tc.cfg(srv.URL))
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if err := tc.invoke(client); err == nil {
			t.Errorf("%s: want the 500 surfaced", tc.name)
		}
		_ = client.Close()
		if got := hits.Load(); got != tc.want {
			t.Errorf("%s: hits = %d, want %d", tc.name, got, tc.want)
		}
	}
}

// Anthropic answers 529 when overloaded. The chat path retries it, DoHTTP did
// not: batch and file calls failed on the first overload (B16). Like a 503 the
// request was not processed, so even a create is resent.
func TestDoHTTPRetries529(t *testing.T) {
	for _, nonIdempotent := range []bool{false, true} {
		srv, hits := hitServer(t, func(hit int32, w http.ResponseWriter) {
			if hit == 1 {
				w.WriteHeader(529)
				_, _ = w.Write([]byte(`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`))
				return
			}
			w.WriteHeader(http.StatusOK)
		})
		resp, err := llm.DoHTTPWithOptions(context.Background(), llm.Config{DisableProxy: true, RetryPolicy: fastPolicy}, nil,
			func(ctx context.Context) (*http.Request, error) {
				return http.NewRequestWithContext(ctx, http.MethodPost, srv.URL, strings.NewReader("{}"))
			}, llm.HTTPCallOptions{NonIdempotent: nonIdempotent})
		if err != nil || resp.StatusCode != http.StatusOK || hits.Load() != 2 {
			t.Fatalf("nonIdempotent=%v: resp=%v err=%v hits=%d, want a retry after the 529", nonIdempotent, resp, err, hits.Load())
		}
		_ = resp.Body.Close()
	}
}
