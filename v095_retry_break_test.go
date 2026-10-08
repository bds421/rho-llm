package llm_test

// Break-the-system tests for the v0.9.5 retry unification (H1), the
// non-idempotent create guard (H5) and Retry-After / RetryBudget handling.
// Every test drives the real retry loop (DoHTTP over httptest, or a
// PooledClient over a scripted inner client) and asserts on hit counts and
// wall-clock, never on internal state.

import (
	"context"
	"errors"
	"iter"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
)

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
	var backoff time.Duration
	cfg := llm.Config{DisableProxy: true, RetryPolicy: fastPolicy, RetryHook: func(e llm.RetryEvent) {
		if e.Type == llm.RetryBackingOff {
			backoff = e.Backoff
		}
	}}
	// Abort the sleep early: only the chosen backoff matters here.
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	_, err := llm.DoHTTP(ctx, cfg, nil, func(ctx context.Context) (*http.Request, error) {
		return http.NewRequestWithContext(ctx, http.MethodGet, srv.URL, nil)
	})
	if !errors.Is(err, context.DeadlineExceeded) {
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

// ---------------------------------------------------------------------------
// PooledClient (chat) — single-key backoff, Retry-After, budget, multi-key.
// ---------------------------------------------------------------------------

// scriptedClient returns errs[i] for the i-th call (then success).
type scriptedClient struct {
	mu    sync.Mutex
	calls int
	errs  []error
}

func (c *scriptedClient) next() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	i := c.calls
	c.calls++
	if i < len(c.errs) {
		return c.errs[i]
	}
	return nil
}

func (c *scriptedClient) Complete(context.Context, llm.Request) (*llm.Response, error) {
	if err := c.next(); err != nil {
		return nil, err
	}
	return &llm.Response{Content: "ok"}, nil
}

func (c *scriptedClient) Stream(context.Context, llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		if err := c.next(); err != nil {
			yield(llm.StreamEvent{}, err)
			return
		}
		yield(llm.StreamEvent{Type: llm.EventDone, StopReason: llm.StopEndTurn}, nil)
	}
}
func (c *scriptedClient) Provider() string { return "test" }
func (c *scriptedClient) Model() string    { return "m" }
func (c *scriptedClient) Close() error     { return nil }

func newScriptedPool(t *testing.T, cfg llm.Config, keys []string, inner *scriptedClient) *llm.PooledClient {
	t.Helper()
	pc, err := llm.NewPooledClient(cfg, keys, func(llm.AuthProfile) (llm.Client, error) { return inner, nil })
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = pc.Close() })
	return pc
}

// H1: a single-key 503 used to sleep the 30 s key cooldown (60 s for 429);
// Config.RetryPolicy was dead configuration. The backoff now follows it.
func TestPoolSingleKeyBackoffFollowsRetryPolicyNotCooldown(t *testing.T) {
	inner := &scriptedClient{errs: []error{llm.NewOverloadedError("test", "busy"), llm.NewRateLimitError("test", "slow")}}
	cfg := llm.Config{Provider: "test", RetryPolicy: fastPolicy, CooldownOverload: time.Hour, CooldownRateLimit: time.Hour}
	pc := newScriptedPool(t, cfg, []string{"k"}, inner)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	start := time.Now()
	resp, err := pc.Complete(ctx, llm.Request{})
	if err != nil || resp.Content != "ok" {
		t.Fatalf("Complete: resp=%v err=%v (backoff used the 1h key cooldown?)", resp, err)
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Fatalf("took %v, want the ms-scale RetryPolicy backoff", elapsed)
	}
	// Same for the pre-data Stream retry path.
	inner2 := &scriptedClient{errs: []error{llm.NewOverloadedError("test", "busy")}}
	pc2 := newScriptedPool(t, cfg, []string{"k"}, inner2)
	for _, err := range pc2.Stream(ctx, llm.Request{}) {
		if err != nil {
			t.Fatalf("Stream: %v", err)
		}
	}
}

// The provider's Retry-After hint stretches the single-key backoff.
func TestPoolHonoursRetryAfterHint(t *testing.T) {
	rl := llm.NewRateLimitError("test", "slow down")
	rl.RetryAfter = 300 * time.Millisecond
	inner := &scriptedClient{errs: []error{rl}}
	var backoff time.Duration
	cfg := llm.Config{Provider: "test", RetryPolicy: fastPolicy, RetryHook: func(e llm.RetryEvent) {
		if e.Type == llm.RetryBackingOff {
			backoff = e.Backoff
		}
	}}
	pc := newScriptedPool(t, cfg, []string{"k"}, inner)
	start := time.Now()
	if _, err := pc.Complete(context.Background(), llm.Request{}); err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed < 280*time.Millisecond || backoff != 300*time.Millisecond {
		t.Fatalf("elapsed=%v backoff=%v, want >= the 300ms Retry-After", elapsed, backoff)
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

// RetryBudget stops the chat retry loop before a backoff that would overrun it.
func TestPoolRetryBudgetExhaustion(t *testing.T) {
	busy := llm.NewOverloadedError("test", "busy")
	cfg := llm.Config{
		Provider: "test", RetryBudget: 100 * time.Millisecond,
		RetryPolicy: &llm.RetryPolicy{BaseDelay: 300 * time.Millisecond, MaxDelay: 300 * time.Millisecond, Factor: 1},
	}
	inner := &scriptedClient{errs: []error{busy, busy, busy}}
	pc := newScriptedPool(t, cfg, []string{"k"}, inner)
	start := time.Now()
	_, err := pc.Complete(context.Background(), llm.Request{})
	if err == nil || !strings.Contains(err.Error(), "retry budget") || !llm.IsOverloaded(err) {
		t.Fatalf("want a budget-exhausted error wrapping the 503, got %v", err)
	}
	if elapsed := time.Since(start); elapsed > 200*time.Millisecond || inner.calls != 1 {
		t.Fatalf("elapsed=%v calls=%d: slept past the budget", elapsed, inner.calls)
	}

	inner2 := &scriptedClient{errs: []error{busy, busy, busy}}
	pc2 := newScriptedPool(t, cfg, []string{"k"}, inner2)
	var streamErr error
	for _, err := range pc2.Stream(context.Background(), llm.Request{}) {
		streamErr = err
	}
	if streamErr == nil || !strings.Contains(streamErr.Error(), "retry budget") || inner2.calls != 1 {
		t.Fatalf("stream: err=%v calls=%d", streamErr, inner2.calls)
	}
}

// The final attempt no longer sleeps a backoff before giving up.
func TestPoolNoBackoffAfterFinalAttempt(t *testing.T) {
	busy := llm.NewOverloadedError("test", "busy")
	var backoffs atomic.Int32
	cfg := llm.Config{
		Provider:    "test",
		RetryPolicy: &llm.RetryPolicy{BaseDelay: 100 * time.Millisecond, MaxDelay: 100 * time.Millisecond, Factor: 1},
		RetryHook: func(e llm.RetryEvent) {
			if e.Type == llm.RetryBackingOff {
				backoffs.Add(1)
			}
		},
	}
	inner := &scriptedClient{errs: []error{busy, busy, busy, busy}}
	pc := newScriptedPool(t, cfg, []string{"k"}, inner)
	start := time.Now()
	_, err := pc.Complete(context.Background(), llm.Request{})
	if err == nil || inner.calls != 3 || backoffs.Load() != 2 {
		t.Fatalf("err=%v calls=%d backoffs=%d, want 3 calls / 2 backoffs", err, inner.calls, backoffs.Load())
	}
	if elapsed := time.Since(start); elapsed > 290*time.Millisecond {
		t.Fatalf("elapsed %v: slept after the last attempt", elapsed)
	}
}

// Multi-key rotation is preserved: with every key cooling down, the pool
// waits for the soonest key (the cooldown), not the RetryPolicy delay.
func TestPoolMultiKeyAllCoolingWaitsForCooldown(t *testing.T) {
	busy := llm.NewOverloadedError("test", "busy")
	inner := &scriptedClient{errs: []error{busy, busy}}
	var backoff time.Duration
	cfg := llm.Config{
		Provider: "test", CooldownOverload: 150 * time.Millisecond,
		RetryPolicy: &llm.RetryPolicy{BaseDelay: time.Hour, MaxDelay: time.Hour, Factor: 1},
		RetryHook: func(e llm.RetryEvent) {
			if e.Type == llm.RetryBackingOff {
				backoff = e.Backoff
			}
		},
	}
	pc := newScriptedPool(t, cfg, []string{"a", "b"}, inner)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	resp, err := pc.Complete(ctx, llm.Request{})
	if err != nil || resp.Content != "ok" {
		t.Fatalf("resp=%v err=%v (used the 1h policy delay instead of the key cooldown?)", resp, err)
	}
	if backoff <= 0 || backoff > 150*time.Millisecond {
		t.Fatalf("backoff = %v, want the ≤150ms soonest-key cooldown", backoff)
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
		{"openai images", openai("gpt-image-1"), func(c llm.ModalityClient) error {
			_, err := c.GenerateImages(context.Background(), llm.ImageRequest{Model: "gpt-image-1", Prompt: "x"})
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
