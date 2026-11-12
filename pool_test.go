package llm_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider"
)

// A pool driven by a client that randomly fails or is cancelled, hammered
// concurrently: must stay race-free, never panic, and never permanently
// disable the only key for a non-auth failure.
func TestPoolConcurrentFailuresAndCancellationRaceFree(t *testing.T) {
	var seq int64
	mu := &sync.Mutex{}
	next := func() int64 { mu.Lock(); defer mu.Unlock(); seq++; return seq }

	cfg := llm.DefaultConfig()
	cfg.CircuitThreshold = 0 // isolate pool/key behavior from breaker fail-fast
	// Tiny backoff + cooldowns so a retry storm doesn't sleep on real 60s
	// rate-limit cooldowns — we're testing concurrency, not wall-clock backoff.
	cfg.RetryPolicy = &llm.RetryPolicy{BaseDelay: time.Millisecond, MaxDelay: 2 * time.Millisecond, Factor: 2, Jitter: 0}
	cfg.CooldownRateLimit = time.Millisecond
	cfg.CooldownOverload = time.Millisecond
	cfg.CooldownDefault = time.Millisecond
	pc, err := llm.NewPooledClient(cfg, []string{"key-a"}, func(llm.AuthProfile) (llm.Client, error) {
		m := llm.NewMockClient("anthropic", "m")
		m.SetResponseFunc(func(llm.Request) (*llm.Response, error) {
			switch next() % 3 {
			case 0:
				return nil, llm.NewOverloadedError("anthropic", "503")
			case 1:
				return nil, llm.NewRateLimitError("anthropic", "429")
			default:
				return &llm.Response{Content: "ok", StopReason: llm.StopEndTurn}, nil
			}
		})
		return m, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient: %v", err)
	}
	defer pc.Close()

	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			ctx := context.Background()
			if n%2 == 0 {
				c, cancel := context.WithCancel(ctx)
				cancel() // pre-cancelled: must not poison the key
				ctx = c
			}
			_, _ = pc.Complete(ctx, llm.Request{})
		}(i)
	}
	wg.Wait()

	// After the storm, a transient/cancellation-only history must leave the key
	// usable — a temporary cooldown is allowed, a permanent disable
	// ("unhealthy", reserved for auth errors) is not. The profile is named
	// "<provider>-<n>" (e.g. anthropic-1), not by the key value.
	status := pc.PoolStatus()
	if status == "" {
		t.Fatal("empty pool status")
	}
	if strings.Contains(status, "unhealthy") {
		t.Fatalf("key permanently disabled by transient/cancelled failures: %s", status)
	}
}

func TestAuthPoolKeyBaseURLSplit(t *testing.T) {
	cases := []struct{ key, wantAPI, wantBase string }{
		{"key", "key", ""},
		{"key|https://base/v1", "key", "https://base/v1"},
		{"key|", "key", ""},
		{"|base", "", "base"}, // empty api (adapter New rejects it downstream)
		{"a|b|c", "a", "b|c"}, // split on FIRST '|' only
	}
	for _, c := range cases {
		pool := llm.NewAuthPool("p", []string{c.key})
		prof, ok := pool.GetCurrent()
		if !ok {
			t.Fatalf("%q: GetCurrent returned false", c.key)
		}
		if prof.APIKey != c.wantAPI || prof.BaseURL != c.wantBase {
			t.Errorf("%q -> APIKey=%q BaseURL=%q, want %q / %q", c.key, prof.APIKey, prof.BaseURL, c.wantAPI, c.wantBase)
		}
	}
}

// TestAuthProfileAvailability verifies auth profile state transitions.
func TestAuthProfileAvailability(t *testing.T) {
	p := &llm.AuthProfile{
		Name:      "test",
		APIKey:    "key",
		IsHealthy: true,
	}

	if !p.IsAvailable() {
		t.Error("Fresh profile should be available")
	}

	p.MarkUsed()
	if !p.IsAvailable() {
		t.Error("Used profile should still be available")
	}

	p.MarkFailed(fmt.Errorf("rate limit"), 1*time.Minute)
	if p.IsAvailable() {
		t.Error("Failed profile should not be available during cooldown")
	}

	p.MarkHealthy()
	if !p.IsAvailable() {
		t.Error("Healthy profile should be available")
	}
}

// TestPooledClientWithMultipleKeys tests auth pool rotation with multiple API keys.
func TestPooledClientWithMultipleKeys(t *testing.T) {
	keys := envKeys("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEYS")
	if testing.Short() {
		t.Skip("integration test: skipped in -short mode")
	}
	if len(keys) < 2 {
		t.Skip("ANTHROPIC_API_KEYS not set or has fewer than 2 keys")
	}

	cfg := llm.Config{
		Provider:    "anthropic",
		Model:       "claude-haiku-4-5-20251001",
		MaxTokens:   100,
		Temperature: floatPtr(0.0),
		Timeout:     30 * time.Second,
	}

	client, err := llm.NewClientWithKeys(cfg, keys)
	if err != nil {
		t.Fatalf("Failed to create pooled client: %v", err)
	}
	defer client.Close()

	t.Logf("Created pooled client with %d keys", len(keys))

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req := llm.Request{
		Messages:    []llm.Message{llm.NewTextMessage("user", TestPrompt)},
		MaxTokens:   100,
		Temperature: floatPtr(0.0),
	}

	resp, err := client.Complete(ctx, req)
	if err != nil {
		t.Fatalf("Pooled completion failed: %v", err)
	}

	if resp.Content == "" {
		t.Error("Empty response content")
	}

	t.Logf("Pooled response: %s (tokens: in=%d, out=%d)", resp.Content, resp.InputTokens, resp.OutputTokens)
}

// TestAuthPoolCreation verifies pool construction and initial state.
func TestAuthPoolCreation(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a", "key-b", "key-c"})

	if pool.Count() != 3 {
		t.Fatalf("Count() = %d, want 3", pool.Count())
	}

	current, ok := pool.GetCurrent()
	if !ok {
		t.Fatal("GetCurrent() returned false for non-empty pool")
	}
	if current.APIKey != "key-a" {
		t.Errorf("GetCurrent().APIKey = %q, want key-a", current.APIKey)
	}
	if current.Name != "test-1" {
		t.Errorf("GetCurrent().Name = %q, want test-1", current.Name)
	}
}

// TestAuthPoolEmptyPool verifies behavior with zero profiles.
func TestAuthPoolEmptyPool(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{})

	if pool.Count() != 0 {
		t.Fatalf("Count() = %d, want 0", pool.Count())
	}
	if _, ok := pool.GetCurrent(); ok {
		t.Error("GetCurrent() should return false for empty pool")
	}

	_, err := pool.GetAvailable()
	if err == nil {
		t.Fatal("GetAvailable() should error for empty pool")
	}
}

// TestAuthPoolGetAvailableRotation verifies rotation when current profile is in cooldown.
func TestAuthPoolGetAvailableRotation(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a", "key-b", "key-c"})

	// Get first profile
	p1, err := pool.GetAvailable()
	if err != nil {
		t.Fatalf("GetAvailable() error: %v", err)
	}
	if p1.APIKey != "key-a" {
		t.Errorf("first profile = %q, want key-a", p1.APIKey)
	}

	// Put first profile in cooldown
	pool.MarkFailedByName(p1.Name, llm.NewRateLimitError("test", "rate limited"))

	// Should rotate to second
	p2, err := pool.GetAvailable()
	if err != nil {
		t.Fatalf("GetAvailable() after rotation error: %v", err)
	}
	if p2.APIKey != "key-b" {
		t.Errorf("rotated profile = %q, want key-b", p2.APIKey)
	}
}

// TestAuthPoolAllInCooldown verifies error when all profiles are exhausted.
func TestAuthPoolAllInCooldown(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a", "key-b"})

	// Exhaust both profiles
	p1, _ := pool.GetAvailable()
	pool.MarkFailedByName(p1.Name, llm.NewRateLimitError("test", "rate limited"))
	p2, _ := pool.GetAvailable()
	pool.MarkFailedByName(p2.Name, llm.NewRateLimitError("test", "rate limited"))

	_, err := pool.GetAvailable()
	if err == nil {
		t.Fatal("GetAvailable() should error when all in cooldown")
	}
	if !strings.Contains(err.Error(), "no available auth profiles") {
		t.Errorf("error = %q, want 'no available auth profiles'", err.Error())
	}
}

// TestAuthPoolMarkSuccess verifies recovery after failure.
func TestAuthPoolMarkSuccess(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a"})

	// Fail the profile
	p1, _ := pool.GetAvailable()
	pool.MarkFailedByName(p1.Name, llm.NewOverloadedError("test", "overloaded"))

	// Should be in cooldown
	_, err := pool.GetAvailable()
	if err == nil {
		t.Fatal("profile should be in cooldown")
	}

	// Mark success resets the profile
	pool.MarkSuccessByName(p1.Name)
	p, err := pool.GetAvailable()
	if err != nil {
		t.Fatalf("after MarkSuccess, GetAvailable() error: %v", err)
	}
	if p.APIKey != "key-a" {
		t.Errorf("recovered profile = %q, want key-a", p.APIKey)
	}
}

// TestAuthPoolStatus verifies the status string format.
func TestAuthPoolStatus(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a", "key-b"})

	status := pool.Status()
	if !strings.Contains(status, "*test-1:ok") {
		t.Errorf("status = %q, want current marker on test-1", status)
	}
	if !strings.Contains(status, "test-2:ok") {
		t.Errorf("status = %q, want test-2:ok", status)
	}
}

func (m *failingMockClient) Complete(_ context.Context, _ llm.Request) (*llm.Response, error) {
	c := m.nextCall()
	return c.resp, c.err
}

func (m *failingMockClient) Stream(_ context.Context, _ llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		c := m.nextCall()
		if c.err != nil {
			yield(llm.StreamEvent{}, c.err)
			return
		}
		if c.midStreamErr != nil {
			// Yield one content event, then error (simulates mid-stream failure)
			if !yield(llm.StreamEvent{Type: llm.EventContent, Text: "partial"}, nil) {
				return
			}
			yield(llm.StreamEvent{}, c.midStreamErr)
			return
		}
		if !yield(llm.StreamEvent{Type: llm.EventContent, Text: "streamed"}, nil) {
			return
		}
		yield(llm.StreamEvent{Type: llm.EventDone, StopReason: "end_turn"}, nil)
	}
}

func (m *failingMockClient) nextCall() mockCall {
	if m.callIdx < len(m.calls) {
		c := m.calls[m.callIdx]
		m.callIdx++
		return c
	}
	return m.calls[len(m.calls)-1]
}

func (m *failingMockClient) Provider() string { return m.provider }

func (m *failingMockClient) Model() string { return m.model }

func (m *failingMockClient) Close() error { m.closed = true; return nil }

func (m *threadSafeMockClient) Complete(_ context.Context, _ llm.Request) (*llm.Response, error) {
	return m.resp, m.err
}

func (m *threadSafeMockClient) Stream(_ context.Context, _ llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		if m.err != nil {
			yield(llm.StreamEvent{}, m.err)
			return
		}
		if !yield(llm.StreamEvent{Type: llm.EventContent, Text: "streamed"}, nil) {
			return
		}
		yield(llm.StreamEvent{Type: llm.EventDone, StopReason: "end_turn"}, nil)
	}
}

func (m *threadSafeMockClient) Provider() string { return m.provider }

func (m *threadSafeMockClient) Model() string { return m.model }

func (m *threadSafeMockClient) Close() error { return nil }

// TestPooledClientCompleteSuccess verifies happy path.
func TestPooledClientCompleteSuccess(t *testing.T) {
	mock := &failingMockClient{
		calls:    []mockCall{{resp: &llm.Response{Content: "ok"}}},
		provider: "test",
		model:    "test-model",
	}

	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return mock, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	resp, err := pc.Complete(context.Background(), llm.Request{})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if resp.Content != "ok" {
		t.Errorf("Content = %q, want ok", resp.Content)
	}
}

// TestPooledClientCompleteRetryOnRetryableError verifies retry with rotation.
func TestPooledClientCompleteRetryOnRetryableError(t *testing.T) {
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		mock := &failingMockClient{
			provider: "test",
			model:    "test-model",
		}
		if callCount <= 1 {
			// First client: initial creation, will be used and fail
			mock.calls = []mockCall{{err: llm.NewRateLimitError("test", "rate limited")}}
		} else {
			// Second client: after rotation, succeeds
			mock.calls = []mockCall{{resp: &llm.Response{Content: "recovered"}}}
		}
		return mock, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	resp, err := pc.Complete(context.Background(), llm.Request{})
	if err != nil {
		t.Fatalf("Complete should recover after rotation: %v", err)
	}
	if resp.Content != "recovered" {
		t.Errorf("Content = %q, want recovered", resp.Content)
	}
}

// TestPooledClientCompleteAuthErrorRotates verifies auth errors rotate to try other keys.
func TestPooledClientCompleteAuthErrorRotates(t *testing.T) {
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		return &failingMockClient{
			calls:    []mockCall{{err: llm.NewAuthError("test", "bad key", 401)}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	_, err = pc.Complete(context.Background(), llm.Request{})
	if err == nil {
		t.Fatal("Complete should return error after exhausting all keys")
	}
	// Wrapped error should still match via errors.As
	if !llm.IsAuthError(err) {
		t.Errorf("expected auth error (wrapped), got: %v", err)
	}
	// Should have rotated to try both keys before giving up
	if callCount != 2 {
		t.Errorf("clientFunc called %d times, want 2 (auth errors should rotate)", callCount)
	}
}

// TestPooledClientCompleteNoRetryOnBadRequest verifies immediate return for bad request errors.
func TestPooledClientCompleteNoRetryOnBadRequest(t *testing.T) {
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		return &failingMockClient{
			calls:    []mockCall{{err: llm.NewAPIErrorFromStatus("test", 400, "invalid json")}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	_, err = pc.Complete(context.Background(), llm.Request{})
	if err == nil {
		t.Fatal("Complete should return error immediately for bad request")
	}
	// Should NOT have rotated (bad request is not key-related)
	if callCount != 1 {
		t.Errorf("clientFunc called %d times, want 1 (no rotation for bad request)", callCount)
	}
}

// TestPooledClientStreamPreDataRetry verifies streams retry on pre-data connection errors.
func TestPooledClientStreamPreDataRetry(t *testing.T) {
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		mock := &failingMockClient{
			provider: "test",
			model:    "test-model",
		}
		if callCount <= 1 {
			// First client: rate limited on connection
			mock.calls = []mockCall{{err: llm.NewRateLimitError("test", "rate limited")}}
		} else {
			// Second client (after rotation): succeeds
			mock.calls = []mockCall{{resp: &llm.Response{}}}
		}
		return mock, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	var chunks int
	for event, err := range pc.Stream(context.Background(), llm.Request{}) {
		if err != nil {
			t.Fatalf("Stream should recover after rotation: %v", err)
		}
		if event.Type == llm.EventContent {
			chunks++
		}
	}
	if chunks != 1 {
		t.Errorf("chunks = %d, want 1", chunks)
	}
	// Should have created a second client (rotation on pre-data error)
	if callCount < 2 {
		t.Errorf("clientFunc called %d times, want >= 2 (should rotate on pre-data error)", callCount)
	}
}

// TestPooledClientStreamMidStreamNoRetry verifies streams are NOT retried after data has been yielded.
func TestPooledClientStreamMidStreamNoRetry(t *testing.T) {
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		return &failingMockClient{
			calls:    []mockCall{{midStreamErr: llm.NewRateLimitError("test", "rate limited")}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	var gotContent bool
	var gotErr bool
	for event, err := range pc.Stream(context.Background(), llm.Request{}) {
		if err != nil {
			gotErr = true
			if !llm.IsRateLimited(err) {
				t.Errorf("expected rate limit error, got: %v", err)
			}
			break
		}
		if event.Type == llm.EventContent {
			gotContent = true
		}
	}
	if !gotContent {
		t.Error("expected at least one content event before error")
	}
	if !gotErr {
		t.Error("expected mid-stream error")
	}
	// Should NOT have rotated (mid-stream error, data already yielded)
	if callCount != 1 {
		t.Errorf("clientFunc called %d times, want 1 (no retry after data yielded)", callCount)
	}
}

// TestPooledClientStreamAuthErrorRotates verifies auth errors rotate to try other keys.
func TestPooledClientStreamAuthErrorRotates(t *testing.T) {
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		return &failingMockClient{
			calls:    []mockCall{{err: llm.NewAuthError("test", "bad key", 401)}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	for _, err := range pc.Stream(context.Background(), llm.Request{}) {
		if err != nil {
			// Wrapped error should still match via errors.As
			if !llm.IsAuthError(err) {
				t.Errorf("expected auth error (wrapped), got: %v", err)
			}
			break
		}
		t.Fatal("Stream should not yield events")
	}
	// Should have rotated to try both keys before giving up
	if callCount != 2 {
		t.Errorf("clientFunc called %d times, want 2 (auth errors should rotate)", callCount)
	}
}

// TestPooledClientStreamBadRequestNoRetry verifies bad request errors are not retried.
func TestPooledClientStreamBadRequestNoRetry(t *testing.T) {
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		return &failingMockClient{
			calls:    []mockCall{{err: llm.NewAPIErrorFromStatus("test", 400, "invalid json")}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	for _, err := range pc.Stream(context.Background(), llm.Request{}) {
		if err != nil {
			break
		}
		t.Fatal("Stream should not yield events")
	}
	// Should NOT have rotated (bad request is not key-related)
	if callCount != 1 {
		t.Errorf("clientFunc called %d times, want 1 (no retry for bad request)", callCount)
	}
}

// TestPooledClientStreamSuccess verifies happy path streaming.
func TestPooledClientStreamSuccess(t *testing.T) {
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &failingMockClient{
			calls:    []mockCall{{resp: &llm.Response{}}}, // Stream uses nil err path
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	var chunks int
	for event, err := range pc.Stream(context.Background(), llm.Request{}) {
		if err != nil {
			t.Fatalf("Stream error: %v", err)
		}
		if event.Type == llm.EventContent {
			chunks++
		}
	}
	if chunks != 1 {
		t.Errorf("chunks = %d, want 1", chunks)
	}
}

// TestPooledClientContextCancellation verifies that a single-key pool returns immediately
// when the key is in cooldown — does not retry with the same profile.
func TestPooledClientContextCancellation(t *testing.T) {
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &failingMockClient{
			calls:    []mockCall{{err: llm.NewOverloadedError("test", "overloaded")}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, err = pc.Complete(ctx, llm.Request{})
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("Complete should error on cancelled context")
	}
	// Should return immediately — not retry with the same cooled-down profile
	if elapsed > 500*time.Millisecond {
		t.Errorf("elapsed = %v, expected < 500ms (should not retry with same profile)", elapsed)
	}
}

// TestPooledClientNoKeys verifies error on empty keys.
func TestPooledClientNoKeys(t *testing.T) {
	_, err := llm.NewPooledClient(llm.DefaultConfig(), []string{}, func(profile llm.AuthProfile) (llm.Client, error) {
		return nil, nil
	})
	if err == nil {
		t.Fatal("NewPooledClient should error with no keys")
	}
}

// TestPooledClientProviderModel verifies delegation.
func TestPooledClientProviderModel(t *testing.T) {
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &failingMockClient{
			calls:    []mockCall{{resp: &llm.Response{}}},
			provider: "my-provider",
			model:    "my-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	if pc.Provider() != "my-provider" {
		t.Errorf("Provider() = %q, want my-provider", pc.Provider())
	}
	if pc.Model() != "my-model" {
		t.Errorf("Model() = %q, want my-model", pc.Model())
	}
}

// TestPooledClientPoolStatus verifies status reporting.
func TestPooledClientPoolStatus(t *testing.T) {
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &failingMockClient{
			calls:    []mockCall{{resp: &llm.Response{}}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	status := pc.PoolStatus()
	if !strings.Contains(status, "ok") {
		t.Errorf("PoolStatus() = %q, want to contain 'ok'", status)
	}
}

// TestCooldownErrorUnwrap verifies error unwrapping.
func TestCooldownErrorUnwrap(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a"})

	// Put profile in cooldown
	p, _ := pool.GetAvailable()
	pool.MarkFailedByName(p.Name, llm.NewRateLimitError("test", "rate limited"))

	// GetAvailable should return CooldownError
	_, err := pool.GetAvailable()
	if err == nil {
		t.Fatal("expected error")
	}

	// Should unwrap to ErrNoAvailableProfiles
	if !strings.Contains(err.Error(), "no available auth profiles") {
		t.Errorf("error = %q, want to contain 'no available auth profiles'", err.Error())
	}
}

// TestAuthPoolWithBaseURL verifies the API_KEY|BASE_URL format.
func TestAuthPoolWithBaseURL(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{
		"key-only",
		"key-with-url|https://custom.example.com/v1",
	})

	if pool.Count() != 2 {
		t.Fatalf("Count() = %d, want 2", pool.Count())
	}

	// First profile: no BaseURL
	p1, _ := pool.GetAvailable()
	if p1.APIKey != "key-only" {
		t.Errorf("first APIKey = %q, want key-only", p1.APIKey)
	}
	if p1.BaseURL != "" {
		t.Errorf("first BaseURL = %q, want empty", p1.BaseURL)
	}

	// Put first in cooldown to get second
	pool.MarkFailedByName(p1.Name, llm.NewRateLimitError("test", "rate limited"))

	// Second profile: has BaseURL
	p2, _ := pool.GetAvailable()
	if p2.APIKey != "key-with-url" {
		t.Errorf("second APIKey = %q, want key-with-url", p2.APIKey)
	}
	if p2.BaseURL != "https://custom.example.com/v1" {
		t.Errorf("second BaseURL = %q, want https://custom.example.com/v1", p2.BaseURL)
	}
}

// TestAuthPoolAuthErrorPermanentDisable verifies auth errors permanently disable profiles.
func TestAuthPoolAuthErrorPermanentDisable(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a"})

	p, _ := pool.GetAvailable()
	pool.MarkFailedByName(p.Name, llm.NewAuthError("test", "invalid key", 401))

	// Profile should be permanently disabled, not just in cooldown
	_, err := pool.GetAvailable()
	if err == nil {
		t.Fatal("expected error after auth failure")
	}
	// Should match ErrNoAvailableProfiles (permanently disabled, not cooldown)
	if !errors.Is(err, llm.ErrNoAvailableProfiles) {
		t.Errorf("error = %q, want errors.Is(ErrNoAvailableProfiles)", err.Error())
	}
	if !strings.Contains(err.Error(), "permanently disabled") {
		t.Errorf("error = %q, want to contain 'permanently disabled'", err.Error())
	}
}

// TestCooldownErrorUnwrapActual verifies errors.Is with CooldownError.
func TestCooldownErrorUnwrapActual(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a"})

	// Put profile in cooldown
	p, _ := pool.GetAvailable()
	pool.MarkFailedByName(p.Name, llm.NewRateLimitError("test", "rate limited"))

	// GetAvailable should return CooldownError
	_, err := pool.GetAvailable()
	if err == nil {
		t.Fatal("expected error")
	}

	// errors.Is should match ErrNoAvailableProfiles via Unwrap
	if !errors.Is(err, llm.ErrNoAvailableProfiles) {
		t.Errorf("errors.Is(err, ErrNoAvailableProfiles) = false, want true")
	}

	// String should contain the message
	if !strings.Contains(err.Error(), "no available auth profiles") {
		t.Errorf("error = %q, want to contain 'no available auth profiles'", err.Error())
	}
}

// TestPooledClientCloseNilClient verifies Close handles nil client.
func TestPooledClientCloseNilClient(t *testing.T) {
	// This tests the case where client might be nil, though in practice
	// NewPooledClient always initializes a client
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &mockClient{provider: "test", model: "test"}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}

	// First close should work
	if err := pc.Close(); err != nil {
		t.Errorf("Close() error: %v", err)
	}
}

// TestPooledClientRotateClientAlreadyRotated verifies short-circuit when already rotated.
func TestPooledClientRotateClientAlreadyRotated(t *testing.T) {
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		mock := &failingMockClient{
			provider: "test",
			model:    "test-model",
		}
		if callCount <= 1 {
			// First client: rate limited
			mock.calls = []mockCall{{err: llm.NewRateLimitError("test", "rate limited")}}
		} else {
			// Second client: succeeds
			mock.calls = []mockCall{{resp: &llm.Response{Content: "ok"}}}
		}
		return mock, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	// First call rotates
	resp, err := pc.Complete(context.Background(), llm.Request{})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if resp.Content != "ok" {
		t.Errorf("Content = %q, want ok", resp.Content)
	}
}

// TestPooledClientStreamCancellation verifies stream respects context cancellation during backoff.
func TestPooledClientStreamCancellation(t *testing.T) {
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &failingMockClient{
			calls:    []mockCall{{err: llm.NewOverloadedError("test", "overloaded")}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	start := time.Now()
	for _, err := range pc.Stream(ctx, llm.Request{}) {
		if err != nil {
			break
		}
	}
	elapsed := time.Since(start)

	// Should return quickly due to context cancellation
	if elapsed > 500*time.Millisecond {
		t.Errorf("elapsed = %v, expected < 500ms", elapsed)
	}
}

// TestAuthPoolStatusUnhealthy verifies status shows unhealthy profiles.
func TestAuthPoolStatusUnhealthy(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a", "key-b"})

	// Mark first profile as permanently unhealthy (auth error)
	p, _ := pool.GetAvailable()
	pool.MarkFailedByName(p.Name, llm.NewAuthError("test", "bad key", 401))

	status := pool.Status()
	if !strings.Contains(status, "unhealthy") {
		t.Errorf("status = %q, want to contain 'unhealthy'", status)
	}
}

// TestNewClientWithKeysEmpty verifies NewClientWithKeys with no keys still
// builds a working client from the config (R-L1: it now routes through a
// PooledClient with a single config-derived profile rather than a bare single
// client, so it gets the same retry/breaker wrapping as a multi-key pool).
func TestNewClientWithKeysEmpty(t *testing.T) {
	cfg := llm.Config{
		Provider: "ollama",
		Model:    "llama3",
	}

	// Empty keys fall back to the config's own credentials, wrapped in a pool.
	client, err := llm.NewClientWithKeys(cfg, []string{})
	if err != nil {
		t.Fatalf("NewClientWithKeys error: %v", err)
	}
	defer client.Close()

	if client.Provider() != "ollama" {
		t.Errorf("Provider() = %q, want ollama", client.Provider())
	}
}

// TestMarkFailedByNameNotFound verifies MarkFailedByName with unknown profile name.
func TestMarkFailedByNameNotFound(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a"})

	// This should not panic - just silently return
	pool.MarkFailedByName("nonexistent-profile", llm.NewRateLimitError("test", "rate limited"))

	// Pool should still be functional
	p, err := pool.GetAvailable()
	if err != nil {
		t.Fatalf("GetAvailable() error: %v", err)
	}
	if p.APIKey != "key-a" {
		t.Errorf("APIKey = %q, want key-a", p.APIKey)
	}
}

// TestAuthPoolStatusCooldown verifies status shows cooldown duration.
func TestAuthPoolStatusCooldown(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a"})

	// Put profile in temporary cooldown (not auth error)
	p, _ := pool.GetAvailable()
	pool.MarkFailedByName(p.Name, llm.NewRateLimitError("test", "rate limited"))

	status := pool.Status()
	// Should contain "cooldown" with a duration, not "unhealthy"
	if !strings.Contains(status, "cooldown") {
		t.Errorf("status = %q, want to contain 'cooldown'", status)
	}
}

// TestNewPooledClientClientFuncError verifies error when clientFunc fails on init.
func TestNewPooledClientClientFuncError(t *testing.T) {
	_, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return nil, fmt.Errorf("client creation failed")
	})
	if err == nil {
		t.Fatal("NewPooledClient should error when clientFunc fails")
	}
	if !strings.Contains(err.Error(), "client creation failed") {
		t.Errorf("error = %q, want to contain 'client creation failed'", err.Error())
	}
}

// TestPooledClientRotateClientFuncError verifies error handling when clientFunc fails during rotation.
func TestPooledClientRotateClientFuncError(t *testing.T) {
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		if callCount == 1 {
			// First call (init): succeed but return failing client
			return &failingMockClient{
				calls:    []mockCall{{err: llm.NewRateLimitError("test", "rate limited")}},
				provider: "test",
				model:    "test-model",
			}, nil
		}
		// Subsequent calls (rotation): fail
		return nil, fmt.Errorf("rotation client creation failed")
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	// Complete should fail because rotation fails
	_, err = pc.Complete(context.Background(), llm.Request{})
	if err == nil {
		t.Fatal("Complete should error when rotation fails")
	}
}

// TestNewPooledClientWithBaseURL verifies per-profile BaseURL is used.
func TestNewPooledClientWithBaseURL(t *testing.T) {
	var capturedBaseURL string
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a|https://custom.example.com/v1"}, func(profile llm.AuthProfile) (llm.Client, error) {
		capturedBaseURL = profile.BaseURL
		return &mockClient{provider: "test", model: "test"}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	if capturedBaseURL != "https://custom.example.com/v1" {
		t.Errorf("BaseURL = %q, want https://custom.example.com/v1", capturedBaseURL)
	}
}

// TestPooledClientCompleteBackoffZeroCooldown verifies backoff when CooldownError has zero wait.
func TestPooledClientCompleteBackoffZeroCooldown(t *testing.T) {
	// Create a pool where all keys are in cooldown with very short duration
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		return &failingMockClient{
			calls:    []mockCall{{err: llm.NewRateLimitError("test", "rate limited")}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	// Should return context deadline error, not hang
	_, err = pc.Complete(ctx, llm.Request{})
	if err == nil {
		t.Fatal("Complete should error")
	}
}

// TestPooledClientStreamBackoffZeroCooldown verifies stream backoff with zero wait.
func TestPooledClientStreamBackoffZeroCooldown(t *testing.T) {
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &failingMockClient{
			calls:    []mockCall{{err: llm.NewRateLimitError("test", "rate limited")}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	// Should return context deadline error, not hang
	for _, err := range pc.Stream(ctx, llm.Request{}) {
		if err != nil {
			break
		}
	}
}

// TestPooledClientStreamAuthErrorNoHealthyKeys verifies stream auth error handling.
func TestPooledClientStreamAuthErrorNoHealthyKeys(t *testing.T) {
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &failingMockClient{
			calls:    []mockCall{{err: llm.NewAuthError("test", "bad key", 401)}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	for _, err := range pc.Stream(context.Background(), llm.Request{}) {
		if err != nil {
			if !llm.IsAuthError(err) {
				t.Errorf("expected auth error, got: %v", err)
			}
			return
		}
	}
	t.Fatal("expected error from stream")
}

// TestPooledClientCompleteAuthErrorNoHealthyKeys verifies Complete auth error with no healthy keys.
func TestPooledClientCompleteAuthErrorNoHealthyKeys(t *testing.T) {
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &failingMockClient{
			calls:    []mockCall{{err: llm.NewAuthError("test", "bad key", 401)}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	_, err = pc.Complete(context.Background(), llm.Request{})
	if err == nil {
		t.Fatal("Complete should error with auth error and no healthy keys")
	}
	if !llm.IsAuthError(err) {
		t.Errorf("expected auth error, got: %v", err)
	}
}

// TestMarkSuccessByNameNotFound verifies MarkSuccessByName with unknown profile name.
func TestMarkSuccessByNameNotFound(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a"})

	// This should not panic - just silently return
	pool.MarkSuccessByName("nonexistent-profile")

	// Pool should still be functional
	p, err := pool.GetAvailable()
	if err != nil {
		t.Fatalf("GetAvailable() error: %v", err)
	}
	if p.APIKey != "key-a" {
		t.Errorf("APIKey = %q, want key-a", p.APIKey)
	}
}

// TestPooledClientStreamRetryWithNewProfile verifies the "retrying with new profile" log path.
func TestPooledClientStreamRetryWithNewProfile(t *testing.T) {
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		mock := &failingMockClient{
			provider: "test",
			model:    "test-model",
		}
		if callCount <= 1 {
			// First client: rate limited
			mock.calls = []mockCall{{err: llm.NewRateLimitError("test", "rate limited")}}
		} else {
			// Second client: succeeds
			mock.calls = []mockCall{{resp: &llm.Response{}}}
		}
		return mock, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	var gotContent bool
	for event, err := range pc.Stream(context.Background(), llm.Request{}) {
		if err != nil {
			t.Fatalf("Stream error: %v", err)
		}
		if event.Type == llm.EventContent {
			gotContent = true
		}
	}
	if !gotContent {
		t.Error("expected content event")
	}
	if callCount != 2 {
		t.Errorf("callCount = %d, want 2", callCount)
	}
}

// TestPooledClientCompleteRetryWithNewProfile verifies the "retrying with new profile" log path.
func TestPooledClientCompleteRetryWithNewProfile(t *testing.T) {
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		mock := &failingMockClient{
			provider: "test",
			model:    "test-model",
		}
		if callCount <= 1 {
			// First client: overloaded (not auth, so rotation should help)
			mock.calls = []mockCall{{err: llm.NewOverloadedError("test", "overloaded")}}
		} else {
			// Second client: succeeds
			mock.calls = []mockCall{{resp: &llm.Response{Content: "recovered"}}}
		}
		return mock, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	resp, err := pc.Complete(context.Background(), llm.Request{})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if resp.Content != "recovered" {
		t.Errorf("Content = %q, want recovered", resp.Content)
	}
	if callCount != 2 {
		t.Errorf("callCount = %d, want 2", callCount)
	}
}

// TestMarkFailedByNameGenericError verifies cooldown for non-rate-limit, non-overload errors.
func TestMarkFailedByNameGenericError(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a"})

	p, _ := pool.GetAvailable()
	// Use a generic API error (not rate limit, not overload, not auth)
	genericErr := llm.NewAPIErrorFromStatus("test", 500, "internal server error")
	pool.MarkFailedByName(p.Name, genericErr)

	// Profile should be in cooldown but still healthy
	_, err := pool.GetAvailable()
	if err == nil {
		t.Fatal("expected error after generic failure")
	}
	// Should be cooldown, not permanent
	if !strings.Contains(err.Error(), "cooldown") && !strings.Contains(err.Error(), "available") {
		t.Errorf("error = %q, want cooldown error", err.Error())
	}
}

// TestPooledClientStreamMidStreamAuthError verifies mid-stream auth error handling.
func TestPooledClientStreamMidStreamAuthError(t *testing.T) {
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		return &failingMockClient{
			calls:    []mockCall{{midStreamErr: llm.NewAuthError("test", "bad key", 401)}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	var gotContent bool
	var gotErr bool
	for event, err := range pc.Stream(context.Background(), llm.Request{}) {
		if err != nil {
			gotErr = true
			if !llm.IsAuthError(err) {
				t.Errorf("expected auth error, got: %v", err)
			}
			break
		}
		if event.Type == llm.EventContent {
			gotContent = true
		}
	}
	if !gotContent {
		t.Error("expected at least one content event before error")
	}
	if !gotErr {
		t.Error("expected mid-stream error")
	}
	// Mid-stream errors should not retry
	if callCount != 1 {
		t.Errorf("callCount = %d, want 1 (mid-stream should not retry)", callCount)
	}
}

// TestPooledClientStreamAllRetriesExhausted verifies stream error after all retries.
func TestPooledClientStreamAllRetriesExhausted(t *testing.T) {
	// Use 5 keys so maxRetries = 5. Each rotation succeeds but the new client also fails.
	// This way we never enter the backoff path (rotation always succeeds).
	// After 5 iterations, the loop ends and we get "all retries exhausted".
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b", "key-c", "key-d", "key-e"}, func(profile llm.AuthProfile) (llm.Client, error) {
		// All clients fail with overloaded error
		return &threadSafeMockClient{
			provider: "test",
			model:    "test-model",
			err:      llm.NewOverloadedError("test", "overloaded"),
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	var lastErr error
	for _, err := range pc.Stream(context.Background(), llm.Request{}) {
		if err != nil {
			lastErr = err
			break
		}
	}
	if lastErr == nil {
		t.Fatal("expected error after all retries exhausted")
	}
	// Should contain "all retries exhausted"
	if !strings.Contains(lastErr.Error(), "all retries exhausted") {
		t.Errorf("error = %q, want to contain 'all retries exhausted'", lastErr.Error())
	}
}

// TestPooledClientStreamEarlyBreakSuccess verifies early break marks success.
func TestPooledClientStreamEarlyBreakSuccess(t *testing.T) {
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &failingMockClient{
			calls:    []mockCall{{resp: &llm.Response{}}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	// Break early after first content
	for event, err := range pc.Stream(context.Background(), llm.Request{}) {
		if err != nil {
			t.Fatalf("Stream error: %v", err)
		}
		if event.Type == llm.EventContent {
			break // Early break
		}
	}
	// Should not panic, profile should be marked success
}

// TestPooledClientStreamMidStreamNonRetryable verifies mid-stream non-retryable non-auth error.
func TestPooledClientStreamMidStreamNonRetryable(t *testing.T) {
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &failingMockClient{
			calls:    []mockCall{{midStreamErr: llm.NewAPIErrorFromStatus("test", 400, "bad request")}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	var gotErr bool
	for _, err := range pc.Stream(context.Background(), llm.Request{}) {
		if err != nil {
			gotErr = true
			break
		}
	}
	if !gotErr {
		t.Error("expected mid-stream error")
	}
}

// TestNewClientWithKeysAndBaseURL verifies per-profile BaseURL through NewClientWithKeys.
func TestNewClientWithKeysAndBaseURL(t *testing.T) {
	cfg := llm.Config{
		Provider: "custom",
		Model:    "test-model",
	}

	// Keys with custom BaseURL
	keys := []string{"key-a|http://localhost:9999/v1"}

	client, err := llm.NewClientWithKeys(cfg, keys)
	if err != nil {
		t.Fatalf("NewClientWithKeys error: %v", err)
	}
	defer client.Close()

	// Should not panic and create a client
	if client.Provider() != "custom" {
		t.Errorf("Provider() = %q, want custom", client.Provider())
	}
}

// TestPooledClientConcurrentRotation verifies thundering herd prevention.
func TestPooledClientConcurrentRotation(t *testing.T) {
	// This test forcefully synchronizes multiple goroutines to ensure they all
	// hit the rate-limit failure at the exact same millisecond, brutally
	// testing the double-checked locking mechanism in rotateClient.
	var clientCreationCount int32
	var mu sync.Mutex

	// We use a gate to hold all goroutines inside their Complete() call
	// until they have all arrived, then we unleash them simultaneously.
	var unleash sync.WaitGroup
	var arrive sync.WaitGroup

	const numConcurrent = 50
	unleash.Add(1)
	arrive.Add(numConcurrent)

	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		mu.Lock()
		clientCreationCount++
		count := clientCreationCount
		mu.Unlock()

		if count <= 1 {
			// First client: forces all callers to synchronize before failing
			return &mockClient{
				provider: "test",
				model:    "test-model",
				completeFunc: func(ctx context.Context, req llm.Request) (*llm.Response, error) {
					arrive.Done()  // Signal that we are ready
					unleash.Wait() // Block until the gate is opened
					return nil, llm.NewRateLimitError("test", "429 rate limited")
				},
			}, nil
		}
		// After rotation, succeed on every call
		return &threadSafeMockClient{
			provider: "test",
			model:    "test-model",
			resp:     &llm.Response{Content: "ok"},
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	var wg sync.WaitGroup
	errChan := make(chan error, numConcurrent)

	for i := 0; i < numConcurrent; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// This complete call will block inside the mock until we open the gate
			_, err := pc.Complete(context.Background(), llm.Request{})
			if err != nil {
				errChan <- err
			}
		}()
	}

	// Wait for all 50 goroutines to enter the Complete method and block
	arrive.Wait()

	// UNLEASH THE HERD
	unleash.Done()

	// Wait for them all to finish resolving the failover
	wg.Wait()
	close(errChan)

	mu.Lock()
	finalCount := clientCreationCount
	mu.Unlock()

	// 1 Initial Client + 1 Failover Client = EXACTLY 2.
	// If it is > 2, the double-checked lock failed and the herd broke through.
	if finalCount != 2 {
		t.Fatalf("Thundering herd protection failed! Expected exactly 2 clients created, got %d", finalCount)
	}
}

// TestNewPooledClientGetAvailableError verifies error when initial GetAvailable fails.
func TestNewPooledClientGetAvailableError(t *testing.T) {
	// Create pool where all profiles are immediately unhealthy
	// This is hard to test directly without modifying internal state,
	// so we'll test the zero-keys case which returns ErrNoAvailableProfiles
	_, err := llm.NewPooledClient(llm.DefaultConfig(), []string{}, func(profile llm.AuthProfile) (llm.Client, error) {
		return nil, nil
	})
	if err == nil {
		t.Fatal("NewPooledClient should error with no keys")
	}
}

// TestLocalProviderRetryProtection verifies that local providers (Ollama, vLLM,
// LM Studio) created via NewClient get retry/backoff protection on transient
// errors, even though they have no API key.
func TestLocalProviderRetryProtection(t *testing.T) {
	providers := []string{"ollama", "vllm", "lmstudio"}

	for _, provider := range providers {
		t.Run(provider, func(t *testing.T) {
			cfg := llm.Config{
				Provider:  provider,
				Model:     "test-model",
				MaxTokens: 100,
				Timeout:   5 * time.Second,
			}

			client, err := llm.NewClient(cfg)
			if err != nil {
				t.Fatalf("NewClient error: %v", err)
			}
			defer client.Close()

			// PooledClient exposes PoolStatus(); a bare single client does not.
			// This proves NewClient wraps local providers in PooledClient for retry.
			type poolStatusReporter interface {
				PoolStatus() string
			}
			if _, ok := client.(poolStatusReporter); !ok {
				t.Errorf("NewClient(%s) returned %T, want *PooledClient (no retry protection)", provider, client)
			}
		})
	}
}

// TestLocalProviderRetryActualTransient verifies that a local provider client
// actually retries on transient HTTP errors (not just that it's wrapped).
func TestLocalProviderRetryActualTransient(t *testing.T) {
	// Server: fail once with 502 (retryable), then succeed
	attempts := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts == 1 {
			w.WriteHeader(http.StatusBadGateway)
			fmt.Fprint(w, `{"error":"bad gateway"}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"chatcmpl-1","object":"chat.completion","model":"llama3","choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`)
	}))
	defer srv.Close()

	cfg := llm.Config{
		Provider:          "ollama",
		Model:             "llama3",
		ModelCapabilities: llm.Capabilities(llm.CapabilityChat),
		BaseURL:           srv.URL,
		MaxTokens:         100,
		Timeout:           5 * time.Second,
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}
	defer client.Close()

	resp, err := client.Complete(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
	})
	if err != nil {
		t.Fatalf("Complete should succeed after retry, got: %v", err)
	}
	if resp.Content != "Hello" {
		t.Errorf("Content = %q, want Hello", resp.Content)
	}
	if attempts < 2 {
		t.Errorf("attempts = %d, want >= 2 (1 failure + 1 success)", attempts)
	}
}

// TestConfigurableCooldowns verifies that custom cooldown values are used.
func TestConfigurableCooldowns(t *testing.T) {
	if llm.DefaultCooldownRateLimit != 60*time.Second {
		t.Errorf("DefaultCooldownRateLimit = %v, want 60s", llm.DefaultCooldownRateLimit)
	}
	if llm.DefaultCooldownOverload != 30*time.Second {
		t.Errorf("DefaultCooldownOverload = %v, want 30s", llm.DefaultCooldownOverload)
	}
	if llm.DefaultCooldownDefault != 10*time.Second {
		t.Errorf("DefaultCooldownDefault = %v, want 10s", llm.DefaultCooldownDefault)
	}
	if llm.DefaultCircuitThreshold != 5 {
		t.Errorf("DefaultCircuitThreshold = %d, want 5", llm.DefaultCircuitThreshold)
	}
	if llm.DefaultCircuitCooldown != 30*time.Second {
		t.Errorf("DefaultCircuitCooldown = %v, want 30s", llm.DefaultCircuitCooldown)
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

// errorFirstClient fails before any content, after the provider reported
// usage: a retryable overloaded error carrying the billed input.
type errorFirstClient struct {
	fakeClient
	calls int
}

func (e *errorFirstClient) Stream(context.Context, llm.Request) iter.Seq2[llm.StreamEvent, error] {
	e.calls++
	return func(yield func(llm.StreamEvent, error) bool) {
		err := llm.NewAPIErrorFromStatus("anthropic", 529, "overloaded")
		yield(llm.StreamEvent{Type: llm.EventError, Error: err.Error(), InputTokens: 70}, err)
	}
}

func TestPoolDoesNotHideABilledAttempt(t *testing.T) {
	inner := &errorFirstClient{fakeClient: fakeClient{provider: "anthropic", model: "claude-opus-5-5"}}
	pooled, err := llm.NewPooledClient(llm.Config{Provider: "anthropic", Model: "claude-opus-5-5"},
		[]string{"k"}, func(llm.AuthProfile) (llm.Client, error) { return inner, nil })
	if err != nil {
		t.Fatal(err)
	}
	resp, err := llm.CollectStream(pooled.Stream(context.Background(), llm.Request{}))
	if err == nil || resp.InputTokens != 70 || inner.calls != 1 {
		t.Errorf("usage %d, calls %d, err %v", resp.InputTokens, inner.calls, err)
	}
}

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

func (c *scriptedClient) Model() string { return "m" }

func (c *scriptedClient) Close() error { return nil }

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

// A burst of 429s on one key is that key's quota, not a broken endpoint.
// Five concurrent 429s on key a (breaker threshold 5) opened the circuit for
// the whole pool although key b was healthy (B10).
func TestRateLimitBurstOnOneKeyDoesNotOpenTheBreaker(t *testing.T) {
	const n = 5
	for _, stream := range []bool{false, true} {
		var gate sync.WaitGroup
		gate.Add(n)
		fn := func(key string, _ llm.Request) (*llm.Response, error) {
			if key == "a" {
				gate.Done()
				gate.Wait() // all n requests are in flight on key a
				return nil, llm.NewRateLimitError("test", "slow down")
			}
			return &llm.Response{Content: "ok"}, nil
		}
		pc := keyPool(t, llm.Config{Provider: "test", CircuitThreshold: n, CircuitCooldown: time.Hour}, []string{"a", "b"}, fn)
		var wg sync.WaitGroup
		var failed atomic.Int32
		for i := 0; i < n; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if err := callPool(pc, stream); err != nil {
					failed.Add(1)
				}
			}()
		}
		wg.Wait()
		if err := callPool(pc, stream); failed.Load() > 0 || err != nil {
			t.Fatalf("stream=%v: healthy key b exists, yet %d burst requests failed and the next one: %v (%s)", stream, failed.Load(), err, pc.PoolStatus())
		}
	}
}

// With a single key there is nothing to rotate to: its 429s still count, so
// the breaker can stop a caller that keeps hammering a quota-exhausted key.
func TestRateLimitsOnTheOnlyKeyStillCount(t *testing.T) {
	var calls atomic.Int32
	fn := func(string, llm.Request) (*llm.Response, error) {
		calls.Add(1)
		return nil, llm.NewRateLimitError("test", "slow down")
	}
	cfg := llm.Config{Provider: "test", CircuitThreshold: 2, CircuitCooldown: time.Hour, MaxRetries: 1}
	pc := keyPool(t, cfg, []string{"a"}, fn)
	for i := 0; i < 5; i++ {
		_, _ = pc.Complete(context.Background(), llm.Request{})
	}
	if calls.Load() > 2 {
		t.Fatalf("%d calls reached the provider; the breaker (threshold 2) never opened", calls.Load())
	}
}

// After waiting for the soonest cooldown, the retry must go to the key the
// pool waited for. It went to the key that failed last and was still cooling:
// a (100 ms overload) → b (30 s rate limit) → wait 100 ms → b again (B13).
func TestRetryAfterCooldownUsesTheKeyThatRecovered(t *testing.T) {
	for _, stream := range []bool{false, true} {
		var mu sync.Mutex
		var order []string
		fn := func(key string, _ llm.Request) (*llm.Response, error) {
			mu.Lock()
			order = append(order, key)
			n := len(order)
			mu.Unlock()
			switch {
			case n == 1:
				return nil, llm.NewOverloadedError("test", "busy")
			case key == "b":
				return nil, llm.NewRateLimitError("test", "slow down")
			}
			return &llm.Response{Content: "ok from " + key}, nil
		}
		cfg := llm.Config{Provider: "test", CooldownOverload: 100 * time.Millisecond, CooldownRateLimit: 30 * time.Second}
		pc := keyPool(t, cfg, []string{"a", "b"}, fn)
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		err := callPoolCtx(ctx, pc, stream)
		cancel()
		if err != nil || strings.Join(order, ",") != "a,b,a" {
			t.Fatalf("stream=%v: order=%v err=%v, want a,b,a and success", stream, order, err)
		}
	}
}

// A 403 for one request (a model the organization may not use) permanently
// disabled every key the pool rotated through; they never recovered (B12).
func TestRequestScoped403DoesNotBurnThePool(t *testing.T) {
	for _, stream := range []bool{false, true} {
		var mu sync.Mutex
		var forbiddenCalls []string
		fn := func(key string, req llm.Request) (*llm.Response, error) {
			if req.Model == "model-without-access" {
				mu.Lock()
				forbiddenCalls = append(forbiddenCalls, key)
				mu.Unlock()
				return nil, llm.NewAPIErrorFromStatus("test", 403, `{"error":"model not available for your organization"}`)
			}
			return &llm.Response{Content: "ok"}, nil
		}
		pc := keyPool(t, llm.Config{Provider: "test"}, []string{"a", "b", "c"}, fn)
		if err := callPoolReq(context.Background(), pc, stream, llm.Request{Model: "model-without-access"}); !llm.IsAuthError(err) {
			t.Fatalf("stream=%v: err = %v, want the 403", stream, err)
		}
		if len(forbiddenCalls) != 2 {
			t.Fatalf("stream=%v: the forbidden request went to %v; want it stopped at the second key", stream, forbiddenCalls)
		}
		st := pc.PoolStatus()
		if strings.Contains(st, "unhealthy") {
			t.Fatalf("stream=%v: a 403 disabled keys permanently: %s", stream, st)
		}
		for i := 0; i < 3; i++ {
			if err := callPool(pc, stream); err != nil {
				t.Fatalf("stream=%v: a normal request failed after the 403: %v", stream, err)
			}
		}
	}
}

// A 401 still disables the key for good: the key is invalid.
func TestUnauthorizedStillDisablesTheKey(t *testing.T) {
	fn := func(key string, _ llm.Request) (*llm.Response, error) {
		if key == "a" {
			return nil, llm.NewAuthError("test", "invalid key", 401)
		}
		return &llm.Response{Content: "ok"}, nil
	}
	pc := keyPool(t, llm.Config{Provider: "test"}, []string{"a", "b"}, fn)
	if _, err := pc.Complete(context.Background(), llm.Request{}); err != nil {
		t.Fatal(err)
	}
	if st := pc.PoolStatus(); !strings.Contains(st, "test-1:unhealthy") {
		t.Fatalf("status = %s, want key a disabled", st)
	}
}

// DisableRetries turns off the retry, not the breaker: failures were never
// recorded, so a breaker with threshold 2 let all ten calls through (B14).
func TestDisableRetriesStillFeedsTheBreaker(t *testing.T) {
	for _, stream := range []bool{false, true} {
		var calls atomic.Int32
		fn := func(string, llm.Request) (*llm.Response, error) {
			calls.Add(1)
			return nil, llm.NewOverloadedError("test", "down")
		}
		cfg := llm.Config{Provider: "test", DisableRetries: true, CircuitThreshold: 2, CircuitCooldown: time.Hour}
		pc := keyPool(t, cfg, []string{"a"}, fn)
		for i := 0; i < 10; i++ {
			_ = callPool(pc, stream)
		}
		if calls.Load() != 2 {
			t.Fatalf("stream=%v: %d calls reached the provider, want 2 (breaker threshold 2, one attempt each)", stream, calls.Load())
		}
	}
}

// NewClient with DisableRetries bypassed the pool, so CircuitThreshold was
// silently ignored (B14).
func TestNewClientDisableRetriesKeepsTheBreaker(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()
	client, err := llm.NewClient(llm.Config{Provider: "openai", Model: "gpt-6.1-sol", APIKey: "k", BaseURL: srv.URL,
		DisableRetries: true, CircuitThreshold: 2, CircuitCooldown: time.Hour})
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	for i := 0; i < 5; i++ {
		_, _ = client.Complete(context.Background(), llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "x")}})
	}
	if calls.Load() != 2 {
		t.Fatalf("%d requests reached the server, want 2", calls.Load())
	}
}

func callPool(pc *llm.PooledClient, stream bool) error {
	return callPoolReq(context.Background(), pc, stream, llm.Request{})
}

func callPoolCtx(ctx context.Context, pc *llm.PooledClient, stream bool) error {
	return callPoolReq(ctx, pc, stream, llm.Request{})
}

func callPoolReq(ctx context.Context, pc *llm.PooledClient, stream bool, req llm.Request) error {
	if !stream {
		_, err := pc.Complete(ctx, req)
		return err
	}
	var last error
	for _, err := range pc.Stream(ctx, req) {
		if err != nil {
			last = err
		}
	}
	return last
}

// A quota 429 on one key used to end the call (non-retryable, no rotation)
// or, before P5, burn retries on the same key. Now the pool rests that key and
// answers from the next; with no other key it gives up after one attempt
// without opening the breaker.
func TestPoolRotatesPastExhaustedQuota(t *testing.T) {
	var hits sync.Map
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		n, _ := hits.LoadOrStore(key, new(atomic.Int32))
		n.(*atomic.Int32).Add(1)
		w.Header().Set("Content-Type", "application/json")
		if key == "key-a" {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"message":"You exceeded your current quota","type":"insufficient_quota","code":"insufficient_quota"}}`))
			return
		}
		body, _ := io.ReadAll(r.Body)
		if strings.Contains(string(body), `"stream":true`) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\ndata: [DONE]\n\n")
			return
		}
		_, _ = w.Write([]byte(`{"id":"x","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`))
	}))
	defer srv.Close()
	count := func(key string) int32 {
		n, ok := hits.Load(key)
		if !ok {
			return 0
		}
		return n.(*atomic.Int32).Load()
	}
	cfg := llm.Config{Provider: "openai", Model: "m", BaseURL: srv.URL, DisableProxy: true, Timeout: 5 * time.Second,
		RetryPolicy:       &llm.RetryPolicy{BaseDelay: time.Millisecond, MaxDelay: 2 * time.Millisecond, Factor: 2},
		ModelCapabilities: llm.Capabilities(llm.CapabilityChat, llm.CapabilityStream)}
	req := llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "q")}}

	two, err := llm.NewClientWithKeys(cfg, []string{"key-a", "key-b"})
	if err != nil {
		t.Fatal(err)
	}
	defer two.Close()
	for i := range 2 { // the second call must skip the resting key
		resp, err := two.Complete(context.Background(), req)
		if err != nil || resp.Content != "ok" {
			t.Fatalf("call %d: resp=%v err=%v", i, resp, err)
		}
	}
	if count("key-a") != 1 || count("key-b") != 2 {
		t.Errorf("hits a=%d b=%d, want 1/2", count("key-a"), count("key-b"))
	}
	// The exhausted key rests as long as a 403 (5m), not a throttle's cooldown.
	if pc, ok := two.(*llm.PooledClient); !ok || !strings.Contains(pc.PoolStatus(), "cooldown 5m0s") {
		t.Errorf("pool status = %v", two)
	}
	var gotErr error
	for _, e := range two.Stream(context.Background(), req) {
		if e != nil {
			gotErr = e
		}
	}
	if gotErr != nil || count("key-a") != 1 {
		t.Errorf("stream: err=%v hits a=%d", gotErr, count("key-a"))
	}

	hits = sync.Map{}
	one, err := llm.NewClientWithKeys(cfg, []string{"key-a"})
	if err != nil {
		t.Fatal(err)
	}
	defer one.Close()
	for range 3 {
		if _, err := one.Complete(context.Background(), req); !llm.IsQuotaExhausted(err) {
			t.Fatalf("single key: err = %v, want the quota error", err)
		}
	}
	// One attempt per call (the resting key is still the only one), and the
	// breaker stays closed: a fourth call still reaches the server.
	if got := count("key-a"); got != 3 {
		t.Errorf("single key: %d attempts over 3 calls, want 3", got)
	}
}
