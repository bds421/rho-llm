package llm_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider"
)

func TestCircuitBreakerStartsClosed(t *testing.T) {
	cb := llm.NewCircuitBreaker(3, 100*time.Millisecond)
	if cb.State() != llm.CircuitClosed {
		t.Errorf("initial state = %v, want closed", cb.State())
	}
}

func TestCircuitBreakerOpensAfterThreshold(t *testing.T) {
	cb := llm.NewCircuitBreaker(3, 100*time.Millisecond)

	// 2 failures: still closed
	cb.RecordFailure()
	cb.RecordFailure()
	if cb.State() != llm.CircuitClosed {
		t.Fatalf("state after 2 failures = %v, want closed", cb.State())
	}

	// 3rd failure: opens
	cb.RecordFailure()
	if cb.State() != llm.CircuitOpen {
		t.Fatalf("state after 3 failures = %v, want open", cb.State())
	}
}

func TestCircuitBreakerAllowRejectsDuringOpen(t *testing.T) {
	cb := llm.NewCircuitBreaker(1, 1*time.Hour) // long cooldown
	cb.RecordFailure()                          // opens

	if cb.Allow() {
		t.Error("Allow() = true, want false when circuit is open")
	}
}

func TestCircuitBreakerTransitionsToHalfOpen(t *testing.T) {
	cb := llm.NewCircuitBreaker(1, 10*time.Millisecond)
	cb.RecordFailure() // opens

	// Wait for cooldown
	time.Sleep(20 * time.Millisecond)

	if !cb.Allow() {
		t.Fatal("Allow() = false after cooldown, want true (half-open)")
	}
	if cb.State() != llm.CircuitHalfOpen {
		t.Errorf("state = %v, want half-open", cb.State())
	}
}

func TestCircuitBreakerHalfOpenRejectsSecondRequest(t *testing.T) {
	cb := llm.NewCircuitBreaker(1, 10*time.Millisecond)
	cb.RecordFailure() // opens
	time.Sleep(20 * time.Millisecond)

	// First Allow → half-open, allowed
	if !cb.Allow() {
		t.Fatal("first Allow after cooldown should succeed")
	}

	// Second Allow → still half-open, rejected (only one probe)
	if cb.Allow() {
		t.Error("second Allow in half-open should be rejected")
	}
}

func TestCircuitBreakerSuccessClosesFromHalfOpen(t *testing.T) {
	cb := llm.NewCircuitBreaker(1, 10*time.Millisecond)
	cb.RecordFailure() // opens
	time.Sleep(20 * time.Millisecond)
	cb.Allow() // → half-open

	cb.RecordSuccess()
	if cb.State() != llm.CircuitClosed {
		t.Errorf("state = %v, want closed after success in half-open", cb.State())
	}

	// Should allow requests again
	if !cb.Allow() {
		t.Error("Allow() = false after closing, want true")
	}
}

func TestCircuitBreakerFailureReopensFromHalfOpen(t *testing.T) {
	cb := llm.NewCircuitBreaker(1, 10*time.Millisecond)
	cb.RecordFailure() // opens
	time.Sleep(20 * time.Millisecond)
	cb.Allow() // → half-open

	cb.RecordFailure()
	if cb.State() != llm.CircuitOpen {
		t.Errorf("state = %v, want open after failure in half-open", cb.State())
	}
}

func TestCircuitBreakerSuccessResetsClosed(t *testing.T) {
	cb := llm.NewCircuitBreaker(3, 100*time.Millisecond)

	// 2 failures, then a success — should reset counter
	cb.RecordFailure()
	cb.RecordFailure()
	cb.RecordSuccess()

	// Now 1 more failure should NOT open (counter was reset)
	cb.RecordFailure()
	if cb.State() != llm.CircuitClosed {
		t.Errorf("state = %v, want closed (success should reset failures)", cb.State())
	}
}

func TestCircuitBreakerNilAllow(t *testing.T) {
	var cb *llm.CircuitBreaker
	if !cb.Allow() {
		t.Error("nil.Allow() = false, want true")
	}
}

func TestCircuitBreakerNilRecordSuccess(t *testing.T) {
	var cb *llm.CircuitBreaker
	cb.RecordSuccess() // should not panic
}

func TestCircuitBreakerNilRecordFailure(t *testing.T) {
	var cb *llm.CircuitBreaker
	cb.RecordFailure() // should not panic
}

func TestCircuitBreakerNilState(t *testing.T) {
	var cb *llm.CircuitBreaker
	if cb.State() != llm.CircuitClosed {
		t.Errorf("nil.State() = %v, want closed", cb.State())
	}
}

func TestCircuitBreakerNilReset(t *testing.T) {
	var cb *llm.CircuitBreaker
	cb.Reset() // should not panic
}

func TestCircuitBreakerNilExecute(t *testing.T) {
	var cb *llm.CircuitBreaker
	called := false
	err := cb.Execute(func() error {
		called = true
		return nil
	})
	if err != nil {
		t.Errorf("nil.Execute() = %v, want nil", err)
	}
	if !called {
		t.Error("nil.Execute() did not call fn")
	}
}

func TestCircuitBreakerExecuteSuccess(t *testing.T) {
	cb := llm.NewCircuitBreaker(3, 100*time.Millisecond)
	err := cb.Execute(func() error { return nil })
	if err != nil {
		t.Errorf("Execute() = %v, want nil", err)
	}
	if cb.State() != llm.CircuitClosed {
		t.Errorf("state = %v, want closed", cb.State())
	}
}

func TestCircuitBreakerExecuteFailure(t *testing.T) {
	cb := llm.NewCircuitBreaker(2, 100*time.Millisecond)
	sentinel := errors.New("boom")

	err := cb.Execute(func() error { return sentinel })
	if !errors.Is(err, sentinel) {
		t.Errorf("Execute() = %v, want %v", err, sentinel)
	}

	err = cb.Execute(func() error { return sentinel })
	if !errors.Is(err, sentinel) {
		t.Errorf("Execute() = %v, want %v", err, sentinel)
	}

	// Circuit should be open now
	if cb.State() != llm.CircuitOpen {
		t.Errorf("state = %v, want open", cb.State())
	}
}

func TestCircuitBreakerExecuteOpen(t *testing.T) {
	cb := llm.NewCircuitBreaker(1, 1*time.Hour)
	cb.RecordFailure() // opens

	called := false
	err := cb.Execute(func() error {
		called = true
		return nil
	})
	if !errors.Is(err, llm.ErrCircuitOpen) {
		t.Errorf("Execute() = %v, want ErrCircuitOpen", err)
	}
	if called {
		t.Error("fn called when circuit is open")
	}
}

func TestCircuitBreakerExecuteWithSuccessPredicate(t *testing.T) {
	authErr := errors.New("auth error")
	cb := llm.NewCircuitBreaker(2, 100*time.Millisecond,
		llm.WithSuccessPredicate(func(err error) bool {
			return errors.Is(err, authErr)
		}),
	)

	// Auth errors should not trip the circuit
	for i := 0; i < 5; i++ {
		err := cb.Execute(func() error { return authErr })
		if !errors.Is(err, authErr) {
			t.Fatalf("Execute() = %v, want %v", err, authErr)
		}
	}

	if cb.State() != llm.CircuitClosed {
		t.Errorf("state = %v, want closed (auth errors should not trip)", cb.State())
	}
}

func TestCircuitBreakerOnStateChange(t *testing.T) {
	type transition struct {
		from, to llm.CircuitState
	}
	var transitions []transition

	cb := llm.NewCircuitBreaker(2, 10*time.Millisecond,
		llm.WithOnStateChange(func(from, to llm.CircuitState) {
			transitions = append(transitions, transition{from, to})
		}),
	)

	cb.RecordFailure()
	cb.RecordFailure()                // → open
	time.Sleep(20 * time.Millisecond) //
	cb.Allow()                        // → half-open
	cb.RecordSuccess()                // → closed

	expected := []transition{
		{llm.CircuitClosed, llm.CircuitOpen},
		{llm.CircuitOpen, llm.CircuitHalfOpen},
		{llm.CircuitHalfOpen, llm.CircuitClosed},
	}

	if len(transitions) != len(expected) {
		t.Fatalf("got %d transitions, want %d: %v", len(transitions), len(expected), transitions)
	}
	for i, want := range expected {
		got := transitions[i]
		if got.from != want.from || got.to != want.to {
			t.Errorf("transition %d: got %v→%v, want %v→%v", i, got.from, got.to, want.from, want.to)
		}
	}
}

func TestCircuitBreakerReset(t *testing.T) {
	cb := llm.NewCircuitBreaker(1, 1*time.Hour)
	cb.RecordFailure() // opens

	if cb.State() != llm.CircuitOpen {
		t.Fatalf("state = %v, want open", cb.State())
	}

	cb.Reset()

	if cb.State() != llm.CircuitClosed {
		t.Errorf("state after Reset = %v, want closed", cb.State())
	}
	if !cb.Allow() {
		t.Error("Allow after Reset = false, want true")
	}
}

func TestCircuitStateString(t *testing.T) {
	tests := []struct {
		state llm.CircuitState
		want  string
	}{
		{llm.CircuitClosed, "closed"},
		{llm.CircuitOpen, "open"},
		{llm.CircuitHalfOpen, "half-open"},
		{llm.CircuitState(99), "unknown"},
	}
	for _, tt := range tests {
		if got := tt.state.String(); got != tt.want {
			t.Errorf("CircuitState(%d).String() = %q, want %q", tt.state, got, tt.want)
		}
	}
}

func TestCircuitBreakerConcurrentAccess(t *testing.T) {
	cb := llm.NewCircuitBreaker(100, 10*time.Millisecond)

	var wg sync.WaitGroup
	const goroutines = 50

	// Half recording failures, half recording successes
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				cb.Allow()
				if n%2 == 0 {
					cb.RecordFailure()
				} else {
					cb.RecordSuccess()
				}
				_ = cb.State()
			}
		}(i)
	}

	wg.Wait()
	// No panic or race detector complaint = pass
}

func TestCircuitBreakerConcurrentExecute(t *testing.T) {
	cb := llm.NewCircuitBreaker(1000, 10*time.Millisecond) // high threshold
	var calls atomic.Int64

	var wg sync.WaitGroup
	const goroutines = 20
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				_ = cb.Execute(func() error {
					calls.Add(1)
					return nil
				})
			}
		}()
	}

	wg.Wait()
	if calls.Load() != int64(goroutines*50) {
		t.Errorf("calls = %d, want %d", calls.Load(), goroutines*50)
	}
}

func TestCircuitBreakerThresholdOne(t *testing.T) {
	cb := llm.NewCircuitBreaker(1, 100*time.Millisecond)
	cb.RecordFailure()
	if cb.State() != llm.CircuitOpen {
		t.Errorf("state = %v, want open after 1 failure with threshold=1", cb.State())
	}
}

func TestCircuitBreakerHighThreshold(t *testing.T) {
	cb := llm.NewCircuitBreaker(1000, 100*time.Millisecond)
	for i := 0; i < 999; i++ {
		cb.RecordFailure()
	}
	if cb.State() != llm.CircuitClosed {
		t.Fatalf("state = %v after 999 failures, want closed (threshold=1000)", cb.State())
	}
	cb.RecordFailure()
	if cb.State() != llm.CircuitOpen {
		t.Errorf("state = %v after 1000 failures, want open", cb.State())
	}
}

// TestCircuitBreakerOpensAfterConsecutiveFailures verifies the circuit opens
// after CircuitThreshold consecutive failures, returning circuit-open error.
func TestCircuitBreakerOpensAfterConsecutiveFailures(t *testing.T) {
	var mu sync.Mutex
	var circuitOpenEvents int

	cfg := llm.DefaultConfig()
	cfg.CircuitThreshold = 2 // opens after 2 failures; attempt 0,1 fail → circuit opens → attempt 2 sees it
	cfg.CircuitCooldown = 50 * time.Millisecond
	cfg.CooldownOverload = 10 * time.Millisecond
	// Single-key backoff follows RetryPolicy (v0.9.5): keep it shorter than
	// the circuit cooldown so the open circuit is observed between attempts.
	cfg.RetryPolicy = &llm.RetryPolicy{BaseDelay: 5 * time.Millisecond, MaxDelay: 10 * time.Millisecond, Factor: 2.0}
	cfg.RetryHook = func(evt llm.RetryEvent) {
		mu.Lock()
		defer mu.Unlock()
		if evt.Type == llm.RetryCircuitOpen {
			circuitOpenEvents++
		}
	}

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

	// Attempts 0 and 1 fail (hitting threshold=2), circuit opens.
	// Attempt 2 sees the open circuit and returns immediately.
	_, err = pc.Complete(ctx, llm.Request{})
	if err == nil {
		t.Fatal("expected error, got nil")
	}

	mu.Lock()
	if circuitOpenEvents == 0 {
		t.Error("expected at least one RetryCircuitOpen event from hook")
	}
	mu.Unlock()

	// Error should mention circuit breaker
	if !strings.Contains(err.Error(), "circuit breaker open") {
		t.Errorf("expected circuit breaker open error, got: %v", err)
	}
}

// TestCircuitBreakerRecovery verifies the circuit transitions half-open → closed
// when a probe succeeds after the cooldown elapses.
func TestCircuitBreakerRecovery(t *testing.T) {
	var mu sync.Mutex
	failCount := 0
	threshold := 2

	cfg := llm.DefaultConfig()
	cfg.CircuitThreshold = threshold
	cfg.CircuitCooldown = 50 * time.Millisecond
	// Short pool cooldowns so the auth pool doesn't block recovery
	cfg.CooldownOverload = 50 * time.Millisecond
	cfg.CooldownDefault = 50 * time.Millisecond
	// Single-key backoff follows RetryPolicy (v0.9.5): it must outlast the
	// 50ms circuit cooldown so the next attempt is admitted as the probe.
	cfg.RetryPolicy = &llm.RetryPolicy{
		BaseDelay: 60 * time.Millisecond,
		MaxDelay:  100 * time.Millisecond,
		Factor:    2.0,
		Jitter:    0,
	}

	pc, err := llm.NewPooledClient(cfg, []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &mockClient{
			provider: "test",
			model:    "m",
			completeFunc: func(ctx context.Context, req llm.Request) (*llm.Response, error) {
				mu.Lock()
				fc := failCount
				failCount++
				mu.Unlock()

				// First 'threshold' calls fail (to trip the circuit), then succeed
				if fc < threshold {
					return nil, llm.NewOverloadedError("test", "overloaded")
				}
				return &llm.Response{Content: "recovered"}, nil
			},
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient: %v", err)
	}
	defer pc.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// This call will exhaust the threshold, open the circuit, then backoff.
	// After cooldown, half-open probe succeeds → circuit closes → returns success.
	resp, err := pc.Complete(ctx, llm.Request{})
	if err != nil {
		t.Fatalf("Complete: %v (expected recovery after circuit half-open)", err)
	}
	if resp.Content != "recovered" {
		t.Errorf("Content = %q, want recovered", resp.Content)
	}
}

// TestCircuitBreakerAuthErrorsDoNotTrip verifies that auth errors (401/403)
// do not count toward the circuit breaker threshold.
func TestCircuitBreakerAuthErrorsDoNotTrip(t *testing.T) {
	cfg := llm.DefaultConfig()
	cfg.CircuitThreshold = 2
	cfg.CircuitCooldown = 1 * time.Hour // should never open

	callCount := 0
	pc, err := llm.NewPooledClient(cfg, []string{"key-a", "key-b", "key-c"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		if callCount <= 2 {
			// First two keys: auth errors
			return &threadSafeMockClient{
				provider: "test",
				model:    "m",
				err:      llm.NewAuthError("test", "invalid key", 401),
			}, nil
		}
		// Third key: success
		return &threadSafeMockClient{
			provider: "test",
			model:    "m",
			resp:     &llm.Response{Content: "ok"},
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient: %v", err)
	}
	defer pc.Close()

	resp, err := pc.Complete(context.Background(), llm.Request{})
	if err != nil {
		t.Fatalf("Complete: %v (auth errors should not trip circuit)", err)
	}
	if resp.Content != "ok" {
		t.Errorf("Content = %q, want ok", resp.Content)
	}
}

// TestCircuitBreakerStreamIntegration verifies the circuit breaker gates
// stream attempts the same as Complete — returning circuit-open error.
func TestCircuitBreakerStreamIntegration(t *testing.T) {
	cfg := llm.DefaultConfig()
	cfg.CircuitThreshold = 2 // opens after 2 failures
	cfg.CircuitCooldown = 50 * time.Millisecond
	cfg.CooldownOverload = 10 * time.Millisecond
	// Single-key backoff follows RetryPolicy (v0.9.5): keep it shorter than
	// the circuit cooldown so the open circuit is observed between attempts.
	cfg.RetryPolicy = &llm.RetryPolicy{BaseDelay: 5 * time.Millisecond, MaxDelay: 10 * time.Millisecond, Factor: 2.0}

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

	var streamErr error
	for _, err := range pc.Stream(ctx, llm.Request{}) {
		if err != nil {
			streamErr = err
			break
		}
	}

	if streamErr == nil {
		t.Fatal("expected error from stream, got nil")
	}
	// After threshold failures, the circuit opens and stream returns circuit-open error
	if !strings.Contains(streamErr.Error(), "circuit breaker open") {
		t.Errorf("expected circuit breaker open error, got: %v", streamErr)
	}
}

// TestDefaultConfigIncludesCircuitBreaker verifies DefaultConfig has the circuit breaker enabled.
func TestDefaultConfigIncludesCircuitBreaker(t *testing.T) {
	cfg := llm.DefaultConfig()
	if cfg.CircuitThreshold != llm.DefaultCircuitThreshold {
		t.Errorf("DefaultConfig().CircuitThreshold = %d, want %d", cfg.CircuitThreshold, llm.DefaultCircuitThreshold)
	}
	if cfg.CircuitCooldown != llm.DefaultCircuitCooldown {
		t.Errorf("DefaultConfig().CircuitCooldown = %v, want %v", cfg.CircuitCooldown, llm.DefaultCircuitCooldown)
	}
}

// TestCircuitBreakerCallbackPanic verifies a panicking callback doesn't crash the caller.
func TestCircuitBreakerCallbackPanic(t *testing.T) {
	cb := llm.NewCircuitBreaker(1, time.Second,
		llm.WithOnStateChange(func(from, to llm.CircuitState) {
			panic("boom from callback")
		}),
	)

	// This should NOT panic — the library recovers
	cb.RecordFailure() // trips threshold=1, fires callback

	// Circuit should still be in the expected state despite panic
	if cb.State() != llm.CircuitOpen {
		t.Errorf("State() = %v, want CircuitOpen", cb.State())
	}
}
