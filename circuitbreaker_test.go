package llm_test

import (
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
)

// --- State transition tests ---

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
	cb.RecordFailure()                           // opens

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

// --- Nil safety tests ---

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

// --- Execute tests ---

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

// --- OnStateChange callback tests ---

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
	cb.RecordFailure()                  // → open
	time.Sleep(20 * time.Millisecond)   //
	cb.Allow()                          // → half-open
	cb.RecordSuccess()                  // → closed

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

// --- Reset tests ---

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

// --- String representation ---

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

// --- Concurrency tests ---

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

// --- Threshold edge cases ---

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
