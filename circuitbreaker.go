package llm

import (
	"errors"
	"sync"
	"time"
)

// ErrCircuitOpen is returned when the circuit breaker is open and rejecting requests.
var ErrCircuitOpen = errors.New("circuit breaker is open")

// CircuitState represents the current state of a circuit breaker.
type CircuitState int

const (
	// CircuitClosed allows all requests through. Failures are counted.
	CircuitClosed CircuitState = iota
	// CircuitOpen rejects all requests. After cooldown, transitions to half-open.
	CircuitOpen
	// CircuitHalfOpen allows one probe request. Success closes, failure re-opens.
	CircuitHalfOpen
)

// String returns a human-readable name for the circuit state.
func (s CircuitState) String() string {
	switch s {
	case CircuitClosed:
		return "closed"
	case CircuitOpen:
		return "open"
	case CircuitHalfOpen:
		return "half-open"
	default:
		return "unknown"
	}
}

// CircuitBreakerOption configures optional circuit breaker behavior.
type CircuitBreakerOption func(*CircuitBreaker)

// WithSuccessPredicate sets a function that determines whether an error
// should be treated as a success (not tripping the circuit). For example,
// authentication errors are bad keys, not endpoint failures.
func WithSuccessPredicate(fn func(error) bool) CircuitBreakerOption {
	return func(cb *CircuitBreaker) {
		cb.isSuccess = fn
	}
}

// WithOnStateChange sets a callback invoked on every state transition.
func WithOnStateChange(fn func(from, to CircuitState)) CircuitBreakerOption {
	return func(cb *CircuitBreaker) {
		cb.onStateChange = fn
	}
}

// CircuitBreaker implements a 3-state circuit breaker pattern.
// It opens after a threshold of consecutive failures, stays open for a
// cooldown period, then allows a single probe (half-open). A successful
// probe closes the circuit; a failed probe re-opens it.
//
// All methods are safe for concurrent use. All methods are nil-safe:
// calling any method on a nil *CircuitBreaker is a no-op that allows
// all requests (equivalent to a permanently closed circuit).
type CircuitBreaker struct {
	mu            sync.Mutex
	state         CircuitState
	failures      int
	threshold     int
	cooldown      time.Duration
	openedAt      time.Time
	isSuccess     func(error) bool
	onStateChange func(from, to CircuitState)
}

// NewCircuitBreaker creates a circuit breaker that opens after threshold
// consecutive failures and stays open for cooldown before allowing a probe.
func NewCircuitBreaker(threshold int, cooldown time.Duration, opts ...CircuitBreakerOption) *CircuitBreaker {
	cb := &CircuitBreaker{
		threshold: threshold,
		cooldown:  cooldown,
	}
	for _, opt := range opts {
		opt(cb)
	}
	return cb
}

// Allow reports whether a request should be attempted.
// Returns true if the circuit is closed or half-open (probe allowed).
// Returns false if the circuit is open and cooldown has not elapsed.
// Nil-safe: always returns true on nil receiver.
func (cb *CircuitBreaker) Allow() bool {
	if cb == nil {
		return true
	}
	cb.mu.Lock()

	switch cb.state {
	case CircuitClosed:
		cb.mu.Unlock()
		return true
	case CircuitOpen:
		if time.Since(cb.openedAt) >= cb.cooldown {
			from, to, changed := cb.setStateLocked(CircuitHalfOpen)
			cb.mu.Unlock()
			if changed {
				cb.fireCallback(from, to)
			}
			return true
		}
		cb.mu.Unlock()
		return false
	case CircuitHalfOpen:
		// Only one probe at a time; additional requests are rejected
		// until the probe completes.
		cb.mu.Unlock()
		return false
	default:
		cb.mu.Unlock()
		return true
	}
}

// RecordSuccess resets the failure counter and closes the circuit if half-open.
// Nil-safe: no-op on nil receiver.
func (cb *CircuitBreaker) RecordSuccess() {
	if cb == nil {
		return
	}
	cb.mu.Lock()
	cb.failures = 0
	from, to, changed := CircuitClosed, CircuitClosed, false
	if cb.state == CircuitHalfOpen {
		from, to, changed = cb.setStateLocked(CircuitClosed)
	}
	cb.mu.Unlock()
	if changed {
		cb.fireCallback(from, to)
	}
}

// RecordFailure increments the consecutive failure counter.
// If the threshold is reached, the circuit opens. If already half-open,
// it re-opens immediately.
// Nil-safe: no-op on nil receiver.
func (cb *CircuitBreaker) RecordFailure() {
	if cb == nil {
		return
	}
	cb.mu.Lock()
	cb.failures++

	var from, to CircuitState
	var changed bool
	switch cb.state {
	case CircuitClosed:
		if cb.failures >= cb.threshold {
			cb.openedAt = time.Now()
			from, to, changed = cb.setStateLocked(CircuitOpen)
		}
	case CircuitHalfOpen:
		// Probe failed — re-open
		cb.openedAt = time.Now()
		from, to, changed = cb.setStateLocked(CircuitOpen)
	}
	cb.mu.Unlock()
	if changed {
		cb.fireCallback(from, to)
	}
}

// Execute is convenience sugar: Allow → fn() → RecordSuccess/RecordFailure.
// If the circuit is open, returns ErrCircuitOpen without calling fn.
// The isSuccess predicate (if configured) determines whether an error
// counts as a failure.
// Nil-safe: calls fn directly on nil receiver.
func (cb *CircuitBreaker) Execute(fn func() error) error {
	if cb == nil {
		return fn()
	}
	if !cb.Allow() {
		return ErrCircuitOpen
	}
	err := fn()
	if err == nil {
		cb.RecordSuccess()
		return nil
	}
	if cb.isSuccess != nil && cb.isSuccess(err) {
		// Error does not represent endpoint failure (e.g., auth error)
		return err
	}
	cb.RecordFailure()
	return err
}

// State returns the current circuit state.
// Nil-safe: returns CircuitClosed on nil receiver.
func (cb *CircuitBreaker) State() CircuitState {
	if cb == nil {
		return CircuitClosed
	}
	cb.mu.Lock()
	defer cb.mu.Unlock()
	return cb.state
}

// Reset resets the circuit breaker to its initial closed state.
// Nil-safe: no-op on nil receiver.
func (cb *CircuitBreaker) Reset() {
	if cb == nil {
		return
	}
	cb.mu.Lock()
	cb.failures = 0
	from, to, changed := cb.setStateLocked(CircuitClosed)
	cb.mu.Unlock()
	if changed {
		cb.fireCallback(from, to)
	}
}

// setStateLocked transitions to a new state. Must be called with cb.mu held.
// Returns the from/to states and whether a transition occurred, so the
// caller can fire the callback after releasing the lock.
func (cb *CircuitBreaker) setStateLocked(to CircuitState) (from, newState CircuitState, changed bool) {
	from = cb.state
	if from == to {
		return from, to, false
	}
	cb.state = to
	return from, to, true
}

// fireCallback invokes onStateChange outside the mutex.
func (cb *CircuitBreaker) fireCallback(from, to CircuitState) {
	if cb.onStateChange != nil {
		cb.onStateChange(from, to)
	}
}
