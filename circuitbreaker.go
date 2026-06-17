package llm

import (
	"errors"
	"log/slog"
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
	probeStarted  time.Time // when the current half-open probe was admitted
	probeGen      uint64    // identity of the current half-open probe (bumped on each admission)
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
	admitted, _ := cb.allow()
	return admitted
}

// bumpProbeGen advances the probe generation and returns the new token,
// skipping the two reserved values so a real probe token is always a distinct
// identity — never 0 ("admitted while closed / no probe") and never anyProbe
// (the wildcard that bypasses the identity check). This holds even across the
// (practically impossible) 2^64 wrap. Must be called with cb.mu held.
func (cb *CircuitBreaker) bumpProbeGen() uint64 {
	cb.probeGen++
	if cb.probeGen == 0 || cb.probeGen == anyProbe {
		cb.probeGen = 1
	}
	return cb.probeGen
}

// allow is Allow with probe identity. When it admits a half-open probe it
// returns a non-zero probeToken uniquely identifying that probe, so the caller
// can later ReleaseProbe(token) for *its own* probe rather than an unrelated one
// admitted by another goroutine in the meantime. A request admitted while the
// circuit is closed (or rejected) returns token 0 — it never held a probe slot.
// Nil-safe: returns (true, 0).
func (cb *CircuitBreaker) allow() (admitted bool, probeToken uint64) {
	if cb == nil {
		return true, 0
	}
	cb.mu.Lock()

	switch cb.state {
	case CircuitClosed:
		cb.mu.Unlock()
		return true, 0
	case CircuitOpen:
		if time.Since(cb.openedAt) >= cb.cooldown {
			from, to, changed := cb.setStateLocked(CircuitHalfOpen)
			cb.probeStarted = time.Now()
			tok := cb.bumpProbeGen()
			cb.mu.Unlock()
			if changed {
				cb.fireCallback(from, to)
			}
			return true, tok
		}
		cb.mu.Unlock()
		return false, 0
	case CircuitHalfOpen:
		// Only one probe at a time; additional requests are rejected until the
		// probe completes. If the probe never reports back (abandoned iterator,
		// crashed goroutine), admit a fresh probe after another cooldown rather
		// than wedging half-open forever.
		if time.Since(cb.probeStarted) >= cb.cooldown {
			cb.probeStarted = time.Now()
			tok := cb.bumpProbeGen()
			cb.mu.Unlock()
			return true, tok
		}
		cb.mu.Unlock()
		return false, 0
	default:
		cb.mu.Unlock()
		return true, 0
	}
}

// anyProbe is the wildcard probe token used by the public RecordSuccess /
// RecordFailure: it bypasses the half-open probe-identity check, preserving
// their original state-blind behavior for direct callers (e.g. Execute, which
// runs Allow→fn→Record in one goroutine). The pool passes the real token from
// allow() so a stale/foreign outcome can't resolve a probe it doesn't own.
// (probeGen would have to wrap 2^64 admissions to collide with this.)
const anyProbe = ^uint64(0)

// RecordSuccess resets the failure counter and closes the circuit if half-open.
// Nil-safe: no-op on nil receiver.
func (cb *CircuitBreaker) RecordSuccess() {
	cb.recordSuccess(anyProbe)
}

// recordSuccess is RecordSuccess scoped to a probe token. When the circuit is
// half-open, only the caller holding the current probe (token == probeGen, or
// the anyProbe wildcard) may close it; a stale/foreign success — from a request
// admitted while the circuit was closed (token 0), or whose probe a newer one
// already superseded — is ignored so it can't close a probe it doesn't own.
// Nil-safe.
func (cb *CircuitBreaker) recordSuccess(token uint64) {
	if cb == nil {
		return
	}
	cb.mu.Lock()
	if cb.state == CircuitHalfOpen && token != anyProbe && cb.probeGen != token {
		cb.mu.Unlock()
		return
	}
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
	cb.recordFailure(anyProbe)
}

// recordFailure is RecordFailure scoped to a probe token. When the circuit is
// half-open, only the caller holding the current probe (or the anyProbe
// wildcard) may re-open it; a stale/foreign failure is ignored — the unhealthy
// signal is already reflected in the open/half-open state, so dropping it can't
// lose information but does stop it re-opening an unrelated goroutine's probe.
// In the closed state the token is irrelevant: every failure counts toward the
// threshold. Nil-safe.
func (cb *CircuitBreaker) recordFailure(token uint64) {
	if cb == nil {
		return
	}
	cb.mu.Lock()
	if cb.state == CircuitHalfOpen && token != anyProbe && cb.probeGen != token {
		cb.mu.Unlock()
		return
	}
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

// ReleaseProbe re-arms the half-open probe identified by token — admitted but
// whose outcome is non-diagnostic (the caller cancelled before it resolved, or
// it drew a client-side error like 400 that says nothing about endpoint health).
// It records neither success nor failure: the circuit stays half-open, but the
// next allow() admits a fresh probe immediately instead of stranding real
// traffic for a full extra cooldown.
//
// token is the value allow() returned when it admitted this request as a probe.
// Re-arming is single-owner: a token of 0 (the request was admitted while
// closed, so it never held a probe slot) is a no-op, and so is a token that a
// newer probe has already superseded. This prevents a stale cancel/error from
// resetting an unrelated goroutine's in-flight probe (which would admit a second
// concurrent probe). No-op unless still half-open with this exact probe. Nil-safe.
func (cb *CircuitBreaker) ReleaseProbe(token uint64) {
	if cb == nil || token == 0 {
		return
	}
	cb.mu.Lock()
	if cb.state == CircuitHalfOpen && cb.probeGen == token {
		// Zero time reads as "probe elapsed" to allow's half-open branch, so the
		// next caller is admitted as a fresh probe.
		cb.probeStarted = time.Time{}
	}
	cb.mu.Unlock()
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
// Recovers from panics in user-provided callbacks to prevent crashing the caller.
func (cb *CircuitBreaker) fireCallback(from, to CircuitState) {
	if cb.onStateChange != nil {
		defer func() {
			if r := recover(); r != nil {
				slog.Error("circuit breaker callback panicked", "from", from, "to", to, "panic", r)
			}
		}()
		cb.onStateChange(from, to)
	}
}
