package llm

import (
	"math/rand/v2"
	"time"
)

// RetryEventType classifies retry lifecycle events for observability hooks.
type RetryEventType int

const (
	// RetryAttemptFailed fires when an individual attempt returns a retryable error.
	RetryAttemptFailed RetryEventType = iota
	// RetryRotating fires when the pool rotates to a different auth profile.
	RetryRotating
	// RetryBackingOff fires when the client sleeps before the next attempt.
	RetryBackingOff
	// RetryCircuitOpen fires when the circuit breaker rejects an attempt.
	RetryCircuitOpen
	// RetryExhausted fires when all retry attempts have been used up.
	RetryExhausted
)

// RetryEvent carries context about a retry lifecycle event.
type RetryEvent struct {
	Type     RetryEventType
	Attempt  int           // 0-indexed attempt number
	Err      error         // the error that triggered this event (may be nil)
	Backoff  time.Duration // sleep duration (only for RetryBackingOff)
	Provider string        // provider name, if available
}

// RetryHook is called during retry lifecycle events for observability.
// Implementations must be safe for concurrent use.
type RetryHook func(RetryEvent)

// RetryPolicy configures exponential backoff with jitter.
type RetryPolicy struct {
	// BaseDelay is the initial backoff delay before the first retry.
	BaseDelay time.Duration

	// MaxDelay caps the backoff delay.
	MaxDelay time.Duration

	// Factor is the exponential multiplier applied each attempt (e.g., 2.0 for doubling).
	Factor float64

	// Jitter is the fraction of delay used for randomization (e.g., 0.25 = +/-25%).
	Jitter float64
}

// DefaultRetryPolicy matches the original hardcoded retry behavior:
// 1s base, 30s max, doubling, +/-25% jitter.
var DefaultRetryPolicy = RetryPolicy{
	BaseDelay: 1 * time.Second,
	MaxDelay:   30 * time.Second,
	Factor:     2.0,
	Jitter:     0.25,
}

// Delay returns the backoff duration for the given 0-indexed attempt.
// The delay grows as BaseDelay * Factor^attempt, capped at MaxDelay,
// with random jitter of +/-Jitter fraction applied.
func (p RetryPolicy) Delay(attempt int) time.Duration {
	if attempt < 0 {
		attempt = 0
	}

	// Exponential: baseDelay * factor^attempt
	delay := p.BaseDelay
	for i := 0; i < attempt; i++ {
		delay = time.Duration(float64(delay) * p.Factor)
		if delay > p.MaxDelay {
			delay = p.MaxDelay
			break
		}
	}

	if delay > p.MaxDelay {
		delay = p.MaxDelay
	}

	// Jitter: (1 - jitter) to (1 + jitter) of calculated delay
	if p.Jitter > 0 {
		jitterRange := time.Duration(float64(delay) * p.Jitter)
		if jitterRange > 0 {
			jitter := time.Duration(rand.Int64N(int64(2*jitterRange))) - jitterRange // #nosec G404 -- backoff jitter does not need crypto randomness
			delay += jitter
		}
	}

	// Ensure we don't exceed maxDelay after jitter
	if delay > p.MaxDelay {
		delay = p.MaxDelay
	}
	if delay <= 0 {
		delay = p.BaseDelay
	}

	return delay
}

// Backoff calculates exponential backoff with jitter.
//
// attempt is 0-indexed. The delay doubles each attempt starting from baseDelay,
// capped at maxDelay. Jitter of +/-25% is applied to prevent thundering herd.
//
// This is a backward-compatible wrapper around RetryPolicy.Delay.
//
// Examples (baseDelay=1s, maxDelay=30s):
//
//	attempt 0: ~1s   (0.75s - 1.25s)
//	attempt 1: ~2s   (1.50s - 2.50s)
//	attempt 2: ~4s   (3.00s - 5.00s)
//	attempt 3: ~8s   (6.00s - 10.0s)
//	attempt 4: ~16s  (12.0s - 20.0s)
//	attempt 5: ~30s  (22.5s - 30.0s) (capped)
func Backoff(attempt int, baseDelay, maxDelay time.Duration) time.Duration {
	p := RetryPolicy{
		BaseDelay: baseDelay,
		MaxDelay:  maxDelay,
		Factor:    2.0,
		Jitter:    0.25,
	}
	return p.Delay(attempt)
}
