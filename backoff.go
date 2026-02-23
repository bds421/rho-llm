package llm

import (
	"math/rand/v2"
	"time"
)

// Backoff calculates exponential backoff with jitter.
//
// attempt is 0-indexed. The delay doubles each attempt starting from baseDelay,
// capped at maxDelay. Jitter of +/-25% is applied to prevent thundering herd.
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
	if attempt < 0 {
		attempt = 0
	}

	// Exponential: baseDelay * 2^attempt
	delay := baseDelay
	for i := 0; i < attempt; i++ {
		delay *= 2
		if delay > maxDelay {
			delay = maxDelay
			break
		}
	}

	if delay > maxDelay {
		delay = maxDelay
	}

	// Jitter: 75%-125% of calculated delay
	quarter := delay / 4
	if quarter > 0 {
		jitter := time.Duration(rand.Int64N(int64(2*quarter))) - quarter
		delay += jitter
	}

	// Ensure we don't exceed maxDelay after jitter
	if delay > maxDelay {
		delay = maxDelay
	}
	if delay <= 0 {
		delay = baseDelay
	}

	return delay
}
