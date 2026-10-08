package llm

import (
	"context"
	"errors"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// DefaultHTTPMaxAttempts is the number of attempts DoHTTP makes when
// Config.MaxRetries is zero. It matches the single-key chat client (three
// attempts), so a modality or batch call and a chat call against the same
// degraded endpoint give up after the same number of tries. An explicit
// Config.MaxRetries (DefaultConfig sets DefaultMaxRetries) still wins.
const DefaultHTTPMaxAttempts = 3

// MaxRetryAfter caps how long a provider's Retry-After header may stretch a
// single backoff. A larger hint is clamped to this value rather than obeyed,
// so a hostile or misconfigured upstream cannot park a request for hours; use
// Config.RetryBudget to bound the whole retry sequence.
const MaxRetryAfter = 60 * time.Second

// parseRetryAfter reads the standard Retry-After header (delta-seconds or an
// HTTP-date) and OpenAI's millisecond variant retry-after-ms. It reports
// false when no usable hint is present. Negative, zero and past values report
// false; a value above MaxRetryAfter is clamped to it.
func parseRetryAfter(header http.Header, now time.Time) (time.Duration, bool) {
	if header == nil {
		return 0, false
	}
	if ms := strings.TrimSpace(header.Get("Retry-After-Ms")); ms != "" {
		// ParseFloat accepts "NaN"/"Inf"; v > 0 rejects NaN, and the clamp is
		// done in float space so +Inf or 1e300 cannot overflow a Duration.
		if v, err := strconv.ParseFloat(ms, 64); err == nil && v > 0 {
			if v*float64(time.Millisecond) >= float64(MaxRetryAfter) {
				return MaxRetryAfter, true
			}
			return time.Duration(v * float64(time.Millisecond)), true
		}
	}
	raw := strings.TrimSpace(header.Get("Retry-After"))
	if raw == "" {
		return 0, false
	}
	if secs, err := strconv.ParseInt(raw, 10, 64); err == nil {
		if secs <= 0 {
			return 0, false
		}
		if secs > int64(MaxRetryAfter/time.Second) {
			return MaxRetryAfter, true
		}
		return time.Duration(secs) * time.Second, true
	}
	if when, err := http.ParseTime(raw); err == nil {
		wait := when.Sub(now)
		if wait <= 0 {
			return 0, false
		}
		return clampRetryAfter(wait), true
	}
	return 0, false
}

func clampRetryAfter(d time.Duration) time.Duration {
	if d > MaxRetryAfter {
		return MaxRetryAfter
	}
	return d
}

// retryAfterOf returns the provider's Retry-After hint carried by err, or 0.
func retryAfterOf(err error) time.Duration {
	var apiErr *APIError
	if errors.As(err, &apiErr) && apiErr.RetryAfter > 0 {
		return clampRetryAfter(apiErr.RetryAfter)
	}
	return 0
}

// backoffFor returns the sleep before the attempt after `attempt`: the
// policy's exponential delay, stretched to the provider's Retry-After hint
// when that is longer (never shortened by it — the hint is a floor).
func backoffFor(policy RetryPolicy, attempt int, retryAfter time.Duration) time.Duration {
	delay := policy.Delay(attempt)
	if retryAfter > delay {
		delay = clampRetryAfter(retryAfter)
	}
	return delay
}

// retryBudget tracks Config.RetryBudget: the total wall-clock time a retry
// sequence may spend before it stops starting new attempts.
type retryBudget struct {
	start  time.Time
	budget time.Duration
}

func newRetryBudget(budget time.Duration) retryBudget {
	return retryBudget{start: time.Now(), budget: budget}
}

// allows reports whether sleeping for delay and then trying again still fits
// in the budget. A zero (or negative) budget is unlimited.
func (b retryBudget) allows(delay time.Duration) bool {
	if b.budget <= 0 {
		return true
	}
	return time.Since(b.start)+delay < b.budget
}

// sleepOutlastsDeadline reports whether sleeping for delay would reach or pass
// ctx's deadline. The retry engines check it before every backoff: the attempt
// after the sleep could not complete anyway, and sleeping into the deadline
// would replace the provider's error (e.g. a 429 carrying Retry-After) with a
// bare context.DeadlineExceeded. A context without a deadline never outlasts.
func sleepOutlastsDeadline(ctx context.Context, delay time.Duration) bool {
	deadline, ok := ctx.Deadline()
	return ok && !time.Now().Add(delay).Before(deadline)
}

// isDialError reports a transport failure that happened before the request
// could have reached the provider (DNS lookup or TCP/TLS connect), so even a
// non-idempotent request is safe to resend.
func isDialError(err error) bool {
	var dnsErr *net.DNSError
	if errors.As(err, &dnsErr) {
		return true
	}
	var opErr *net.OpError
	if errors.As(err, &opErr) && opErr.Op == "dial" {
		return true
	}
	// A proxy CONNECT failure is reported as "proxyconnect" by net/http.
	return errors.As(err, &opErr) && opErr.Op == "proxyconnect"
}
