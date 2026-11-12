package llm

import (
	"net/http"
	"testing"
	"time"
)

// PIN (Round3): RetryPolicy.Delay must never return a non-positive duration even
// under maximal jitter — a zero/negative backoff would busy-spin retries. Pins
// the post-jitter floor clamp.
func TestRetryPolicyDelayNeverNonPositive(t *testing.T) {
	rp := RetryPolicy{BaseDelay: 100 * time.Millisecond, MaxDelay: 100 * time.Millisecond, Factor: 2, Jitter: 0.99}
	for attempt := 0; attempt < 5; attempt++ {
		for i := 0; i < 1000; i++ {
			if d := rp.Delay(attempt); d <= 0 {
				t.Fatalf("Delay(%d) = %v, must stay > 0 (jitter underflow must be clamped)", attempt, d)
			}
		}
	}
}

// Hostile Retry-After values must never yield a negative, zero-but-ok, or
// overflowing wait, and never exceed MaxRetryAfter.
func TestParseRetryAfterHostileValues(t *testing.T) {
	now := time.Date(2026, 10, 8, 12, 0, 0, 0, time.UTC)
	cases := []struct {
		name, key, value string
		want             time.Duration
		ok               bool
	}{
		{"seconds", "Retry-After", "5", 5 * time.Second, true},
		{"zero", "Retry-After", "0", 0, false},
		{"negative", "Retry-After", "-3", 0, false},
		{"huge", "Retry-After", "9223372036854775807", MaxRetryAfter, true},
		// Beyond int64 is a very long wait, like "huge" (it read as no hint, so
		// a quota 429 asking for days was retried; P5).
		{"overflow", "Retry-After", "99999999999999999999999", MaxRetryAfter, true},
		{"overflow negative", "Retry-After", "-99999999999999999999999", 0, false},
		{"garbage", "Retry-After", "soon", 0, false},
		{"fraction", "Retry-After", "1.5", 0, false},
		{"date future", "Retry-After", now.Add(10 * time.Second).Format(http.TimeFormat), 10 * time.Second, true},
		{"date past", "Retry-After", now.Add(-time.Hour).Format(http.TimeFormat), 0, false},
		{"date far", "Retry-After", now.Add(48 * time.Hour).Format(http.TimeFormat), MaxRetryAfter, true},
		{"ms", "Retry-After-Ms", "250", 250 * time.Millisecond, true},
		{"ms NaN", "Retry-After-Ms", "NaN", 0, false},
		{"ms Inf", "Retry-After-Ms", "+Inf", MaxRetryAfter, true},
		{"ms huge", "Retry-After-Ms", "1e300", MaxRetryAfter, true},
		{"ms negative", "Retry-After-Ms", "-5", 0, false},
	}
	for _, tc := range cases {
		h := http.Header{}
		h.Set(tc.key, tc.value)
		got, ok := parseRetryAfter(h, now)
		if got != tc.want || ok != tc.ok {
			t.Errorf("%s: got (%v,%v), want (%v,%v)", tc.name, got, ok, tc.want, tc.ok)
		}
	}
	if _, ok := parseRetryAfter(nil, now); ok {
		t.Error("nil header reported a hint")
	}
}
