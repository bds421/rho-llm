package llm_test

import (
	"testing"
	"time"

	llm "gitlab2024.bds421-cloud.com/bds421/rho/llm"
)

func TestRetryPolicyExponentialGrowth(t *testing.T) {
	p := llm.DefaultRetryPolicy

	// Each attempt should roughly double from the previous.
	// Average over many samples to smooth jitter.
	for attempt := 0; attempt < 5; attempt++ {
		var total time.Duration
		const samples = 200
		for i := 0; i < samples; i++ {
			total += p.Delay(attempt)
		}
		avg := total / time.Duration(samples)

		expected := p.BaseDelay
		for j := 0; j < attempt; j++ {
			expected = time.Duration(float64(expected) * p.Factor)
			if expected > p.MaxDelay {
				expected = p.MaxDelay
				break
			}
		}

		low := time.Duration(float64(expected) * 0.60)
		high := time.Duration(float64(expected) * 1.40)
		if high > p.MaxDelay {
			high = p.MaxDelay
		}

		if avg < low || avg > high {
			t.Errorf("attempt %d: avg=%v, expected ~%v (range %v-%v)", attempt, avg, expected, low, high)
		}
	}
}

func TestRetryPolicyMaxCap(t *testing.T) {
	p := llm.RetryPolicy{
		BaseDelay: 1 * time.Second,
		MaxDelay:  5 * time.Second,
		Factor:    2.0,
		Jitter:    0.25,
	}

	for i := 0; i < 100; i++ {
		d := p.Delay(20)
		if d > 5*time.Second {
			t.Errorf("Delay(20) = %v, exceeds max 5s", d)
		}
		if d <= 0 {
			t.Errorf("Delay(20) = %v, should be positive", d)
		}
	}
}

func TestRetryPolicyNegativeAttempt(t *testing.T) {
	p := llm.DefaultRetryPolicy
	d := p.Delay(-5)
	if d <= 0 || d > p.MaxDelay {
		t.Errorf("Delay(-5) = %v, expected positive value <= %v", d, p.MaxDelay)
	}
}

func TestRetryPolicyZeroJitter(t *testing.T) {
	p := llm.RetryPolicy{
		BaseDelay: 100 * time.Millisecond,
		MaxDelay:  10 * time.Second,
		Factor:    2.0,
		Jitter:    0, // no jitter — deterministic
	}

	tests := []struct {
		attempt int
		want    time.Duration
	}{
		{0, 100 * time.Millisecond},
		{1, 200 * time.Millisecond},
		{2, 400 * time.Millisecond},
		{3, 800 * time.Millisecond},
	}

	for _, tt := range tests {
		got := p.Delay(tt.attempt)
		if got != tt.want {
			t.Errorf("Delay(%d) = %v, want %v", tt.attempt, got, tt.want)
		}
	}
}

func TestRetryPolicyCustomFactor(t *testing.T) {
	p := llm.RetryPolicy{
		BaseDelay: 100 * time.Millisecond,
		MaxDelay:  10 * time.Second,
		Factor:    3.0, // triple each time
		Jitter:    0,
	}

	tests := []struct {
		attempt int
		want    time.Duration
	}{
		{0, 100 * time.Millisecond},
		{1, 300 * time.Millisecond},
		{2, 900 * time.Millisecond},
		{3, 2700 * time.Millisecond},
	}

	for _, tt := range tests {
		got := p.Delay(tt.attempt)
		if got != tt.want {
			t.Errorf("Delay(%d) = %v, want %v", tt.attempt, got, tt.want)
		}
	}
}

func TestRetryPolicyBaseExceedsMax(t *testing.T) {
	p := llm.RetryPolicy{
		BaseDelay: 10 * time.Second,
		MaxDelay:  5 * time.Second,
		Factor:    2.0,
		Jitter:    0.25,
	}

	for i := 0; i < 20; i++ {
		d := p.Delay(0)
		if d > 5*time.Second {
			t.Errorf("Delay(0) = %v, exceeds max 5s", d)
		}
		if d <= 0 {
			t.Errorf("Delay(0) = %v, should be positive", d)
		}
	}
}

func TestRetryPolicyVeryHighAttempt(t *testing.T) {
	p := llm.RetryPolicy{
		BaseDelay: 1 * time.Second,
		MaxDelay:  5 * time.Second,
		Factor:    2.0,
		Jitter:    0.25,
	}

	for i := 0; i < 10; i++ {
		d := p.Delay(1000)
		if d > 5*time.Second {
			t.Errorf("Delay(1000) = %v, exceeds max 5s", d)
		}
		if d <= 0 {
			t.Errorf("Delay(1000) = %v, should be positive", d)
		}
	}
}

func TestRetryPolicyJitterDistribution(t *testing.T) {
	p := llm.RetryPolicy{
		BaseDelay: 1 * time.Second,
		MaxDelay:  30 * time.Second,
		Factor:    2.0,
		Jitter:    0.25,
	}

	// Attempt 0: base = 1s, jitter ±25% → range [750ms, 1250ms]
	var minSeen, maxSeen time.Duration
	const samples = 1000
	for i := 0; i < samples; i++ {
		d := p.Delay(0)
		if minSeen == 0 || d < minSeen {
			minSeen = d
		}
		if d > maxSeen {
			maxSeen = d
		}
	}

	if minSeen >= 1*time.Second {
		t.Errorf("jitter never went below base: min=%v", minSeen)
	}
	if maxSeen <= 1*time.Second {
		t.Errorf("jitter never went above base: max=%v", maxSeen)
	}
}

func TestDefaultRetryPolicyValues(t *testing.T) {
	p := llm.DefaultRetryPolicy
	if p.BaseDelay != 1*time.Second {
		t.Errorf("BaseDelay = %v, want 1s", p.BaseDelay)
	}
	if p.MaxDelay != 30*time.Second {
		t.Errorf("MaxDelay = %v, want 30s", p.MaxDelay)
	}
	if p.Factor != 2.0 {
		t.Errorf("Factor = %v, want 2.0", p.Factor)
	}
	if p.Jitter != 0.25 {
		t.Errorf("Jitter = %v, want 0.25", p.Jitter)
	}
}

func TestBackoffMatchesRetryPolicy(t *testing.T) {
	// Backoff wrapper should produce values in the same range as a
	// RetryPolicy with factor=2.0, jitter=0.25.
	base := 1 * time.Second
	max := 30 * time.Second

	for attempt := 0; attempt < 5; attempt++ {
		var backoffTotal, policyTotal time.Duration
		const samples = 500
		for i := 0; i < samples; i++ {
			backoffTotal += llm.Backoff(attempt, base, max)
		}
		p := llm.RetryPolicy{BaseDelay: base, MaxDelay: max, Factor: 2.0, Jitter: 0.25}
		for i := 0; i < samples; i++ {
			policyTotal += p.Delay(attempt)
		}

		backoffAvg := backoffTotal / time.Duration(samples)
		policyAvg := policyTotal / time.Duration(samples)

		// Averages should be within 20% of each other
		diff := backoffAvg - policyAvg
		if diff < 0 {
			diff = -diff
		}
		tolerance := time.Duration(float64(policyAvg) * 0.20)
		if diff > tolerance {
			t.Errorf("attempt %d: Backoff avg=%v, Policy avg=%v, diff=%v exceeds 20%%", attempt, backoffAvg, policyAvg, diff)
		}
	}
}
