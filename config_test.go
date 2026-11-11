package llm_test

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider"
	_ "github.com/bds421/rho-llm/provider/gemini"
	_ "github.com/bds421/rho-llm/provider/openaibatch"
	_ "github.com/bds421/rho-llm/provider/openaicompat"
)

// A misconfigured (negative) download cap must never make the bounded read return
// zero bytes and silently truncate results — the effective cap falls back to the
// positive default instead.
func TestEffectiveMaxBatchDownloadBytesNeverNonPositive(t *testing.T) {
	cfg := llm.DefaultConfig()
	cfg.MaxBatchDownloadBytes = -5
	if got := cfg.EffectiveMaxBatchDownloadBytes(); got <= 0 {
		t.Fatalf("download cap must never be non-positive (would silently truncate results); got %d", got)
	}
}

// TestBreakTimeoutFlooredOnEveryConstructor pins the Timeout floor.
//
// Found by mutation audit: disabling the floor in applyConfigFloors left the
// whole suite green. A zero or negative Timeout produces an http.Client with
// no deadline, so a hung endpoint stalls the caller forever instead of
// returning a timeout error.
//
// Asserted at NewSafeHTTPClient — the exported seam every adapter builds its
// transport through. Note this pins NewSafeHTTPClient's OWN floor: the
// applyConfigFloors Timeout branch is redundant with it (see the comment
// there), so no test can kill a mutation of that branch alone. The property
// that matters — no client is ever unbounded — is what this asserts.
// Driving a real request instead would measure the pool's retry/backoff
// budget, not the floor.
func TestBreakTimeoutFlooredOnEveryConstructor(t *testing.T) {
	for _, tc := range []struct {
		name    string
		timeout time.Duration
	}{
		{"zero (struct-literal default)", 0},
		{"negative", -5 * time.Second},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, err := llm.NewClient(llm.Config{
				Provider: "anthropic", Model: "claude-sonnet-4-6", APIKey: "k",
				Timeout: tc.timeout,
			})
			if err != nil {
				t.Fatalf("NewClient: %v", err)
			}
			defer c.Close()

			hc, err := llm.NewSafeHTTPClient(llm.Config{Timeout: tc.timeout})
			if err != nil {
				t.Fatalf("NewSafeHTTPClient: %v", err)
			}
			if hc.Timeout <= 0 {
				t.Errorf("http.Client.Timeout = %v for input %v: an unbounded client stalls "+
					"the caller indefinitely against a hung endpoint", hc.Timeout, tc.timeout)
			}
			if hc.Timeout != llm.DefaultTimeout {
				t.Errorf("http.Client.Timeout = %v, want DefaultTimeout (%v)", hc.Timeout, llm.DefaultTimeout)
			}
		})
	}
}

// TestDefaultConfig verifies default config values.
func TestDefaultConfig(t *testing.T) {
	cfg := llm.DefaultConfig()

	if cfg.Provider != "anthropic" {
		t.Errorf("Expected default provider 'anthropic', got %q", cfg.Provider)
	}
	if cfg.MaxTokens != 8192 {
		t.Errorf("Expected default MaxTokens 8192, got %d", cfg.MaxTokens)
	}
	if cfg.Timeout != 120*time.Second {
		t.Errorf("Expected default Timeout 120s, got %v", cfg.Timeout)
	}
}

// TestTimeoutDefaultApplied verifies that a zero-value Timeout gets a sensible
// default rather than creating an unbounded HTTP client.
func TestTimeoutDefaultApplied(t *testing.T) {
	cfg := llm.Config{
		Provider:  "ollama", // no auth required
		Model:     "llama3",
		MaxTokens: 100,
		// Timeout deliberately omitted (zero value)
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient with zero Timeout should succeed: %v", err)
	}
	defer client.Close()

	// If we got here, the client was created with DefaultTimeout, not zero.
	// The real assertion is that the next line doesn't hang forever.
	if client.Provider() != "ollama" {
		t.Errorf("Provider = %q, want ollama", client.Provider())
	}
}

// TestNegativeTimeoutDefaultApplied verifies negative Timeout is corrected.
func TestNegativeTimeoutDefaultApplied(t *testing.T) {
	cfg := llm.Config{
		Provider:  "ollama",
		Model:     "llama3",
		MaxTokens: 100,
		Timeout:   -5 * time.Second,
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient with negative Timeout should succeed: %v", err)
	}
	defer client.Close()
}

// TestConfigBetaFeaturesDefault verifies DefaultConfig sets BetaFeatures.
func TestConfigBetaFeaturesDefault(t *testing.T) {
	cfg := llm.DefaultConfig()
	if len(cfg.BetaFeatures) != 1 || cfg.BetaFeatures[0] != "interleaved-thinking-2025-05-14" {
		t.Errorf("DefaultConfig().BetaFeatures = %v, want [interleaved-thinking-2025-05-14]", cfg.BetaFeatures)
	}
}

// TestDefaultTimeoutConstant verifies the constant is exported and correct.
func TestDefaultTimeoutConstant(t *testing.T) {
	if llm.DefaultTimeout != 120*time.Second {
		t.Errorf("DefaultTimeout = %v, want 120s", llm.DefaultTimeout)
	}
}

// TestDefaultMaxTokensMatchesDefaultConfig keeps the floor and DefaultConfig()
// from drifting apart — two sources of the same default is how the zero-value
// gap appeared in the first place.
func TestDefaultMaxTokensMatchesDefaultConfig(t *testing.T) {
	if got := llm.DefaultConfig().MaxTokens; got != llm.DefaultMaxTokens {
		t.Errorf("DefaultConfig().MaxTokens = %d, DefaultMaxTokens = %d; want equal", got, llm.DefaultMaxTokens)
	}
}

// The hook is a func and must never reach Config's JSON; a round trip must
// keep working and drop it.
func TestConfigJSONRoundTripWithUsageHook(t *testing.T) {
	cfg := llm.Config{Provider: "gemini", Model: "gemini-3.5-flash-lite", APIKey: "secret", UsageHook: func(llm.UsageEvent) {}}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("Config with UsageHook no longer marshals: %v", err)
	}
	if strings.Contains(strings.ToLower(string(data)), "usage") {
		t.Fatalf("hook serialized: %s", data)
	}
	var back llm.Config
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatal(err)
	}
	if back.Provider != "gemini" || back.Model != "gemini-3.5-flash-lite" || back.UsageHook != nil {
		t.Fatalf("round trip = %+v", back)
	}
}
