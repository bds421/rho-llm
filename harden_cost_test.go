package llm_test

// Hardening pass 14 — EstimateCost math across all token types + sentinel clamp.

import (
	"math"
	"testing"

	llm "github.com/bds421/rho-llm"
)

func TestEstimateCostAllTokenTypesAndSentinel(t *testing.T) {
	id := uniqueModelID("test-only-cost-allTypes")
	if err := llm.RegisterModel(llm.ModelInfo{
		ID: id, Provider: "custom",
		InputPricePer1M: 3, OutputPricePer1M: 15, CacheWritePricePer1M: 3.75, CacheReadPricePer1M: 0.3,
	}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}

	// 1M of each; thinking is billed at the output rate (output+thinking)*Pout.
	got := llm.EstimateCost(llm.CostInput{
		Model: id, InputTokens: 1_000_000, OutputTokens: 1_000_000, ThinkingTokens: 1_000_000,
		CacheCreateTokens: 1_000_000, CacheReadTokens: 1_000_000,
	})
	want := 3.0 + (15.0 + 15.0) + 3.75 + 0.3 // = 37.05
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("EstimateCost all-types = %v, want %v", got, want)
	}
	if math.IsNaN(got) || math.IsInf(got, 0) {
		t.Fatalf("EstimateCost produced a non-finite cost: %v", got)
	}

	// The TokensNotReported (-1) sentinel and stray negatives must clamp to 0,
	// never producing a negative cost.
	sentinel := llm.EstimateCost(llm.CostInput{
		Model: id, InputTokens: -1, OutputTokens: -1, ThinkingTokens: -1, CacheCreateTokens: -1, CacheReadTokens: -1,
	})
	if sentinel != 0 {
		t.Fatalf("EstimateCost with all -1 sentinels = %v, want 0 (clamped)", sentinel)
	}
}
