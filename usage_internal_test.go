package llm

import (
	"context"
	"math"
	"testing"
)

// C4 (v0.9.3): repeated ReportModalityUsage calls for one operation are
// summed. Two MaxFloat64-second reports used to sum to +Inf, which
// EstimateCost then treats as "non-finite = 0" — so more audio cost $0.
// Seconds must saturate at MaxFloat64, and token sums at MaxInt.
func TestReportModalityUsageSaturatesSums(t *testing.T) {
	recorder := &usageRecorder{}
	ctx := context.WithValue(context.Background(), usageRecorderKey{}, recorder)
	for range 3 {
		ReportModalityUsage(ctx, ModalityUsage{
			InputTokens: math.MaxInt, OutputTokens: math.MaxInt,
			AudioInputTokens: math.MaxInt, AudioSeconds: math.MaxFloat64,
		})
	}
	got := recorder.snapshot()
	if got.InputTokens != math.MaxInt || got.OutputTokens != math.MaxInt || got.AudioInputTokens != math.MaxInt {
		t.Fatalf("token sums wrapped: %+v", got)
	}
	if got.AudioSeconds != math.MaxFloat64 {
		t.Fatalf("AudioSeconds = %v, want saturated at MaxFloat64", got.AudioSeconds)
	}
	cost := EstimateCost(CostInput{Model: "whisper-1", AudioSeconds: got.AudioSeconds})
	if cost <= 0 || math.IsInf(cost, 0) || math.IsNaN(cost) {
		t.Fatalf("summed audio cost = %v, want finite and > 0", cost)
	}
}
