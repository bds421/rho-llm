package llm_test

import (
	"math"
	"testing"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider"
	_ "github.com/bds421/rho-llm/provider/openaibatch"
)

func TestEstimateCostBatchIsHalf(t *testing.T) {
	in := llm.CostInput{Model: "gpt-4.1-mini", InputTokens: 1_000_000, OutputTokens: 1_000_000}
	full := llm.EstimateCost(in)
	in.Batch = true
	half := llm.EstimateCost(in)
	if full <= 0 {
		t.Fatal("expected nonzero sync cost (model must have pricing)")
	}
	if math.Abs(half-full*0.5) > 1e-9 {
		t.Fatalf("batch cost %v should be half of sync cost %v", half, full)
	}
}

func assertSaneCost(t *testing.T, label string, cost float64) {
	t.Helper()
	if cost < 0 || math.IsNaN(cost) || math.IsInf(cost, 0) {
		t.Fatalf("%s: cost = %v, want finite and >= 0", label, cost)
	}
}

// pricedModel registers a model with a distinct non-zero price for every
// billable dimension.
func pricedModel(t *testing.T) string {
	t.Helper()
	id := uniqueModelID("test-only-cost-overflow")
	if err := llm.RegisterModel(llm.ModelInfo{
		ID: id, Provider: "custom",
		InputPricePer1M: 3, OutputPricePer1M: 15, CacheWritePricePer1M: 3.75,
		CacheReadPricePer1M: 0.3, AudioInputPricePer1M: 1.5, AudioPricePerMinute: 0.006,
	}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}
	return id
}

// Each token field at math.MaxInt / math.MaxInt64 alone must cost about
// count*price/1e6 — positive, finite, and never less than half the count.
func TestEstimateCostMaxIntInEachField(t *testing.T) {
	id := pricedModel(t)
	const maxF = float64(math.MaxInt)
	cases := map[string]struct {
		set  func(*llm.CostInput, int)
		want float64
	}{
		"input":       {func(c *llm.CostInput, n int) { c.InputTokens = n }, maxF * 3 / 1e6},
		"output":      {func(c *llm.CostInput, n int) { c.OutputTokens = n }, maxF * 15 / 1e6},
		"thinking":    {func(c *llm.CostInput, n int) { c.ThinkingTokens = n }, maxF * 15 / 1e6},
		"cache write": {func(c *llm.CostInput, n int) { c.CacheCreateTokens = n }, maxF * 3.75 / 1e6},
		"cache read":  {func(c *llm.CostInput, n int) { c.CacheReadTokens = n }, maxF * 0.3 / 1e6},
		"audio share": {func(c *llm.CostInput, n int) { c.InputTokens = n; c.AudioInputTokens = n }, maxF * 1.5 / 1e6},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			for _, n := range []int{math.MaxInt, int(int64(math.MaxInt64))} {
				full := llm.CostInput{Model: id}
				tc.set(&full, n)
				half := llm.CostInput{Model: id}
				tc.set(&half, n/2)
				got := llm.EstimateCost(full)
				assertSaneCost(t, name, got)
				if math.Abs(got-tc.want) > tc.want*1e-9 {
					t.Fatalf("%s MaxInt cost = %v, want %v", name, got, tc.want)
				}
				if h := llm.EstimateCost(half); got < h {
					t.Fatalf("%s not monotonic: cost(MaxInt)=%v < cost(MaxInt/2)=%v", name, got, h)
				}
			}
		})
	}
}

// Fields that are summed before pricing must not wrap: OutputTokens and
// ThinkingTokens share the output rate, and MaxInt+MaxInt overflowed to -2.
func TestEstimateCostSummedFieldsDoNotWrap(t *testing.T) {
	id := pricedModel(t)
	const maxF = float64(math.MaxInt)
	for _, batch := range []bool{false, true} {
		cases := []struct {
			name string
			in   llm.CostInput
			want float64
		}{
			{"output+thinking", llm.CostInput{OutputTokens: math.MaxInt, ThinkingTokens: math.MaxInt}, 2 * maxF * 15 / 1e6},
			{"output+thinking off by one", llm.CostInput{OutputTokens: math.MaxInt, ThinkingTokens: 1}, (maxF + 1) * 15 / 1e6},
			{"everything", llm.CostInput{
				InputTokens: math.MaxInt, OutputTokens: math.MaxInt, ThinkingTokens: math.MaxInt,
				CacheCreateTokens: math.MaxInt, CacheReadTokens: math.MaxInt, AudioInputTokens: math.MaxInt,
				AudioSeconds: math.MaxFloat64,
			}, maxF*(1.5+15+15+3.75+0.3)/1e6 + math.MaxFloat64/60*0.006},
		}
		for _, tc := range cases {
			tc.in.Model = id
			tc.in.Batch = batch
			want := tc.want
			if batch {
				want *= 0.5
			}
			got := llm.EstimateCost(tc.in)
			assertSaneCost(t, tc.name, got)
			if math.Abs(got-want) > want*1e-9 {
				t.Fatalf("%s (batch=%v) = %v, want %v", tc.name, batch, got, want)
			}
		}
	}
}

// Negative counts (MinInt, the -1 sentinel) count as 0 and never subtract.
func TestEstimateCostNegativeInputsClampToZero(t *testing.T) {
	id := pricedModel(t)
	for _, n := range []int{-1, math.MinInt, math.MinInt + 1} {
		got := llm.EstimateCost(llm.CostInput{
			Model: id, InputTokens: n, OutputTokens: n, ThinkingTokens: n,
			CacheCreateTokens: n, CacheReadTokens: n, AudioInputTokens: n, AudioSeconds: float64(n),
		})
		if got != 0 {
			t.Fatalf("all fields %d: cost = %v, want 0", n, got)
		}
		// A negative in one field must not reduce what the others cost.
		base := llm.EstimateCost(llm.CostInput{Model: id, InputTokens: 1000, OutputTokens: 1000})
		mixed := llm.EstimateCost(llm.CostInput{Model: id, InputTokens: 1000, OutputTokens: 1000, ThinkingTokens: n, AudioInputTokens: n})
		if mixed != base {
			t.Fatalf("negative field %d changed the cost: %v vs %v", n, mixed, base)
		}
	}
}

// AudioSeconds: NaN, ±Inf, negatives and -0 count as 0; huge finite values
// stay finite even when seconds/60*price exceeds MaxFloat64.
func TestEstimateCostAudioSecondsHostile(t *testing.T) {
	id := pricedModel(t)
	for _, s := range []float64{math.NaN(), math.Inf(1), math.Inf(-1), -1, math.Copysign(0, -1), -math.MaxFloat64} {
		if got := llm.EstimateCost(llm.CostInput{Model: id, AudioSeconds: s}); got != 0 {
			t.Fatalf("AudioSeconds %v: cost = %v, want 0", s, got)
		}
	}
	got := llm.EstimateCost(llm.CostInput{Model: id, AudioSeconds: math.MaxFloat64})
	assertSaneCost(t, "MaxFloat64 seconds", got)
	if got <= 0 {
		t.Fatalf("MaxFloat64 seconds priced at %v", got)
	}

	// A per-minute price above 60 makes MaxFloat64/60*price overflow to +Inf.
	pricey := uniqueModelID("test-only-cost-pricey-minute")
	if err := llm.RegisterModel(llm.ModelInfo{ID: pricey, Provider: "custom", AudioPricePerMinute: 1e6}); err != nil {
		t.Fatal(err)
	}
	huge := llm.EstimateCost(llm.CostInput{Model: pricey, AudioSeconds: math.MaxFloat64})
	assertSaneCost(t, "MaxFloat64 seconds at $1e6/min", huge)
	if smaller := llm.EstimateCost(llm.CostInput{Model: pricey, AudioSeconds: math.MaxFloat64 / 2}); huge < smaller {
		t.Fatalf("not monotonic in seconds: %v < %v", huge, smaller)
	}
}

// RegisterModel accepts any float, so a hostile or typo'd registration (NaN,
// ±Inf, negative, absurdly large prices) must still yield a sane cost.
func TestEstimateCostHostilePrices(t *testing.T) {
	prices := map[string]float64{
		"NaN": math.NaN(), "+Inf": math.Inf(1), "-Inf": math.Inf(-1),
		"negative": -15, "MaxFloat64": math.MaxFloat64,
	}
	for name, p := range prices {
		t.Run(name, func(t *testing.T) {
			id := uniqueModelID("test-only-cost-hostile-price")
			if err := llm.RegisterModel(llm.ModelInfo{
				ID: id, Provider: "custom",
				InputPricePer1M: p, OutputPricePer1M: p, CacheWritePricePer1M: p,
				CacheReadPricePer1M: p, AudioInputPricePer1M: p, AudioPricePerMinute: p,
			}); err != nil {
				t.Fatal(err)
			}
			for _, n := range []int{0, 1, 1_000_000, math.MaxInt} {
				in := llm.CostInput{
					Model: id, InputTokens: n, OutputTokens: n, ThinkingTokens: n,
					CacheCreateTokens: n, CacheReadTokens: n, AudioInputTokens: n / 2, AudioSeconds: float64(n),
				}
				assertSaneCost(t, name, llm.EstimateCost(in))
				in.Batch = true
				assertSaneCost(t, name+" batch", llm.EstimateCost(in))
			}
		})
	}
}

// A model with every price at zero, and an unknown model, cost exactly 0
// whatever the counts.
func TestEstimateCostZeroPriceAndUnknownModel(t *testing.T) {
	free := uniqueModelID("test-only-cost-free")
	if err := llm.RegisterModel(llm.ModelInfo{ID: free, Provider: "custom"}); err != nil {
		t.Fatal(err)
	}
	maxed := llm.CostInput{
		InputTokens: math.MaxInt, OutputTokens: math.MaxInt, ThinkingTokens: math.MaxInt,
		CacheCreateTokens: math.MaxInt, CacheReadTokens: math.MaxInt, AudioInputTokens: math.MaxInt,
		AudioSeconds: math.MaxFloat64,
	}
	for _, model := range []string{free, "no-such-model-c4", ""} {
		in := maxed
		in.Model = model
		if got := llm.EstimateCost(in); got != 0 {
			t.Fatalf("model %q: cost = %v, want 0", model, got)
		}
	}
}

// EstimateCost must NOT silently normalize a whitespace-padded model ID to the
// trimmed registry key — a padded ID is an unknown model and must cost 0, never
// quietly resolve to the real price (which would mask a caller bug, or worse let
// a padded ID dodge a price lookup). Pins the no-silent-trim contract.
func TestEstimateCostDoesNotSilentlyNormalizeWhitespace(t *testing.T) {
	const M = 1_000_000
	if base := llm.EstimateCost(llm.CostInput{Model: "glm-5.2", InputTokens: M}); base <= 0 {
		t.Fatalf("precondition: glm-5.2 must have a price, got %v", base)
	}
	for _, padded := range []string{" glm-5.2", "glm-5.2 ", " glm-5.2 ", "\tglm-5.2"} {
		if c := llm.EstimateCost(llm.CostInput{Model: padded, InputTokens: M, OutputTokens: M}); c != 0 {
			t.Errorf("EstimateCost(%q) = %v, want 0 — padded ID must not silently resolve to the trimmed model", padded, c)
		}
	}
}

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

// TestEstimateCost verifies cost calculation for known models.
func TestEstimateCost(t *testing.T) {
	tests := []struct {
		name    string
		input   llm.CostInput
		wantMin float64
		wantMax float64
	}{
		// claude-sonnet-4-6: $3/1M input, $15/1M output
		// 1000 input: 3 * 1000/1M = 0.003, 500 output: 15 * 500/1M = 0.0075
		{"sonnet basic", llm.CostInput{Model: "claude-sonnet-4-6", InputTokens: 1000, OutputTokens: 500}, 0.0104, 0.0106},
		// gemini-2.5-flash-lite: $0.10/1M input, $0.40/1M output
		// 10000 input: 0.001, 5000 output: 0.002 = 0.003
		{"flash-lite", llm.CostInput{Model: "gemini-2.5-flash-lite", InputTokens: 10000, OutputTokens: 5000}, 0.002, 0.004},
		// unknown model: 0
		{"unknown", llm.CostInput{Model: "unknown-model", InputTokens: 1000, OutputTokens: 500}, 0, 0},
		// claude-opus-4-6: $5/1M input, $25/1M output
		// 10000 input: 0.05, 2000 output: 0.05
		{"opus basic", llm.CostInput{Model: "claude-opus-4-6", InputTokens: 10000, OutputTokens: 2000}, 0.099, 0.101},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cost := llm.EstimateCost(tc.input)
			if cost < tc.wantMin || cost > tc.wantMax {
				t.Errorf("EstimateCost(%+v) = %f, want [%f, %f]",
					tc.input, cost, tc.wantMin, tc.wantMax)
			}
		})
	}
}

// TestEstimateCostNegativeTokens verifies negative tokens are clamped to 0.
func TestEstimateCostNegativeTokens(t *testing.T) {
	// -1 tokens (sentinel for "not reported") should not produce negative cost
	cost := llm.EstimateCost(llm.CostInput{Model: "claude-sonnet-4-6", InputTokens: -1, OutputTokens: -1})
	if cost < 0 {
		t.Errorf("EstimateCost with negative tokens = %f, want >= 0", cost)
	}
	if cost != 0 {
		t.Errorf("EstimateCost with -1 tokens = %f, want 0", cost)
	}
}

// TestEstimateCostWithThinkingTokens verifies thinking tokens are priced at output rate.
func TestEstimateCostWithThinkingTokens(t *testing.T) {
	input := llm.CostInput{
		Model:          "gemini-2.5-flash",
		InputTokens:    1000,
		OutputTokens:   500,
		ThinkingTokens: 2000,
	}
	cost := llm.EstimateCost(input)

	// input: 1000 * 0.30 / 1M = 0.0003
	// output+thinking: (500+2000) * 2.50 / 1M = 0.00625
	// total: 0.00655
	expected := 0.00655
	if cost < expected-0.0005 || cost > expected+0.0005 {
		t.Errorf("EstimateCost with thinking tokens = %f, want ~%f", cost, expected)
	}
}

func TestIterationPricing(t *testing.T) {
	near := func(a, b float64) bool { return math.Abs(a-b) < 1e-9 }
	// Haiku 5.5: input $0.10, output $0.50, 5m write $0.125, 1h write $0.20, read $0.01, search $10/1k.
	it := []llm.UsageIteration{{Type: "message", Model: "claude-haiku-5-5", InputTokens: 1_000_000,
		CacheCreationTokens: 1_000_000, CacheCreation1hTokens: 400_000}}
	got := llm.EstimateCost(llm.CostInput{Model: "claude-haiku-5-5", Iterations: it})
	if want := 0.10 + 0.6*0.125 + 0.4*0.20; !near(got, want) {
		t.Errorf("1h share per iteration: %v, want %v", got, want)
	}
	// An unregistered top-level model: searches priced at the iteration's model.
	got = llm.EstimateCost(llm.CostInput{Model: "claude-example-1", Iterations: it[:1], WebSearchRequests: 2})
	if want := 0.10 + 0.6*0.125 + 0.4*0.20 + 0.02; !near(got, want) {
		t.Errorf("searches with unregistered top model: %v, want %v", got, want)
	}
	if got := llm.EstimateCost(llm.CostInput{Model: "claude-example-1", WebSearchRequests: 2}); got != 0 {
		t.Errorf("nothing registered: %v, want 0", got)
	}
}
