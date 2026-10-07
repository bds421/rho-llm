package llm_test

import (
	"context"
	"math"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// Break-the-system tests for cost arithmetic (C4, v0.9.3). A consumer saw
// EstimateCost return about -5e-06 for math.MaxInt token counts: output and
// thinking tokens were summed as int before the float conversion and wrapped.
// Contract pinned here: a cost is never negative, NaN or Inf, and is monotonic
// in every input (more tokens or seconds never cost less).

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

// Conversation/Session usage accumulation: running token totals saturate at
// MaxInt instead of wrapping negative, and the running cost stays finite.
func TestUsageAccumulationSaturates(t *testing.T) {
	id := pricedModel(t)
	maxResp := &llm.Response{
		Model: id, InputTokens: math.MaxInt, OutputTokens: math.MaxInt, ThinkingTokens: math.MaxInt,
		CacheCreationTokens: math.MaxInt, CacheReadTokens: math.MaxInt,
	}
	var u llm.Usage
	u.AddResponse(maxResp)
	first := u.Cost
	u.AddResponse(maxResp)
	u.AddBatchResponse(maxResp)
	for name, v := range map[string]int{
		"input": u.InputTokens, "output": u.OutputTokens, "thinking": u.ThinkingTokens,
		"cache write": u.CacheCreationTokens, "cache read": u.CacheReadTokens,
	} {
		if v != math.MaxInt {
			t.Fatalf("%s tokens = %d, want saturated at MaxInt", name, v)
		}
	}
	assertSaneCost(t, "accumulated", u.Cost)
	if u.Cost < first {
		t.Fatalf("accumulated cost shrank: %v < %v", u.Cost, first)
	}

	// A price so high each response saturates at MaxFloat64: the running sum
	// must saturate too, not become +Inf.
	pricey := uniqueModelID("test-only-cost-pricey")
	if err := llm.RegisterModel(llm.ModelInfo{ID: pricey, Provider: "custom", OutputPricePer1M: math.MaxFloat64}); err != nil {
		t.Fatal(err)
	}
	var p llm.Usage
	for range 3 {
		p.AddResponse(&llm.Response{Model: pricey, OutputTokens: math.MaxInt})
	}
	assertSaneCost(t, "saturated running cost", p.Cost)
	if p.Cost != math.MaxFloat64 {
		t.Fatalf("running cost = %v, want saturated at MaxFloat64", p.Cost)
	}
}

// Session.Send feeds Usage.AddResponse; repeated MaxInt turns must not wrap.
func TestSessionUsageSaturatesAcrossTurns(t *testing.T) {
	id := pricedModel(t)
	mock := llm.NewMockClient("custom", id)
	mock.SetResponseFunc(func(llm.Request) (*llm.Response, error) {
		return &llm.Response{Model: id, Content: "ok", StopReason: llm.StopEndTurn, InputTokens: math.MaxInt, OutputTokens: math.MaxInt}, nil
	})
	s := llm.NewSession(mock)
	for range 3 {
		if _, err := s.Send(context.Background(), "hi"); err != nil {
			t.Fatal(err)
		}
	}
	u := s.Usage()
	if u.InputTokens != math.MaxInt || u.OutputTokens != math.MaxInt {
		t.Fatalf("session usage wrapped: %+v", u)
	}
	assertSaneCost(t, "session", u.Cost)
}

// Gemini usageMetadata whose candidates+thoughts would overflow must
// saturate at MaxInt rather than silently drop the thought tokens.
func TestUsageHookGeminiThoughtsSaturate(t *testing.T) {
	srv := geminiServer(t, 200, okTranscript+`,"usageMetadata":{"candidatesTokenCount":9223372036854775806,"thoughtsTokenCount":5}}`)
	var log eventLog
	client := geminiUsageClient(t, srv, "gemini-3.5-flash-lite", log.hook)
	if _, err := transcribe(client); err != nil {
		t.Fatal(err)
	}
	e := log.all()[0]
	if e.OutputTokens != math.MaxInt {
		t.Fatalf("output tokens = %d, want saturated at MaxInt", e.OutputTokens)
	}
	assertSaneCost(t, "gemini event", e.CostUSD)
}

// satAdd adds a non-negative delta to n, saturating at MaxInt.
func satAdd(n, d int) int {
	if d > 0 && n > math.MaxInt-d {
		return math.MaxInt
	}
	return n + d
}

// FuzzEstimateCostMonotonic asserts that EstimateCost is finite, >= 0, and
// monotonic: raising any single input never lowers the cost. Audio tokens are a
// share of InputTokens (billed at the audio rate instead of the text rate), so
// they are raised together with InputTokens.
func FuzzEstimateCostMonotonic(f *testing.F) {
	id := uniqueModelID("test-only-cost-fuzz")
	if err := llm.RegisterModel(llm.ModelInfo{
		ID: id, Provider: "custom",
		InputPricePer1M: 3, OutputPricePer1M: 15, CacheWritePricePer1M: 3.75,
		CacheReadPricePer1M: 0.3, AudioInputPricePer1M: 1.5, AudioPricePerMinute: 120,
	}); err != nil {
		f.Fatal(err)
	}
	f.Add(int64(0), int64(0), int64(0), int64(0), int64(0), int64(0), 0.0, false, uint8(0), int64(1), 1.0)
	f.Add(int64(math.MaxInt64), int64(math.MaxInt64), int64(math.MaxInt64), int64(1), int64(-1), int64(5), math.MaxFloat64, true, uint8(2), int64(math.MaxInt64), 1e300)
	f.Add(int64(1000), int64(-1), int64(math.MaxInt64), int64(0), int64(0), int64(2000), math.NaN(), false, uint8(1), int64(1), math.Inf(1))
	f.Add(int64(math.MinInt64), int64(500), int64(0), int64(math.MaxInt64), int64(math.MaxInt64), int64(math.MaxInt64), -3.5, true, uint8(6), int64(7), 60.0)

	f.Fuzz(func(t *testing.T, in, out, think, cw, cr, audio int64, secs float64, batch bool, field uint8, delta int64, dsecs float64) {
		base := llm.CostInput{
			Model: id, InputTokens: int(in), OutputTokens: int(out), ThinkingTokens: int(think),
			CacheCreateTokens: int(cw), CacheReadTokens: int(cr), AudioInputTokens: int(audio),
			AudioSeconds: secs, Batch: batch,
		}
		c0 := llm.EstimateCost(base)
		assertSaneCost(t, "base", c0)

		// Monotonicity is defined over valid (clamped) inputs: normalize the
		// field being raised so a negative or non-finite start does not make
		// "raise" mean "move toward zero".
		d := int(delta)
		if d < 0 {
			d = -(d + 1) // map to >= 0 without overflow
		}
		nn := func(n int) int { return max(n, 0) }
		raised := base
		switch field % 7 {
		case 0:
			base.InputTokens = nn(base.InputTokens)
			raised.InputTokens = satAdd(base.InputTokens, d)
		case 1:
			base.OutputTokens = nn(base.OutputTokens)
			raised.OutputTokens = satAdd(base.OutputTokens, d)
		case 2:
			base.ThinkingTokens = nn(base.ThinkingTokens)
			raised.ThinkingTokens = satAdd(base.ThinkingTokens, d)
		case 3:
			base.CacheCreateTokens = nn(base.CacheCreateTokens)
			raised.CacheCreateTokens = satAdd(base.CacheCreateTokens, d)
		case 4:
			base.CacheReadTokens = nn(base.CacheReadTokens)
			raised.CacheReadTokens = satAdd(base.CacheReadTokens, d)
		case 5: // add d tokens of audio to the prompt
			base.InputTokens = nn(base.InputTokens)
			base.AudioInputTokens = min(nn(base.AudioInputTokens), base.InputTokens)
			raised.InputTokens = satAdd(base.InputTokens, d)
			raised.AudioInputTokens = base.AudioInputTokens + (raised.InputTokens - base.InputTokens)
		case 6:
			if math.IsNaN(base.AudioSeconds) || math.IsInf(base.AudioSeconds, 0) || base.AudioSeconds < 0 {
				base.AudioSeconds = 0
			}
			ds := dsecs
			if math.IsNaN(ds) || ds < 0 {
				ds = 0
			}
			raised.AudioSeconds = math.Min(base.AudioSeconds+ds, math.MaxFloat64)
		}
		if field%7 != 6 {
			raised.AudioSeconds = base.AudioSeconds
		}
		lo := llm.EstimateCost(base)
		hi := llm.EstimateCost(raised)
		assertSaneCost(t, "normalized base", lo)
		assertSaneCost(t, "raised", hi)
		if hi < lo {
			t.Fatalf("not monotonic in field %d: cost(%+v)=%v < cost(%+v)=%v", field%7, raised, hi, base, lo)
		}
	})
}
