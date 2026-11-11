package llm_test

import (
	"context"
	"errors"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider"
	_ "github.com/bds421/rho-llm/provider/gemini"
	_ "github.com/bds421/rho-llm/provider/openaicompat"
)

func TestUsageAddResponseNilSafe(t *testing.T) {
	var u llm.Usage
	u.AddResponse(nil) // must not panic
	if u.InputTokens != 0 || u.Cost != 0 {
		t.Errorf("nil response mutated usage: %+v", u)
	}
}

func TestUsageIgnoresTokensNotReportedSentinel(t *testing.T) {
	var u llm.Usage
	u.AddResponse(&llm.Response{
		Model:        "claude-sonnet-4-6",
		InputTokens:  llm.TokensNotReported, // -1 sentinel from a stream without usage
		OutputTokens: llm.TokensNotReported,
	})
	if u.InputTokens != 0 || u.OutputTokens != 0 {
		t.Errorf("TokensNotReported polluted usage: %+v", u)
	}
	if u.Cost < 0 {
		t.Errorf("negative cost from sentinel tokens: %v", u.Cost)
	}
}

func TestUsageAddBatchResponseHalfCostSameTokens(t *testing.T) {
	resp := &llm.Response{Model: "gpt-4.1-mini", InputTokens: 1_000_000, OutputTokens: 1_000_000}

	var syncU, batchU llm.Usage
	syncU.AddResponse(resp)
	batchU.AddBatchResponse(resp)

	if syncU.InputTokens != batchU.InputTokens || syncU.OutputTokens != batchU.OutputTokens {
		t.Fatalf("token accumulation must be identical: sync=%+v batch=%+v", syncU, batchU)
	}
	if syncU.Cost <= 0 {
		t.Fatal("expected a nonzero sync cost (model must have pricing)")
	}
	if math.Abs(batchU.Cost-syncU.Cost*0.5) > 1e-9 {
		t.Fatalf("batch cost %v must be half of sync cost %v", batchU.Cost, syncU.Cost)
	}
}

func TestUsageAddBatchResponseClampsTokensNotReported(t *testing.T) {
	// A streamed turn that never reported usage carries the -1 sentinel; folding it at
	// batch pricing must not push tokens or cost negative.
	resp := &llm.Response{Model: "gpt-4.1-mini", InputTokens: llm.TokensNotReported, OutputTokens: llm.TokensNotReported}
	var u llm.Usage
	u.AddBatchResponse(resp)
	if u.InputTokens != 0 || u.OutputTokens != 0 {
		t.Fatalf("sentinel (-1) must clamp to 0, got %+v", u)
	}
	if u.Cost < 0 {
		t.Fatalf("cost must never be negative, got %v", u.Cost)
	}
}

// Sanity: AddBatchResponse/AddResponse are nil-safe (shared add()).
func TestAddBatchResponseNilSafe(t *testing.T) {
	var u llm.Usage
	u.AddBatchResponse(nil)
	u.AddResponse(nil)
	if (u != llm.Usage{}) {
		t.Fatalf("nil response must be a no-op, got %+v", u)
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

// TestTokensNotReported verifies the sentinel constant value and StreamEvent usage.
func TestTokensNotReported(t *testing.T) {
	if llm.TokensNotReported != -1 {
		t.Errorf("TokensNotReported = %d, want -1", llm.TokensNotReported)
	}

	// Verify sentinel is distinguishable from zero in StreamEvent
	event := llm.StreamEvent{
		Type:         llm.EventDone,
		InputTokens:  llm.TokensNotReported,
		OutputTokens: llm.TokensNotReported,
	}
	if event.InputTokens == 0 {
		t.Error("TokensNotReported should not equal 0")
	}
	if event.OutputTokens == 0 {
		t.Error("TokensNotReported should not equal 0")
	}
}

type eventLog struct {
	mu     sync.Mutex
	events []llm.UsageEvent
}

func (l *eventLog) hook(e llm.UsageEvent) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.events = append(l.events, e)
}

func (l *eventLog) all() []llm.UsageEvent {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]llm.UsageEvent(nil), l.events...)
}

// geminiBody answers every generateContent call with body (status 200) or, for
// a non-200 status, a Gemini error object.
func geminiServer(t *testing.T, status int, body string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func geminiUsageClient(t *testing.T, srv *httptest.Server, model string, hook llm.UsageHook) llm.ModalityClient {
	t.Helper()
	client, err := llm.NewModalityClient(llm.Config{
		Provider: "gemini", Model: model, APIKey: "k", BaseURL: srv.URL,
		DisableProxy: true, DisableRetries: true, Timeout: 5 * time.Second, UsageHook: hook,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = client.Close() })
	return client
}

const okTranscript = `{"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":"zwei Weckerl"}]}}]`

func transcribe(client llm.ModalityClient) (string, error) {
	return client.TranscribeAudio(context.Background(), llm.TranscriptionRequest{Audio: fbWAV, MediaType: "audio/wav"})
}

// A panicking hook must neither crash the caller nor change the result: the
// transcript and nil error come back exactly as without a hook, and a failed
// call keeps its own error rather than a panic.
func TestUsageHookPanicDoesNotChangeTheResult(t *testing.T) {
	srv := geminiServer(t, 200, okTranscript+`,"usageMetadata":{"promptTokenCount":10}}`)
	var calls int
	client := geminiUsageClient(t, srv, "gemini-3.5-flash-lite", func(llm.UsageEvent) { calls++; panic("hook bug") })
	text, err := transcribe(client)
	if err != nil || text != "zwei Weckerl" {
		t.Fatalf("panicking hook changed the result: %q %v", text, err)
	}
	if calls != 1 {
		t.Fatalf("hook calls = %d, want 1", calls)
	}

	failing := geminiServer(t, 503, `{"error":{"code":503,"message":"overloaded"}}`)
	client = geminiUsageClient(t, failing, "gemini-3.5-flash-lite", func(llm.UsageEvent) { panic(42) })
	if _, err := transcribe(client); !llm.IsOverloaded(err) {
		t.Fatalf("panicking hook replaced the provider error: %v", err)
	}
}

// No hook configured: the call must work and nothing may dereference nil.
func TestUsageHookNilIsSafe(t *testing.T) {
	srv := geminiServer(t, 200, okTranscript+`,"usageMetadata":{"promptTokenCount":10}}`)
	client := geminiUsageClient(t, srv, "gemini-3.5-flash-lite", nil)
	if text, err := transcribe(client); err != nil || text != "zwei Weckerl" {
		t.Fatalf("%q %v", text, err)
	}
}

// A provider failure is still an attempt: one event, Err set to the real
// error, no usage, zero cost.
func TestUsageHookReportsFailedAttemptWithErrAndZeroCost(t *testing.T) {
	srv := geminiServer(t, 400, `{"error":{"code":400,"message":"Thinking is not enabled for this model"}}`)
	var log eventLog
	client := geminiUsageClient(t, srv, "gemini-3.5-transcribe", log.hook)
	_, callErr := transcribe(client)
	if callErr == nil {
		t.Fatal("expected the 400")
	}
	events := log.all()
	if len(events) != 1 {
		t.Fatalf("events = %d, want 1 for the failed attempt", len(events))
	}
	e := events[0]
	if e.Err == nil || e.Err.Error() != callErr.Error() {
		t.Fatalf("event Err = %v, call err = %v", e.Err, callErr)
	}
	if e.CostUSD != 0 || e.InputTokens != 0 || e.OutputTokens != 0 {
		t.Fatalf("failed attempt carries usage/cost: %+v", e)
	}
	if e.Operation != llm.OperationTranscription || e.Provider != "gemini" || e.Model != "gemini-3.5-transcribe" || e.Attempt != 0 || e.Fallback {
		t.Fatalf("event identity wrong: %+v", e)
	}
	if e.Latency <= 0 {
		t.Fatalf("latency not measured: %v", e.Latency)
	}
}

// A billed-but-unusable response (finish MAX_TOKENS) fails the call, yet its
// tokens were charged: the event must carry Err AND the cost.
func TestUsageHookKeepsCostOfABilledFailure(t *testing.T) {
	srv := geminiServer(t, 200, `{"candidates":[{"finishReason":"MAX_TOKENS","content":{"parts":[{"text":"zwei"}]}}],"usageMetadata":{"promptTokenCount":1000,"candidatesTokenCount":100,"promptTokensDetails":[{"modality":"AUDIO","tokenCount":1000}]}}`)
	var log eventLog
	client := geminiUsageClient(t, srv, "gemini-3.5-transcribe", log.hook)
	if _, err := transcribe(client); err == nil {
		t.Fatal("truncated transcript accepted")
	}
	events := log.all()
	if len(events) != 1 || events[0].Err == nil || events[0].CostUSD <= 0 {
		t.Fatalf("billed failure lost its cost: %+v", events)
	}
}

// Failover must be visible: a failed primary and a successful fallback are two
// events, Attempt 0/1, Fallback false/true, and the fallback (no hook of its
// own) inherits the primary's hook.
func TestUsageHookFailoverEmitsOneEventPerAttempt(t *testing.T) {
	s := newGeminiStub(t, map[string]string{"gemini-3.5-transcribe": "broken"})
	var log eventLog
	primary := s.cfg("gemini-3.5-transcribe")
	primary.UsageHook = log.hook
	client, err := llm.NewFallbackModalityClient(primary, s.cfg("gemini-3.5-flash-lite"))
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	if text, err := transcribe(client); err != nil || text != "from gemini-3.5-flash-lite" {
		t.Fatalf("%q %v", text, err)
	}
	events := log.all()
	if len(events) != 2 {
		t.Fatalf("events = %d (%+v), want 2", len(events), events)
	}
	if events[0].Attempt != 0 || events[0].Fallback || events[0].Err == nil || events[0].Model != "gemini-3.5-transcribe" {
		t.Fatalf("primary event wrong: %+v", events[0])
	}
	if events[1].Attempt != 1 || !events[1].Fallback || events[1].Err != nil || events[1].Model != "gemini-3.5-flash-lite" {
		t.Fatalf("fallback event wrong: %+v", events[1])
	}
}

// A fallback with its own hook reports there, not to the primary's.
func TestUsageHookEachDeploymentUsesItsOwnHook(t *testing.T) {
	s := newGeminiStub(t, map[string]string{"gemini-3.5-transcribe": "overloaded"})
	var first, second eventLog
	primary, fallback := s.cfg("gemini-3.5-transcribe"), s.cfg("gemini-3.5-flash-lite")
	primary.UsageHook, fallback.UsageHook = first.hook, second.hook
	client, err := llm.NewFallbackModalityClient(primary, fallback)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	if _, err := transcribe(client); err != nil {
		t.Fatal(err)
	}
	if a, b := first.all(), second.all(); len(a) != 1 || a[0].Attempt != 0 || len(b) != 1 || b[0].Attempt != 1 || !b[0].Fallback {
		t.Fatalf("primary hook %+v, fallback hook %+v", a, b)
	}
}

// Hostile usageMetadata: absent, null, wrong types, overflowing numbers,
// negative counts, an audio share larger than the prompt. The transcript must
// survive every one, and the event must carry zero or capped tokens — never
// negatives or an audio count above the input count.
func TestUsageHookHostileGeminiUsageMetadata(t *testing.T) {
	cases := map[string]string{
		"absent":         okTranscript + `}`,
		"null":           okTranscript + `,"usageMetadata":null}`,
		"string":         okTranscript + `,"usageMetadata":"lots"}`,
		"wrong types":    okTranscript + `,"usageMetadata":{"promptTokenCount":"12","candidatesTokenCount":true}}`,
		"overflow":       okTranscript + `,"usageMetadata":{"promptTokenCount":1e300}}`,
		"details string": okTranscript + `,"usageMetadata":{"promptTokenCount":5,"promptTokensDetails":"AUDIO"}}`,
		"negative":       okTranscript + `,"usageMetadata":{"promptTokenCount":-5,"candidatesTokenCount":-7,"thoughtsTokenCount":-1,"promptTokensDetails":[{"modality":"AUDIO","tokenCount":-9}]}}`,
		"audio > input":  okTranscript + `,"usageMetadata":{"promptTokenCount":10,"promptTokensDetails":[{"modality":"AUDIO","tokenCount":500}]}}`,
		"maxint thought": okTranscript + `,"usageMetadata":{"candidatesTokenCount":9223372036854775807,"thoughtsTokenCount":9223372036854775807}}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			srv := geminiServer(t, 200, body)
			var log eventLog
			client := geminiUsageClient(t, srv, "gemini-3.5-flash-lite", log.hook)
			text, err := transcribe(client)
			if err != nil || text != "zwei Weckerl" {
				t.Fatalf("malformed usage broke the transcript: %q %v", text, err)
			}
			events := log.all()
			if len(events) != 1 {
				t.Fatalf("events = %d", len(events))
			}
			e := events[0]
			if e.InputTokens < 0 || e.OutputTokens < 0 || e.AudioInputTokens < 0 || e.AudioInputTokens > e.InputTokens {
				t.Fatalf("garbage usage: %+v", e)
			}
			if e.CostUSD < 0 || math.IsNaN(e.CostUSD) || math.IsInf(e.CostUSD, 0) {
				t.Fatalf("garbage cost: %v", e.CostUSD)
			}
			switch name {
			case "absent", "null", "string", "wrong types", "overflow", "negative":
				if e.InputTokens != 0 || e.OutputTokens != 0 || e.CostUSD != 0 {
					t.Fatalf("%s produced usage %+v", name, e)
				}
			case "details string":
				if e.InputTokens != 0 {
					t.Fatalf("partially decoded usage reported: %+v", e)
				}
			case "maxint thought":
				if e.OutputTokens != math.MaxInt {
					t.Fatalf("output tokens wrapped instead of saturating: %+v", e)
				}
			case "audio > input":
				if e.InputTokens != 10 || e.AudioInputTokens != 10 {
					t.Fatalf("audio share not capped: %+v", e)
				}
			}
		})
	}
}

// Real Gemini usage: prompt tokens split by modality, thinking billed as
// output, cost from the registry's audio rate.
func TestUsageHookGeminiAudioTokensAndCost(t *testing.T) {
	srv := geminiServer(t, 200, okTranscript+`,"usageMetadata":{"promptTokenCount":1100,"candidatesTokenCount":40,"thoughtsTokenCount":10,"promptTokensDetails":[{"modality":"TEXT","tokenCount":100},{"modality":"AUDIO","tokenCount":1000}]}}`)
	var log eventLog
	client := geminiUsageClient(t, srv, "gemini-3.1-flash-lite", log.hook)
	if _, err := transcribe(client); err != nil {
		t.Fatal(err)
	}
	e := log.all()[0]
	if e.InputTokens != 1100 || e.AudioInputTokens != 1000 || e.OutputTokens != 50 {
		t.Fatalf("usage = %+v", e)
	}
	// gemini-3.1-flash-lite: $0.25 text, $0.50 audio, $1.50 output per 1M.
	want := (100*0.25 + 1000*0.50 + 50*1.50) / 1e6
	if math.Abs(e.CostUSD-want) > 1e-12 {
		t.Fatalf("cost = %v, want %v", e.CostUSD, want)
	}
}

// Unknown price → 0, never a guess: an unregistered model, and audio tokens on
// a model whose audio rate is unknown must not be billed at the text rate.
func TestUsageCostZeroWhenPriceUnknown(t *testing.T) {
	if got := llm.EstimateCost(llm.CostInput{Model: "no-such-model-xyz", InputTokens: 1e6, AudioInputTokens: 1e6, AudioSeconds: 600}); got != 0 {
		t.Fatalf("unknown model priced at %v", got)
	}
	// gemini-3.1-pro-preview has a text price but no verified audio price.
	if got := llm.EstimateCost(llm.CostInput{Model: "gemini-3.1-pro-preview", InputTokens: 1e6, AudioInputTokens: 1e6}); got != 0 {
		t.Fatalf("audio tokens billed at a guessed rate: %v", got)
	}
	// Hostile durations never produce negative or NaN cost.
	for _, secs := range []float64{-60, math.NaN(), math.Inf(1), math.Inf(-1)} {
		if got := llm.EstimateCost(llm.CostInput{Model: "whisper-1", AudioSeconds: secs}); got != 0 {
			t.Fatalf("AudioSeconds %v priced at %v", secs, got)
		}
	}
	// An audio share larger than the input total is capped, not double-billed.
	capped := llm.EstimateCost(llm.CostInput{Model: "gemini-3.1-flash-lite", InputTokens: 10, AudioInputTokens: 1e9})
	if want := 10 * 0.50 / 1e6; math.Abs(capped-want) > 1e-15 {
		t.Fatalf("audio overcount: %v, want %v", capped, want)
	}
}

func openAITranscriptionClient(t *testing.T, provider, model, body string, hook llm.UsageHook) llm.ModalityClient {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	client, err := llm.NewModalityClient(llm.Config{
		Provider: provider, Model: model, APIKey: "k", BaseURL: srv.URL,
		DisableProxy: true, DisableRetries: true, UsageHook: hook,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = client.Close() })
	return client
}

func transcribeMP3(client llm.ModalityClient) (string, error) {
	return client.TranscribeAudio(context.Background(), llm.TranscriptionRequest{Audio: []byte("ID3\x04\x00\x00"), MediaType: "audio/mpeg"})
}

// OpenAI / xAI transcription usage: token-billed, duration-billed, and
// hostile shapes that must not fail the transcript or produce garbage.
func TestUsageHookOpenAICompatTranscriptionUsage(t *testing.T) {
	cases := []struct {
		name, provider, model, body string
		in, out, audio              int
		seconds                     float64
	}{
		{"tokens", "openai", "gpt-4o-transcribe", `{"text":"hi","usage":{"type":"tokens","input_tokens":120,"output_tokens":8,"total_tokens":128,"input_token_details":{"audio_tokens":100,"text_tokens":20}}}`, 120, 8, 100, 0},
		{"duration", "openai", "whisper-1", `{"text":"hi","usage":{"type":"duration","seconds":90}}`, 0, 0, 0, 90},
		{"xai duration", "xai", "grok-voice-transcribe-2.0", `{"text":"hi","duration":36}`, 0, 0, 0, 36},
		{"usage string", "openai", "gpt-4o-transcribe", `{"text":"hi","usage":"many"}`, 0, 0, 0, 0},
		{"unknown type", "openai", "gpt-4o-transcribe", `{"text":"hi","usage":{"type":"credits","input_tokens":99}}`, 0, 0, 0, 0},
		{"negative", "openai", "gpt-4o-transcribe", `{"text":"hi","usage":{"type":"tokens","input_tokens":-1,"output_tokens":-2,"input_token_details":{"audio_tokens":-3}}}`, 0, 0, 0, 0},
		{"negative seconds", "openai", "whisper-1", `{"text":"hi","usage":{"type":"duration","seconds":-30}}`, 0, 0, 0, 0},
		{"xai duration string", "xai", "grok-voice-transcribe-2.0", `{"text":"hi","duration":"1.2s"}`, 0, 0, 0, 0},
		{"openai duration ignored", "openai", "whisper-1", `{"text":"hi","duration":50}`, 0, 0, 0, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var log eventLog
			client := openAITranscriptionClient(t, c.provider, c.model, c.body, log.hook)
			text, err := transcribeMP3(client)
			if err != nil || text != "hi" {
				t.Fatalf("usage field broke the transcript: %q %v", text, err)
			}
			events := log.all()
			if len(events) != 1 {
				t.Fatalf("events = %d", len(events))
			}
			e := events[0]
			if e.InputTokens != c.in || e.OutputTokens != c.out || e.AudioInputTokens != c.audio || e.AudioSeconds != c.seconds {
				t.Fatalf("usage = in %d out %d audio %d secs %v; want %d %d %d %v", e.InputTokens, e.OutputTokens, e.AudioInputTokens, e.AudioSeconds, c.in, c.out, c.audio, c.seconds)
			}
			if e.CostUSD < 0 || (c.in == 0 && c.seconds == 0 && e.CostUSD != 0) {
				t.Fatalf("cost = %v", e.CostUSD)
			}
		})
	}
	// Duration billing: whisper-1 90 s at $0.006/min.
	var log eventLog
	client := openAITranscriptionClient(t, "openai", "whisper-1", `{"text":"hi","usage":{"type":"duration","seconds":90}}`, log.hook)
	if _, err := transcribeMP3(client); err != nil {
		t.Fatal(err)
	}
	if got, want := log.all()[0].CostUSD, 1.5*0.006; math.Abs(got-want) > 1e-12 {
		t.Fatalf("whisper cost = %v, want %v", got, want)
	}
}

// Embeddings report the provider's prompt_tokens (and only those).
func TestUsageHookEmbeddingsReportPromptTokens(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"model":"text-embedding-3-small","data":[{"index":0,"embedding":[0.1]}],"usage":{"prompt_tokens":7}}`)
	}))
	defer srv.Close()
	var log eventLog
	client, err := llm.NewModalityClient(llm.Config{Provider: "openai", Model: "text-embedding-3-small", APIKey: "k", BaseURL: srv.URL, DisableProxy: true, DisableRetries: true, UsageHook: log.hook})
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	if _, err := client.GenerateEmbeddings(context.Background(), llm.EmbeddingRequest{Input: []string{"x"}}); err != nil {
		t.Fatal(err)
	}
	events := log.all()
	if len(events) != 1 || events[0].Operation != llm.OperationEmbeddings || events[0].InputTokens != 7 {
		t.Fatalf("events = %+v", events)
	}
}

// Decision (documented on Config.UsageHook): a request rejected by validation
// never reached the provider, cost nothing, and is not reported.
func TestUsageHookNotCalledForValidationErrors(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { hits.Add(1) }))
	defer srv.Close()
	var log eventLog
	client, err := llm.NewModalityClient(llm.Config{Provider: "gemini", Model: "gemini-3.5-flash-lite", APIKey: "k", BaseURL: srv.URL, DisableProxy: true, UsageHook: log.hook})
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	if _, err := client.TranscribeAudio(context.Background(), llm.TranscriptionRequest{Audio: []byte("not audio"), MediaType: "audio/wav"}); err == nil {
		t.Fatal("invalid audio accepted")
	}
	if _, err := client.GenerateEmbeddings(context.Background(), llm.EmbeddingRequest{}); err == nil {
		t.Fatal("empty embedding request accepted")
	}
	if hits.Load() != 0 || len(log.all()) != 0 {
		t.Fatalf("validation error reached provider (%d) or hook (%d)", hits.Load(), len(log.all()))
	}
}

// Concurrent calls (some failing over) under -race: every attempt reported
// exactly once, usage never bleeds between concurrent calls.
func TestUsageHookConcurrentCallsUnderRace(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		if strings.Contains(r.URL.Path, "gemini-3.5-transcribe") {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = io.WriteString(w, `{"error":{"code":503,"message":"overloaded"}}`)
			return
		}
		_, _ = io.WriteString(w, okTranscript+`,"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":5,"promptTokensDetails":[{"modality":"AUDIO","tokenCount":90}]}}`)
	}))
	defer srv.Close()
	var log eventLog
	cfg := func(model string) llm.Config {
		return llm.Config{Provider: "gemini", Model: model, APIKey: "k", BaseURL: srv.URL, DisableProxy: true, DisableRetries: true, UsageHook: log.hook}
	}
	chain, err := llm.NewFallbackModalityClient(cfg("gemini-3.5-transcribe"), cfg("gemini-3.5-flash-lite"))
	if err != nil {
		t.Fatal(err)
	}
	defer chain.Close()
	const n = 40
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := transcribe(chain); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	events := log.all()
	if len(events) != 2*n {
		t.Fatalf("events = %d, want %d", len(events), 2*n)
	}
	var primaries, fallbacks int
	for _, e := range events {
		switch {
		case e.Attempt == 0 && !e.Fallback && e.Err != nil && e.InputTokens == 0:
			primaries++
		case e.Attempt == 1 && e.Fallback && e.Err == nil && e.InputTokens == 100 && e.AudioInputTokens == 90:
			fallbacks++
		default:
			t.Fatalf("unexpected or cross-contaminated event: %+v", e)
		}
	}
	if primaries != n || fallbacks != n {
		t.Fatalf("primaries=%d fallbacks=%d", primaries, fallbacks)
	}
}

// ReportModalityUsage outside a recorder (custom adapters, nil ctx) is a no-op.
func TestReportModalityUsageWithoutRecorderIsNoOp(t *testing.T) {
	llm.ReportModalityUsage(context.Background(), llm.ModalityUsage{InputTokens: 5})
	var nilCtx context.Context
	llm.ReportModalityUsage(nilCtx, llm.ModalityUsage{InputTokens: 5})
}

type tenantKey struct{}

// M1: concurrent callers sharing ONE client must each see their own context
// on their own event — the attribution the consumer used to need a
// per-caller client pool for. Both hooks fire, UsageHook first.
func TestUsageHookCtxAttributesConcurrentCallers(t *testing.T) {
	srv := geminiServer(t, 200, okTranscript+`,"usageMetadata":{"promptTokenCount":10}}`)
	var mu sync.Mutex
	var order []string
	seen := map[string]int{}
	client, err := llm.NewModalityClient(llm.Config{
		Provider: "gemini", Model: "gemini-3.5-flash-lite", APIKey: "k", BaseURL: srv.URL,
		DisableProxy: true, DisableRetries: true, Timeout: 5 * time.Second,
		UsageHook: func(llm.UsageEvent) {
			mu.Lock()
			order = append(order, "plain")
			mu.Unlock()
		},
		UsageHookCtx: func(ctx context.Context, e llm.UsageEvent) {
			tenant, _ := ctx.Value(tenantKey{}).(string)
			mu.Lock()
			defer mu.Unlock()
			order = append(order, "ctx")
			if e.InputTokens == 10 {
				seen[tenant]++
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	const callers = 16
	var wg sync.WaitGroup
	for i := 0; i < callers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ctx := context.WithValue(context.Background(), tenantKey{}, "tenant-"+string(rune('a'+i)))
			if _, err := client.TranscribeAudio(ctx, llm.TranscriptionRequest{Audio: fbWAV, MediaType: "audio/wav"}); err != nil {
				t.Error(err)
			}
		}(i)
	}
	wg.Wait()
	if len(seen) != callers {
		t.Fatalf("attributed tenants = %d (%v), want %d distinct", len(seen), seen, callers)
	}
	for tenant, n := range seen {
		if n != 1 {
			t.Fatalf("tenant %s got %d events, want 1", tenant, n)
		}
	}
	if len(order) != 2*callers {
		t.Fatalf("hook calls = %d, want both hooks per event", len(order))
	}
}

// M1: a panicking UsageHookCtx neither changes the result nor starves the
// plain hook, and the ctx hook still fires for a cancelled attempt.
func TestUsageHookCtxPanicAndCancelledContext(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body) // lets the server notice the client hang up
		<-r.Context().Done()
	}))
	defer srv.Close()
	var plain, ctxHook int
	var gotErr error
	client, err := llm.NewModalityClient(llm.Config{
		Provider: "gemini", Model: "gemini-3.5-flash-lite", APIKey: "k", BaseURL: srv.URL,
		DisableProxy: true, DisableRetries: true,
		UsageHook: func(llm.UsageEvent) { plain++ },
		UsageHookCtx: func(ctx context.Context, e llm.UsageEvent) {
			ctxHook++
			gotErr = ctx.Err()
			panic("hook bug")
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_, err = client.TranscribeAudio(ctx, llm.TranscriptionRequest{Audio: fbWAV, MediaType: "audio/wav"})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("result changed by the hook: %v", err)
	}
	if plain != 1 || ctxHook != 1 || !errors.Is(gotErr, context.DeadlineExceeded) {
		t.Fatalf("plain=%d ctx=%d ctxErr=%v", plain, ctxHook, gotErr)
	}
}

// M1: a fallback chain's deployments inherit the primary's UsageHookCtx, and
// the ctx carries the caller's values on every attempt.
func TestUsageHookCtxInheritedAcrossFallbackChain(t *testing.T) {
	s := newGeminiStub(t, map[string]string{"gemini-3.5-transcribe": "overloaded"})
	var mu sync.Mutex
	var tenants []string
	primary := s.cfg("gemini-3.5-transcribe")
	primary.UsageHookCtx = func(ctx context.Context, e llm.UsageEvent) {
		mu.Lock()
		defer mu.Unlock()
		tenant, _ := ctx.Value(tenantKey{}).(string)
		tenants = append(tenants, tenant+"/"+e.Model)
	}
	client, err := llm.NewFallbackModalityClient(primary, s.cfg("gemini-3.5-flash-lite"))
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	ctx := context.WithValue(context.Background(), tenantKey{}, "t1")
	if _, err := client.TranscribeAudio(ctx, llm.TranscriptionRequest{Audio: fbWAV, MediaType: "audio/wav"}); err != nil {
		t.Fatal(err)
	}
	if strings.Join(tenants, ",") != "t1/gemini-3.5-transcribe,t1/gemini-3.5-flash-lite" {
		t.Fatalf("events = %v", tenants)
	}
}

// M5: Gemini's promptTokenCount includes cachedContentTokenCount. The modality
// usage must split it like the chat adapter does (v0.9.4 contract) and price
// cached tokens at the cache-read rate — not bill them as full input.
func TestUsageHookGeminiCachedTokensSplitAndPriced(t *testing.T) {
	srv := geminiServer(t, 200, okTranscript+`,"usageMetadata":{"promptTokenCount":10000,"cachedContentTokenCount":9000,"candidatesTokenCount":100}}`)
	var log eventLog
	client := geminiUsageClient(t, srv, "gemini-3.5-flash-lite", log.hook)
	if _, err := transcribe(client); err != nil {
		t.Fatal(err)
	}
	e := log.all()[0]
	if e.InputTokens != 1000 || e.CacheReadTokens != 9000 || e.OutputTokens != 100 {
		t.Fatalf("usage = %+v, want 1000 uncached / 9000 cached / 100 out", e)
	}
	// gemini-3.5-flash-lite: $0.30 input, $0.03 cache read, $2.50 output per 1M.
	want := (1000*0.30 + 9000*0.03 + 100*2.50) / 1e6
	if math.Abs(e.CostUSD-want) > 1e-12 {
		t.Fatalf("cost = %v, want %v", e.CostUSD, want)
	}
}

// N1: cached audio is subtracted from the audio share using
// cacheTokensDetails, rather than clamping total audio to the uncached input
// (which billed cached audio — or text — as uncached audio).
func TestUsageHookGeminiCachedAudioSplit(t *testing.T) {
	for name, tc := range map[string]struct {
		usage                string
		input, cached, audio int
	}{
		"audio partly cached": {`{"promptTokenCount":1000,"cachedContentTokenCount":600,` +
			`"promptTokensDetails":[{"modality":"AUDIO","tokenCount":800},{"modality":"TEXT","tokenCount":200}],` +
			`"cacheTokensDetails":[{"modality":"AUDIO","tokenCount":500},{"modality":"TEXT","tokenCount":100}]}`, 400, 600, 300},
		"only text cached": {`{"promptTokenCount":1000,"cachedContentTokenCount":200,` +
			`"promptTokensDetails":[{"modality":"AUDIO","tokenCount":800},{"modality":"TEXT","tokenCount":200}],` +
			`"cacheTokensDetails":[{"modality":"TEXT","tokenCount":200}]}`, 800, 200, 800},
		"all audio cached": {`{"promptTokenCount":1000,"cachedContentTokenCount":800,` +
			`"promptTokensDetails":[{"modality":"AUDIO","tokenCount":800},{"modality":"TEXT","tokenCount":200}],` +
			`"cacheTokensDetails":[{"modality":"AUDIO","tokenCount":800}]}`, 200, 800, 0},
		"cached audio > audio": {`{"promptTokenCount":1000,"cachedContentTokenCount":100,` +
			`"promptTokensDetails":[{"modality":"AUDIO","tokenCount":50}],` +
			`"cacheTokensDetails":[{"modality":"AUDIO","tokenCount":90},{"modality":"AUDIO","tokenCount":-500}]}`, 900, 100, 0},
	} {
		srv := geminiServer(t, 200, okTranscript+`,"usageMetadata":`+tc.usage+`}`)
		var log eventLog
		client := geminiUsageClient(t, srv, "gemini-3.5-flash-lite", log.hook)
		if _, err := transcribe(client); err != nil {
			t.Fatal(err)
		}
		e := log.all()[0]
		if e.InputTokens != tc.input || e.CacheReadTokens != tc.cached || e.AudioInputTokens != tc.audio {
			t.Fatalf("%s: usage = %+v, want input %d / cached %d / audio %d", name, e, tc.input, tc.cached, tc.audio)
		}
	}
}

// M5: hostile cache counts never produce negative or overlapping usage.
func TestUsageHookGeminiHostileCachedTokens(t *testing.T) {
	for name, usage := range map[string]string{
		"cached > prompt":  `{"promptTokenCount":10,"cachedContentTokenCount":500}`,
		"negative cached":  `{"promptTokenCount":10,"cachedContentTokenCount":-500}`,
		"audio and cached": `{"promptTokenCount":100,"cachedContentTokenCount":90,"promptTokensDetails":[{"modality":"AUDIO","tokenCount":80}]}`,
	} {
		srv := geminiServer(t, 200, okTranscript+`,"usageMetadata":`+usage+`}`)
		var log eventLog
		client := geminiUsageClient(t, srv, "gemini-3.5-flash-lite", log.hook)
		if _, err := transcribe(client); err != nil {
			t.Fatal(err)
		}
		e := log.all()[0]
		if e.InputTokens < 0 || e.CacheReadTokens < 0 || e.AudioInputTokens > e.InputTokens {
			t.Fatalf("%s: garbage usage %+v", name, e)
		}
		switch name {
		case "cached > prompt":
			if e.InputTokens != 0 || e.CacheReadTokens != 10 {
				t.Fatalf("%s: %+v", name, e)
			}
		case "negative cached":
			if e.InputTokens != 10 || e.CacheReadTokens != 0 {
				t.Fatalf("%s: %+v", name, e)
			}
		}
	}
}

// M5: Gemini embedContent reports no usage; the adapter must not invent one
// (it used to report len(text)/4 as provider-reported InputTokens).
func TestGeminiEmbeddingsDoNotFabricateUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"embedding":{"values":[0.1,0.2]}}`)
	}))
	defer srv.Close()
	_ = llm.RegisterModel(llm.ModelInfo{ID: "gemini-embedding-001", Provider: "gemini",
		InputPricePer1M: 0.15, Capabilities: llm.Capabilities(llm.CapabilityEmbeddings)})
	var log eventLog
	client, err := llm.NewModalityClient(llm.Config{Provider: "gemini", Model: "gemini-embedding-001", APIKey: "k",
		BaseURL: srv.URL, DisableProxy: true, DisableRetries: true, UsageHook: log.hook})
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	resp, err := client.GenerateEmbeddings(context.Background(), llm.EmbeddingRequest{
		Input: []string{strings.Repeat("long text ", 400), "more"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.InputTokens != 0 {
		t.Fatalf("EmbeddingResponse.InputTokens = %d, want 0 (not reported by embedContent)", resp.InputTokens)
	}
	if e := log.all(); len(e) != 1 || e[0].InputTokens != 0 || e[0].CostUSD != 0 {
		t.Fatalf("events = %+v", e)
	}
}
