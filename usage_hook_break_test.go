package llm_test

import (
	"context"
	"encoding/json"
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
	_ "github.com/bds421/rho-llm/provider/gemini"
	_ "github.com/bds421/rho-llm/provider/openaicompat"
)

// Break-the-system tests for Config.UsageHook (v0.9.2). Every test here was
// confirmed red against the code without the feature or the specific guard.

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
	// gemini-2.0-flash has a text price but no verified audio price.
	if got := llm.EstimateCost(llm.CostInput{Model: "gemini-2.0-flash", InputTokens: 1e6, AudioInputTokens: 1e6}); got != 0 {
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

// The registry fields the cost display relies on, set from primary sources;
// a typo or dropped field silently zeroes the live cost.
func TestRegistryAudioPricesPinned(t *testing.T) {
	cases := []struct {
		model              string
		audio1M, perMinute float64
	}{
		{"gemini-3.5-flash-lite", 0.30, 0},
		{"gemini-3.5-transcribe", 2.00, 0},
		{"gemini-3.1-flash-lite", 0.50, 0},
		{"gpt-4o-transcribe", 2.50, 0},
		{"gpt-4o-mini-transcribe", 1.25, 0},
		{"whisper-1", 0, 0.006},
		{"gpt-transcribe", 0, 0.0045},
		{"grok-voice-transcribe-2.0", 0, 0.10 / 60},
	}
	for _, c := range cases {
		info, ok := llm.GetModelInfo(c.model)
		if !ok || info.AudioInputPricePer1M != c.audio1M || info.AudioPricePerMinute != c.perMinute {
			t.Errorf("%s: audio=%v perMinute=%v, want %v %v", c.model, info.AudioInputPricePer1M, info.AudioPricePerMinute, c.audio1M, c.perMinute)
		}
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
