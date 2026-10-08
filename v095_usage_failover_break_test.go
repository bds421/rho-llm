package llm_test

// Break tests for v0.9.5: the context-carrying usage hook (M1), the failover
// policy knob (M3) and cached tokens in modality usage (M5).

import (
	"context"
	"errors"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
)

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

// M3: FailoverTransientOnly does NOT re-send a payload the primary rejected
// as malformed (400) or oversized (413) — no second charge, no second vendor
// — but still fails over on 5xx/429. The default policy keeps failing over on
// the 400 (compatibility with the 2026-10-07 broken-model case).
func TestFailoverPolicyTransientOnly(t *testing.T) {
	cases := []struct {
		mode        string
		wantCalls   int
		wantSuccess bool
	}{
		{"broken", 1, false},     // 400
		{"overloaded", 2, true},  // 503
		{"ratelimited", 2, true}, // 429
	}
	for _, tc := range cases {
		s := newGeminiStub(t, map[string]string{"gemini-3.5-transcribe": tc.mode})
		client, err := llm.NewFallbackModalityClientWithPolicy(llm.FailoverTransientOnly,
			s.cfg("gemini-3.5-transcribe"), s.cfg("gemini-3.5-flash-lite"))
		if err != nil {
			t.Fatal(err)
		}
		_, err = client.TranscribeAudio(context.Background(), llm.TranscriptionRequest{Audio: fbWAV, MediaType: "audio/wav"})
		_ = client.Close()
		if (err == nil) != tc.wantSuccess {
			t.Errorf("%s: err = %v", tc.mode, err)
		}
		if got := s.seen(); len(got) != tc.wantCalls {
			t.Errorf("%s: calls = %v, want %d", tc.mode, got, tc.wantCalls)
		}
	}

	// nil policy == default == NewFallbackModalityClient: a 400 fails over.
	s := newGeminiStub(t, map[string]string{"gemini-3.5-transcribe": "broken"})
	client, err := llm.NewFallbackModalityClientWithPolicy(nil, s.cfg("gemini-3.5-transcribe"), s.cfg("gemini-3.5-flash-lite"))
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	if _, err := client.TranscribeAudio(context.Background(), llm.TranscriptionRequest{Audio: fbWAV, MediaType: "audio/wav"}); err != nil {
		t.Fatalf("default policy stopped failing over on 400: %v", err)
	}
}

// M3: a custom policy is consulted, but can never resurrect caller
// cancellation into a failover.
func TestFailoverPolicyCannotOverrideCancellation(t *testing.T) {
	s := newGeminiStub(t, map[string]string{"gemini-3.5-transcribe": "hang"})
	var consulted int
	always := func(context.Context, error) bool { consulted++; return true }
	client, err := llm.NewFallbackModalityClientWithPolicy(always, s.cfg("gemini-3.5-transcribe"), s.cfg("gemini-3.5-flash-lite"))
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_, err = client.TranscribeAudio(ctx, llm.TranscriptionRequest{Audio: fbWAV, MediaType: "audio/wav"})
	if err == nil || len(s.seen()) != 1 || consulted != 0 {
		t.Fatalf("err=%v calls=%v consulted=%d: cancellation failed over", err, s.seen(), consulted)
	}
}

// FailoverTransientOnly classification table, including wrapped errors.
func TestFailoverTransientOnlyTable(t *testing.T) {
	ctx := context.Background()
	for status, want := range map[int]bool{400: false, 413: false, 415: false, 422: false,
		401: true, 403: true, 404: true, 408: true, 429: true, 500: true, 502: true, 503: true, 504: true} {
		err := llm.NewAPIErrorFromStatus("p", status, "x")
		if got := llm.FailoverTransientOnly(ctx, errors.Join(errors.New("wrapped"), err)); got != want {
			t.Errorf("status %d: got %v want %v", status, got, want)
		}
	}
	if llm.FailoverTransientOnly(ctx, errors.New("validation: bad request shape")) {
		t.Error("a non-provider error failed over")
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if llm.FailoverTransientOnly(cancelled, llm.NewOverloadedError("p", "x")) {
		t.Error("failed over with a cancelled caller context")
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
