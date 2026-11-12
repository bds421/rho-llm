package llm_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider/gemini"
)

var fbWAV = []byte("RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00")

// geminiStub answers per model: "broken" is the 2026-10-07 outage of
// gemini-3.5-transcribe (400 on every call), "overloaded" a 503, "hang" never
// answers, anything else a transcript.
type geminiStub struct {
	mu      sync.Mutex
	calls   []string
	mode    map[string]string
	release chan struct{}
	srv     *httptest.Server
}

func newGeminiStub(t *testing.T, mode map[string]string) *geminiStub {
	t.Helper()
	s := &geminiStub{mode: mode, release: make(chan struct{})}
	s.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		model := r.URL.Path[strings.LastIndex(r.URL.Path, "/")+1:]
		model = model[:strings.Index(model, ":")]
		s.mu.Lock()
		s.calls = append(s.calls, model)
		m := s.mode[model]
		s.mu.Unlock()
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		switch m {
		case "broken":
			w.WriteHeader(http.StatusBadRequest)
			_, _ = io.WriteString(w, `{"error":{"code":400,"message":"Thinking is not enabled for this model"}}`)
		case "overloaded":
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = io.WriteString(w, `{"error":{"code":503,"message":"overloaded"}}`)
		case "ratelimited":
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = io.WriteString(w, `{"error":{"code":429,"message":"quota"}}`)
		case "hang":
			select {
			case <-r.Context().Done():
			case <-s.release:
			}
		default:
			_, _ = io.WriteString(w, `{"candidates":[{"finishReason":"STOP","content":{"parts":[{"audioTranscription":{"text":"from `+model+`"}},{"text":""}]}}]}`)
		}
	}))
	t.Cleanup(s.srv.Close)
	t.Cleanup(func() { close(s.release) })
	return s
}

func (s *geminiStub) seen() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.calls...)
}

func (s *geminiStub) cfg(model string) llm.Config {
	return llm.Config{Provider: "gemini", Model: model, APIKey: "k", BaseURL: s.srv.URL, DisableProxy: true, Timeout: 5 * time.Second}
}

func fallbackChain(t *testing.T, s *geminiStub, models ...string) llm.ModalityClient {
	t.Helper()
	var cfgs []llm.Config
	for _, m := range models {
		cfgs = append(cfgs, s.cfg(m))
	}
	client, err := llm.NewFallbackModalityClient(cfgs[0], cfgs[1:]...)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = client.Close() })
	return client
}

// The outage that motivated the feature: the dedicated model rejects every
// request; the chain must deliver the fallback's transcript, and the fallback
// must be asked under its OWN model name, not the primary's.
func TestFallbackSurvivesABrokenPrimaryModel(t *testing.T) {
	s := newGeminiStub(t, map[string]string{"gemini-3.5-transcribe": "broken"})
	client := fallbackChain(t, s, "gemini-3.5-transcribe", "gemini-3.5-flash-lite")
	text, err := client.TranscribeAudio(context.Background(), llm.TranscriptionRequest{
		Model: "gemini-3.5-transcribe", Audio: fbWAV, MediaType: "audio/wav", Vocabulary: []string{"Weckerl"},
	})
	if err != nil || text != "from gemini-3.5-flash-lite" {
		t.Fatalf("recording lost despite a healthy fallback: %q %v", text, err)
	}
	if got := s.seen(); strings.Join(got, ",") != "gemini-3.5-transcribe,gemini-3.5-flash-lite" {
		t.Fatalf("calls = %v", got)
	}
}

// A 5xx or 429 on the primary must hand over at once, not after rho-llm's
// default backoff (three attempts, or MaxRetries, with exponential backoff).
func TestFallbackFailsOverWithoutBackingOffOnThePrimary(t *testing.T) {
	for _, mode := range []string{"overloaded", "ratelimited"} {
		s := newGeminiStub(t, map[string]string{"gemini-3.5-transcribe": mode})
		client := fallbackChain(t, s, "gemini-3.5-transcribe", "gemini-3.5-flash-lite")
		start := time.Now()
		if _, err := client.TranscribeAudio(context.Background(), llm.TranscriptionRequest{Audio: fbWAV, MediaType: "audio/wav"}); err != nil {
			t.Fatalf("%s: %v", mode, err)
		}
		if elapsed := time.Since(start); elapsed > 900*time.Millisecond {
			t.Errorf("%s: failover took %v — the primary backed off before handing over", mode, elapsed)
		}
		if got := s.seen(); len(got) != 2 {
			t.Errorf("%s: calls = %v; the primary was retried instead of failing over", mode, got)
		}
	}
}

// When every deployment fails, the error must name each attempt and keep the
// last APIError reachable so callers can still classify it.
func TestFallbackAllFailedKeepsEveryAttemptAndClassification(t *testing.T) {
	s := newGeminiStub(t, map[string]string{"gemini-3.5-transcribe": "broken", "gemini-3.5-flash-lite": "ratelimited"})
	cfgFallback := s.cfg("gemini-3.5-flash-lite")
	cfgFallback.DisableRetries = true // keep the test fast; the last deployment's retries are its own
	client, err := llm.NewFallbackModalityClient(s.cfg("gemini-3.5-transcribe"), cfgFallback)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	_, err = client.TranscribeAudio(context.Background(), llm.TranscriptionRequest{Audio: fbWAV, MediaType: "audio/wav"})
	if err == nil || !strings.Contains(err.Error(), "gemini-3.5-transcribe") || !strings.Contains(err.Error(), "gemini-3.5-flash-lite") {
		t.Fatalf("error does not name both attempts: %v", err)
	}
	if !llm.IsRateLimited(err) {
		t.Fatalf("joined error lost the rate-limit classification: %v", err)
	}
	var apiErr *llm.APIError
	if !errors.As(err, &apiErr) {
		t.Fatal("APIError not reachable through the joined error")
	}
}

// Caller mistakes and cancellation fail every deployment the same way: no
// second call may reach any provider.
func TestFallbackNeverRetriesCallerMistakesOrCancellation(t *testing.T) {
	s := newGeminiStub(t, map[string]string{"gemini-3.5-transcribe": "hang"})
	client := fallbackChain(t, s, "gemini-3.5-transcribe", "gemini-3.5-flash-lite")
	png := []byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0}
	for name, req := range map[string]llm.TranscriptionRequest{
		"non-audio bytes":  {Audio: png, MediaType: "audio/wav"},
		"bad language":     {Audio: fbWAV, MediaType: "audio/wav", Language: "German"},
		"control in vocab": {Audio: fbWAV, MediaType: "audio/wav", Vocabulary: []string{"a\nb"}},
	} {
		if _, err := client.TranscribeAudio(context.Background(), req); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
	if got := s.seen(); len(got) != 0 {
		t.Fatalf("invalid requests reached providers: %v", got)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if _, err := client.TranscribeAudio(ctx, llm.TranscriptionRequest{Audio: fbWAV, MediaType: "audio/wav"}); err == nil {
		t.Fatal("cancelled request reported success")
	}
	if got := s.seen(); len(got) != 1 {
		t.Fatalf("calls = %v; a cancelled request was sent to the fallback too", got)
	}
}

// A primary that runs into its own Config.Timeout is a slow provider, not a
// cancelled request: net/http's Client.Timeout error matches
// context.DeadlineExceeded, which was read as caller cancellation, so the
// healthy fallback was never tried (B11).
func TestFallbackFailsOverWhenThePrimaryHitsItsOwnTimeout(t *testing.T) {
	s := newGeminiStub(t, map[string]string{"gemini-3.5-transcribe": "hang"})
	primary := s.cfg("gemini-3.5-transcribe")
	primary.Timeout = 150 * time.Millisecond
	client, err := llm.NewFallbackModalityClient(primary, s.cfg("gemini-3.5-flash-lite"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = client.Close() })

	text, err := client.TranscribeAudio(context.Background(), llm.TranscriptionRequest{Audio: fbWAV, MediaType: "audio/wav"})
	if err != nil || text != "from gemini-3.5-flash-lite" {
		t.Fatalf("primary timed out and the fallback was not used: %q %v", text, err)
	}
	if got := s.seen(); len(got) == 0 || got[len(got)-1] != "gemini-3.5-flash-lite" {
		t.Fatalf("calls = %v, want the fallback last", got)
	}
	for _, policy := range []llm.FailoverPolicy{llm.FailoverAnyProviderError, llm.FailoverTransientOnly} {
		if !policy(context.Background(), fmt.Errorf("wrapped: %w", context.DeadlineExceeded)) {
			t.Error("a deployment's own timeout must count as a provider failure")
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if policy(ctx, context.DeadlineExceeded) {
			t.Error("with the caller's ctx done, nothing may fail over")
		}
		if policy(context.Background(), context.Canceled) {
			t.Error("cancellation must not fail over")
		}
	}
}

// A healthy primary never costs a second call; a three-deep chain walks in order.
func TestFallbackChainOrderAndNoSpuriousCalls(t *testing.T) {
	s := newGeminiStub(t, map[string]string{"gemini-3.5-transcribe": "broken", "gemini-3.5-flash-lite": "broken"})
	client := fallbackChain(t, s, "gemini-3.5-transcribe", "gemini-3.5-flash-lite", "gemini-2.5-flash")
	text, err := client.TranscribeAudio(context.Background(), llm.TranscriptionRequest{Audio: fbWAV, MediaType: "audio/wav"})
	if err != nil || text != "from gemini-2.5-flash" {
		t.Fatalf("third deployment not reached: %q %v", text, err)
	}
	healthy := newGeminiStub(t, nil)
	c2 := fallbackChain(t, healthy, "gemini-3.5-transcribe", "gemini-3.5-flash-lite")
	if _, err := c2.TranscribeAudio(context.Background(), llm.TranscriptionRequest{Audio: fbWAV, MediaType: "audio/wav"}); err != nil {
		t.Fatal(err)
	}
	if got := healthy.seen(); len(got) != 1 {
		t.Fatalf("calls = %v; a healthy primary still cost a fallback call", got)
	}
	if c2.Model() != "gemini-3.5-transcribe" || c2.Provider() != "gemini" {
		t.Fatalf("chain reports %s/%s, want the primary", c2.Provider(), c2.Model())
	}
}

// A deployment that cannot be built must fail construction (and not leak the
// clients already built), so a misconfiguration surfaces at startup.
func TestFallbackConstructionFailsOnAnInvalidDeployment(t *testing.T) {
	s := newGeminiStub(t, nil)
	for name, bad := range map[string]llm.Config{
		"no model":         {Provider: "gemini", APIKey: "k", BaseURL: s.srv.URL},
		"unknown provider": {Provider: "nope", Model: "m", APIKey: "k"},
		"private base url": {Provider: "gemini", Model: "gemini-3.5-flash-lite", APIKey: "k", BaseURL: "http://169.254.169.254", BlockPrivateBaseURL: true},
	} {
		if c, err := llm.NewFallbackModalityClient(s.cfg("gemini-3.5-transcribe"), bad); err == nil {
			_ = c.Close()
			t.Errorf("%s: invalid fallback accepted", name)
		}
	}
}

func TestFallbackConcurrentUseAndDoubleClose(t *testing.T) {
	s := newGeminiStub(t, map[string]string{"gemini-3.5-transcribe": "broken"})
	client := fallbackChain(t, s, "gemini-3.5-transcribe", "gemini-3.5-flash-lite")
	var wg sync.WaitGroup
	for i := 0; i < 24; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if text, err := client.TranscribeAudio(context.Background(), llm.TranscriptionRequest{Audio: fbWAV, MediaType: "audio/wav"}); err != nil || text == "" {
				t.Errorf("concurrent failover: %q %v", text, err)
			}
		}()
	}
	wg.Wait()
	if err := client.Close(); err != nil {
		t.Fatal(err)
	}
	if err := client.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
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
