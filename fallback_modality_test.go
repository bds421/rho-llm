package llm_test

import (
	"context"
	"errors"
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
