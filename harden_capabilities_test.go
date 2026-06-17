package llm_test

// Hardening pass 3 — capabilities.go HTTP body-read integrity.

import (
	"bufio"
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
)

// truncatedBodyServer answers with a 200 + a Content-Length larger than the body
// it actually sends, then closes the connection — so a client reading the body
// observes io.ErrUnexpectedEOF (a mid-read failure / truncation).
func truncatedBodyServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hj, ok := w.(http.Hijacker)
		if !ok {
			t.Fatal("ResponseWriter is not a Hijacker")
		}
		conn, _, err := hj.Hijack()
		if err != nil {
			t.Fatalf("hijack: %v", err)
		}
		defer conn.Close()
		bw := bufio.NewWriter(conn)
		// Claim 5000 bytes; send only "PART". The client read aborts with
		// ErrUnexpectedEOF.
		_, _ = bw.WriteString("HTTP/1.1 200 OK\r\nContent-Length: 5000\r\n\r\nPART")
		_ = bw.Flush()
	}))
}

// A truncated audio response must surface an error, not be returned as if it were
// complete audio. SynthesizeSpeech returns raw bytes (no JSON decode to catch the
// truncation), so swallowing the body-read error is silent data corruption.
func TestSynthesizeSpeechRejectsTruncatedBody(t *testing.T) {
	srv := truncatedBodyServer(t)
	defer srv.Close()

	cfg := llm.Config{Provider: "openai", BaseURL: srv.URL, APIKey: "k", Timeout: 5 * time.Second}
	audio, err := llm.SynthesizeSpeech(context.Background(), cfg, llm.SpeechRequest{
		Model: "tts-1", Input: "hello", Voice: "alloy",
	})
	if err == nil {
		t.Fatalf("want an error on a truncated response; got %d bytes of (silently truncated) audio", len(audio))
	}
	if audio != nil {
		t.Fatalf("truncated response must return nil audio, got %d bytes", len(audio))
	}
}
