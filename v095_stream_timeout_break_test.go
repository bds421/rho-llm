package llm_test

// H3 break tests: http.Client.Timeout covers reading the whole body, so a
// stream that legitimately outlived Config.Timeout was cut off mid-response.
// Streams now run on NewStreamingHTTPClient: no whole-body timeout, but a
// silent gap longer than Timeout still fails. Driven over real httptest SSE
// servers for all four wire protocols.

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
)

type streamFixture struct {
	name   string
	cfg    func(url string) llm.Config
	chunks []string // SSE data payloads; the last completes the turn
}

func streamFixtures() []streamFixture {
	const n = 8
	repeat := func(chunk string) []string {
		out := make([]string, n)
		for i := range out {
			out[i] = chunk
		}
		return out
	}
	return []streamFixture{
		{
			name: "openai_compat",
			cfg: func(url string) llm.Config {
				return llm.Config{Provider: "openai", Model: "gpt-4.1", APIKey: "k", BaseURL: url}
			},
			chunks: append(repeat(`{"choices":[{"index":0,"delta":{"content":"a"}}]}`),
				`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`, `[DONE]`),
		},
		{
			name: "anthropic",
			cfg: func(url string) llm.Config {
				return llm.Config{Provider: "anthropic", Model: "claude-sonnet-5", APIKey: "k", BaseURL: url}
			},
			chunks: append(append([]string{
				`{"type":"message_start","message":{"usage":{"input_tokens":1}}}`,
				`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			}, repeat(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"a"}}`)...),
				`{"type":"content_block_stop","index":0}`,
				`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
				`{"type":"message_stop"}`),
		},
		{
			name: "gemini",
			cfg: func(url string) llm.Config {
				return llm.Config{Provider: "gemini", Model: "gemini-2.5-flash", APIKey: "k", BaseURL: url}
			},
			chunks: append(repeat(`{"candidates":[{"content":{"role":"model","parts":[{"text":"a"}]}}]}`),
				`{"candidates":[{"content":{"role":"model","parts":[{"text":"."}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}`),
		},
		{
			name: "openai_responses",
			cfg: func(url string) llm.Config {
				return llm.Config{Provider: "openai_responses", Model: "gpt-5", APIKey: "k", BaseURL: url}
			},
			chunks: append(repeat(`{"type":"response.output_text.delta","delta":"a"}`),
				`{"type":"response.completed","response":{"id":"r","status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}`),
		},
	}
}

// pacedSSEServer writes chunks with a gap between them; stallAfter >= 0 makes it
// go silent (without closing) after that many chunks.
func pacedSSEServer(t *testing.T, chunks []string, gap time.Duration, stallAfter int) *httptest.Server {
	t.Helper()
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		flusher.Flush()
		for i, chunk := range chunks {
			if stallAfter >= 0 && i == stallAfter {
				select {
				case <-release:
				case <-r.Context().Done():
				}
				return
			}
			time.Sleep(gap)
			_, _ = fmt.Fprintf(w, "data: %s\n\n", chunk)
			flusher.Flush()
		}
	}))
	t.Cleanup(func() { close(release); srv.Close() })
	return srv
}

func newStreamClient(t *testing.T, cfg llm.Config, timeout time.Duration) llm.Client {
	t.Helper()
	cfg.Timeout = timeout
	cfg.DisableProxy = true
	cfg.DisableRetries = true
	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() { _ = client.Close() })
	return client
}

// A stream whose total duration (~9 × 60ms ≈ 540ms) far exceeds a 200ms
// Config.Timeout completes, because no single gap exceeds it.
func TestStreamOutlivesClientTimeout(t *testing.T) {
	for _, fx := range streamFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			srv := pacedSSEServer(t, fx.chunks, 60*time.Millisecond, -1)
			client := newStreamClient(t, fx.cfg(srv.URL), 200*time.Millisecond)
			start := time.Now()
			var text strings.Builder
			var done bool
			for ev, err := range client.Stream(context.Background(), llm.Request{
				Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
			}) {
				if err != nil {
					t.Fatalf("stream died after %v (Timeout=200ms): %v", time.Since(start), err)
				}
				text.WriteString(ev.Text)
				done = done || ev.Type == llm.EventDone
			}
			if !done || !strings.HasPrefix(text.String(), "aaaaaaaa") {
				t.Fatalf("incomplete stream: done=%v text=%q", done, text.String())
			}
			if elapsed := time.Since(start); elapsed < 400*time.Millisecond {
				t.Fatalf("stream took only %v; fixture did not outlive Timeout", elapsed)
			}
		})
	}
}

// A stream that goes silent mid-response fails after ~Timeout instead of
// hanging until the context (here: 5s) runs out.
func TestStreamIdleGapFailsAfterTimeout(t *testing.T) {
	for _, fx := range streamFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			srv := pacedSSEServer(t, fx.chunks, 0, 3)
			client := newStreamClient(t, fx.cfg(srv.URL), 200*time.Millisecond)
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			start := time.Now()
			var streamErr error
			for _, err := range client.Stream(ctx, llm.Request{
				Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
			}) {
				if err != nil {
					streamErr = err
					break
				}
			}
			elapsed := time.Since(start)
			if streamErr == nil {
				t.Fatal("stalled stream ended without an error")
			}
			if elapsed > 2*time.Second {
				t.Fatalf("stalled stream took %v to fail; idle timeout not applied", elapsed)
			}
			var idleErr *llm.StreamIdleTimeoutError
			var netErr net.Error
			if !errors.As(streamErr, &idleErr) || !errors.As(streamErr, &netErr) || !netErr.Timeout() {
				t.Fatalf("want a StreamIdleTimeoutError (net.Error timeout), got %T: %v", streamErr, streamErr)
			}
		})
	}
}

// Non-stream calls keep the whole-request Timeout.
func TestCompleteStillBoundedByTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		select { // headers sent, body never finishes
		case <-time.After(3 * time.Second):
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()
	client := newStreamClient(t, llm.Config{Provider: "openai", Model: "gpt-4.1", APIKey: "k", BaseURL: srv.URL}, 200*time.Millisecond)
	start := time.Now()
	_, err := client.Complete(context.Background(), llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")}})
	if err == nil || time.Since(start) > 2*time.Second {
		t.Fatalf("Complete err=%v after %v, want a timeout near 200ms", err, time.Since(start))
	}
}

// A stream whose headers never arrive fails via ResponseHeaderTimeout.
func TestStreamHeaderTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-time.After(3 * time.Second):
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()
	client := newStreamClient(t, llm.Config{Provider: "openai", Model: "gpt-4.1", APIKey: "k", BaseURL: srv.URL}, 200*time.Millisecond)
	start := time.Now()
	var streamErr error
	for _, err := range client.Stream(context.Background(), llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")}}) {
		if err != nil {
			streamErr = err
			break
		}
	}
	if streamErr == nil || time.Since(start) > 2*time.Second {
		t.Fatalf("err=%v after %v, want a header timeout near 200ms", streamErr, time.Since(start))
	}
}
