package llm_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"gitlab2024.bds421-cloud.com/bds421/rho/llm"
	"gitlab2024.bds421-cloud.com/bds421/rho/llm/provider/anthropic"
	"gitlab2024.bds421-cloud.com/bds421/rho/llm/provider/gemini"
	"gitlab2024.bds421-cloud.com/bds421/rho/llm/provider/openaicompat"
)

// =============================================================================
// SECURITY TESTS
// =============================================================================

// TestGeminiAPIKeyNotInURL verifies that the Gemini adapter sends the API key
// via the x-goog-api-key header, NOT as a URL query parameter. API keys in
// URLs leak into server logs, proxy logs, referer headers, and browser history.
func TestGeminiAPIKeyNotInURL(t *testing.T) {
	const testKey = "test-secret-key-12345"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// FAIL if the key appears anywhere in the URL
		if strings.Contains(r.URL.String(), testKey) {
			t.Errorf("API key leaked into URL: %s", r.URL.String())
		}
		if r.URL.Query().Get("key") != "" {
			t.Errorf("API key sent as query parameter 'key=%s'", r.URL.Query().Get("key"))
		}

		// PASS if the key is in the header
		got := r.Header.Get("x-goog-api-key")
		if got != testKey {
			t.Errorf("x-goog-api-key header = %q, want %q", got, testKey)
		}

		// Return a minimal valid Gemini response
		w.Header().Set("Content-Type", "application/json")
		resp := map[string]interface{}{
			"candidates": []map[string]interface{}{
				{
					"content": map[string]interface{}{
						"parts": []map[string]string{{"text": "Hello"}},
						"role":  "model",
					},
					"finishReason": "STOP",
				},
			},
			"usageMetadata": map[string]int{
				"promptTokenCount":     5,
				"candidatesTokenCount": 1,
				"totalTokenCount":      6,
			},
		}
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			t.Fatalf("failed to encode response: %v", err)
		}
	}))
	defer srv.Close()

	cfg := llm.Config{
		Provider: "gemini",
		Model:    "gemini-2.5-flash",
		APIKey:   testKey,
		BaseURL:  srv.URL,
	}

	client, err := gemini.New(cfg)
	if err != nil {
		t.Fatalf("gemini.New() error: %v", err)
	}

	resp, err := client.Complete(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
	})
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}
	if resp.Content != "Hello" {
		t.Errorf("unexpected content: %q", resp.Content)
	}
}

// TestGeminiStreamAPIKeyNotInURL verifies the streaming path also uses headers.
func TestGeminiStreamAPIKeyNotInURL(t *testing.T) {
	const testKey = "stream-secret-key-67890"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.String(), testKey) {
			t.Errorf("API key leaked into URL: %s", r.URL.String())
		}
		if r.URL.Query().Get("key") != "" {
			t.Errorf("API key sent as query parameter")
		}

		got := r.Header.Get("x-goog-api-key")
		if got != testKey {
			t.Errorf("x-goog-api-key header = %q, want %q", got, testKey)
		}

		// Return minimal SSE stream
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":1}}\n\n")
	}))
	defer srv.Close()

	cfg := llm.Config{
		Provider: "gemini",
		Model:    "gemini-2.5-flash",
		APIKey:   testKey,
		BaseURL:  srv.URL,
	}

	client, err := gemini.New(cfg)
	if err != nil {
		t.Fatalf("gemini.New() error: %v", err)
	}

	for event, err := range client.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
	}) {
		if err != nil {
			t.Fatalf("Stream error: %v", err)
		}
		_ = event
	}
}

// TestErrorBodyTruncation verifies that enormous error response bodies are
// truncated before being stored in APIError.Message. Without truncation,
// a malicious endpoint could return a multi-GB error body causing OOM.
func TestErrorBodyTruncation(t *testing.T) {
	// 2MB body — well over the expected cap
	bigBody := strings.Repeat("x", 2*1024*1024)

	apiErr := llm.NewAPIErrorFromStatus("test", 500, bigBody)

	// Message should be capped (we expect 4KB max)
	const maxExpected = 4096 + 100 // allow small overhead for truncation marker
	if len(apiErr.Message) > maxExpected {
		t.Errorf("APIError.Message length = %d, want <= %d", len(apiErr.Message), maxExpected)
	}
}

// TestErrorBodyReadBounded verifies that io.ReadAll on error responses is
// bounded. We test this via the Gemini adapter with a server that streams
// an enormous error body.
func TestErrorBodyReadBounded(t *testing.T) {
	// Server that returns a 500 with a never-ending body
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		// Write 2MB of error data
		chunk := strings.Repeat("E", 64*1024)
		for i := 0; i < 32; i++ {
			if _, err := io.WriteString(w, chunk); err != nil {
				return
			}
		}
	}))
	defer srv.Close()

	cfg := llm.Config{
		Provider: "gemini",
		Model:    "gemini-2.5-flash",
		APIKey:   "test-key",
		BaseURL:  srv.URL,
	}

	client, err := gemini.New(cfg)
	if err != nil {
		t.Fatalf("gemini.New() error: %v", err)
	}

	_, err = client.Complete(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
	})
	if err == nil {
		t.Fatal("expected error from 500 response")
	}

	// The error message should be bounded, not the full 2MB
	var apiErr *llm.APIError
	if ok := errors.As(err, &apiErr); !ok {
		t.Fatalf("expected *llm.APIError, got %T", err)
	}

	const maxExpected = 1024*1024 + 4096 // 1MB read limit + truncation overhead
	if len(apiErr.Message) > maxExpected {
		t.Errorf("error body length = %d, want <= %d (unbounded read)", len(apiErr.Message), maxExpected)
	}
}

// TestRedirectDoesNotLeakAuthHeaders verifies that HTTP redirects to a
// different host strip sensitive authentication headers. Without this,
// a MITM or malicious redirect could steal API keys.
func TestRedirectDoesNotLeakAuthHeaders(t *testing.T) {
	var capturedHeaders http.Header

	// Attacker server that captures headers
	attacker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedHeaders = r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{"candidates":[{"content":{"parts":[{"text":"pwned"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}`)
	}))
	defer attacker.Close()

	// Legitimate-looking server that redirects to attacker
	redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, attacker.URL+r.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer redirector.Close()

	cfg := llm.Config{
		Provider: "gemini",
		Model:    "gemini-2.5-flash",
		APIKey:   "secret-key-do-not-leak",
		BaseURL:  redirector.URL,
	}

	client, err := gemini.New(cfg)
	if err != nil {
		t.Fatalf("gemini.New() error: %v", err)
	}

	// The request may succeed or fail depending on redirect policy —
	// what matters is that the attacker didn't get our auth headers.
	_, _ = client.Complete(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
	})

	if capturedHeaders != nil {
		if got := capturedHeaders.Get("x-goog-api-key"); got != "" {
			t.Errorf("x-goog-api-key header leaked to redirect target: %q", got)
		}
		if got := capturedHeaders.Get("Authorization"); got != "" {
			t.Errorf("Authorization header leaked to redirect target: %q", got)
		}
		if got := capturedHeaders.Get("x-api-key"); got != "" {
			t.Errorf("x-api-key header leaked to redirect target: %q", got)
		}
	}
}

// TestRedirectDoesNotLeakAnthropicKey tests redirect stripping for Anthropic's x-api-key.
func TestRedirectDoesNotLeakAnthropicKey(t *testing.T) {
	var capturedHeaders http.Header

	attacker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedHeaders = r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{"id":"msg_test","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[{"type":"text","text":"pwned"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer attacker.Close()

	redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, attacker.URL+r.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer redirector.Close()

	cfg := llm.Config{
		Provider: "anthropic",
		Model:    "claude-sonnet-4-6",
		APIKey:   "sk-secret-anthropic-key",
		BaseURL:  redirector.URL,
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient() error: %v", err)
	}
	defer client.Close()

	_, _ = client.Complete(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
	})

	if capturedHeaders != nil {
		if got := capturedHeaders.Get("x-api-key"); got != "" {
			t.Errorf("x-api-key header leaked to redirect target: %q", got)
		}
		if got := capturedHeaders.Get("Authorization"); got != "" {
			t.Errorf("Authorization header leaked to redirect target: %q", got)
		}
	}
}

// TestRedirectDoesNotLeakOpenAIKey tests redirect stripping for OpenAI Bearer tokens.
func TestRedirectDoesNotLeakOpenAIKey(t *testing.T) {
	var capturedHeaders http.Header

	attacker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedHeaders = r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{"id":"chatcmpl-test","object":"chat.completion","model":"gpt-4","choices":[{"index":0,"message":{"role":"assistant","content":"pwned"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer attacker.Close()

	redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, attacker.URL+r.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer redirector.Close()

	cfg := llm.Config{
		Provider:  "openai",
		Model:     "gpt-4",
		APIKey:    "sk-openai-secret-key",
		BaseURL:   redirector.URL,
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient() error: %v", err)
	}
	defer client.Close()

	_, _ = client.Complete(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
	})

	if capturedHeaders != nil {
		if got := capturedHeaders.Get("Authorization"); got != "" {
			t.Errorf("Authorization header leaked to redirect target: %q", got)
		}
	}
}

// =============================================================================
// ROUND 2: Unbounded success body decode, inputBuffer accumulation, SSE errors
// =============================================================================

// TestSuccessResponseBodyBounded verifies that JSON decoding of success
// response bodies is bounded. Without a limit, a malicious endpoint returning
// a multi-GB JSON response causes OOM via json.Decoder.
func TestSuccessResponseBodyBounded(t *testing.T) {
	t.Run("gemini", func(t *testing.T) {
		// Server returns a 200 with a huge JSON body (> MaxResponseBodyBytes)
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			// Write valid JSON prefix, then a huge text field
			fmt.Fprint(w, `{"candidates":[{"content":{"parts":[{"text":"`)
			// Write 40MB of 'A' — well over any sane response limit
			chunk := strings.Repeat("A", 64*1024)
			for i := 0; i < 640; i++ {
				if _, err := io.WriteString(w, chunk); err != nil {
					return
				}
			}
			fmt.Fprint(w, `"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}`)
		}))
		defer srv.Close()

		cfg := llm.Config{
			Provider: "gemini",
			Model:    "gemini-2.5-flash",
			APIKey:   "test-key",
			BaseURL:  srv.URL,
			Timeout:  10 * time.Second,
		}
		client, err := gemini.New(cfg)
		if err != nil {
			t.Fatalf("New() error: %v", err)
		}

		_, err = client.Complete(context.Background(), llm.Request{
			Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
		})
		// Should get a decode error because the reader was truncated, not OOM
		if err == nil {
			t.Fatal("expected error from oversized response body, got nil")
		}
		// Verify we got an error (not a panic/OOM)
		t.Logf("got expected error: %v", err)
	})

	t.Run("anthropic", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"id":"msg_test","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[{"type":"text","text":"`)
			chunk := strings.Repeat("A", 64*1024)
			for i := 0; i < 640; i++ {
				if _, err := io.WriteString(w, chunk); err != nil {
					return
				}
			}
			fmt.Fprint(w, `"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
		}))
		defer srv.Close()

		cfg := llm.Config{
			Provider: "anthropic",
			Model:    "claude-sonnet-4-6",
			APIKey:   "test-key",
			BaseURL:  srv.URL,
			Timeout:  10 * time.Second,
		}
		client, err := anthropic.New(cfg)
		if err != nil {
			t.Fatalf("New() error: %v", err)
		}

		_, err = client.Complete(context.Background(), llm.Request{
			Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
		})
		if err == nil {
			t.Fatal("expected error from oversized response body, got nil")
		}
		t.Logf("got expected error: %v", err)
	})

	t.Run("openai", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"id":"chatcmpl-test","object":"chat.completion","model":"gpt-4","choices":[{"index":0,"message":{"role":"assistant","content":"`)
			chunk := strings.Repeat("A", 64*1024)
			for i := 0; i < 640; i++ {
				if _, err := io.WriteString(w, chunk); err != nil {
					return
				}
			}
			fmt.Fprint(w, `"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
		}))
		defer srv.Close()

		cfg := llm.Config{
			Provider: "openai",
			Model:    "gpt-4",
			APIKey:   "test-key",
			BaseURL:  srv.URL,
			Timeout:  10 * time.Second,
		}
		client, err := openaicompat.New(cfg)
		if err != nil {
			t.Fatalf("New() error: %v", err)
		}

		_, err = client.Complete(context.Background(), llm.Request{
			Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
		})
		if err == nil {
			t.Fatal("expected error from oversized response body, got nil")
		}
		t.Logf("got expected error: %v", err)
	})
}

// TestStreamInputBufferBounded verifies that tool input accumulation during
// streaming is bounded. Without a cap, a malicious endpoint sending thousands
// of input_json_delta events can exhaust memory.
func TestStreamInputBufferBounded(t *testing.T) {
	t.Run("anthropic", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			// Start a tool call
			fmt.Fprint(w, "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_1\",\"name\":\"big_tool\"}}\n\n")
			// Send 2000 input_json_delta events, each ~1KB → total ~2MB
			for i := 0; i < 2000; i++ {
				chunk := strings.Repeat("x", 1024)
				fmt.Fprintf(w, "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"%s\"}}\n\n", chunk)
			}
			// End the tool call
			fmt.Fprint(w, "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n")
			fmt.Fprint(w, "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":100}}\n\n")
			fmt.Fprint(w, "data: {\"type\":\"message_stop\"}\n\n")
		}))
		defer srv.Close()

		cfg := llm.Config{
			Provider: "anthropic",
			Model:    "claude-sonnet-4-6",
			APIKey:   "test-key",
			BaseURL:  srv.URL,
			Timeout:  10 * time.Second,
		}
		client, err := anthropic.New(cfg)
		if err != nil {
			t.Fatalf("New() error: %v", err)
		}

		var gotError bool
		for _, err := range client.Stream(context.Background(), llm.Request{
			Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
		}) {
			if err != nil {
				gotError = true
				t.Logf("got expected error from oversized tool input: %v", err)
				break
			}
		}
		if !gotError {
			t.Error("expected error from oversized tool input buffer, stream completed without error")
		}
	})

	t.Run("openai", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			// Start a tool call
			fmt.Fprint(w, "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"big_tool\",\"arguments\":\"\"}}]}}]}\n\n")
			// Send 2000 argument chunks, each ~1KB → total ~2MB
			for i := 0; i < 2000; i++ {
				chunk := strings.Repeat("y", 1024)
				fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"%s\"}}]}}]}\n\n", chunk)
			}
			fmt.Fprint(w, "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"finish_reason\":\"tool_calls\"}]}\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
		}))
		defer srv.Close()

		cfg := llm.Config{
			Provider: "openai",
			Model:    "gpt-4",
			APIKey:   "test-key",
			BaseURL:  srv.URL,
			Timeout:  10 * time.Second,
		}
		client, err := openaicompat.New(cfg)
		if err != nil {
			t.Fatalf("New() error: %v", err)
		}

		var gotError bool
		for _, err := range client.Stream(context.Background(), llm.Request{
			Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
		}) {
			if err != nil {
				gotError = true
				t.Logf("got expected error from oversized tool input: %v", err)
				break
			}
		}
		if !gotError {
			t.Error("expected error from oversized tool input buffer, stream completed without error")
		}
	})
}

// TestMalformedSSEEventYieldsError verifies that malformed SSE events are
// reported as errors to the caller, not silently dropped. Silent drops cause
// data loss without the caller's knowledge.
func TestMalformedSSEEventYieldsError(t *testing.T) {
	t.Run("gemini", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			// Valid event, then malformed, then valid done
			fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}],\"role\":\"model\"}}]}\n\n")
			fmt.Fprint(w, "data: {this is not valid json}\n\n")
			fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\" world\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":1,\"candidatesTokenCount\":1}}\n\n")
		}))
		defer srv.Close()

		cfg := llm.Config{
			Provider: "gemini",
			Model:    "gemini-2.5-flash",
			APIKey:   "test-key",
			BaseURL:  srv.URL,
			Timeout:  10 * time.Second,
		}
		client, err := gemini.New(cfg)
		if err != nil {
			t.Fatalf("New() error: %v", err)
		}

		var gotError bool
		for _, err := range client.Stream(context.Background(), llm.Request{
			Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
		}) {
			if err != nil {
				gotError = true
				break
			}
		}
		if !gotError {
			t.Error("expected error from malformed SSE event, stream completed silently")
		}
	})

	t.Run("anthropic", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10}}}\n\n")
			fmt.Fprint(w, "data: {not valid json at all}\n\n")
			fmt.Fprint(w, "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n")
		}))
		defer srv.Close()

		cfg := llm.Config{
			Provider: "anthropic",
			Model:    "claude-sonnet-4-6",
			APIKey:   "test-key",
			BaseURL:  srv.URL,
			Timeout:  10 * time.Second,
		}
		client, err := anthropic.New(cfg)
		if err != nil {
			t.Fatalf("New() error: %v", err)
		}

		var gotError bool
		for _, err := range client.Stream(context.Background(), llm.Request{
			Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
		}) {
			if err != nil {
				gotError = true
				break
			}
		}
		if !gotError {
			t.Error("expected error from malformed SSE event, stream completed silently")
		}
	})

	t.Run("openai", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"}}]}\n\n")
			fmt.Fprint(w, "data: {broken json\n\n")
			fmt.Fprint(w, "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"finish_reason\":\"stop\"}]}\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
		}))
		defer srv.Close()

		cfg := llm.Config{
			Provider: "openai",
			Model:    "gpt-4",
			APIKey:   "test-key",
			BaseURL:  srv.URL,
			Timeout:  10 * time.Second,
		}
		client, err := openaicompat.New(cfg)
		if err != nil {
			t.Fatalf("New() error: %v", err)
		}

		var gotError bool
		for _, err := range client.Stream(context.Background(), llm.Request{
			Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
		}) {
			if err != nil {
				gotError = true
				break
			}
		}
		if !gotError {
			t.Error("expected error from malformed SSE event, stream completed silently")
		}
	})
}
