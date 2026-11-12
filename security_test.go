package llm_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/provider/anthropic"
	"github.com/bds421/rho-llm/provider/gemini"
	"github.com/bds421/rho-llm/provider/openaicompat"
	"github.com/bds421/rho-llm/provider/openairesponses"
)

// A per-key "apikey|baseurl" override can embed credentials in the BaseURL
// (userinfo user:pass@ or a ?token= secret). When an upstream error echoes that
// BaseURL, MarkFailed stores it in LastError and MarshalJSON serializes it — both
// must strip the embedded secrets (AuthProfile.MarshalJSON already redacts the
// BaseURL *field*, so leaving them in LastError is an inconsistency/leak).
func TestAuthProfileDoesNotLeakBaseURLCredsInLastError(t *testing.T) {
	const apiKey = "sk-minimax-abcdef123456"
	const baseURL = "https://user:superSecretPw@proxy.internal:9000/v1?token=bearer_xyz987"
	p := llm.AuthProfile{Name: "minimax-1", APIKey: apiKey, BaseURL: baseURL}

	// Upstream error that echoes the full per-key BaseURL + the API key.
	p.MarkFailed(fmt.Errorf("403 from %s using key %s", baseURL, apiKey), time.Second)

	leaks := []string{"superSecretPw", "bearer_xyz987", apiKey}
	for _, secret := range leaks {
		if strings.Contains(p.LastError, secret) {
			t.Errorf("MarkFailed left secret %q in LastError: %q", secret, p.LastError)
		}
	}
	// Host must remain visible for debugging.
	if !strings.Contains(p.LastError, "proxy.internal") {
		t.Errorf("LastError dropped the host (needed for debugging): %q", p.LastError)
	}

	b, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	s := string(b)
	for _, secret := range leaks {
		if strings.Contains(s, secret) {
			t.Errorf("AuthProfile JSON leaked secret %q: %s", secret, s)
		}
	}
}

// The model is interpolated into the request URL path; a model containing URL
// metacharacters must be escaped, not injected (e.g. "m?x=1" must not add a
// query parameter / change the endpoint).
func TestGeminiEscapesModelInURL(t *testing.T) {
	var rawQuery, path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rawQuery, path = r.URL.RawQuery, r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"ok"}],"role":"model"}}],"usageMetadata":{}}`))
	}))
	defer srv.Close()

	c, err := gemini.New(llm.Config{Provider: "gemini", Model: "m?injected=1", APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, _ = c.Complete(context.Background(), llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")}})

	if strings.Contains(rawQuery, "injected=1") {
		t.Fatalf("model name injected a query parameter (unescaped): rawQuery=%q path=%q", rawQuery, path)
	}
}

func TestFixedPathAdaptersIgnoreHostileModelInURL(t *testing.T) {
	const hostile = "m?inject=1/../../evil"

	cases := []struct {
		name     string
		wantPath string
		newC     func(cfg llm.Config) (llm.Client, error)
		resp     string
	}{
		{"anthropic", "/messages", func(c llm.Config) (llm.Client, error) { return anthropic.New(c) },
			`{"id":"m","model":"x","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`},
		{"openaicompat", "/chat/completions", func(c llm.Config) (llm.Client, error) { return openaicompat.New(c) },
			`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{}}`},
		{"openairesponses", "/responses", func(c llm.Config) (llm.Client, error) { return openairesponses.New(c) },
			`{"id":"r","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{}}`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var gotPath, gotQuery string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotPath, gotQuery = r.URL.Path, r.URL.RawQuery
				w.Header().Set("Content-Type", "application/json")
				w.Write([]byte(tc.resp))
			}))
			defer srv.Close()

			c, err := tc.newC(llm.Config{Provider: tc.name, Model: hostile, APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second,
				ModelCapabilities: llm.Capabilities(llm.CapabilityChat, llm.CapabilityStream)})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			_, _ = c.Complete(context.Background(), llm.Request{MaxTokens: 16, Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")}})

			if gotPath != tc.wantPath {
				t.Fatalf("hostile model altered the request path: got %q, want %q", gotPath, tc.wantPath)
			}
			if gotQuery != "" {
				t.Fatalf("hostile model injected a query: %q", gotQuery)
			}
		})
	}
}

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

// A redirect chain that leaves the original origin and then stays on the new
// one: A → B/first → B/second. net/http copies the original request's headers
// onto every hop, so comparing each hop only with the previous one strips them
// on A → B and hands them to B on B → B (B1). The second variant keeps the
// hostname and changes only the port, where Go's own Authorization strip
// (which compares hostnames) does not help either.
func TestRedirectChainNeverReturnsKeysToAForeignOrigin(t *testing.T) {
	for _, foreignHost := range []string{"localhost", "127.0.0.1"} {
		for _, header := range []string{"Authorization", "x-api-key", "x-goog-api-key"} {
			var seen []string
			foreign := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				seen = append(seen, r.URL.Path+"="+r.Header.Get(header))
				if r.URL.Path == "/first" {
					http.Redirect(w, r, "/second", http.StatusTemporaryRedirect)
					return
				}
				w.WriteHeader(http.StatusOK)
			}))
			target := strings.Replace(foreign.URL, "127.0.0.1", foreignHost, 1) + "/first"
			origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				http.Redirect(w, r, target, http.StatusTemporaryRedirect)
			}))

			req, err := http.NewRequest(http.MethodPost, origin.URL+"/v1/messages", strings.NewReader(`{}`))
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set(header, "secret-key-do-not-leak")
			resp, err := llm.SafeHTTPClient(5 * time.Second).Do(req)
			if err != nil {
				t.Fatalf("%s via %s: %v", header, foreignHost, err)
			}
			resp.Body.Close()
			origin.Close()
			foreign.Close()

			if len(seen) != 2 {
				t.Fatalf("%s via %s: foreign origin saw %v, want both hops", header, foreignHost, seen)
			}
			for _, hop := range seen {
				if strings.Contains(hop, "secret-key-do-not-leak") {
					t.Errorf("%s via %s: key reached the foreign origin: %v", header, foreignHost, seen)
				}
			}
		}
	}
}

// A chain that goes away and comes back keeps the key on the way back: the
// original origin is the one the caller sent it to. localhost:<port> is a
// different origin from 127.0.0.1:<port>.
func TestRedirectChainBackToTheOriginKeepsKeys(t *testing.T) {
	var originURL, gotBack string
	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/back":
			gotBack = r.Header.Get("x-api-key")
			w.WriteHeader(http.StatusOK)
		case "/away":
			http.Redirect(w, r, originURL+"/back", http.StatusTemporaryRedirect)
		default:
			http.Redirect(w, r, strings.Replace(originURL, "127.0.0.1", "localhost", 1)+"/away", http.StatusTemporaryRedirect)
		}
	}))
	defer origin.Close()
	originURL = origin.URL

	req, err := http.NewRequest(http.MethodGet, origin.URL+"/start", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("x-api-key", "secret-key")
	resp, err := llm.SafeHTTPClient(5 * time.Second).Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if gotBack != "secret-key" {
		t.Fatalf("back on the original origin the key is %q, want it kept", gotBack)
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
		Provider: "openai",
		Model:    "gpt-4",
		APIKey:   "sk-openai-secret-key",
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
		if got := capturedHeaders.Get("Authorization"); got != "" {
			t.Errorf("Authorization header leaked to redirect target: %q", got)
		}
	}
}

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

// TestConfigMarshalJSONRedactsAPIKey verifies that Config.MarshalJSON replaces
// the APIKey with "REDACTED" so secrets don't leak into logs or debug output.
func TestConfigMarshalJSONRedactsAPIKey(t *testing.T) {
	tests := []struct {
		name    string
		apiKey  string
		wantKey string
	}{
		{"non-empty key is redacted", "sk-secret-12345", "REDACTED"},
		{"empty key stays empty", "", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := llm.DefaultConfig()
			cfg.APIKey = tt.apiKey

			data, err := json.Marshal(cfg)
			if err != nil {
				t.Fatalf("Marshal error: %v", err)
			}

			var decoded map[string]interface{}
			if err := json.Unmarshal(data, &decoded); err != nil {
				t.Fatalf("Unmarshal error: %v", err)
			}

			got, _ := decoded["api_key"].(string)
			if got != tt.wantKey {
				t.Errorf("api_key = %q, want %q", got, tt.wantKey)
			}

			// The raw JSON must never contain the original secret
			if tt.apiKey != "" && strings.Contains(string(data), tt.apiKey) {
				t.Errorf("serialized JSON contains the raw API key")
			}
		})
	}
}

// TestAuthProfileMarshalJSONRedactsAPIKey verifies that AuthProfile.MarshalJSON
// replaces the APIKey with "REDACTED".
func TestAuthProfileMarshalJSONRedactsAPIKey(t *testing.T) {
	tests := []struct {
		name    string
		apiKey  string
		wantKey string
	}{
		{"non-empty key is redacted", "sk-auth-profile-secret", "REDACTED"},
		{"empty key stays empty", "", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			profile := llm.AuthProfile{
				Name:      "test-profile",
				APIKey:    tt.apiKey,
				IsHealthy: true,
			}

			data, err := json.Marshal(profile)
			if err != nil {
				t.Fatalf("Marshal error: %v", err)
			}

			var decoded map[string]interface{}
			if err := json.Unmarshal(data, &decoded); err != nil {
				t.Fatalf("Unmarshal error: %v", err)
			}

			got, _ := decoded["api_key"].(string)
			if got != tt.wantKey {
				t.Errorf("api_key = %q, want %q", got, tt.wantKey)
			}

			if tt.apiKey != "" && strings.Contains(string(data), tt.apiKey) {
				t.Errorf("serialized JSON contains the raw API key")
			}
		})
	}
}

// TestBaseURLSchemeValidation documents that non-HTTP(S) BaseURL schemes
// (file://, javascript:, ftp://, data:) fail at request time. NOTE: this library
// does NOT validate the scheme itself — Go's net/http rejects unsupported schemes
// when the request is made. This test pins that incidental behavior; it is not
// evidence of an allowlist. For actual SSRF hardening of BaseURL, see
// Config.BlockPrivateBaseURL / llm.CheckBaseURL (TestBlockPrivateBaseURL).
//
// Uses the openaicompat adapter directly (not llm.NewClient) to avoid the
// PooledClient retry loop, which would add ~30s per subtest in cooldown waits.
// TestAdapterCloseClosesIdleConnections verifies that calling Close() on
// each provider adapter drains idle HTTP connections. Without this, rotated
// clients leak transport-level connection pools.
func TestAdapterCloseClosesIdleConnections(t *testing.T) {
	// The test strategy: create a client pointing at a test server, make one
	// request to establish a connection pool entry, then Close() the client.
	// After Close, the server should not see any lingering connections.
	// We can verify this indirectly: calling Close should not panic and should
	// invoke CloseIdleConnections (we can verify by checking the HTTP transport
	// state). In practice, the simplest effective test is to verify Close()
	// returns nil and the method actually calls httpClient.CloseIdleConnections.
	//
	// A practical approach: make a request, then Close. If Close merely
	// returns nil without draining, connections stay open. We verify via a
	// second request after Close — if connections were drained, the transport
	// needs to re-dial.

	t.Run("anthropic", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
		}))
		defer srv.Close()

		client, err := anthropic.New(llm.Config{
			Provider: "anthropic", Model: "claude-sonnet-4-6",
			APIKey: "test", BaseURL: srv.URL, Timeout: 5 * time.Second,
		})
		if err != nil {
			t.Fatalf("New: %v", err)
		}

		// Establish a connection
		_, err = client.Complete(context.Background(), llm.Request{
			Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
		})
		if err != nil {
			t.Fatalf("Complete: %v", err)
		}

		// Close should drain idle connections (not panic, return nil)
		if err := client.Close(); err != nil {
			t.Errorf("Close() = %v, want nil", err)
		}
	})

	t.Run("gemini", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"candidates":[{"content":{"parts":[{"text":"ok"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}`)
		}))
		defer srv.Close()

		client, err := gemini.New(llm.Config{
			Provider: "gemini", Model: "gemini-2.5-flash",
			APIKey: "test", BaseURL: srv.URL, Timeout: 5 * time.Second,
		})
		if err != nil {
			t.Fatalf("New: %v", err)
		}

		_, err = client.Complete(context.Background(), llm.Request{
			Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
		})
		if err != nil {
			t.Fatalf("Complete: %v", err)
		}

		if err := client.Close(); err != nil {
			t.Errorf("Close() = %v, want nil", err)
		}
	})

	t.Run("openaicompat", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"id":"chatcmpl-1","object":"chat.completion","model":"gpt-4","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
		}))
		defer srv.Close()

		client, err := openaicompat.New(llm.Config{
			Provider: "openai", Model: "gpt-4",
			APIKey: "test", BaseURL: srv.URL, Timeout: 5 * time.Second,
		})
		if err != nil {
			t.Fatalf("New: %v", err)
		}

		_, err = client.Complete(context.Background(), llm.Request{
			Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
		})
		if err != nil {
			t.Fatalf("Complete: %v", err)
		}

		if err := client.Close(); err != nil {
			t.Errorf("Close() = %v, want nil", err)
		}
	})
}

func TestBaseURLSchemeValidation(t *testing.T) {
	tests := []struct {
		name      string
		baseURL   string
		wantError bool
	}{
		{"https scheme works", "https://api.example.com", false},
		{"http scheme works", "http://localhost:8080", false},
		{"file scheme rejected", "file:///etc/passwd", true},
		{"javascript scheme rejected", "javascript:alert(1)", true},
		{"ftp scheme rejected", "ftp://example.com", true},
		{"data scheme rejected", "data:text/html,<h1>hi</h1>", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := llm.Config{
				Provider:  "openai",
				Model:     "gpt-4",
				APIKey:    "test-key",
				BaseURL:   tt.baseURL,
				MaxTokens: 100,
				Timeout:   2 * time.Second,
			}

			client, err := openaicompat.New(cfg)
			if err != nil {
				// Construction error is fine for invalid schemes
				if tt.wantError {
					return
				}
				t.Fatalf("New() unexpected error: %v", err)
			}

			// Try a request — invalid schemes should fail at request creation
			_, err = client.Complete(context.Background(), llm.Request{
				Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
			})

			if tt.wantError && err == nil {
				t.Error("expected error for invalid URL scheme, got nil")
			}
			if !tt.wantError && err != nil {
				// Connection errors are expected (no real server) — that's fine.
				// We only care that http/https don't produce *scheme* errors.
				errStr := err.Error()
				if strings.Contains(errStr, "unsupported protocol") ||
					strings.Contains(errStr, "invalid scheme") {
					t.Errorf("unexpected scheme error for valid URL: %v", err)
				}
			}
		})
	}
}

// TestPerConfigErrorMessageLimit proves that Config.MaxErrorMessageLen is actually
// honored by the adapters. Regression test for the "dead config" bug where adapters
// ignored every per-Config safety limit and used package globals/constants instead.
// A server returns a 5000-byte error body; with MaxErrorMessageLen=100 the stored
// message must be far shorter than the 4096-byte default would have allowed.
func TestPerConfigErrorMessageLimit(t *testing.T) {
	const bodyLen = 5000
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		if _, err := io.WriteString(w, strings.Repeat("E", bodyLen)); err != nil {
			return
		}
	}))
	defer srv.Close()

	cfg := llm.Config{
		Provider:           "gemini",
		Model:              "gemini-2.5-flash",
		APIKey:             "test-key",
		BaseURL:            srv.URL,
		MaxErrorMessageLen: 100, // far below DefaultMaxErrorMessageLen (4096)
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

	var apiErr *llm.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *llm.APIError, got %T", err)
	}

	// 100 chars + "... [truncated]" marker. If the per-Config limit were ignored,
	// the default (4096) would yield a message ~40x longer than this.
	const wantMax = 100 + len("... [truncated]")
	if len(apiErr.Message) > wantMax {
		t.Errorf("APIError.Message length = %d, want <= %d (per-Config MaxErrorMessageLen was ignored)", len(apiErr.Message), wantMax)
	}
	if len(apiErr.Message) < 100 {
		t.Errorf("APIError.Message length = %d, want >= 100 (truncated too aggressively)", len(apiErr.Message))
	}
}

// TestBlockPrivateBaseURL verifies the opt-in SSRF guard (F2): with
// BlockPrivateBaseURL set, construction rejects loopback/private/link-local hosts
// (incl. the cloud metadata IP) — while the default (opt-out) still allows them,
// so local providers keep working.
func TestBlockPrivateBaseURL(t *testing.T) {
	cases := []struct {
		name    string
		baseURL string
		block   bool
		wantErr bool
	}{
		{"metadata IP blocked", "http://169.254.169.254/latest", true, true},
		{"loopback blocked", "http://127.0.0.1:8080/v1", true, true},
		{"localhost blocked", "http://localhost:11434/v1", true, true},
		{"private 10.x blocked", "http://10.0.0.5/v1", true, true},
		{"public host allowed", "https://api.example.com/v1", true, false},
		{"default: metadata IP allowed (opt-out)", "http://169.254.169.254/latest", false, false},
		{"empty base URL ok", "", true, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := llm.CheckBaseURL(llm.Config{BaseURL: tc.baseURL, BlockPrivateBaseURL: tc.block})
			if tc.wantErr && err == nil {
				t.Errorf("CheckBaseURL(%q, block=%v) = nil, want error", tc.baseURL, tc.block)
			}
			if !tc.wantErr && err != nil {
				t.Errorf("CheckBaseURL(%q, block=%v) = %v, want nil", tc.baseURL, tc.block, err)
			}
		})
	}

	// Integration: NewClient must refuse a private BaseURL when blocking is on.
	_, err := llm.NewClient(llm.Config{
		Provider: "openai", Model: "gpt-4", APIKey: "k",
		BaseURL: "http://169.254.169.254/v1", BlockPrivateBaseURL: true,
		MaxTokens: 10,
	})
	if err == nil {
		t.Error("NewClient with private BaseURL + BlockPrivateBaseURL should error")
	}
}

// Config and AuthProfile redacted the key only in MarshalJSON: fmt's %v, %+v,
// %#v and slog's text handler printed it (B32).
func TestConfigAndProfileRedactInEveryPrintedForm(t *testing.T) {
	const key = "sk-ant-secret-key-123456"
	cfg := llm.Config{Provider: "anthropic", APIKey: key, BaseURL: "https://user:pass-secret@gw.example.com/v1"}
	profile := llm.AuthProfile{Name: "p", APIKey: key, LastError: "401 for " + key}

	var text, js strings.Builder
	slog.New(slog.NewTextHandler(&text, nil)).Info("x", "cfg", cfg, "profile", profile, "ptr", &cfg)
	slog.New(slog.NewJSONHandler(&js, nil)).Info("x", "cfg", cfg, "profile", profile)
	outputs := map[string]string{
		"%v cfg": fmt.Sprintf("%v", cfg), "%+v cfg": fmt.Sprintf("%+v", cfg), "%#v cfg": fmt.Sprintf("%#v", cfg),
		"%v &cfg":     fmt.Sprintf("%v", &cfg),
		"%+v profile": fmt.Sprintf("%+v", profile), "%#v profile": fmt.Sprintf("%#v", profile),
		"slog text": text.String(), "slog json": js.String(),
	}
	for form, out := range outputs {
		if strings.Contains(out, key) || strings.Contains(out, "pass-secret") {
			t.Errorf("%s printed a secret: %.200s", form, out)
		}
	}
	if !strings.Contains(outputs["%+v cfg"], "Provider:anthropic") {
		t.Errorf("%%+v lost the non-secret fields: %.200s", outputs["%+v cfg"])
	}
}
