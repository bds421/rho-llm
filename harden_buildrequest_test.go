package llm_test

// Hardening pass 6 — adapter buildRequest message integrity.

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/provider/anthropic"
)

// A message whose content all gets filtered out (here: an assistant turn with
// only an empty-text part) must not be emitted as an empty-content message —
// Anthropic rejects "content": [] / null. The adapter should drop it (matching
// NormalizeForProvider), not send a malformed message.
func TestAnthropicBuildRequestDropsEmptyContentMessage(t *testing.T) {
	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"m","model":"claude-sonnet-4-6","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer srv.Close()

	client, err := anthropic.New(llm.Config{Provider: "anthropic", Model: "claude-sonnet-4-6", APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = client.Complete(context.Background(), llm.Request{
		MaxTokens: 16,
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "hi"),
			{Role: llm.RoleAssistant, Content: []llm.ContentPart{{Type: llm.ContentText, Text: ""}}}, // all-empty
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if bytes.Contains(body, []byte(`"content":[]`)) || bytes.Contains(body, []byte(`"content":null`)) {
		t.Fatalf("emitted an empty-content message (Anthropic rejects it): %s", body)
	}
}
