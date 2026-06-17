package llm_test

// Hardening pass 7 — OpenAI-family buildRequest empty-message integrity.

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
	"github.com/bds421/rho-llm/provider/openaicompat"
	"github.com/bds421/rho-llm/provider/openairesponses"
)

func captureBody(t *testing.T, jsonResp string) (*httptest.Server, *[]byte) {
	t.Helper()
	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, jsonResp)
	}))
	return srv, &body
}

// A message whose content all filtered out must not be emitted with a null/empty
// content field that the provider rejects.
func TestOpenAIFamilyBuildRequestDropsEmptyContentMessage(t *testing.T) {
	emptyAsst := llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentPart{{Type: llm.ContentText, Text: ""}}}
	msgs := []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi"), emptyAsst}

	t.Run("openaicompat", func(t *testing.T) {
		srv, body := captureBody(t, `{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{}}`)
		defer srv.Close()
		c, err := openaicompat.New(llm.Config{Provider: "openai", Model: "gpt-4.1", APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if _, err := c.Complete(context.Background(), llm.Request{MaxTokens: 16, Messages: msgs}); err != nil {
			t.Fatalf("Complete: %v", err)
		}
		if bytes.Contains(*body, []byte(`"content":null`)) {
			t.Fatalf("openaicompat emitted a null-content message: %s", *body)
		}
	})

	t.Run("openairesponses", func(t *testing.T) {
		srv, body := captureBody(t, `{"id":"r","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{}}`)
		defer srv.Close()
		c, err := openairesponses.New(llm.Config{Provider: "openai_responses", Model: "gpt-5.2", APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if _, err := c.Complete(context.Background(), llm.Request{MaxTokens: 16, Messages: msgs}); err != nil {
			t.Fatalf("Complete: %v", err)
		}
		if bytes.Contains(*body, []byte(`"content":null`)) {
			t.Fatalf("openairesponses emitted a null-content message: %s", *body)
		}
	})
}
