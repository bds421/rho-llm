package llm_test

// Hardening pass 8 — structured output request building.

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
)

// A json_schema response format with no Schema is meaningless and serializes to
// "schema": null, which the provider rejects. The adapter should degrade to plain
// JSON mode (json_object) rather than emit a null schema.
func TestOpenAICompatJSONSchemaWithoutSchemaDegrades(t *testing.T) {
	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"{}"},"finish_reason":"stop"}],"usage":{}}`)
	}))
	defer srv.Close()

	c, err := openaicompat.New(llm.Config{Provider: "openai", Model: "gpt-4.1", APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = c.Complete(context.Background(), llm.Request{
		MaxTokens:      16,
		Messages:       []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
		ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSONSchema, Name: "x"}, // no Schema
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if bytes.Contains(body, []byte(`"schema":null`)) {
		t.Fatalf("emitted a null json_schema (provider rejects it): %s", body)
	}
}
