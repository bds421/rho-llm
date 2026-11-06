package anthropic_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/provider/anthropic"
)

// A client built directly with New checks capabilities like the factory: a
// ResponseFormat the model cannot honour is refused before the request.
func TestDirectClientChecksCapabilities(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("request sent")
	}))
	defer srv.Close()
	c, err := anthropic.New(llm.Config{Model: "claude-example-1", APIKey: "k", BaseURL: srv.URL, Timeout: 5 * time.Second,
		ModelCapabilities: llm.Capabilities(llm.CapabilityChat, llm.CapabilityStream)})
	if err != nil {
		t.Fatal(err)
	}
	req := llm.Request{
		ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSONSchema, Schema: map[string]any{"type": "object"}},
		Messages:       []llm.Message{llm.NewTextMessage(llm.RoleUser, "q")},
	}
	if _, err := c.Complete(context.Background(), req); err == nil {
		t.Error("Complete accepted a ResponseFormat without the capability")
	}
	if _, err := llm.CollectStream(c.Stream(context.Background(), req)); err == nil {
		t.Error("Stream accepted a ResponseFormat without the capability")
	}
}
