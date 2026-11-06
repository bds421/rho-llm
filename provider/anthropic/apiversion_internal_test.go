package anthropic

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
)

// API version 2026-10-21 is the default. What it rejects with a 400 is
// refused before sending; pinning 2023-06-01 keeps the old rules.
func TestAPIVersion20261021(t *testing.T) {
	temp := 0.3
	cited := llm.ContentPart{Type: llm.ContentText, Text: "quoted", Citations: []llm.Citation{{Type: "char_location", CitedText: "x"}}}
	doc := llm.NewTextDocument("", "text")
	refused := map[string]llm.Request{
		"temperature":          {Temperature: &temp},
		"top_p":                {SamplingParams: map[string]any{"top_p": 0.5}},
		"top_k":                {SamplingParams: map[string]any{"top_k": 5}},
		"sampling temperature": {SamplingParams: map[string]any{"temperature": 1}},
		"document in assistant": {Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "q"),
			{Role: llm.RoleAssistant, Content: []llm.ContentPart{doc}}}},
		"citations on user text": {Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentPart{cited}}}},
		"citations in a tool result": {Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentPart{
			{Type: llm.ContentToolResult, ToolResultID: "t1", ToolResultParts: []llm.ContentPart{cited}}}}}},
	}
	strict := &Client{config: llm.Config{Model: "claude-haiku-4-5", MaxTokens: 256}, providerName: "anthropic"}
	pinned := &Client{config: llm.Config{Model: "claude-haiku-4-5", MaxTokens: 256, AnthropicVersion: llm.AnthropicVersion20230601}, providerName: "anthropic"}
	for name, req := range refused {
		if len(req.Messages) == 0 {
			req.Messages = []llm.Message{llm.NewTextMessage(llm.RoleUser, "q")}
		}
		if _, err := strict.buildRequest(req, false); err == nil || !strings.Contains(err.Error(), "2026-10-21") {
			t.Errorf("%s on 2026-10-21: %v", name, err)
		}
		if _, err := pinned.buildRequest(req, false); err != nil {
			t.Errorf("%s on 2023-06-01: %v", name, err)
		}
	}
	// Citations on assistant text, as Claude returned them, stay accepted.
	ok := llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "q"), {Role: llm.RoleAssistant, Content: []llm.ContentPart{cited}}}}
	if _, err := strict.buildRequest(ok, false); err != nil {
		t.Errorf("assistant citations: %v", err)
	}
}

func TestAPIVersionHeader(t *testing.T) {
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Get("anthropic-version")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"m","type":"message","role":"assistant","model":"claude-haiku-5-5","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer srv.Close()
	for version, want := range map[string]string{"": "2026-10-21", llm.AnthropicVersion20230601: "2023-06-01"} {
		c, err := New(llm.Config{Model: "claude-haiku-5-5", APIKey: "k", BaseURL: srv.URL, Timeout: 5 * time.Second, AnthropicVersion: version})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := c.Complete(context.Background(), llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "q")}}); err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Errorf("version %q: header %q, want %q", version, got, want)
		}
	}
}
