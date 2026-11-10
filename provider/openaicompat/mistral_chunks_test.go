package openaicompat_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider/openaicompat"
)

// Mistral reasoning models (mistral-large-4-0) send content as typed chunks —
// a thinking chunk holding a list of text chunks, then the answer as a text
// chunk — in both responses and stream deltas (captured live 2026-11-10).
// Before, the answer was dropped (content was read only as a string) and
// every streamed chunk failed to decode.
const mistralChunkedResponse = `{"id":"x","model":"mistral-large-4-0","object":"chat.completion",
"usage":{"prompt_tokens":28,"completion_tokens":123,"completion_tokens_details":{"reasoning_tokens":119}},
"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","tool_calls":null,"content":[
{"type":"thinking","thinking":[{"type":"text","text":"17 x 23 "},{"type":"text","text":"= 391"}],"closed":true},
{"type":"text","text":"391"}]}}]}`

var mistralChunkedStream = strings.Join([]string{
	`data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
	`data: {"choices":[{"index":0,"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"We"}]}]},"finish_reason":null}]}`,
	`data: {"choices":[{"index":0,"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":" add."}],"closed":true}]},"finish_reason":null}]}`,
	`data: {"choices":[{"index":0,"delta":{"content":"4"},"finish_reason":null}]}`,
	`data: {"choices":[{"index":0,"delta":{"content":[{"type":"text","text":"!"}]},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":9}}`,
	`data: [DONE]`,
}, "\n\n") + "\n\n"

func mistralClient(t *testing.T, body, contentType string) llm.Client {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", contentType)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	c, err := llm.NewClient(llm.Config{
		Provider: "mistral", Model: "mistral-large-4-0", APIKey: "k",
		BaseURL: srv.URL, DisableProxy: true, DisableRetries: true, Timeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c
}

func TestMistralChunkedContentComplete(t *testing.T) {
	c := mistralClient(t, mistralChunkedResponse, "application/json")
	resp, err := c.Complete(context.Background(), llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "17*23?")}})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "391" {
		t.Errorf("Content = %q, want %q", resp.Content, "391")
	}
	if resp.Thinking != "17 x 23 = 391" {
		t.Errorf("Thinking = %q, want %q", resp.Thinking, "17 x 23 = 391")
	}
}

func TestMistralChunkedContentStream(t *testing.T) {
	c := mistralClient(t, mistralChunkedStream, "text/event-stream")
	var text, thinking strings.Builder
	for ev, err := range c.Stream(context.Background(), llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "2+2?")}}) {
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		switch ev.Type {
		case llm.EventContent:
			text.WriteString(ev.Text)
		case llm.EventThinking:
			thinking.WriteString(ev.Thinking)
		}
	}
	if text.String() != "4!" {
		t.Errorf("content = %q, want %q", text.String(), "4!")
	}
	if thinking.String() != "We add." {
		t.Errorf("thinking = %q, want %q", thinking.String(), "We add.")
	}
}
