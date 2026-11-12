package openairesponses

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// A refusal content block was dropped: the caller got an empty end_turn and no
// hint the model had declined (B4).
func TestRefusalIsContentWithStopRefusal(t *testing.T) {
	c := &Client{providerName: "openai"}
	var r responsesResponse
	if err := json.Unmarshal([]byte(`{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"refusal","refusal":"I can't help with that."}]}]}`), &r); err != nil {
		t.Fatal(err)
	}
	resp := c.parseResponse(&r)
	if resp.Content != "I can't help with that." || resp.StopReason != llm.StopRefusal {
		t.Fatalf("Complete: content=%q stop=%q, want the refusal text and stop refusal", resp.Content, resp.StopReason)
	}

	sse := "data: {\"type\":\"response.refusal.delta\",\"delta\":\"I can't \"}\n\n" +
		"data: {\"type\":\"response.refusal.delta\",\"delta\":\"help with that.\"}\n\n" +
		"data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":3,\"output_tokens\":5}}}\n\n"
	var text strings.Builder
	var done llm.StreamEvent
	c.parseStream(strings.NewReader(sse), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			t.Fatal(err)
		}
		switch ev.Type {
		case llm.EventContent:
			text.WriteString(ev.Text)
		case llm.EventDone:
			done = ev
		}
		return true
	})
	if text.String() != "I can't help with that." || done.StopReason != llm.StopRefusal {
		t.Fatalf("Stream: text=%q stop=%q", text.String(), done.StopReason)
	}
}

// Complete joined text blocks with "\n" while Stream concatenated them, so the
// same answer differed between the two paths (B4).
func TestCompleteJoinsTextLikeStream(t *testing.T) {
	c := &Client{providerName: "openai"}
	var r responsesResponse
	if err := json.Unmarshal([]byte(`{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"Hello "},{"type":"output_text","text":"world"}]}]}`), &r); err != nil {
		t.Fatal(err)
	}
	if got := c.parseResponse(&r).Content; got != "Hello world" {
		t.Fatalf("Content = %q, want %q", got, "Hello world")
	}
}

// A failed response returned stop reason "error" with no error and no detail
// from Complete, and a Done event from Stream (B4).
func TestFailedResponseIsAnAPIError(t *testing.T) {
	for _, tc := range []struct {
		code      string
		status    int
		retryable bool
	}{
		{"server_error", 500, true},
		{"rate_limit_exceeded", 429, true},
		{"invalid_prompt", 400, false},
		{"", 500, true},
	} {
		body := `{"id":"r","status":"failed","error":{"code":"` + tc.code + `","message":"it failed"},"usage":{"input_tokens":4,"output_tokens":0}}`
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(body))
		}))
		c, err := New(llm.Config{Provider: "openai_responses", Model: "gpt-6.1-sol", APIKey: "k", BaseURL: srv.URL})
		if err != nil {
			t.Fatal(err)
		}
		resp, err := c.Complete(context.Background(), llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "x")}})
		srv.Close()
		var apiErr *llm.APIError
		if resp != nil || !errors.As(err, &apiErr) || apiErr.StatusCode != tc.status || llm.IsRetryable(err) != tc.retryable {
			t.Fatalf("%q: resp=%v err=%v, want status %d retryable=%v", tc.code, resp, err, tc.status, tc.retryable)
		}
		if !strings.Contains(err.Error(), "it failed") {
			t.Fatalf("%q: provider message lost: %v", tc.code, err)
		}

		if _, err := c.ParseResponsesBatchResultBody(json.RawMessage(body)); !errors.As(err, &apiErr) {
			t.Fatalf("%q: batch result body: err = %v, want an APIError", tc.code, err)
		}
	}
}

// A stream "error" event was a plain error: rate limits and auth failures lost
// their classification (B4).
func TestStreamErrorEventIsClassified(t *testing.T) {
	c := &Client{providerName: "openai", config: llm.Config{APIKey: "sk-secret-123456789"}}
	var got error
	c.parseStream(strings.NewReader(`data: {"type":"error","error":{"code":"rate_limit_exceeded","message":"slow down sk-secret-123456789"}}`+"\n\n"), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			got = err
		}
		return true
	})
	if !llm.IsRateLimited(got) || strings.Contains(got.Error(), "sk-secret-123456789") {
		t.Fatalf("err = %v, want a rate-limit error without the key", got)
	}
}
