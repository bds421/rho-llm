package openaicompat

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

func streamAll(c *Client, sse string) (evs []llm.StreamEvent, errs []error) {
	c.parseStream(strings.NewReader(sse), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			errs = append(errs, err)
		}
		evs = append(evs, ev)
		return true
	})
	return
}

// A server that fails after the 200 puts the error in a chunk (OpenRouter,
// vLLM, Together), often followed by [DONE]. Before B2 the chunk was decoded
// as an empty delta: the turn ended as a successful end_turn with truncated
// content, or as a retryable "unexpected EOF" that re-sent a 400.
func TestStreamErrorChunkEndsTheTurnWithAnAPIError(t *testing.T) {
	text := "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"}}]}\n\n"
	cases := []struct {
		name      string
		sse       string
		status    int
		retryable bool
		message   string
	}{
		{"numeric code then DONE", text + "data: {\"error\":{\"code\":429,\"message\":\"Rate limit exceeded\"}}\n\ndata: [DONE]\n\n", 429, true, "Rate limit exceeded"},
		{"error with finish_reason error", text + "data: {\"error\":{\"code\":502,\"message\":\"Provider disconnected\"},\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"error\"}]}\n\ndata: [DONE]\n\n", 502, true, "Provider disconnected"},
		{"400 without DONE", "data: {\"error\":{\"code\":400,\"message\":\"context length exceeded\"}}\n\n", 400, false, "context length exceeded"},
		{"string code", text + "data: {\"error\":{\"message\":\"Incorrect API key\",\"type\":\"invalid_request_error\",\"code\":\"invalid_api_key\"}}\n\n", 401, false, "Incorrect API key"},
		{"type only", "data: {\"error\":{\"message\":\"slow down\",\"type\":\"rate_limit_error\"}}\n\n", 429, true, "slow down"},
		{"unknown shape", "data: {\"error\":{\"message\":\"boom\"}}\n\n", 500, true, "boom"},
		{"code out of range", "data: {\"error\":{\"code\":42,\"message\":\"odd\"}}\n\n", 500, true, "odd"},
		{"empty error object", "data: {\"error\":{}}\n\n", 500, true, "provider reported an error"},
	}
	for _, tc := range cases {
		c := &Client{providerName: "openrouter"}
		evs, errs := streamAll(c, tc.sse)
		if len(errs) != 1 {
			t.Fatalf("%s: errors = %v, want exactly one", tc.name, errs)
		}
		for _, ev := range evs {
			if ev.Type == llm.EventDone {
				t.Fatalf("%s: a failed turn yielded EventDone (stop %q)", tc.name, ev.StopReason)
			}
		}
		var apiErr *llm.APIError
		if !errors.As(errs[0], &apiErr) || apiErr.StatusCode != tc.status {
			t.Fatalf("%s: err = %#v, want *APIError with status %d", tc.name, errs[0], tc.status)
		}
		if llm.IsRetryable(errs[0]) != tc.retryable {
			t.Errorf("%s: IsRetryable = %v, want %v", tc.name, llm.IsRetryable(errs[0]), tc.retryable)
		}
		if !strings.Contains(errs[0].Error(), tc.message) {
			t.Errorf("%s: err %q lost the provider's message %q", tc.name, errs[0], tc.message)
		}
	}
}

// finish_reason "error" without an error object is still not a completion.
func TestFinishReasonErrorIsNotACompletion(t *testing.T) {
	c := &Client{providerName: "openrouter"}
	sse := "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"}}]}\n\n" +
		"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"error\"}]}\n\n" +
		"data: [DONE]\n\n"
	evs, errs := streamAll(c, sse)
	if len(errs) != 1 || !llm.IsRetryable(errs[0]) {
		t.Fatalf("errors = %v, want one retryable server error", errs)
	}
	for _, ev := range evs {
		if ev.Type == llm.EventDone {
			t.Fatalf("finish_reason error yielded EventDone")
		}
	}
}

// The provider may echo the key in its error text; it must not reach the caller.
func TestStreamErrorChunkRedactsTheKey(t *testing.T) {
	c := &Client{providerName: "openrouter", config: llm.Config{APIKey: "sk-or-secret-123456789"}}
	_, errs := streamAll(c, "data: {\"error\":{\"code\":401,\"message\":\"bad key sk-or-secret-123456789\"}}\n\n")
	if len(errs) != 1 || strings.Contains(errs[0].Error(), "sk-or-secret-123456789") {
		t.Fatalf("errors = %v, want one error without the key", errs)
	}
}

// Complete: some servers answer 200 with only an error object. Before B2 that
// decoded as an empty response with no error.
func TestCompleteErrorBodyWithStatus200(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"error":{"code":503,"message":"upstream unavailable"}}`))
	}))
	defer srv.Close()
	c, err := New(llm.Config{Provider: "openrouter", Model: "m", APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Complete(context.Background(), llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "x")}})
	var apiErr *llm.APIError
	if !errors.As(err, &apiErr) || apiErr.StatusCode != 503 || resp != nil {
		t.Fatalf("resp=%v err=%v, want a 503 APIError", resp, err)
	}
}

// A normal response that happens to carry an empty error field is unaffected.
func TestCompleteChoicesWinOverAnEmptyErrorField(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"error":null}`))
	}))
	defer srv.Close()
	c, err := New(llm.Config{Provider: "openrouter", Model: "m", APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Complete(context.Background(), llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "x")}})
	if err != nil || resp.Content != "hi" {
		t.Fatalf("resp=%+v err=%v", resp, err)
	}
}
