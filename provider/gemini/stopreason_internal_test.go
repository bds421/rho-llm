package gemini

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

func streamEvents(c *Client, sse string) (evs []llm.StreamEvent, errs []error) {
	c.parseStream(strings.NewReader(sse), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			errs = append(errs, err)
		} else {
			evs = append(evs, ev)
		}
		return true
	})
	return
}

const blocked = `{"promptFeedback":{"blockReason":"PROHIBITED_CONTENT"},"usageMetadata":{"promptTokenCount":7,"totalTokenCount":7}}`

// A blocked prompt has no candidates, only promptFeedback.blockReason. Complete
// returned an empty response with no stop reason, and Stream ran to EOF and
// surfaced a retryable truncation, so the blocked prompt was sent again (B3).
func TestBlockedPromptIsARefusal(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gemini-3.8-flash"}, providerName: "gemini"}

	var api geminiResponse
	if err := json.Unmarshal([]byte(blocked), &api); err != nil {
		t.Fatal(err)
	}
	resp := c.parseResponse(&api, "gemini-3.8-flash")
	if resp.StopReason != llm.StopRefusal || resp.RawStopReason != "PROHIBITED_CONTENT" || resp.InputTokens != 7 {
		t.Fatalf("Complete: stop=%q raw=%q in=%d, want refusal / PROHIBITED_CONTENT / 7", resp.StopReason, resp.RawStopReason, resp.InputTokens)
	}

	evs, errs := streamEvents(c, "data: "+blocked+"\n\n")
	if len(errs) != 0 || len(evs) != 1 || evs[0].Type != llm.EventDone {
		t.Fatalf("Stream: events=%+v errs=%v, want one Done", evs, errs)
	}
	if evs[0].StopReason != llm.StopRefusal || evs[0].RawStopReason != "PROHIBITED_CONTENT" || evs[0].InputTokens != 7 {
		t.Fatalf("Stream Done = %+v", evs[0])
	}
}

// An error chunk after the 200 became "stream ended without completion"; the
// status and message were lost and IsRateLimited was false (B3).
func TestStreamErrorChunkIsAnAPIError(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gemini-3.8-flash", APIKey: "AIza-secret-key-123"}, providerName: "gemini"}
	for _, tc := range []struct {
		chunk     string
		status    int
		retryable bool
	}{
		{`{"error":{"code":429,"message":"Resource has been exhausted","status":"RESOURCE_EXHAUSTED"}}`, 429, true},
		{`{"error":{"code":400,"message":"Request contains an invalid argument. key AIza-secret-key-123","status":"INVALID_ARGUMENT"}}`, 400, false},
		{`{"error":{"code":503,"message":"The model is overloaded","status":"UNAVAILABLE"}}`, 503, true},
		{`{"error":{"message":"no code"}}`, 500, true},
	} {
		sse := `data: {"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}` + "\n\n" + "data: " + tc.chunk + "\n\n"
		evs, errs := streamEvents(c, sse)
		if len(errs) != 1 {
			t.Fatalf("%s: errs = %v, want one", tc.chunk, errs)
		}
		var apiErr *llm.APIError
		if !errors.As(errs[0], &apiErr) || apiErr.StatusCode != tc.status || llm.IsRetryable(errs[0]) != tc.retryable {
			t.Fatalf("%s: err = %v, want status %d retryable=%v", tc.chunk, errs[0], tc.status, tc.retryable)
		}
		if strings.Contains(errs[0].Error(), "AIza-secret-key-123") {
			t.Fatalf("key leaked into the error: %v", errs[0])
		}
		for _, ev := range evs {
			if ev.Type == llm.EventDone {
				t.Fatalf("%s: failed stream yielded Done", tc.chunk)
			}
		}
	}
}

// Complete: a 200 body with only an error object is that error.
func TestCompleteErrorBodyIsAnAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"error":{"code":503,"message":"overloaded","status":"UNAVAILABLE"}}`))
	}))
	defer srv.Close()
	c, err := New(llm.Config{Provider: "gemini", Model: "gemini-3.8-flash", APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Complete(context.Background(), llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "x")}})
	var apiErr *llm.APIError
	if resp != nil || !errors.As(err, &apiErr) || apiErr.StatusCode != 503 {
		t.Fatalf("resp=%v err=%v, want a 503 APIError", resp, err)
	}
}

// A failed turn is not a normal stop: MALFORMED_FUNCTION_CALL passed through as
// its raw string with err=nil, so a tool loop ended on it as if it completed.
func TestFailedFinishReasonsAreStopError(t *testing.T) {
	for raw, want := range map[string]string{
		"STOP":                      llm.StopEndTurn,
		"MAX_TOKENS":                llm.StopMaxTokens,
		"SAFETY":                    llm.StopRefusal,
		"RECITATION":                llm.StopRefusal,
		"BLOCKLIST":                 llm.StopRefusal,
		"SPII":                      llm.StopRefusal,
		"IMAGE_SAFETY":              llm.StopRefusal,
		"MALFORMED_FUNCTION_CALL":   llm.StopError,
		"UNEXPECTED_TOOL_CALL":      llm.StopError,
		"TOO_MANY_TOOL_CALLS":       llm.StopError,
		"OTHER":                     llm.StopError,
		"FINISH_REASON_UNSPECIFIED": llm.StopError,
		"SOMETHING_NEW":             llm.StopError,
	} {
		if got := normalizeStopReason(raw); got != want {
			t.Errorf("normalizeStopReason(%s) = %q, want %q", raw, got, want)
		}
	}
	c := &Client{config: llm.Config{Model: "gemini-3.8-flash"}, providerName: "gemini"}
	evs, errs := streamEvents(c, `data: {"candidates":[{"content":{"parts":[]},"finishReason":"MALFORMED_FUNCTION_CALL"}]}`+"\n\n")
	if len(errs) != 0 || len(evs) != 1 || evs[0].StopReason != llm.StopError || evs[0].RawStopReason != "MALFORMED_FUNCTION_CALL" {
		t.Fatalf("events=%+v errs=%v", evs, errs)
	}
}
