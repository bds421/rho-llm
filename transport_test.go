package llm_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

func TestErrorFromResponseClassifiesAndBounds(t *testing.T) {
	// 429 → rate-limit APIError.
	resp := &http.Response{StatusCode: 429, Body: io.NopCloser(strings.NewReader("slow down"))}
	if err := llm.ErrorFromResponse("test", resp, llm.Config{}); !llm.IsRateLimited(err) {
		t.Errorf("429 → %v, want rate-limit error", err)
	}

	// Oversized error body must be bounded.
	big := strings.Repeat("E", 5*1024*1024)
	resp2 := &http.Response{StatusCode: 500, Body: io.NopCloser(strings.NewReader(big))}
	err := llm.ErrorFromResponse("test", resp2, llm.Config{})
	var apiErr *llm.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %T", err)
	}
	if len(apiErr.Message) > 1024*1024+4096 {
		t.Errorf("error message not bounded: %d bytes", len(apiErr.Message))
	}
}

func TestDecodeJSONResponseBounded(t *testing.T) {
	resp := &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(`{"x":7}`))}
	var out struct {
		X int `json:"x"`
	}
	if err := llm.DecodeJSONResponse(resp, llm.Config{}, &out); err != nil || out.X != 7 {
		t.Errorf("decode = %v, x=%d", err, out.X)
	}

	// A JSON body far over the response limit must be truncated → decode error (not OOM).
	cfg := llm.Config{MaxResponseBodyBytes: 64}
	huge := `{"x":"` + strings.Repeat("A", 10000) + `"}`
	resp2 := &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(huge))}
	var out2 struct {
		X string `json:"x"`
	}
	if err := llm.DecodeJSONResponse(resp2, cfg, &out2); err == nil {
		t.Error("oversized body should fail to decode under a small limit")
	}
}

func TestNewJSONRequest(t *testing.T) {
	req, err := llm.NewJSONRequest(context.Background(), "https://example.com/v1/x", []byte(`{"a":1}`))
	if err != nil {
		t.Fatal(err)
	}
	if req.Method != http.MethodPost {
		t.Errorf("method = %q", req.Method)
	}
	if req.Header.Get("Content-Type") != "application/json" {
		t.Errorf("content-type = %q", req.Header.Get("Content-Type"))
	}
	body, _ := io.ReadAll(req.Body)
	if string(body) != `{"a":1}` {
		t.Errorf("body = %q", body)
	}
}

// The space after "data:" is optional in the SSE spec. Lines without it were
// skipped, so a spec-valid stream from a gateway that omits the space ended
// as a retryable "unexpected EOF" in every adapter (B5).
func TestSSEDataAcceptsBothSpacings(t *testing.T) {
	cases := []struct {
		line string
		data string
		ok   bool
	}{
		{`data: {"a":1}`, `{"a":1}`, true},
		{`data:{"a":1}`, `{"a":1}`, true},
		{`data:  {"a":1}`, ` {"a":1}`, true}, // only one space is part of the syntax
		{`data: [DONE]`, `[DONE]`, true},
		{`data:[DONE]`, `[DONE]`, true},
		{`data: `, ``, false}, // keep-alive
		{`data:`, ``, false},
		{`event: message_start`, ``, false},
		{`: comment`, ``, false},
		{``, ``, false},
		{`datax: {}`, ``, false},
		{` data: {}`, ``, false},
	}
	for _, tc := range cases {
		data, ok := llm.SSEData(tc.line)
		if data != tc.data || ok != tc.ok {
			t.Errorf("SSEData(%q) = %q, %v; want %q, %v", tc.line, data, ok, tc.data, tc.ok)
		}
	}
}
