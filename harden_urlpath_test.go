package llm_test

// Hardening pass 11 — confirm the fixed-path adapters can't be URL-injected via
// the model name (only Gemini puts the model in the path; the others use a fixed
// endpoint). No defect expected — this pins the safe-by-construction property.

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/provider/anthropic"
	"github.com/bds421/rho-llm/provider/openaicompat"
	"github.com/bds421/rho-llm/provider/openairesponses"
)

func TestFixedPathAdaptersIgnoreHostileModelInURL(t *testing.T) {
	const hostile = "m?inject=1/../../evil"

	cases := []struct {
		name     string
		wantPath string
		newC     func(cfg llm.Config) (llm.Client, error)
		resp     string
	}{
		{"anthropic", "/messages", func(c llm.Config) (llm.Client, error) { return anthropic.New(c) },
			`{"id":"m","model":"x","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`},
		{"openaicompat", "/chat/completions", func(c llm.Config) (llm.Client, error) { return openaicompat.New(c) },
			`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{}}`},
		{"openairesponses", "/responses", func(c llm.Config) (llm.Client, error) { return openairesponses.New(c) },
			`{"id":"r","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{}}`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var gotPath, gotQuery string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotPath, gotQuery = r.URL.Path, r.URL.RawQuery
				w.Header().Set("Content-Type", "application/json")
				w.Write([]byte(tc.resp))
			}))
			defer srv.Close()

			c, err := tc.newC(llm.Config{Provider: tc.name, Model: hostile, APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			_, _ = c.Complete(context.Background(), llm.Request{MaxTokens: 16, Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")}})

			if gotPath != tc.wantPath {
				t.Fatalf("hostile model altered the request path: got %q, want %q", gotPath, tc.wantPath)
			}
			if gotQuery != "" {
				t.Fatalf("hostile model injected a query: %q", gotQuery)
			}
		})
	}
}
