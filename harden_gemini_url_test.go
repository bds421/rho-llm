package llm_test

// Hardening pass 10 — gemini URL construction must escape the model name.

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/provider/gemini"
)

// The model is interpolated into the request URL path; a model containing URL
// metacharacters must be escaped, not injected (e.g. "m?x=1" must not add a
// query parameter / change the endpoint).
func TestGeminiEscapesModelInURL(t *testing.T) {
	var rawQuery, path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rawQuery, path = r.URL.RawQuery, r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"ok"}],"role":"model"}}],"usageMetadata":{}}`))
	}))
	defer srv.Close()

	c, err := gemini.New(llm.Config{Provider: "gemini", Model: "m?injected=1", APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, _ = c.Complete(context.Background(), llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")}})

	if strings.Contains(rawQuery, "injected=1") {
		t.Fatalf("model name injected a query parameter (unescaped): rawQuery=%q path=%q", rawQuery, path)
	}
}
