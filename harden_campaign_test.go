package llm_test

// Hardening pass 16 — break-3-rounds campaign findings on the new provider wiring.
// Each test encodes a confirmed defect (no happy path) and was written to FAIL
// before its fix landed.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
)

// A per-request Model override that is an ALIAS (or mixed-case variant) must be
// resolved to its canonical registry ID before it hits the provider wire — the
// provider only recognizes the exact ID (MiniMax wants "MiniMax-M3", not
// "minimax"/"minimax-m3"). factory.go resolves cfg.Model at construction, but a
// Request{Model: alias} override bypassed that and shipped the alias verbatim.
func TestPerRequestAliasResolvesToCanonicalWireModel(t *testing.T) {
	cases := []struct{ provider, reqModel, wantWire string }{
		{"zai", "glm", "glm-5.2"},
		{"minimax", "minimax-m3", "MiniMax-M3"}, // lowercase variant of the mixed-case key
		{"moonshot", "kimi", "kimi-k2.7-code"},
	}
	for _, tc := range cases {
		t.Run(tc.provider+"/"+tc.reqModel, func(t *testing.T) {
			var gotModel string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				var req struct {
					Model string `json:"model"`
				}
				_ = json.Unmarshal(body, &req)
				gotModel = req.Model
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprint(w, `{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
			}))
			defer srv.Close()

			client, err := llm.NewClient(llm.Config{Provider: tc.provider, Model: tc.wantWire, APIKey: "sk-test", BaseURL: srv.URL})
			if err != nil {
				t.Fatalf("NewClient(%s): %v", tc.provider, err)
			}
			defer client.Close()

			_, _ = client.Complete(context.Background(), llm.Request{
				Model:    tc.reqModel, // per-request alias/variant override
				Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
			})
			if gotModel != tc.wantWire {
				t.Fatalf("wire model = %q, want canonical %q — per-request alias not resolved before the wire", gotModel, tc.wantWire)
			}
		})
	}
}

// A per-key "apikey|baseurl" override can embed credentials in the BaseURL
// (userinfo user:pass@ or a ?token= secret). When an upstream error echoes that
// BaseURL, MarkFailed stores it in LastError and MarshalJSON serializes it — both
// must strip the embedded secrets (AuthProfile.MarshalJSON already redacts the
// BaseURL *field*, so leaving them in LastError is an inconsistency/leak).
func TestAuthProfileDoesNotLeakBaseURLCredsInLastError(t *testing.T) {
	const apiKey = "sk-minimax-abcdef123456"
	const baseURL = "https://user:superSecretPw@proxy.internal:9000/v1?token=bearer_xyz987"
	p := llm.AuthProfile{Name: "minimax-1", APIKey: apiKey, BaseURL: baseURL}

	// Upstream error that echoes the full per-key BaseURL + the API key.
	p.MarkFailed(fmt.Errorf("403 from %s using key %s", baseURL, apiKey), time.Second)

	leaks := []string{"superSecretPw", "bearer_xyz987", apiKey}
	for _, secret := range leaks {
		if strings.Contains(p.LastError, secret) {
			t.Errorf("MarkFailed left secret %q in LastError: %q", secret, p.LastError)
		}
	}
	// Host must remain visible for debugging.
	if !strings.Contains(p.LastError, "proxy.internal") {
		t.Errorf("LastError dropped the host (needed for debugging): %q", p.LastError)
	}

	b, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	s := string(b)
	for _, secret := range leaks {
		if strings.Contains(s, secret) {
			t.Errorf("AuthProfile JSON leaked secret %q: %s", secret, s)
		}
	}
}

// EstimateCost must NOT silently normalize a whitespace-padded model ID to the
// trimmed registry key — a padded ID is an unknown model and must cost 0, never
// quietly resolve to the real price (which would mask a caller bug, or worse let
// a padded ID dodge a price lookup). Pins the no-silent-trim contract.
func TestEstimateCostDoesNotSilentlyNormalizeWhitespace(t *testing.T) {
	const M = 1_000_000
	if base := llm.EstimateCost(llm.CostInput{Model: "glm-5.2", InputTokens: M}); base <= 0 {
		t.Fatalf("precondition: glm-5.2 must have a price, got %v", base)
	}
	for _, padded := range []string{" glm-5.2", "glm-5.2 ", " glm-5.2 ", "\tglm-5.2"} {
		if c := llm.EstimateCost(llm.CostInput{Model: padded, InputTokens: M, OutputTokens: M}); c != 0 {
			t.Errorf("EstimateCost(%q) = %v, want 0 — padded ID must not silently resolve to the trimmed model", padded, c)
		}
	}
}
