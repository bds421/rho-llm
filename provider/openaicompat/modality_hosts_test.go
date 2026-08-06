package openaicompat_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider/openaicompat"
)

// Proves Mistral and DashScope/Qwen embeddings ride the shared openai_compat
// modality driver (POST /embeddings) with reviewed registry capabilities —
// the real path for "Chinese + Mistral more support" beyond chat presets.
func TestOpenAICompatEmbeddingsForMistralAndDashScope(t *testing.T) {
	var sawPath string
	var sawAuth string
	var sawBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawPath = r.URL.Path
		sawAuth = r.Header.Get("Authorization")
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &sawBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[0.1,0.2,0.3]}],"usage":{"prompt_tokens":2}}`))
	}))
	defer srv.Close()

	cases := []struct {
		provider string
		model    string
	}{
		{"mistral", "mistral-embed"},
		{"dashscope", "text-embedding-v3"},
		{"dashscope-cn", "text-embedding-v3"},
	}
	for _, tc := range cases {
		t.Run(tc.provider, func(t *testing.T) {
			sawPath, sawAuth, sawBody = "", "", nil
			client, err := llm.NewModalityClient(llm.Config{
				Provider: tc.provider, Model: tc.model, APIKey: "secret-key",
				BaseURL: srv.URL, DisableProxy: true, DisableRetries: true,
				Timeout: 5 * time.Second,
			})
			if err != nil {
				t.Fatalf("NewModalityClient: %v", err)
			}
			defer client.Close()
			out, err := client.GenerateEmbeddings(context.Background(), llm.EmbeddingRequest{
				Model: tc.model, Input: []string{"hello rho"},
			})
			if err != nil {
				t.Fatalf("GenerateEmbeddings: %v", err)
			}
			if len(out.Embeddings) != 1 || len(out.Embeddings[0].Vector) != 3 {
				t.Fatalf("out=%+v", out)
			}
			if !strings.HasSuffix(sawPath, "/embeddings") {
				t.Fatalf("path=%q want .../embeddings", sawPath)
			}
			if sawAuth != "Bearer secret-key" {
				t.Fatalf("auth=%q", sawAuth)
			}
			if sawBody["model"] != tc.model {
				t.Fatalf("body model=%v", sawBody["model"])
			}
		})
	}
}

func TestOpenAICompatEmbeddingRejectsChatOnlyDeepSeek(t *testing.T) {
	// DeepSeek has no first-party embeddings model in the registry — fail closed.
	err := llm.ValidateEmbeddingRequest(llm.Config{
		Provider: "deepseek", Model: "deepseek-chat",
	}, llm.EmbeddingRequest{Model: "deepseek-chat", Input: []string{"x"}})
	if err == nil || !strings.Contains(err.Error(), "embeddings") {
		t.Fatalf("want embeddings rejected for deepseek-chat, got %v", err)
	}
}

func TestRegionalChinesePresetsResolve(t *testing.T) {
	for _, p := range []string{"dashscope-cn", "qwen-cn", "moonshot-cn", "kimi-cn"} {
		preset, ok := llm.PresetFor(p)
		if !ok || preset.Protocol != "openai_compat" || preset.BaseURL == "" {
			t.Fatalf("preset %q missing or incomplete: %+v ok=%v", p, preset, ok)
		}
		if llm.GetDefaultModel(p) == "" {
			t.Fatalf("no default model for %q", p)
		}
	}
	cn, _ := llm.PresetFor("dashscope-cn")
	intl, _ := llm.PresetFor("dashscope")
	if cn.BaseURL == intl.BaseURL {
		t.Fatal("dashscope-cn must use mainland base, not intl")
	}
	if !strings.Contains(cn.BaseURL, "dashscope.aliyuncs.com") {
		t.Fatalf("cn base=%q", cn.BaseURL)
	}
}
