package llm_test

import (
	"testing"

	llm "github.com/bds421/rho-llm"
)

func TestCatalogDefaultsAndDiscoveryForPhaseCProviders(t *testing.T) {
	cases := []struct {
		provider string
		wantDef  string
		sample   string
		caps     []llm.Capability
	}{
		{"mistral", "mistral-small-2603", "pixtral-large-2411", []llm.Capability{llm.CapabilityChat, llm.CapabilityVision}},
		{"zai", "glm-5.2", "glm-4.7", []llm.Capability{llm.CapabilityChat, llm.CapabilityVision}},
		{"minimax", "MiniMax-M3", "MiniMax-M2.5", []llm.Capability{llm.CapabilityChat}},
		{"moonshot", "kimi-k2.7-code", "kimi-k2.5", []llm.Capability{llm.CapabilityChat, llm.CapabilityVision}},
		{"dashscope", "qwen3.6-plus", "qwen3.6-flash", []llm.Capability{llm.CapabilityChat}},
		{"deepseek", "deepseek-chat", "deepseek-reasoner", []llm.Capability{llm.CapabilityChat}},
		{"meta", "muse-spark-1.2", "muse-spark-1.1", []llm.Capability{llm.CapabilityChat, llm.CapabilityVision, llm.CapabilityDocumentInput}},
	}
	for _, tc := range cases {
		if got := llm.GetDefaultModel(tc.provider); got != tc.wantDef {
			t.Errorf("GetDefaultModel(%s)=%q want %q", tc.provider, got, tc.wantDef)
		}
		models := llm.GetAvailableModels(tc.provider)
		found := false
		for _, m := range models {
			if m == tc.sample {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("GetAvailableModels(%s) missing %q: %v", tc.provider, tc.sample, models)
		}
		cfg := llm.Config{Provider: tc.provider, Model: tc.sample}
		if err := llm.RequireCapabilities(cfg, tc.caps...); err != nil {
			t.Errorf("%s/%s capabilities: %v", tc.provider, tc.sample, err)
		}
	}
	// Unlisted still fail closed.
	if err := llm.RequireCapabilities(llm.Config{Provider: "mistral", Model: "totally-unknown-mistral"}, llm.CapabilityChat); err == nil {
		t.Fatal("unknown model must fail closed")
	}
}

func TestCatalogAliasesPhaseC(t *testing.T) {
	if got := llm.ResolveModelAlias("pixtral"); got != "pixtral-large-2411" {
		t.Fatalf("pixtral alias = %q", got)
	}
	if got := llm.ResolveModelAlias("muse"); got != "muse-spark-1.2" {
		t.Fatalf("muse alias = %q", got)
	}
	if got := llm.ResolveModelAlias("glm4.7"); got != "glm-4.7" {
		t.Fatalf("glm4.7 alias = %q", got)
	}
}

func TestChineseRegionalPresetsAndEmbeddingModels(t *testing.T) {
	for _, p := range []string{"dashscope-cn", "moonshot-cn"} {
		if llm.GetDefaultModel(p) == "" {
			t.Fatalf("default for %s empty", p)
		}
		if len(llm.GetAvailableModels(p)) == 0 {
			t.Fatalf("available models for %s empty", p)
		}
	}
	// Cross-region capability: dashscope-cn may use dashscope-registered embedding models.
	cfg := llm.Config{Provider: "dashscope-cn", Model: "text-embedding-v3"}
	if err := llm.RequireCapabilities(cfg, llm.CapabilityEmbeddings); err != nil {
		t.Fatalf("dashscope-cn embeddings: %v", err)
	}
	cfg = llm.Config{Provider: "mistral", Model: "mistral-embed"}
	if err := llm.RequireCapabilities(cfg, llm.CapabilityEmbeddings); err != nil {
		t.Fatalf("mistral-embed: %v", err)
	}
	// DeepSeek chat is not an embeddings model.
	if err := llm.RequireCapabilities(llm.Config{Provider: "deepseek", Model: "deepseek-chat"}, llm.CapabilityEmbeddings); err == nil {
		t.Fatal("deepseek-chat must not claim embeddings")
	}
}
