package llm_test

import (
	"math"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// Kimi K3 fixes temperature, top_p, n and the penalties ("omit them from
// requests", platform.kimi.ai K3.1 Pro quickstart, 2026-11-10). An explicit
// temperature must fail before the wire, through aliases and every
// moonshot/kimi preset; the K2 rows still accept one.
func TestKimiK3RejectsTemperature(t *testing.T) {
	for _, temperature := range []float64{0, 0.2, 1.0, 2} {
		temp := temperature
		for _, provider := range []string{"moonshot", "kimi", "moonshot-cn", "kimi-cn"} {
			for _, model := range []string{"kimi-k3.1-pro", "kimi-k3", "kimi", "kimi-k3.1"} {
				cfg := llm.Config{Provider: provider, Model: model}
				if err := llm.ValidateRequestCapabilities(cfg, llm.Request{Temperature: &temp}, false); err == nil {
					t.Errorf("%s/%s temperature %v: want an error, got nil", provider, model, temp)
				}
				if err := llm.ValidateRequestCapabilities(cfg, llm.Request{}, true); err != nil {
					t.Errorf("%s/%s without temperature: %v", provider, model, err)
				}
			}
		}
		cfg := llm.Config{Provider: "moonshot", Model: "kimi-k2.7-code"}
		if err := llm.ValidateRequestCapabilities(cfg, llm.Request{Temperature: &temp}, false); err != nil {
			t.Errorf("kimi-k2.7-code temperature %v is accepted by the API, got %v", temp, err)
		}
	}
}

// Moonshot retired kimi-k2.5 and moonshot-v1 in August 2026 (404 "model not
// found"); DeepInfra deprecated Meta-Llama-3.1-8B-Instruct on 2026-07-16.
func TestRetiredOn20261110(t *testing.T) {
	cases := map[string]string{
		"kimi-k2.5":                             "kimi-k3.1-pro",
		"moonshot-v1-128k":                      "kimi-k3.1-pro",
		"meta-llama/Meta-Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
	}
	for id, want := range cases {
		got, ok := llm.RetiredModelReplacement(id)
		if !ok || got != want {
			t.Errorf("RetiredModelReplacement(%q) = %q, %v; want %q", id, got, ok, want)
		}
		if _, listed := llm.GetModelInfo(id); listed {
			t.Errorf("%q is retired but still has a registry row", id)
		}
		if _, ok := llm.GetModelInfo(want); !ok {
			t.Errorf("replacement %q for %q is not registered", want, id)
		}
	}
	for _, provider := range []string{"moonshot", "moonshot-cn", "deepinfra"} {
		for _, id := range llm.GetAvailableModels(provider) {
			if _, retired := cases[id]; retired {
				t.Errorf("GetAvailableModels(%s) still lists retired %q", provider, id)
			}
		}
	}
	for _, provider := range []string{"moonshot", "kimi", "moonshot-cn"} {
		_, err := llm.ResolveCapabilityProfile(llm.Config{Provider: provider, Model: "kimi-k2.5"})
		if err == nil || !strings.Contains(err.Error(), "retired") || !strings.Contains(err.Error(), "kimi-k3.1-pro") {
			t.Errorf("%s/kimi-k2.5: err = %v, want the retired error naming kimi-k3.1-pro", provider, err)
		}
	}
}

// docs.together.ai now documents api.together.ai; api.together.xyz still
// answers but is no longer named anywhere in the docs.
func TestTogetherPresetUsesDocumentedEndpoint(t *testing.T) {
	if got := llm.ResolveBaseURL(llm.Config{Provider: "together"}); got != "https://api.together.ai/v1" {
		t.Errorf("ResolveBaseURL(together) = %q, want https://api.together.ai/v1", got)
	}
	us := "https://us-api.together.ai/v1"
	if got := llm.ResolveBaseURL(llm.Config{Provider: "together", BaseURL: us}); got != us {
		t.Errorf("explicit BaseURL overridden: %q", got)
	}
}

// Prices changed or corrected on 2026-11-10, each from the provider's page.
func TestPricesRefreshed20261110(t *testing.T) {
	const M = 1_000_000
	cases := []struct {
		model                          string
		in, out, cacheRead, cacheWrite float64
	}{
		{"kimi-k3.1-pro", 2.00, 10.00, 0.20, 2.00},
		{"kimi-k3", 3.00, 15.00, 0.30, 3.00},
		{"kimi-k2.7-code", 0.95, 4.00, 0.19, 0},
		{"mistral-medium-latest", 1.50, 7.50, 0, 0},
		{"deepseek-ai/DeepSeek-V4-Flash", 0.09, 0.18, 0.018, 0},
		{"command-a-plus-05-2026", 2.50, 10.00, 0, 0},
		{"muse-spark-1.3", 1.25, 4.25, 0.15, 0},
		{"muse-ember-1.0", 0.10, 0.50, 0.004, 0},
		{"qwen/qwen3.8-27b", 0.80, 4.00, 0, 0},
	}
	for _, c := range cases {
		got := llm.EstimateCost(llm.CostInput{Model: c.model, InputTokens: M, OutputTokens: M, CacheReadTokens: M, CacheCreateTokens: M})
		// A row without a cache-read price bills cached tokens at the input rate.
		cacheRead := c.cacheRead
		if cacheRead == 0 {
			cacheRead = c.in
		}
		want := c.in + c.out + cacheRead + c.cacheWrite
		if math.Abs(got-want) > 1e-9 {
			t.Errorf("%s: cost = %v, want %v", c.model, got, want)
		}
	}
	for model, ctx := range map[string]int{
		"deepseek-ai/DeepSeek-V4-Flash": 1048576,
		"codestral-2508":                131072,
		"codestral-latest":              131072,
		"kimi-k3.1-pro":                 1048576,
	} {
		info, ok := llm.GetModelInfo(model)
		if !ok || info.ContextWindow != ctx {
			t.Errorf("%s: ContextWindow = %d, want %d", model, info.ContextWindow, ctx)
		}
	}
}
