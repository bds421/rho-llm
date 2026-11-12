package llm_test

import (
	"context"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider"
	_ "github.com/bds421/rho-llm/provider/gemini"
	_ "github.com/bds421/rho-llm/provider/openaicompat"
)

// Attack: the retirement error must not leak into unrelated failures, and a
// retired ID must never silently dispatch to the network.
func TestBreakRetiredModelNeverDispatches(t *testing.T) {
	c, err := llm.NewClient(llm.Config{
		Provider: "anthropic",
		Model:    "claude-sonnet-4-20250514",
		APIKey:   "k",
		BaseURL:  "http://127.0.0.1:1", // any dial here means validation was bypassed
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer c.Close()

	_, err = c.Complete(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "x")},
	})
	if err == nil {
		t.Fatal("retired model dispatched successfully")
	}
	if !strings.Contains(err.Error(), "retired") {
		t.Errorf("error = %q; want the retirement error, not a transport error — "+
			"the caller must learn the model is gone, not that the network failed", err.Error())
	}
}

// Attack: RetiredModelReplacement under concurrent load. The map is read by
// every dispatch; a data race here corrupts every request.
func TestBreakRetiredModelReplacementConcurrent(t *testing.T) {
	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 2000; i++ {
				if r, ok := llm.RetiredModelReplacement("claude-sonnet-4-20250514"); !ok || r == "" {
					t.Error("retired lookup returned empty under concurrency")
					return
				}
				if _, ok := llm.RetiredModelReplacement("claude-sonnet-4-6"); ok {
					t.Error("live model reported as retired under concurrency")
					return
				}
				_, _ = llm.RetiredModelReplacement("")
			}
		}()
	}
	wg.Wait()
}

// Attack: RegisterModel must win over a retirement entry — otherwise the
// documented escape hatch ("re-add it with RegisterModel") is a lie.
//
// This mutates the process-global registry and there is no UnregisterModel, so
// it deliberately uses the ID no other test asserts on; picking a shared one
// leaks into TestRetiredModelGivesActionableError.
func TestBreakRegisterModelOverridesRetirement(t *testing.T) {
	const id = "claude-3-haiku-20240307"
	if _, retired := llm.RetiredModelReplacement(id); !retired {
		t.Fatalf("%s is not a retired ID; this test no longer exercises the escape hatch", id)
	}
	if err := llm.RegisterModel(llm.ModelInfo{
		ID: id, Provider: "anthropic", MaxTokens: 4096, ContextWindow: 200000,
		Capabilities: llm.CapabilitySet(llm.CapabilityChat),
	}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}
	c, err := llm.NewClient(llm.Config{Provider: "anthropic", Model: id, APIKey: "k"})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer c.Close()
	_, err = c.Complete(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "x")},
	})
	if err != nil && strings.Contains(err.Error(), "retired") {
		t.Error("RegisterModel could not resurrect a retired ID: the documented escape hatch does not work")
	}
}

// Concurrent registration of the SAME model id plus readers: the override must
// not double-append to the provider's available-models list, and reads must be
// race-free.
func TestRegistryConcurrentSameIDNoDoubleAppend(t *testing.T) {
	const id = "break-dup-model"
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			_ = llm.RegisterModel(llm.ModelInfo{ID: id, Provider: "break-dup-prov", InputPricePer1M: float64(n)})
			_ = llm.ModelsByProvider("break-dup-prov")
			_, _ = llm.GetModelInfo(id)
		}(i)
	}
	wg.Wait()

	n := 0
	for _, m := range llm.ModelsByProvider("break-dup-prov") {
		if m.ID == id {
			n++
		}
	}
	if n != 1 {
		t.Fatalf("model %q appears %d times in ModelsByProvider — concurrent override double-registered", id, n)
	}
}

// DefaultConfig's model must equal the registry's Anthropic default. The bug
// this guards: config.go hardcodes a model ID that silently drifts from
// defaultModels every time the flagship moves — which is exactly what had
// happened (DefaultConfig said claude-sonnet-4-6 long after the registry
// default moved on).
func TestDefaultConfigTracksRegistryDefault(t *testing.T) {
	got := llm.DefaultConfig().Model
	want := llm.GetDefaultModel("anthropic")
	if got != want {
		t.Errorf("DefaultConfig().Model = %q, registry default = %q — the two have drifted apart", got, want)
	}
	if _, ok := llm.GetModelInfo(got); !ok {
		t.Errorf("DefaultConfig().Model = %q is not in the registry at all", got)
	}
}

// Two published prices were wrong by large factors on their own provider's
// DEFAULT model, which is the worst place for a pricing error to hide: every
// cost estimate for the out-of-the-box model is wrong, and nothing fails.
// Verified 2026-09-17 against platform.claude.com/docs pricing and
// console.groq.com/docs/models.
func TestDefaultModelPricesMatchPublishedRates(t *testing.T) {
	cases := []struct {
		model           string
		wantIn, wantOut float64
		note            string
	}{
		// Sonnet 5's introductory $2/$10 became the standard price; the
		// scheduled 2026-09-01 rise to $3/$15 was cancelled. Registry had $3/$15.
		{"claude-sonnet-5", 2.00, 10.00, "Anthropic default — was 50% over"},
		// Groq publishes $0.15/$0.60. Registry had $3.00/$8.00 — 20x/13x over,
		// on the `groq` default model.
		{"openai/gpt-oss-120b", 0.15, 0.60, "Groq default — was 20x over"},
		{"openai/gpt-oss-20b", 0.075, 0.30, "was 4x over"},
	}
	for _, c := range cases {
		t.Run(c.model, func(t *testing.T) {
			info, ok := llm.GetModelInfo(c.model)
			if !ok {
				t.Fatalf("%s missing from the registry", c.model)
			}
			if info.InputPricePer1M != c.wantIn || info.OutputPricePer1M != c.wantOut {
				t.Errorf("%s = $%g/$%g, want $%g/$%g (%s)",
					c.model, info.InputPricePer1M, info.OutputPricePer1M, c.wantIn, c.wantOut, c.note)
			}
		})
	}
}

// TestResolveModelAlias verifies alias resolution.
func TestResolveModelAlias(t *testing.T) {
	tests := []struct {
		alias    string
		expected string
	}{
		{"opus", "claude-opus-5-5"},
		{"sonnet", "claude-sonnet-5-5"},
		{"fable", "claude-fable-5-2"},
		{"haiku", "claude-haiku-5-5"},
		{"grok", "grok-4.7"},
		{"gpt6.1", "gpt-6.1-sol"},
		{"grok4.7", "grok-4.7"},
		{"grok4.5", "grok-4.5"},
		{"grok4.3", "grok-4.3"},
		{"gpt5.6", "gpt-5.6-sol"},
		{"gpt5.5", "gpt-5.5"},
		{"muse", "muse-spark-1.3"},
		{"gemini3.5", "gemini-3.5-flash"},
		{"gemini3.6", "gemini-3.6-flash"},
		{"gemini3.5-lite", "gemini-3.5-flash-lite"},
		{"gemini3.1-lite", "gemini-3.1-flash-lite"},
		{"grok-code", "grok-build-0.1"},
		{"grok-4.20-beta", "grok-4.20-0309-reasoning"},
		{"mistral-large", "mistral-large-4-0"},
		{"gemini-pro", "gemini-3.1-pro-preview"},
		{"flash-lite", "gemini-3.5-flash-lite"},
		// Non-alias should pass through
		{"claude-opus-4-6", "claude-opus-4-6"},
		{"unknown-model", "unknown-model"},
	}

	for _, tc := range tests {
		t.Run(tc.alias, func(t *testing.T) {
			got := llm.ResolveModelAlias(tc.alias)
			if got != tc.expected {
				t.Errorf("ResolveModelAlias(%q) = %q, want %q", tc.alias, got, tc.expected)
			}
		})
	}
}

// TestProviderForModel verifies model-to-provider resolution.
func TestProviderForModel(t *testing.T) {
	tests := []struct {
		model    string
		expected string
	}{
		{"claude-sonnet-4-6", "anthropic"},
		{"grok-4.20-0309-non-reasoning", "xai"},
		{"gemini-2.5-flash", "gemini"},
		{"unknown-model", ""},
	}

	for _, tc := range tests {
		t.Run(tc.model, func(t *testing.T) {
			got := llm.ProviderForModel(tc.model)
			if got != tc.expected {
				t.Errorf("ProviderForModel(%q) = %q, want %q", tc.model, got, tc.expected)
			}
		})
	}
}

// TestModelInfoContextWindow verifies all Anthropic/Gemini models have ContextWindow set.
func TestModelInfoContextWindow(t *testing.T) {
	modelsToCheck := []string{
		"claude-fable-5", "claude-opus-5", "claude-sonnet-5",
		"claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001",
		"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
		"muse-spark-1.2", "muse-spark-1.1",
		"gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
		"grok-4.5", "grok-4.3", "grok-build-0.1",
	}

	for _, model := range modelsToCheck {
		info, ok := llm.GetModelInfo(model)
		if !ok {
			t.Errorf("Model %s not in registry", model)
			continue
		}
		if info.ContextWindow == 0 {
			t.Errorf("Model %s has ContextWindow=0", model)
		}
	}
}

// TestModelInfoThinkingFlags verifies reasoning flags are correctly set in the registry.
func TestModelInfoThinkingFlags(t *testing.T) {
	// 1. API-controlled (SupportsThinking)
	apiControlled := []string{
		"claude-opus-5", "claude-sonnet-5", "claude-fable-5",
		"claude-opus-4-6", "claude-sonnet-4-6", "claude-sonnet-4-5",
	}
	for _, model := range apiControlled {
		info, ok := llm.GetModelInfo(model)
		if !ok || !info.SupportsThinking || info.Thinking {
			t.Errorf("Model %s should support API-controlled thinking", model)
		}
	}

	// 2. Intrinsic reasoning (Thinking)
	intrinsic := []string{
		"grok-4.3", "grok-4.20-0309-reasoning", "grok-build-0.1",
	}
	for _, model := range intrinsic {
		info, ok := llm.GetModelInfo(model)
		if !ok || !info.Thinking || info.SupportsThinking {
			t.Errorf("Model %s should have intrinsic reasoning", model)
		}
	}

	// 3. Intrinsic thinking (Gemini 2.5 — API rejects thinkingConfig)
	gemini25 := []string{
		"gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite",
	}
	for _, model := range gemini25 {
		info, ok := llm.GetModelInfo(model)
		if !ok || !info.Thinking || info.SupportsThinking {
			t.Errorf("Model %s should have intrinsic reasoning (Thinking=true, SupportsThinking=false)", model)
		}
	}

	// 4. No reasoning
	none := []string{
		"gpt-4.1",
	}
	for _, model := range none {
		info, ok := llm.GetModelInfo(model)
		if !ok || info.Thinking || info.SupportsThinking {
			t.Errorf("Model %s should not have any reasoning flags", model)
		}
	}
}

// TestGetDefaultModelUnknown verifies fallback for unknown provider.
func TestGetDefaultModelUnknown(t *testing.T) {
	// Unknown providers fall back to the Anthropic default. It was a stale
	// hard-coded "claude-sonnet-5" (B33).
	model := llm.GetDefaultModel("nonexistent-provider")
	if model != llm.GetDefaultModel("anthropic") {
		t.Errorf("GetDefaultModel(unknown) = %q, want the Anthropic default", model)
	}
}

// A multi-vendor or local preset got a Claude model ID as its default, which
// that host rejects; openai_responses had none at all (B33).
func TestGetDefaultModelForPresetsWithoutOne(t *testing.T) {
	for _, p := range []string{"together", "openrouter", "fireworks", "deepinfra", "nvidia", "cerebras", "perplexity", "vllm", "lmstudio"} {
		if got := llm.GetDefaultModel(p); got != "" {
			t.Errorf("GetDefaultModel(%q) = %q, want \"\" (no default on a multi-vendor or local host)", p, got)
		}
	}
	if got, want := llm.GetDefaultModel("openai_responses"), llm.GetDefaultModel("openai"); got != want {
		t.Errorf("GetDefaultModel(openai_responses) = %q, want %q", got, want)
	}
}

// RegisterModel accepted an ID an alias shadows ("sonnet") and IDs with
// surrounding whitespace; neither could ever be looked up (B33).
func TestRegisterModelRefusesUnreachableIDs(t *testing.T) {
	for _, id := range []string{"sonnet", "opus", " my-model", "my-model\t"} {
		if err := llm.RegisterModel(llm.ModelInfo{ID: id, Provider: "anthropic"}); err == nil {
			t.Errorf("RegisterModel(%q) accepted", id)
		}
	}
	if err := llm.RegisterModelAlias(" x ", "claude-sonnet-5-5"); err == nil {
		t.Error("alias with whitespace accepted")
	}
}

// TestRetiredModelGivesActionableError verifies that a model ID the provider
// has retired fails with a message naming the retirement and a replacement,
// rather than the generic "no reviewed capability metadata" error.
//
// Registry lookup is fail-closed, so simply deleting a retired entry turns the
// provider's own clear 404 ("model: claude-sonnet-4-20250514") into a local
// error that gives the caller nothing to act on.
func TestRetiredModelGivesActionableError(t *testing.T) {
	for _, model := range []string{
		// xAI answers these with grok-4.3 / grok-build-0.1 (checked 2026-11-10).
		"grok-3", "grok-3-mini", "grok-4-0709", "grok-4-fast-reasoning",
		"grok-4-1-fast-non-reasoning", "grok-code-fast-1",
		"claude-sonnet-4-20250514",
		"claude-opus-4-20250514",
		"claude-opus-4-1-20250805",
		// claude-3-haiku-20240307 is exercised by
		// TestBreakRegisterModelOverridesRetirement, which re-registers it.
		"claude-sonnet-4-0",
		"claude-opus-4-0",
		"claude-opus-4-1",
	} {
		t.Run(model, func(t *testing.T) {
			if _, ok := llm.GetModelInfo(model); ok {
				t.Fatalf("GetModelInfo(%q) still registered; retired IDs must be removed", model)
			}
			// Capability validation is a dispatch-time gate (Complete/Stream),
			// not a construction-time one, so assert at that boundary.
			provider := "anthropic"
			if strings.HasPrefix(model, "grok-") {
				provider = "xai"
			}
			c, err := llm.NewClient(llm.Config{
				Provider: provider,
				Model:    model,
				APIKey:   "test-key-not-used",
				BaseURL:  "http://127.0.0.1:9", // never reached: retirement fails first
			})
			if err != nil {
				t.Fatalf("NewClient: %v", err)
			}
			defer c.Close()

			_, err = c.Complete(context.Background(), llm.Request{
				Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
			})
			if err == nil {
				t.Fatalf("Complete(%q) = nil error, want retirement error", model)
			}
			if !strings.Contains(err.Error(), "retired") {
				t.Errorf("error = %q, want it to mention %q", err.Error(), "retired")
			}
			if replacement, ok := llm.RetiredModelReplacement(model); !ok {
				t.Errorf("RetiredModelReplacement(%q) = not retired, want retired", model)
			} else if !strings.Contains(err.Error(), replacement) {
				t.Errorf("error = %q, want it to name replacement %q", err.Error(), replacement)
			}
		})
	}
}

// TestLiveModelsStillRegistered guards the inverse: IDs the provider still
// serves (verified against the live models endpoint) must remain registered,
// so the retirement sweep cannot over-delete.
func TestLiveModelsStillRegistered(t *testing.T) {
	for _, model := range []string{
		"claude-sonnet-4-6",
		"claude-sonnet-4-5",
		"claude-opus-4-5",
		"claude-haiku-4-5",
		// Live on 2026-11-10 (GET /v1/models, /v1/language-models).
		"claude-fable-5-2", "grok-4.3", "grok-build-0.1", "grok-4.20-0309-reasoning",
		"mistral-large-4-0", "mistral-large-2512",
	} {
		if _, ok := llm.GetModelInfo(model); !ok {
			t.Errorf("GetModelInfo(%q) = not registered, want registered (still live)", model)
		}
	}
}

func TestCatalogDefaultsAndDiscoveryForPhaseCProviders(t *testing.T) {
	cases := []struct {
		provider string
		wantDef  string
		sample   string
		caps     []llm.Capability
	}{
		{"mistral", "mistral-small-2603", "ministral-14b-2512", []llm.Capability{llm.CapabilityChat, llm.CapabilityVision}},
		{"zai", "glm-5.3", "glm-4.7", []llm.Capability{llm.CapabilityChat, llm.CapabilityVision}},
		{"minimax", "MiniMax-M3", "MiniMax-M2.5", []llm.Capability{llm.CapabilityChat}},
		{"moonshot", "kimi-k3.1-pro", "kimi-k3", []llm.Capability{llm.CapabilityChat, llm.CapabilityVision}},
		{"dashscope", "qwen3.6-plus", "qwen3.6-flash", []llm.Capability{llm.CapabilityChat}},
		{"deepseek", "deepseek-flash", "deepseek-v4-pro", []llm.Capability{llm.CapabilityChat}},
		{"meta", "muse-spark-1.3", "muse-spark-1.2", []llm.Capability{llm.CapabilityChat, llm.CapabilityVision, llm.CapabilityDocumentInput}},
	}
	for _, tc := range cases {
		if got := llm.GetDefaultModel(tc.provider); got != tc.wantDef {
			t.Errorf("GetDefaultModel(%s)=%q want %q", tc.provider, got, tc.wantDef)
		}
		models := llm.ModelsByProvider(tc.provider)
		if !slices.ContainsFunc(models, func(m llm.ModelInfo) bool { return m.ID == tc.sample }) {
			t.Errorf("ModelsByProvider(%s) missing %q: %v", tc.provider, tc.sample, models)
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
	if got := llm.ResolveModelAlias("muse"); got != "muse-spark-1.3" {
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
		if len(llm.ModelsByProvider(p)) == 0 {
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
		for _, m := range llm.ModelsByProvider(provider) {
			if _, retired := cases[m.ID]; retired {
				t.Errorf("ModelsByProvider(%s) still lists retired %q", provider, m.ID)
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

// regTestSeq hands out process-unique suffixes so a test whose precondition is
// "this model is not registered yet" stays valid across repeated runs (e.g.
// `go test -count=2`), since RegisterModel mutates the process-global registry.
var regTestSeq atomic.Uint64

func uniqueModelID(prefix string) string {
	return fmt.Sprintf("%s-%d", prefix, regTestSeq.Add(1))
}

// countModel reports how many times id appears in models.
func countModel(models []llm.ModelInfo, id string) int {
	n := 0
	for _, m := range models {
		if m.ID == id {
			n++
		}
	}
	return n
}

// R-M5: an unlisted model must be registrable at runtime so it gets a real
// cost estimate instead of the silent 0.
func TestRegisterModelEnablesCostEstimate(t *testing.T) {
	id := uniqueModelID("test-only-custom-model")
	if got := llm.EstimateCost(llm.CostInput{Model: id, InputTokens: 1_000_000}); got != 0 {
		t.Fatalf("unlisted model should cost 0 before registration, got %v", got)
	}

	if err := llm.RegisterModel(llm.ModelInfo{
		ID:               id,
		Provider:         "custom",
		InputPricePer1M:  2.0,
		OutputPricePer1M: 6.0,
	}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}

	got := llm.EstimateCost(llm.CostInput{Model: id, InputTokens: 1_000_000, OutputTokens: 1_000_000})
	if want := 8.0; got != want {
		t.Fatalf("after registration cost = %v, want %v", got, want)
	}

	info, ok := llm.GetModelInfo(id)
	if !ok || info.Provider != "custom" {
		t.Fatalf("GetModelInfo(%q) = %+v, %v", id, info, ok)
	}

	// It must also surface through discovery for its provider.
	found := false
	for _, m := range llm.ModelsByProvider("custom") {
		if m.ID == id {
			found = true
		}
	}
	if !found {
		t.Fatal("registered model missing from ModelsByProvider")
	}
}

// BUG 12: EstimateCost looked up the registry without resolving aliases, so a
// caller passing an alias model name got a silent 0 cost instead of the target's
// pricing.
func TestEstimateCostResolvesAlias(t *testing.T) {
	id := uniqueModelID("test-only-cost-target")
	if err := llm.RegisterModel(llm.ModelInfo{ID: id, Provider: "custom", InputPricePer1M: 3.0}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}
	alias := uniqueModelID("test-only-cost-alias")
	if err := llm.RegisterModelAlias(alias, id); err != nil {
		t.Fatalf("RegisterModelAlias: %v", err)
	}
	if got := llm.EstimateCost(llm.CostInput{Model: alias, InputTokens: 1_000_000}); got != 3.0 {
		t.Fatalf("EstimateCost(alias) = %v, want 3.0 (alias not resolved → silent 0 cost)", got)
	}
}

// BUG 3/4: a whitespace-only ID or Provider passed the bare != "" check and got
// stored under a meaningless key. Registration must reject blank-after-trim.
func TestRegisterModelRejectsBlankIDOrProvider(t *testing.T) {
	for _, ws := range []string{" ", "\t", "\n", "  \t  "} {
		if err := llm.RegisterModel(llm.ModelInfo{ID: ws, Provider: "p"}); err == nil {
			t.Errorf("RegisterModel with whitespace-only ID %q must error", ws)
		}
		if _, ok := llm.GetModelInfo(ws); ok {
			t.Errorf("whitespace-only ID %q must not be registered", ws)
		}
		if err := llm.RegisterModel(llm.ModelInfo{ID: "ok-id-x", Provider: ws}); err == nil {
			t.Errorf("RegisterModel with whitespace-only Provider %q must error", ws)
		}
	}
}

// R-M5: registering with missing required fields must error, not corrupt the
// registry.
func TestRegisterModelRejectsInvalid(t *testing.T) {
	if err := llm.RegisterModel(llm.ModelInfo{Provider: "custom"}); err == nil {
		t.Error("RegisterModel without ID must error")
	}
	if err := llm.RegisterModel(llm.ModelInfo{ID: "x-no-provider"}); err == nil {
		t.Error("RegisterModel without Provider must error")
	}
	// An alias to an unknown model must be rejected (else it silently resolves
	// to a non-existent id).
	if err := llm.RegisterModelAlias("test-alias-x", "no-such-model-xyz"); err == nil {
		t.Error("RegisterModelAlias to unknown model must error")
	}
}

// R-M5: a registered alias must resolve and feed cost estimation.
func TestRegisterModelAliasResolves(t *testing.T) {
	const id = "test-only-alias-target-v1"
	if err := llm.RegisterModel(llm.ModelInfo{ID: id, Provider: "custom", InputPricePer1M: 1.0}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}
	if err := llm.RegisterModelAlias("test-only-alias-v1", id); err != nil {
		t.Fatalf("RegisterModelAlias: %v", err)
	}
	if got := llm.ResolveModelAlias("test-only-alias-v1"); got != id {
		t.Fatalf("ResolveModelAlias = %q, want %q", got, id)
	}
}

// registryMu race guard: concurrent RegisterModel and readers must be
// data-race-free. Run under -race; without the lock the map access trips the
// detector (or panics on concurrent map read/write).
func TestRegistryConcurrentAccessRaceFree(t *testing.T) {
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			id := "race-model-" + string(rune('a'+n))
			_ = llm.RegisterModel(llm.ModelInfo{ID: id, Provider: "race", InputPricePer1M: float64(n)})
			_ = llm.EstimateCost(llm.CostInput{Model: id, InputTokens: 100})
			_, _ = llm.GetModelInfo(id)
			_ = llm.Models()
			_ = llm.ModelsByProvider("race")
			_ = llm.ResolveModelAlias(id)
		}(i)
	}
	wg.Wait()
}

// An override that moves a model to a different provider must update both
// providers' discovery — add to the new, remove from the old — and never
// list it twice.
func TestRegisterModelProviderChangeUpdatesDiscovery(t *testing.T) {
	const id = "test-only-move-provider-v1"
	const p1, p2 = "test-only-p1", "test-only-p2"

	if err := llm.RegisterModel(llm.ModelInfo{ID: id, Provider: p1, InputPricePer1M: 1}); err != nil {
		t.Fatalf("RegisterModel p1: %v", err)
	}
	if countModel(llm.ModelsByProvider(p1), id) != 1 {
		t.Fatal("new model missing from its provider's discovery")
	}

	// Move it to a different provider.
	if err := llm.RegisterModel(llm.ModelInfo{ID: id, Provider: p2, InputPricePer1M: 1}); err != nil {
		t.Fatalf("RegisterModel p2: %v", err)
	}
	if countModel(llm.ModelsByProvider(p2), id) != 1 {
		t.Fatal("override to a new provider did not add the model to the new provider's discovery")
	}
	if countModel(llm.ModelsByProvider(p1), id) != 0 {
		t.Fatal("override to a new provider left a stale entry in the old provider's discovery")
	}

	// Re-register under the same provider: still exactly one entry.
	if err := llm.RegisterModel(llm.ModelInfo{ID: id, Provider: p2, InputPricePer1M: 2}); err != nil {
		t.Fatalf("RegisterModel p2 again: %v", err)
	}
	if c := countModel(llm.ModelsByProvider(p2), id); c != 1 {
		t.Fatalf("model appears %d times after same-provider re-register, want 1", c)
	}
}

// #10: an alias key equal to a real model ID would shadow that model
// (ResolveModelAlias checks aliases first) — RegisterModelAlias must reject it,
// and the real model must keep resolving to itself.
func TestRegisterModelAliasRejectsShadowingRealModel(t *testing.T) {
	const a = "test-only-shadow-a-v1"
	const b = "test-only-shadow-b-v1"
	if err := llm.RegisterModel(llm.ModelInfo{ID: a, Provider: "custom", InputPricePer1M: 1}); err != nil {
		t.Fatalf("RegisterModel a: %v", err)
	}
	if err := llm.RegisterModel(llm.ModelInfo{ID: b, Provider: "custom", InputPricePer1M: 2}); err != nil {
		t.Fatalf("RegisterModel b: %v", err)
	}
	if err := llm.RegisterModelAlias(a, b); err == nil {
		t.Fatal("RegisterModelAlias with an alias key equal to an existing model ID must be rejected (would shadow it)")
	}
	if got := llm.ResolveModelAlias(a); got != a {
		t.Fatalf("real model %q was shadowed by an alias: ResolveModelAlias(%q) = %q", a, a, got)
	}
}

// R-M5 end-to-end: a client built for an unlisted model/provider (via BaseURL)
// must work, and registering metadata for it must light up cost in the
// session usage. Uses a mock client so no network is needed.
func TestRegisteredModelFeedsSessionUsage(t *testing.T) {
	const id = "test-only-session-model-v1"
	if err := llm.RegisterModel(llm.ModelInfo{ID: id, Provider: "custom", InputPricePer1M: 10.0, OutputPricePer1M: 30.0}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}
	mock := llm.NewMockClient("custom", id)
	mock.PushResponse(&llm.Response{Model: id, Content: "hi", StopReason: llm.StopEndTurn, InputTokens: 1_000_000, OutputTokens: 1_000_000})
	sess := llm.NewSession(mock, llm.WithBaseRequest(llm.Request{Model: id}))
	if _, err := sess.Send(context.Background(), "x"); err != nil {
		t.Fatalf("Send: %v", err)
	}
	if got := sess.Usage().Cost; got != 40.0 {
		t.Fatalf("session cost = %v, want 40.0 (registered pricing not applied)", got)
	}
}

// The registry fields the cost display relies on, set from primary sources;
// a typo or dropped field silently zeroes the live cost.
func TestRegistryAudioPricesPinned(t *testing.T) {
	cases := []struct {
		model              string
		audio1M, perMinute float64
	}{
		{"gemini-3.5-flash-lite", 0.30, 0},
		{"gemini-3.5-transcribe", 2.00, 0},
		{"gemini-3.1-flash-lite", 0.50, 0},
		{"gpt-4o-transcribe", 2.50, 0},
		{"gpt-4o-mini-transcribe", 1.25, 0},
		{"whisper-1", 0, 0.006},
		{"gpt-transcribe", 0, 0.0045},
		{"grok-voice-transcribe-2.0", 0, 0.10 / 60},
	}
	for _, c := range cases {
		info, ok := llm.GetModelInfo(c.model)
		if !ok || info.AudioInputPricePer1M != c.audio1M || info.AudioPricePerMinute != c.perMinute {
			t.Errorf("%s: audio=%v perMinute=%v, want %v %v", c.model, info.AudioInputPricePer1M, info.AudioPricePerMinute, c.audio1M, c.perMinute)
		}
	}
}

// Providers answer with dated snapshot IDs; the exact-match lookup priced them
// at $0, so Session.Usage and LoggingClient reported free turns (B22).
func TestEstimateCostPricesSnapshotIDs(t *testing.T) {
	for _, tc := range []struct{ snapshot, base string }{
		{"claude-opus-5-5-20260901", "claude-opus-5-5"},
		{"gpt-5.4-2026-03-05", "gpt-5.4"},
		{"claude-sonnet-5-5-20261001", "claude-sonnet-5-5"},
	} {
		in := llm.CostInput{Model: tc.base, InputTokens: 1000, OutputTokens: 100}
		want := llm.EstimateCost(in)
		if want == 0 {
			t.Fatalf("%s is not priced in the registry", tc.base)
		}
		in.Model = tc.snapshot
		if got := llm.EstimateCost(in); got != want {
			t.Errorf("%s = %v, want %v (the price of %s)", tc.snapshot, got, want, tc.base)
		}
	}
	for _, model := range []string{"claude-opus-5-5-x", "claude-opus-5-5-12", "gpt-5.4-mini-preview", "no-such-model-20260101", "-20260101", "claude-opus-5-5-"} {
		if got := llm.EstimateCost(llm.CostInput{Model: model, InputTokens: 1000}); got != 0 && !registered(model) {
			t.Errorf("%q priced at %v; only a numeric snapshot suffix may borrow a price", model, got)
		}
	}
}

func registered(model string) bool {
	_, ok := llm.GetModelInfo(model)
	return ok
}
