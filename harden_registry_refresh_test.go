package llm_test

// Break-the-system, Round 3 — observable contract of the 2026-07-08 model-registry
// refresh, asserted at the PUBLIC seam (GetModelInfo / EstimateCost / GetDefaultModel /
// ResolveModelAlias / ResolveBaseURL), not the internal maps. A refresh changes what
// callers see: which model a bare provider resolves to, what a cost estimate returns
// for a model whose price is deliberately unknown, and how a newly-wired provider
// endpoint is built. Each test pins the NEW behavior so a later edit that regresses it
// (re-guessing a price, flipping a default back, mistyping the new base URL) fails loudly.

import (
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// The refresh deliberately shipped three models with NO price (Cohere/DashScope
// pricing pages were unconfirmed). The contract that makes that safe: they must be
// REGISTERED (so routing/capabilities work) yet EstimateCost must return exactly 0 —
// and 0 here means "price unknown", identical to an unregistered model. This pins the
// seam so that (a) if a real price is added later, this test forces a conscious update,
// and (b) nobody "fixes" the 0 by guessing a number without touching this test.
func TestUnpricedRefreshModelsRegisteredButCostZero(t *testing.T) {
	unpriced := []struct{ model, provider string }{
		{"command-a-plus-05-2026", "cohere"},
		{"qwen3.7-max", "dashscope"},
		{"qwen3.7-plus", "dashscope"},
	}
	for _, m := range unpriced {
		t.Run(m.model, func(t *testing.T) {
			info, ok := llm.GetModelInfo(m.model)
			if !ok {
				t.Fatalf("GetModelInfo(%q) not found — an unpriced model must still be registered so it routes", m.model)
			}
			if info.Provider != m.provider {
				t.Fatalf("GetModelInfo(%q).Provider=%q, want %q", m.model, info.Provider, m.provider)
			}
			if info.InputPricePer1M != 0 || info.OutputPricePer1M != 0 {
				t.Fatalf("%q now carries a price (in=%g out=%g) — if that is real, update this test on purpose",
					m.model, info.InputPricePer1M, info.OutputPricePer1M)
			}
			// Non-trivial token counts must still cost 0 — proving the estimate reflects
			// the missing price, not merely zero usage.
			cost := llm.EstimateCost(llm.CostInput{Model: m.model, InputTokens: 1_000_000, OutputTokens: 1_000_000})
			if cost != 0 {
				t.Fatalf("EstimateCost(%q, 1M+1M tok)=%g, want 0 for an unpriced model", m.model, cost)
			}
			if llm.ProviderForModel(m.model) != m.provider {
				t.Fatalf("ProviderForModel(%q)=%q, want %q", m.model, llm.ProviderForModel(m.model), m.provider)
			}
		})
	}
}

// The refresh retired the previous flagships and repointed each provider's bare
// default. This asserts the NEW default at the seam GetDefaultModel/ResolveModelAlias
// actually uses — the most user-visible behavior change (a bare `Provider:"deepseek"`
// now selects a different model). A default that silently stayed on a now-removed
// model would send a dead ID on the first request.
func TestRefreshedProviderDefaultsAndAliases(t *testing.T) {
	defaults := map[string]string{
		"deepseek": "deepseek-v4-flash",
		"groq":     "openai/gpt-oss-120b",
	}
	for provider, want := range defaults {
		if got := llm.GetDefaultModel(provider); got != want {
			t.Errorf("GetDefaultModel(%q)=%q, want %q (refresh repointed the default)", provider, got, want)
		}
		// The default must resolve to a registered model, not a dangling ID.
		if _, ok := llm.GetModelInfo(llm.ResolveModelAlias(want)); !ok {
			t.Errorf("default %q for %q is not registered", want, provider)
		}
	}
	aliases := map[string]string{
		"deepseek-cloud":  "deepseek-v4-flash",
		"deepseek-v4":     "deepseek-v4-flash",
		"deepseek-flash":  "deepseek-v4-flash",
		"deepseek-v4-pro": "deepseek-v4-pro", // now a bare model ID, self-resolving (NOT an alias)
		"groq":            "openai/gpt-oss-120b",
		"gpt-oss":         "openai/gpt-oss-120b",
		"qwen3.6-27b":     "qwen/qwen3.6-27b",
		"command-a-plus":  "command-a-plus-05-2026",
		"command-a":       "command-a-03-2025", // unchanged: adding A+ must NOT move the old default alias
	}
	for alias, want := range aliases {
		if got := llm.ResolveModelAlias(alias); got != want {
			t.Errorf("ResolveModelAlias(%q)=%q, want %q", alias, got, want)
		}
	}
}

// Cohere's bare default must NOT have drifted to the new, UNPRICED command-a-plus.
// Making an unpriced model the default would make bare cohere requests silently
// un-estimatable — the refresh notes explicitly keep command-a-03-2025 as default.
func TestCohereDefaultStaysOnPricedModel(t *testing.T) {
	def := llm.GetDefaultModel("cohere")
	if def != "command-a-03-2025" {
		t.Fatalf("GetDefaultModel(cohere)=%q, want command-a-03-2025 (must stay on a PRICED model)", def)
	}
	info, ok := llm.GetModelInfo(def)
	if !ok || info.InputPricePer1M == 0 {
		t.Fatalf("cohere default %q must be priced; got ok=%v price=%g", def, ok, info.InputPricePer1M)
	}
}

// Groq's retired Llama aliases must be DEAD passthroughs, not resolve to a stale ID.
// ResolveModelAlias returns its input unchanged for a non-alias, so "llama" must come
// back "llama" (an unknown model that fails downstream), NOT a removed llama-3.x ID.
func TestRetiredGroqAliasesAreDeadPassthrough(t *testing.T) {
	for _, dead := range []string{"llama", "llama-70b", "llama-8b", "llama4", "llama-4-scout"} {
		if got := llm.ResolveModelAlias(dead); got != dead {
			t.Errorf("ResolveModelAlias(%q)=%q — a retired alias must be a passthrough, not resolve to a stale model", dead, got)
		}
		if _, ok := llm.GetModelInfo(llm.ResolveModelAlias(dead)); ok {
			t.Errorf("retired alias %q still resolves to a registered model", dead)
		}
	}
}

// The new antling/ling preset must build the EXACT documented endpoint, protocol, and
// auth, and ling must be a true synonym of antling (same base URL). A typo in the base
// URL ("ant-ling.com" vs "antling.com") or a wrong protocol would silently ship keys to
// the wrong host or frame requests in the wrong wire format.
func TestAntlingPresetWiring(t *testing.T) {
	const wantBase = "https://api.ant-ling.com/v1"
	for _, p := range []string{"antling", "ling"} {
		if got := llm.ResolveBaseURL(llm.Config{Provider: p}); got != wantBase {
			t.Errorf("ResolveBaseURL(%q)=%q, want %q", p, got, wantBase)
		}
		if got := llm.ResolveProtocol(llm.Config{Provider: p}); got != "openai_compat" {
			t.Errorf("ResolveProtocol(%q)=%q, want openai_compat", p, got)
		}
		if got := llm.ResolveAuthHeader(llm.Config{Provider: p}); got != "Bearer" {
			t.Errorf("ResolveAuthHeader(%q)=%q, want Bearer", p, got)
		}
		if llm.IsNoAuthProvider(p) {
			t.Errorf("IsNoAuthProvider(%q)=true — antling is a cloud provider that needs a key", p)
		}
	}
}

// The new provider name must FAIL CLOSED on mis-case: "Antling"/"ANTLING"/" ling " are
// NOT the preset key and must not inherit the real base URL and ship a key there — they
// must demand an explicit BaseURL. Mirrors TestNewProviderNamesFailClosedOnMiscase for
// the newly-added family.
func TestAntlingNameFailsClosedOnMiscase(t *testing.T) {
	for _, name := range []string{"Antling", "ANTLING", " antling", "antling ", "Ling", "LING", " ling"} {
		t.Run(name, func(t *testing.T) {
			_, err := llm.NewClient(llm.Config{Provider: name, Model: "x", APIKey: "sk-secret"})
			if err == nil {
				t.Fatalf("provider %q built a client with no explicit BaseURL — a mis-cased name must fail closed", name)
			}
			if !strings.Contains(err.Error(), "unknown provider") {
				t.Fatalf("provider %q: want 'unknown provider', got %v", name, err)
			}
		})
	}
}
