package llm_test

// Hardening pass 15 — adversarial coverage for the native zai/glm, minimax, and
// moonshot/kimi provider wiring (presets, alias/default/registry maps, cost math,
// cross-host auth-strip, concurrency). NO happy-path: every test encodes a way
// the wiring can silently break and asserts the break is absent. Each test was
// proven to go red against an injected regression before being committed.

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// newProvider is the shared fixture: a native provider, its canonical default
// model (exact registry-key case), and the provider its model is owned by.
type newProvider struct {
	provider string // preset/provider name
	alias    string // a model-alias that ALSO collides with a provider/default key
	model    string // canonical registry-key model ID (case-exact)
	key      string // a throwaway API key, distinct per provider
}

var newProviders = []newProvider{
	{provider: "zai", alias: "glm", model: "glm-5.2", key: "sk-zai-secret-key"},
	{provider: "minimax", alias: "minimax", model: "MiniMax-M3", key: "sk-minimax-secret-key"},
	{provider: "moonshot", alias: "kimi", model: "kimi-k2.7-code", key: "sk-moonshot-secret-key"},
}

// Provider names resolve CASE-SENSITIVELY and must FAIL CLOSED. A mis-cased or
// space-padded name ("ZAI", "Kimi", " minimax") is NOT the real provider: it must
// not silently inherit a real base URL and ship a secret there — it must demand
// an explicit BaseURL. Guards against anyone normalizing/lowercasing preset keys.
func TestNewProviderNamesFailClosedOnMiscase(t *testing.T) {
	miscased := []string{
		"ZAI", "Zai", "GLM", "Glm",
		"Minimax", "MINIMAX", " minimax", "minimax ",
		"Moonshot", "MOONSHOT", "Kimi", "KIMI", " kimi",
	}
	for _, name := range miscased {
		t.Run(name, func(t *testing.T) {
			_, err := llm.NewClient(llm.Config{Provider: name, Model: "x", APIKey: "sk-secret"})
			if err == nil {
				t.Fatalf("provider %q built a client with no explicit BaseURL — a mis-cased/"+
					"padded name must fail closed, not inherit a real endpoint and leak the key", name)
			}
			if !strings.Contains(err.Error(), "unknown provider") {
				t.Fatalf("provider %q: want an 'unknown provider' rejection, got %v", name, err)
			}
		})
	}
}

// Each new family's key string lives in THREE maps at once: presets,
// defaultModels, and modelAliases. If those copies ever diverge, a caller who
// types "glm"/"minimax"/"kimi" gets one model for routing and another for
// pricing. Pin that every colliding key resolves identically across all seams,
// and that the target is a real registry entry owned by the right provider.
func TestNewProviderAliasCollisionsAgree(t *testing.T) {
	for _, p := range newProviders {
		t.Run(p.alias, func(t *testing.T) {
			gotAlias := llm.ResolveModelAlias(p.alias)
			gotDefault := llm.GetDefaultModel(p.alias)
			if gotAlias != p.model {
				t.Errorf("ResolveModelAlias(%q) = %q, want %q", p.alias, gotAlias, p.model)
			}
			if gotDefault != p.model {
				t.Errorf("GetDefaultModel(%q) = %q, want %q", p.alias, gotDefault, p.model)
			}
			if gotAlias != gotDefault {
				t.Fatalf("seam disagreement for %q: alias→%q but default→%q — routing and "+
					"pricing would diverge", p.alias, gotAlias, gotDefault)
			}
			info, ok := llm.GetModelInfo(p.model)
			if !ok {
				t.Fatalf("resolved model %q is absent from the registry", p.model)
			}
			if info.Provider != p.provider {
				t.Fatalf("model %q claims provider %q, want %q", p.model, info.Provider, p.provider)
			}
			// GetDefaultModel(provider-name) must also land on the same model.
			if d := llm.GetDefaultModel(p.provider); d != p.model {
				t.Fatalf("GetDefaultModel(%q) = %q, want %q", p.provider, d, p.model)
			}
		})
	}
}

// ProviderForModel is the reverse map, case-sensitive on the EXACT registry key.
// MiniMax-M3 and kimi-k2.7-code are the non-trivial keys; if anyone lowercases or
// rewrites a key, the canonical lookup breaks while the lowercase alias (not a
// key) keeps returning "". Pin both directions for every new model.
func TestProviderForModelExactCaseKeys(t *testing.T) {
	for _, p := range newProviders {
		if got := llm.ProviderForModel(p.model); got != p.provider {
			t.Errorf("ProviderForModel(%q) = %q, want %q — canonical-case key regressed", p.model, got, p.provider)
		}
		// The alias is NOT a registry key — it must never reverse-map to a provider.
		if got := llm.ProviderForModel(p.alias); got != "" {
			t.Errorf("ProviderForModel(%q) = %q, want \"\" — an alias must not masquerade as a registry key", p.alias, got)
		}
	}
	// A lowercased variant of the mixed-case keys must not resolve either.
	for _, bad := range []string{"minimax-m3", "KIMI-K2.7-CODE", "GLM-5.2"} {
		if got := llm.ProviderForModel(bad); got != "" {
			t.Errorf("ProviderForModel(%q) = %q, want \"\" — case-variant must not match the registry key", bad, got)
		}
	}
}

// All three new models carry a cache-READ price but NO cache-WRITE price. The
// bug this guards: billing cache-create tokens the provider gives away for free
// (write price 0 ⇒ must be skipped). Also: alias pricing must equal canonical
// pricing, and the TokensNotReported(-1) sentinel must clamp to 0.
func TestNewProviderCostMathCacheAndSentinel(t *testing.T) {
	const M = 1_000_000
	// Exact expected cost for 1M input + 1M output + 1M cacheRead (+1M cacheCreate
	// which MUST contribute nothing because write price is 0).
	want := map[string]float64{
		"glm-5.2":        1.40 + 4.40 + 0.26,
		"MiniMax-M3":     0.60 + 2.40 + 0.12,
		"kimi-k2.7-code": 0.95 + 4.00 + 0.16,
	}
	for _, p := range newProviders {
		t.Run(p.model, func(t *testing.T) {
			in := llm.CostInput{Model: p.model, InputTokens: M, OutputTokens: M, CacheReadTokens: M, CacheCreateTokens: M}
			got := llm.EstimateCost(in)
			if math.Abs(got-want[p.model]) > 1e-9 {
				t.Fatalf("%s cost = %v, want %v — a cache-create charge on a model with no write price",
					p.model, got, want[p.model])
			}
			// Alias must price identically to the canonical ID.
			viaAlias := llm.EstimateCost(llm.CostInput{Model: p.alias, InputTokens: M, OutputTokens: M, CacheReadTokens: M, CacheCreateTokens: M})
			if math.Abs(viaAlias-want[p.model]) > 1e-9 {
				t.Fatalf("EstimateCost(%q) = %v, want %v — alias priced differently than canonical ID", p.alias, viaAlias, want[p.model])
			}
			// -1 sentinels and stray negatives clamp to 0; never negative/non-finite.
			s := llm.EstimateCost(llm.CostInput{Model: p.model, InputTokens: -1, OutputTokens: -1, ThinkingTokens: -1, CacheReadTokens: -1, CacheCreateTokens: -1})
			if s != 0 {
				t.Fatalf("%s with all -1 sentinels = %v, want 0 (clamped)", p.model, s)
			}
		})
	}
}

// Every model listed in availableModels for a new provider must be a real
// registry entry owned by that same provider and appear exactly once. Guards a
// copy-paste that lists a model under the wrong provider or twice.
func TestNewProviderAvailableModelsConsistent(t *testing.T) {
	for _, p := range newProviders {
		models := llm.GetAvailableModels(p.provider)
		if len(models) == 0 {
			t.Fatalf("GetAvailableModels(%q) is empty — provider not discoverable", p.provider)
		}
		seen := map[string]int{}
		for _, id := range models {
			seen[id]++
			info, ok := llm.GetModelInfo(id)
			if !ok {
				t.Errorf("%s lists %q which is absent from the registry", p.provider, id)
				continue
			}
			if info.Provider != p.provider {
				t.Errorf("%s lists %q but its registry Provider is %q", p.provider, id, info.Provider)
			}
		}
		for id, n := range seen {
			if n != 1 {
				t.Errorf("%s lists %q %d times — duplicate discovery entry", p.provider, id, n)
			}
		}
	}
}

// The new providers must inherit SafeHTTPClient's cross-host redirect auth
// stripping. Drive a real client at a redirector that bounces to an attacker
// host; the Bearer key must NOT reach the attacker. Non-vacuous: the attacker
// host MUST be reached (the policy strips-and-follows), else the test is empty.
func TestNewProviderRedirectDoesNotLeakKey(t *testing.T) {
	for _, p := range newProviders {
		t.Run(p.provider, func(t *testing.T) {
			var captured http.Header
			attacker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				captured = r.Header.Clone()
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprint(w, `{"id":"x","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"pwned"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
			}))
			defer attacker.Close()
			redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				http.Redirect(w, r, attacker.URL+r.URL.Path, http.StatusTemporaryRedirect)
			}))
			defer redirector.Close()

			client, err := llm.NewClient(llm.Config{Provider: p.provider, Model: p.model, APIKey: p.key, BaseURL: redirector.URL})
			if err != nil {
				t.Fatalf("NewClient(%s): %v", p.provider, err)
			}
			defer client.Close()
			_, _ = client.Complete(context.Background(), llm.Request{
				Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
			})
			if captured == nil {
				t.Fatalf("%s: attacker host was never reached — redirect not followed, auth-strip assertion would be vacuous", p.provider)
			}
			if got := captured.Get("Authorization"); got != "" {
				t.Errorf("%s leaked Authorization header to cross-host redirect target: %q", p.provider, got)
			}
			if strings.Contains(fmt.Sprint(captured), p.key) {
				t.Errorf("%s leaked the raw API key to the attacker host", p.provider)
			}
		})
	}
}

// The alias lookup inside EstimateCost is inlined under the registry RWMutex
// (documented: the RWMutex is not re-entrant, so ResolveModelAlias can't be
// called while holding it). Hammer that read path with concurrent registry
// WRITES to prove the glm/minimax/kimi alias resolution stays race-free and
// never loses its mapping under contention. Worthless without -race.
func TestNewProviderConcurrentAliasResolutionRace(t *testing.T) {
	var writers, readers sync.WaitGroup
	stop := make(chan struct{})

	for w := 0; w < 4; w++ {
		writers.Add(1)
		go func(w int) {
			defer writers.Done()
			for i := 0; ; i++ {
				select {
				case <-stop:
					return
				default:
				}
				id := fmt.Sprintf("race-model-%d-%d", w, i)
				_ = llm.RegisterModel(llm.ModelInfo{ID: id, Provider: "custom", InputPricePer1M: 1})
				_ = llm.RegisterModelAlias(fmt.Sprintf("race-alias-%d-%d", w, i), id)
			}
		}(w)
	}

	for r := 0; r < 8; r++ {
		readers.Add(1)
		go func() {
			defer readers.Done()
			for i := 0; i < 4000; i++ {
				for _, p := range newProviders {
					_ = llm.ResolveModelAlias(p.alias)
					_ = llm.ProviderForModel(p.model)
					_ = llm.GetDefaultModel(p.provider)
					_ = llm.GetAvailableModels(p.provider)
					if c := llm.EstimateCost(llm.CostInput{Model: p.alias, InputTokens: 100, OutputTokens: 100}); c <= 0 {
						t.Errorf("EstimateCost(%q) = %v under concurrent registry writes — alias resolution lost", p.alias, c)
						return
					}
				}
			}
		}()
	}

	readers.Wait() // finite work
	close(stop)    // release the writers
	writers.Wait()
}
