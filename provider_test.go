package llm_test

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider"
)

func TestNewProviderPresets(t *testing.T) {
	newOnes := []string{"together", "fireworks", "nvidia", "perplexity", "deepinfra"}
	for _, p := range newOnes {
		preset, ok := llm.PresetFor(p)
		if !ok {
			t.Errorf("preset %q missing", p)
			continue
		}
		if preset.Protocol != "openai_compat" {
			t.Errorf("%q protocol = %q, want openai_compat", p, preset.Protocol)
		}
		if !strings.HasPrefix(preset.BaseURL, "https://") {
			t.Errorf("%q baseURL = %q, want https://...", p, preset.BaseURL)
		}
		// Must construct without any network call.
		c, err := llm.NewClient(llm.Config{Provider: p, Model: "some-model", APIKey: "k", MaxTokens: 10})
		if err != nil {
			t.Errorf("NewClient(%q): %v", p, err)
			continue
		}
		if c.Provider() != p {
			t.Errorf("Provider() = %q, want %q", c.Provider(), p)
		}
		_ = c.Close()
	}
	// Discovery must include them.
	got := map[string]bool{}
	for _, p := range llm.Providers() {
		got[p] = true
	}
	for _, p := range newOnes {
		if !got[p] {
			t.Errorf("Providers() missing %q", p)
		}
	}
}

func TestNewProvidersHaveRegistryEntries(t *testing.T) {
	for _, p := range []string{"perplexity", "fireworks", "together", "deepinfra", "nvidia"} {
		models := llm.ModelsByProvider(p)
		if len(models) == 0 {
			t.Errorf("ModelsByProvider(%q) is empty — provider has no registry entries", p)
		}
		for _, m := range models {
			if m.Provider != p {
				t.Errorf("%q query returned a %q model: %s", p, m.Provider, m.ID)
			}
		}
	}
	// Cost estimation now works for a priced model (sonar-pro = $3/$15 per 1M) — was 0 before.
	if cost := llm.EstimateCost(llm.CostInput{Model: "sonar-pro", InputTokens: 1_000_000, OutputTokens: 1_000_000}); cost <= 0 {
		t.Errorf("EstimateCost(sonar-pro, 1M/1M) = %v, want > 0", cost)
	}
	// They appear via GetModelInfo too.
	if _, ok := llm.GetModelInfo("accounts/fireworks/models/kimi-k3"); !ok {
		t.Error("fireworks model not found via GetModelInfo")
	}
}

// newProvider is the shared fixture: a native provider, its canonical default
// model (exact registry-key case), and the provider its model is owned by.
type newProvider struct {
	provider string // preset/provider name
	alias    string // a model-alias that ALSO collides with a provider/default key
	model    string // canonical registry-key model ID (case-exact)
	key      string // a throwaway API key, distinct per provider
}

var newProviders = []newProvider{
	{provider: "zai", alias: "glm", model: "glm-5.3", key: "sk-zai-secret-key"},
	{provider: "minimax", alias: "minimax", model: "MiniMax-M3", key: "sk-minimax-secret-key"},
	{provider: "moonshot", alias: "kimi", model: "kimi-k3.1-pro", key: "sk-moonshot-secret-key"},
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
// MiniMax-M3 and kimi-k3.1-pro are the non-trivial keys; if anyone lowercases or
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
	for _, bad := range []string{"minimax-m3", "KIMI-K3.1-PRO", "GLM-5.2"} {
		if got := llm.ProviderForModel(bad); got != "" {
			t.Errorf("ProviderForModel(%q) = %q, want \"\" — case-variant must not match the registry key", bad, got)
		}
	}
}

// glm-5.3 and MiniMax-M3 carry a cache-READ price but NO cache-WRITE price. The
// bug this guards: billing cache-create tokens the provider gives away for free
// (write price 0 ⇒ must be skipped). kimi-k3.1-pro bills writes (1× input for the
// default 5-minute TTL), so its 1M cache-create tokens must add exactly $2. Also: alias pricing must equal canonical
// pricing, and the TokensNotReported(-1) sentinel must clamp to 0.
func TestNewProviderCostMathCacheAndSentinel(t *testing.T) {
	const M = 1_000_000
	// Exact expected cost for 1M input + 1M output + 1M cacheRead + 1M cacheCreate.
	want := map[string]float64{
		"glm-5.3":       1.40 + 4.40 + 0.26,
		"MiniMax-M3":    0.30 + 1.20 + 0.06,
		"kimi-k3.1-pro": 2.00 + 10.00 + 0.20 + 2.00,
	}
	for _, p := range newProviders {
		t.Run(p.model, func(t *testing.T) {
			in := llm.CostInput{Model: p.model, InputTokens: M, OutputTokens: M, CacheReadTokens: M, CacheCreateTokens: M}
			got := llm.EstimateCost(in)
			if math.Abs(got-want[p.model]) > 1e-9 {
				t.Fatalf("%s cost = %v, want %v — cache-create tokens billed at the wrong write price",
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

// Every new provider is discoverable, and each model it lists is registered
// under that provider. Guards a registry row filed under the wrong provider.
func TestNewProviderModelsConsistent(t *testing.T) {
	for _, p := range newProviders {
		models := llm.ModelsByProvider(p.provider)
		if len(models) == 0 {
			t.Fatalf("ModelsByProvider(%q) is empty — provider not discoverable", p.provider)
		}
		for _, m := range models {
			if m.Provider != p.provider {
				t.Errorf("%s lists %q but its registry Provider is %q", p.provider, m.ID, m.Provider)
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
					_ = llm.ModelsByProvider(p.provider)
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

// TestResolveProtocol verifies protocol resolution for all known providers.
func TestResolveProtocol(t *testing.T) {
	tests := []struct {
		provider string
		expected string
	}{
		{"anthropic", "anthropic"},
		{"claude", "anthropic"},
		{"gemini", "gemini"},
		{"google", "gemini"},
		{"openai", "openai_compat"},
		{"xai", "openai_compat"},
		{"grok", "openai_compat"},
		{"groq", "openai_compat"},
		{"cerebras", "openai_compat"},
		{"mistral", "openai_compat"},
		{"openrouter", "openai_compat"},
		{"ollama", "openai_compat"},
		{"vllm", "openai_compat"},
		{"lmstudio", "openai_compat"},
	}

	for _, tc := range tests {
		t.Run(tc.provider, func(t *testing.T) {
			cfg := llm.Config{Provider: tc.provider}
			got := llm.ResolveProtocol(cfg)
			if got != tc.expected {
				t.Errorf("ResolveProtocol(%q) = %q, want %q", tc.provider, got, tc.expected)
			}
		})
	}
}

// TestResolveProtocolCustomBaseURL verifies unknown providers with BaseURL resolve to openai_compat.
func TestResolveProtocolCustomBaseURL(t *testing.T) {
	cfg := llm.Config{
		Provider: "my-custom-provider",
		BaseURL:  "http://localhost:9999/v1",
	}
	got := llm.ResolveProtocol(cfg)
	if got != "openai_compat" {
		t.Errorf("ResolveProtocol(custom) = %q, want openai_compat", got)
	}
}

// TestProviderPresets verifies all known presets have valid protocols.
func TestProviderPresets(t *testing.T) {
	validProtocols := map[string]bool{
		"anthropic":     true,
		"gemini":        true,
		"openai_compat": true,
	}

	providers := []string{
		"anthropic", "claude", "gemini", "google",
		"openai", "xai", "grok", "groq", "cerebras", "mistral", "openrouter",
		"ollama", "vllm", "lmstudio",
	}

	for _, name := range providers {
		preset, ok := llm.PresetFor(name)
		if !ok {
			t.Errorf("Provider %q has no preset", name)
			continue
		}
		if !validProtocols[preset.Protocol] {
			t.Errorf("Provider %q has invalid protocol %q", name, preset.Protocol)
		}
	}
}

// TestOllamaNoAuthRequired verifies Ollama client creation without API key.
func TestOllamaNoAuthRequired(t *testing.T) {
	cfg := llm.Config{
		Provider:  "ollama",
		Model:     "llama3",
		MaxTokens: 100,
		Timeout:   10 * time.Second,
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create Ollama client without API key: %v", err)
	}
	defer client.Close()

	if client.Provider() != "ollama" {
		t.Errorf("Expected provider 'ollama', got %q", client.Provider())
	}
}

// TestVLLMNoAuthRequired verifies vLLM client creation without API key.
func TestVLLMNoAuthRequired(t *testing.T) {
	cfg := llm.Config{
		Provider:  "vllm",
		Model:     "meta-llama/Llama-3-8b",
		MaxTokens: 100,
		Timeout:   10 * time.Second,
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create vLLM client without API key: %v", err)
	}
	defer client.Close()

	if client.Provider() != "vllm" {
		t.Errorf("Expected provider 'vllm', got %q", client.Provider())
	}
}

// TestLMStudioNoAuthRequired verifies LM Studio client creation without API key.
func TestLMStudioNoAuthRequired(t *testing.T) {
	cfg := llm.Config{
		Provider:  "lmstudio",
		Model:     "local-model",
		MaxTokens: 100,
		Timeout:   10 * time.Second,
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create LM Studio client without API key: %v", err)
	}
	defer client.Close()

	if client.Provider() != "lmstudio" {
		t.Errorf("Expected provider 'lmstudio', got %q", client.Provider())
	}
}

// TestIsNoAuthProvider verifies no-auth provider detection.
func TestIsNoAuthProvider(t *testing.T) {
	tests := []struct {
		provider string
		expected bool
	}{
		{"ollama", true},
		{"vllm", true},
		{"lmstudio", true},
		{"anthropic", false},
		{"openai", false},
		{"xai", false},
	}

	for _, tc := range tests {
		t.Run(tc.provider, func(t *testing.T) {
			got := llm.IsNoAuthProvider(tc.provider)
			if got != tc.expected {
				t.Errorf("IsNoAuthProvider(%q) = %v, want %v", tc.provider, got, tc.expected)
			}
		})
	}
}

// TestAllProviders runs a simple completion test against all configured providers.
func TestAllProviders(t *testing.T) {
	// Live, paid calls: never in -short mode (`make test`), even when the
	// developer's shell exports provider keys.
	if testing.Short() {
		t.Skip("integration test: skipped in -short mode")
	}
	anthropicKey := envKey("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEYS")
	xaiKey := envKey("XAI_API_KEY", "XAI_API_KEYS")
	geminiKey := envKey("GEMINI_API_KEY", "GEMINI_API_KEYS")

	tests := []struct {
		name     string
		provider string
		model    string
		apiKey   string
		skip     bool
	}{
		{
			name:     "Anthropic/Claude-Sonnet",
			provider: "anthropic",
			model:    "claude-sonnet-4-6",
			apiKey:   anthropicKey,
			skip:     anthropicKey == "",
		},
		{
			name:     "Anthropic/Claude-Haiku",
			provider: "anthropic",
			model:    "claude-haiku-4-5-20251001",
			apiKey:   anthropicKey,
			skip:     anthropicKey == "",
		},
		{
			name:     "xAI/Grok-4.20",
			provider: "xai",
			model:    "grok-4.20-0309-non-reasoning",
			apiKey:   xaiKey,
			skip:     xaiKey == "",
		},
		{
			name:     "xAI/Grok-Build",
			provider: "xai",
			model:    "grok-build-0.1",
			apiKey:   xaiKey,
			skip:     xaiKey == "",
		},
		{
			name:     "Gemini/2.5-Flash-Lite",
			provider: "gemini",
			model:    "gemini-2.5-flash-lite",
			apiKey:   geminiKey,
			skip:     geminiKey == "",
		},
		{
			name:     "Gemini/2.5-Flash",
			provider: "gemini",
			model:    "gemini-2.5-flash",
			apiKey:   geminiKey,
			skip:     geminiKey == "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.skip {
				t.Skipf("Skipping %s: API key not set", tc.name)
			}

			cfg := llm.Config{
				Provider:    tc.provider,
				Model:       tc.model,
				APIKey:      tc.apiKey,
				MaxTokens:   100,
				Temperature: floatPtr(0.0),
				Timeout:     30 * time.Second,
			}

			client, err := llm.NewClient(cfg)
			if err != nil {
				t.Fatalf("Failed to create client: %v", err)
			}
			defer client.Close()

			if client.Provider() != tc.provider {
				t.Errorf("Expected provider %s, got %s", tc.provider, client.Provider())
			}

			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()

			req := llm.Request{
				Model:       tc.model,
				Messages:    []llm.Message{llm.NewTextMessage("user", TestPrompt)},
				MaxTokens:   100,
				Temperature: floatPtr(0.0),
			}

			resp, err := client.Complete(ctx, req)
			if err != nil {
				t.Fatalf("Completion failed: %v", err)
			}

			if resp.Content == "" {
				t.Error("Empty response content")
			}

			lower := strings.ToLower(resp.Content)
			if !strings.Contains(lower, "hello") && !strings.Contains(lower, "hi") {
				t.Errorf("Expected response containing 'hello' or 'hi', got: %s", resp.Content)
			}

			t.Logf("Provider: %s, Model: %s", client.Provider(), client.Model())
			t.Logf("Response: %s", resp.Content)
			t.Logf("Tokens: in=%d, out=%d", resp.InputTokens, resp.OutputTokens)
		})
	}
}

// TestResolveAuthHeader verifies auth header resolution.
func TestResolveAuthHeader(t *testing.T) {
	// With explicit AuthHeader
	cfg := llm.Config{
		Provider:   "openai",
		AuthHeader: "X-Custom-Auth",
	}
	header := llm.ResolveAuthHeader(cfg)
	if header != "X-Custom-Auth" {
		t.Errorf("ResolveAuthHeader() = %q, want X-Custom-Auth", header)
	}

	// Without explicit AuthHeader, uses preset
	cfg = llm.Config{Provider: "openai"}
	header = llm.ResolveAuthHeader(cfg)
	if header != "Bearer" {
		t.Errorf("ResolveAuthHeader() = %q, want Bearer", header)
	}

	// Unknown provider falls back to "Bearer"
	cfg = llm.Config{Provider: "unknown"}
	header = llm.ResolveAuthHeader(cfg)
	if header != "Bearer" {
		t.Errorf("ResolveAuthHeader() = %q, want Bearer", header)
	}
}

// TestResolveBaseURL verifies base URL resolution.
func TestResolveBaseURL(t *testing.T) {
	// Explicit BaseURL takes precedence
	cfg := llm.Config{
		Provider: "anthropic",
		BaseURL:  "https://custom.example.com",
	}
	url := llm.ResolveBaseURL(cfg)
	if url != "https://custom.example.com" {
		t.Errorf("ResolveBaseURL() = %q, want https://custom.example.com", url)
	}

	// Falls back to preset
	cfg = llm.Config{Provider: "anthropic"}
	url = llm.ResolveBaseURL(cfg)
	if url != "https://api.anthropic.com/v1" {
		t.Errorf("ResolveBaseURL() = %q, want https://api.anthropic.com/v1", url)
	}

	// Unknown provider without BaseURL
	cfg = llm.Config{Provider: "unknown"}
	url = llm.ResolveBaseURL(cfg)
	if url != "" {
		t.Errorf("ResolveBaseURL() = %q, want empty", url)
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

// api.z.ai/api/openai/v1 does not exist: it answers HTTP 200 with
// {"code":500,"msg":"404 NOT_FOUND"} for every request, so the zai preset
// could never complete a call. The documented OpenAI-compatible base is
// /api/paas/v4 (docs.z.ai; live-probed 2026-10-31: 401 without a key).
func TestZaiPresetUsesDocumentedEndpoint(t *testing.T) {
	for _, provider := range []string{"zai", "z-ai", "glm"} {
		got := llm.ResolveBaseURL(llm.Config{Provider: provider})
		if got != "https://api.z.ai/api/paas/v4" {
			t.Errorf("ResolveBaseURL(%s) = %q, want https://api.z.ai/api/paas/v4", provider, got)
		}
	}
	// An explicit BaseURL (e.g. the coding plan) still wins.
	coding := "https://api.z.ai/api/coding/paas/v4"
	if got := llm.ResolveBaseURL(llm.Config{Provider: "zai", BaseURL: coding}); got != coding {
		t.Errorf("explicit BaseURL overridden: %q", got)
	}
}
