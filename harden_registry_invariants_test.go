package llm

// Break-the-system, Round 1 — cross-layer consistency of the registry data maps.
//
// A model-registry refresh is a hand-edit of four maps that reference each other
// (modelRegistry, availableModels, defaultModels, modelAliases) plus provider.go's
// presets. No existing test sweeps them globally, so a typo — an alias pointing at
// a model that was renamed/removed, a discovery-list entry that never got a registry
// row, a default that resolves to nothing, a provider-key mismatch — ships silently
// and only surfaces as a mispriced request or an empty client at runtime. These
// tests assert the invariants that must hold across ALL entries, not a hand-picked
// sample, so the next refresh can't quietly break one.

import (
	"testing"
)

// Every alias target must be a real, registered model ID. A dangling alias makes
// ResolveModelAlias hand a nonexistent model ID to the wire, and EstimateCost
// silently returns 0 for it.
func TestEveryAliasResolvesToRegisteredModel(t *testing.T) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	for alias, target := range modelAliases {
		if _, ok := modelRegistry[target]; !ok {
			t.Errorf("alias %q -> %q: target is not a registered model ID (dangling alias)", alias, target)
		}
	}
}

// Every entry in a provider's discovery list must be a real registered model ID.
// An availableModels entry with no registry row is undiscoverable metadata:
// GetModelInfo/EstimateCost/ProviderForModel all miss it.
func TestEveryAvailableModelIsRegistered(t *testing.T) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	for provider, ids := range availableModels {
		for _, id := range ids {
			if _, ok := modelRegistry[id]; !ok {
				t.Errorf("availableModels[%q] lists %q, which is not in modelRegistry", provider, id)
			}
		}
	}
}

// The map key of a discovery list must equal the embedded ModelInfo.Provider of
// every model in it. A mismatch means discovery says provider X but cost/routing
// metadata says Y — the refresh moved a model between providers and left it in the
// wrong list, or copied a row without fixing Provider.
func TestAvailableModelProviderMatchesRegistryProvider(t *testing.T) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	for provider, ids := range availableModels {
		for _, id := range ids {
			info, ok := modelRegistry[id]
			if !ok {
				continue // covered by TestEveryAvailableModelIsRegistered
			}
			if info.Provider != provider {
				t.Errorf("availableModels[%q] lists %q whose ModelInfo.Provider=%q — provider/discovery mismatch", provider, id, info.Provider)
			}
		}
	}
}

// A provider's default model must resolve (through aliases) to a registered model.
// GetDefaultModel is what NewClient uses when the caller omits a model; a default
// that points at a removed model sends a dead model ID on the very first request.
func TestEveryDefaultModelResolvesToRegistered(t *testing.T) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	for provider, def := range defaultModels {
		resolved := def
		if full, isAlias := modelAliases[def]; isAlias {
			resolved = full
		}
		if _, ok := modelRegistry[resolved]; !ok {
			t.Errorf("defaultModels[%q]=%q resolves to %q, not a registered model", provider, def, resolved)
		}
	}
}

// A default model must actually belong to the provider it is the default FOR —
// i.e. resolve to a model that talks to the SAME endpoint. defaultModels["deepseek"]
// ="deepseek-v4-flash" is a bug if that model's registry row says Provider:"openai":
// the client would send an OpenAI model ID to api.deepseek.com. Provider-synonym keys
// (claude/anthropic, grok/xai, google/gemini, kimi/moonshot, z-ai/glm/zai, qwen/
// dashscope, gpt/openai) are collapsed by comparing their preset BaseURL, so the test
// self-maintains as synonyms are added rather than tracking a hand list.
func TestDefaultModelBelongsToItsProvider(t *testing.T) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	// endpoint identity of a provider key: its preset BaseURL. "gpt" is an
	// OpenAI model-name synonym with no preset of its own — fold it into openai.
	baseFor := func(p string) (string, bool) {
		if p == "gpt" {
			p = "openai"
		}
		pr, ok := presets[p]
		return pr.BaseURL, ok
	}
	for provider, def := range defaultModels {
		resolved := def
		if full, isAlias := modelAliases[def]; isAlias {
			resolved = full
		}
		info, ok := modelRegistry[resolved]
		if !ok {
			continue // covered by TestEveryDefaultModelResolvesToRegistered
		}
		keyBase, keyOK := baseFor(provider)
		modelBase, modelOK := baseFor(info.Provider)
		if !keyOK || !modelOK {
			continue // a preset-less key is covered by TestProviderWithModelsHasPreset
		}
		if keyBase != modelBase {
			t.Errorf("defaultModels[%q]=%q resolves to a %q model (endpoint %s), but %q's endpoint is %s — default points at the wrong provider",
				provider, def, info.Provider, modelBase, provider, keyBase)
		}
	}
}

// No alias key may collide with a real model ID. ResolveModelAlias checks aliases
// FIRST, so an alias named after a real model silently shadows it — every request
// for the real model is redirected to the alias target. RegisterModelAlias rejects
// this at runtime (registry.go:534); the built-in table must obey the same rule.
func TestNoAliasShadowsARealModelID(t *testing.T) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	for alias := range modelAliases {
		if _, ok := modelRegistry[alias]; ok {
			t.Errorf("alias key %q is also a registered model ID — it shadows the real model on every lookup", alias)
		}
	}
}

// A registry row's map key must equal its own ModelInfo.ID. GetModelInfo keys by
// the map key but callers read info.ID back and send THAT on the wire; a mismatch
// (copy/paste that renamed the key but not the ID) sends the wrong model.
func TestRegistryKeyEqualsModelInfoID(t *testing.T) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	for key, info := range modelRegistry {
		if info.ID != key {
			t.Errorf("modelRegistry[%q].ID=%q — key and embedded ID disagree", key, info.ID)
		}
	}
}

// A provider's discovery list must not contain duplicates. The refresh reorders
// and rewrites these slices by hand; a duplicated ID inflates the list and can
// desync RegisterModel's slices.Contains dedup guard.
func TestNoDuplicateInAvailableModels(t *testing.T) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	for provider, ids := range availableModels {
		seen := make(map[string]bool, len(ids))
		for _, id := range ids {
			if seen[id] {
				t.Errorf("availableModels[%q] lists %q more than once", provider, id)
			}
			seen[id] = true
		}
	}
}

// Every provider that has a default model or a discovery list must have a preset
// (base URL + protocol) in provider.go — otherwise NewClient(provider) with no
// explicit BaseURL builds a client with an empty endpoint. This catches a new
// provider added to registry.go but never wired into presets, and vice-versa a
// default whose provider key was misspelled.
func TestProviderWithModelsHasPreset(t *testing.T) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	check := func(provider string) {
		// gpt/qwen/z-ai/glm are alias keys for openai/dashscope/zai — presets exist
		// under the canonical name; skip the synonyms.
		switch provider {
		case "gpt", "qwen", "z-ai", "glm":
			return
		}
		if _, ok := presets[provider]; !ok {
			t.Errorf("provider %q has registry data but no preset in provider.go — NewClient(%q) has no BaseURL", provider, provider)
		}
	}
	for provider, ids := range availableModels {
		if len(ids) > 0 {
			check(provider)
		}
	}
	for provider := range defaultModels {
		check(provider)
	}
}

// A registered model whose Provider names a real preset provider should be
// discoverable via that provider's list. Catches the reverse gap: a row added to
// modelRegistry (so it prices/routes) but forgotten in availableModels, so no user
// can enumerate it. Scoped to models added in this refresh window to stay focused.
func TestNewlyAddedModelsAreDiscoverable(t *testing.T) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	added := []string{
		"grok-build-0.1", "qwen/qwen3.6-27b", "deepseek-v4-flash", "deepseek-v4-pro",
		"command-a-plus-05-2026", "qwen3.7-max", "qwen3.7-plus",
		"gemma4:12b", "gemma4:e2b",
	}
	for _, id := range added {
		info, ok := modelRegistry[id]
		if !ok {
			t.Errorf("expected refreshed model %q to be registered", id)
			continue
		}
		found := false
		for _, listed := range availableModels[info.Provider] {
			if listed == id {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("model %q (provider %q) is registered but absent from availableModels[%q] — undiscoverable", id, info.Provider, info.Provider)
		}
	}
}

// Sanity: the removed models must be gone from EVERY map, not just the registry.
// A half-removed model (dropped from modelRegistry but left in availableModels or
// an alias) is exactly the dangling reference the other invariants catch generally;
// this pins the specific IDs this refresh retired so a future re-add is deliberate.
func TestRetiredModelsFullyRemoved(t *testing.T) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	retired := []string{
		"deepseek-chat", "llama-3.3-70b-versatile", "llama-3.1-8b-instant",
		"meta-llama/llama-4-scout-17b-16e-instruct",
	}
	for _, id := range retired {
		if _, ok := modelRegistry[id]; ok {
			t.Errorf("retired model %q is still in modelRegistry", id)
		}
		for provider, ids := range availableModels {
			for _, listed := range ids {
				if listed == id {
					t.Errorf("retired model %q still listed in availableModels[%q]", id, provider)
				}
			}
		}
		for alias, target := range modelAliases {
			if target == id {
				t.Errorf("alias %q still targets retired model %q", alias, id)
			}
		}
		for provider, def := range defaultModels {
			if def == id {
				t.Errorf("defaultModels[%q] still points at retired model %q", provider, id)
			}
		}
	}
	// The retired Llama aliases must be gone (they had no replacement).
	for _, deadAlias := range []string{"llama", "llama-70b", "llama-8b", "llama4", "llama-4-scout"} {
		if _, ok := modelAliases[deadAlias]; ok {
			t.Errorf("alias %q should have been removed (Groq dropped all Llama models)", deadAlias)
		}
	}
}
