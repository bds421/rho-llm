package llm

import "sort"

// Models returns metadata for every model in the registry, sorted by ID. Use it
// to enumerate or filter the registry programmatically — e.g. by capability:
//
//	for _, m := range llm.Models() {
//	    if m.SupportsThinking { ... }
//	}
func Models() []ModelInfo {
	registryMu.RLock()
	out := make([]ModelInfo, 0, len(modelRegistry))
	for _, m := range modelRegistry {
		out = append(out, m)
	}
	registryMu.RUnlock()
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// ModelsByProvider returns the registered models of one provider, sorted by ID.
// A preset that shares a provider's catalog lists that catalog: "dashscope-cn"
// and "qwen" list DashScope's models, "kimi-cn" Moonshot's. An unknown provider
// yields an empty (non-nil) slice.
func ModelsByProvider(provider string) []ModelInfo {
	family := canonicalProvider(provider)
	registryMu.RLock()
	out := make([]ModelInfo, 0)
	for _, m := range modelRegistry {
		if canonicalProvider(m.Provider) == family {
			out = append(out, m)
		}
	}
	registryMu.RUnlock()
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// Providers returns the sorted list of known provider names (those with a built-in
// preset). Unknown providers can still be used by setting Config.BaseURL.
func Providers() []string {
	out := make([]string, 0, len(presets))
	for name := range presets {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}
