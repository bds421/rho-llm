package llm_test

import (
	"testing"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider"
)

func TestModelsNonEmptySortedUnique(t *testing.T) {
	models := llm.Models()
	if len(models) == 0 {
		t.Fatal("Models() is empty")
	}
	seen := map[string]bool{}
	for i, m := range models {
		if m.ID == "" {
			t.Errorf("model[%d] has empty ID", i)
		}
		if seen[m.ID] {
			t.Errorf("duplicate model ID %q", m.ID)
		}
		seen[m.ID] = true
		if i > 0 && models[i-1].ID > m.ID {
			t.Errorf("Models() not sorted at %d: %q > %q", i, models[i-1].ID, m.ID)
		}
	}
}

func TestModelsByProviderUnknownIsEmptyNotNilPanic(t *testing.T) {
	got := llm.ModelsByProvider("does-not-exist")
	if got == nil {
		t.Error("ModelsByProvider(unknown) returned nil, want empty slice")
	}
	if len(got) != 0 {
		t.Errorf("ModelsByProvider(unknown) = %d models, want 0", len(got))
	}
}

func TestModelsByProviderConsistent(t *testing.T) {
	for _, m := range llm.ModelsByProvider("anthropic") {
		if m.Provider != "anthropic" {
			t.Errorf("ModelsByProvider(anthropic) returned %q model %q", m.Provider, m.ID)
		}
	}
	if len(llm.ModelsByProvider("anthropic")) == 0 {
		t.Error("expected anthropic models")
	}
}

func TestProvidersIncludesKnownSorted(t *testing.T) {
	ps := llm.Providers()
	want := map[string]bool{"anthropic": false, "openai": false, "gemini": false, "ollama": false}
	for i, p := range ps {
		if _, ok := want[p]; ok {
			want[p] = true
		}
		if i > 0 && ps[i-1] > p {
			t.Errorf("Providers() not sorted at %d: %q > %q", i, ps[i-1], p)
		}
	}
	for p, found := range want {
		if !found {
			t.Errorf("Providers() missing %q", p)
		}
	}
}

// TestAvailableModels verifies the available models list.
func TestAvailableModels(t *testing.T) {
	providers := []string{"anthropic", "xai", "gemini"}

	for _, provider := range providers {
		models := llm.ModelsByProvider(provider)
		if len(models) == 0 {
			t.Errorf("Empty model list for provider %s", provider)
		}

		t.Logf("%s models: %v", provider, models)
	}
}
