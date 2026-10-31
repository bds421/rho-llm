package llm_test

import (
	"testing"

	llm "github.com/bds421/rho-llm"
)

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
