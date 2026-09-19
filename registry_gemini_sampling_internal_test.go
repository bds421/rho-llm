package llm

import (
	"strings"
	"testing"
)

// Google removed sampling controls (temperature/top_p/top_k) from the July 2026
// GA Gemini models onward: those endpoints ignore or reject the parameter. The
// exclusion list is therefore load-bearing — a Gemini model registered without
// a decision about it silently advertises a capability the API does not have,
// and callers only find out when a live request 400s.
//
// This test fails when a new Gemini chat model is registered without being
// classified, so the decision cannot be skipped by omission.
func TestEveryRegisteredGeminiChatModelHasASamplingDecision(t *testing.T) {
	// Models predating the removal legitimately support sampling controls.
	legacySupported := map[string]struct{}{
		"gemini-3.5-flash":       {},
		"gemini-3.1-pro-preview": {},
		"gemini-3.1-flash-lite":  {},
		"gemini-3-pro-preview":   {},
		"gemini-3-flash-preview": {},
		"gemini-2.5-pro":         {},
		"gemini-2.5-flash":       {},
		"gemini-2.5-flash-lite":  {},
		"gemini-2.0-flash":       {},
	}

	seen := 0
	for id, info := range modelRegistry {
		if canonicalProvider(info.Provider) != "gemini" {
			continue
		}
		// Non-chat modalities (embeddings, image generation) never take a
		// sampling temperature and are not part of this decision.
		if info.Capabilities != 0 && !info.Capabilities.Supports(CapabilityChat) {
			continue
		}
		seen++
		_, denied := geminiWithoutSamplingControls[id]
		_, legacy := legacySupported[id]
		if !denied && !legacy {
			t.Errorf("gemini model %q is registered but not classified for sampling controls: "+
				"add it to geminiWithoutSamplingControls (Google removed the controls from the "+
				"July 2026 GA models onward) or to this test's legacySupported set", id)
		}
		if denied && legacy {
			t.Errorf("gemini model %q is in BOTH the denied and legacy-supported sets", id)
		}
		// The classification must actually drive the behaviour.
		if got := supportsSamplingTemperature(info); got == denied {
			t.Errorf("gemini model %q: supportsSamplingTemperature() = %t but denied = %t; "+
				"the list is not wired to the behaviour", id, got, denied)
		}
	}
	if seen < 8 {
		t.Fatalf("only %d gemini chat models inspected; the registry walk is probably broken", seen)
	}
}

// The denied list must not name a model that is not registered at all —
// otherwise a typo silently protects nothing.
func TestGeminiSamplingDenyListHasNoStaleEntries(t *testing.T) {
	for id := range geminiWithoutSamplingControls {
		if _, ok := modelRegistry[id]; !ok {
			t.Errorf("geminiWithoutSamplingControls names %q, which is not in modelRegistry: "+
				"either a typo or a model that was removed", id)
		}
		if !strings.HasPrefix(id, "gemini-") {
			t.Errorf("geminiWithoutSamplingControls names %q, which is not a Gemini ID", id)
		}
	}
}

// The three models added on 2026-09-19 are the reason this file exists; pin
// their classification explicitly so a later refactor cannot quietly flip them.
func TestSeptember2026GeminiAdditionsRejectSampling(t *testing.T) {
	for _, id := range []string{"gemini-3.7-flash", "gemini-3.8-flash"} {
		info, ok := modelRegistry[id]
		if !ok {
			t.Fatalf("%s is not registered", id)
		}
		if supportsSamplingTemperature(info) {
			t.Errorf("%s must reject a sampling temperature: Google removed the controls "+
				"from the July 2026 GA models onward", id)
		}
	}
	// Positive anchor: an older model must still accept it, so this is not a
	// blanket denial of every Gemini model.
	if info, ok := modelRegistry["gemini-2.5-flash"]; ok && !supportsSamplingTemperature(info) {
		t.Error("gemini-2.5-flash must still accept a sampling temperature")
	}
}
