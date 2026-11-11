package llm

import (
	"slices"
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
		"gemini-3.5-flash":                   {},
		"gemini-3.1-pro-preview":             {},
		"gemini-3.1-pro-preview-customtools": {},
		"gemini-3.1-flash-lite":              {},
		"gemini-3-flash-preview":             {},
		"gemini-2.5-pro":                     {},
		"gemini-2.5-flash":                   {},
		"gemini-2.5-flash-lite":              {},
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

func TestCurrentGeminiCatalogExcludesShutdownPreview(t *testing.T) {
	shutdownPreview := "gemini-3.1-flash-lite-" + "preview"
	if _, ok := GetModelInfo(shutdownPreview); ok {
		t.Fatal("shutdown Gemini 3.1 Flash-Lite preview remains registered")
	}
	available := ModelsByProvider("gemini")
	listed := func(id string) bool {
		return slices.ContainsFunc(available, func(m ModelInfo) bool { return m.ID == id })
	}
	if listed(shutdownPreview) {
		t.Fatal("shutdown Gemini preview remains discoverable")
	}
	for _, model := range []string{
		"gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
	} {
		if _, ok := GetModelInfo(model); !ok || !listed(model) {
			t.Fatalf("current Gemini model %q is not registered and discoverable", model)
		}
	}
}

// TestComprehensiveThinkingFlags ensures that thinking flags are correctly and completely applied across the entire registry.
func TestComprehensiveThinkingFlags(t *testing.T) {
	for id, info := range modelRegistry {
		// 1. Mutually Exclusive Properties
		// A model cannot logically have both API-controlled thinking and Intrinsic reasoning simultaneously.
		if info.SupportsThinking && info.Thinking {
			t.Errorf("Architectural conflict: Model %s has both SupportsThinking (API-controlled) and Thinking (Intrinsic) set to true.", id)
		}

		// 2. Claude Models (API-Controlled)
		// All non-Haiku Claude models must support API-controlled thinking
		if strings.HasPrefix(id, "claude-") && !strings.Contains(id, "haiku") {
			if !info.SupportsThinking {
				t.Errorf("Model %s should have SupportsThinking=true", id)
			}
		}

		// 3. Grok Reasoning Models (Intrinsic)
		// Any Grok model labeled "reasoning" (and not "non-reasoning") must have intrinsic thinking.
		if strings.HasPrefix(id, "grok-") && strings.Contains(id, "reasoning") && !strings.Contains(id, "non-reasoning") {
			if !info.Thinking {
				t.Errorf("Grok reasoning model %s must have Thinking=true", id)
			}
		}

		// 4. DeepSeek Reasoning Models (Intrinsic)
		if strings.HasPrefix(id, "deepseek-r1") {
			if !info.Thinking {
				t.Errorf("DeepSeek model %s must have Thinking=true", id)
			}
		}

		// Skip non-chat Gemini modality SKUs (embeddings / image gen) — they do not
		// participate in chat thinking controls.
		isGeminiModalityOnly := info.Capabilities != 0 &&
			!info.Capabilities.Supports(CapabilityChat) &&
			(info.Capabilities.Supports(CapabilityEmbeddings) || info.Capabilities.Supports(CapabilityImageGeneration) ||
				info.Capabilities.Supports(CapabilityTranscription))

		// 5. Gemini 2.5 chat models think intrinsically (Thinking=true, not SupportsThinking)
		if strings.HasPrefix(id, "gemini-2.5") && !isGeminiModalityOnly {
			if !info.Thinking {
				t.Errorf("Model %s (Gemini 2.5) should have Thinking=true", id)
			}
			if info.SupportsThinking {
				t.Errorf("Model %s (Gemini 2.5) should not have SupportsThinking (API rejects thinkingConfig)", id)
			}
		}

		// 6. Gemini 3+ chat models expose API-controlled thinkingLevel.
		if isGemini3Plus(id) && !isGeminiModalityOnly {
			if !info.SupportsThinking {
				t.Errorf("Model %s (Gemini 3+) should have SupportsThinking=true", id)
			}
			if info.Thinking {
				t.Errorf("Model %s (Gemini 3+) should not use intrinsic-only Thinking", id)
			}
		}

		// 7. Explicit Negatives (Models that should NEVER have either)
		// Haiku 4.5+ supports extended thinking; only Claude 3 Haiku does not.
		isLegacyHaiku := strings.Contains(id, "claude-3-haiku")
		isGeminiWithoutThinking := strings.HasPrefix(id, "gemini-") &&
			!strings.HasPrefix(id, "gemini-2.5") && !isGemini3Plus(id)
		if isGeminiWithoutThinking || isLegacyHaiku || strings.Contains(id, "non-reasoning") {
			if info.SupportsThinking || info.Thinking {
				t.Errorf("Model %s should not have any thinking flags set", id)
			}
		}
	}
}

// isGemini3Plus reports a Gemini 3.x or later model ID (gemini-3*, gemini-4*):
// the generations with API-controlled thinkingLevel and thought signatures.
func isGemini3Plus(id string) bool {
	return strings.HasPrefix(id, "gemini-3") || strings.HasPrefix(id, "gemini-4")
}

// TestThoughtSignatureFlags ensures ThoughtSignature is set correctly across the registry.
// Gemini 3.x models require thought_signature in function call responses; older models do not.
func TestThoughtSignatureFlags(t *testing.T) {
	for id, info := range modelRegistry {
		isGeminiModalityOnly := info.Capabilities != 0 &&
			!info.Capabilities.Supports(CapabilityChat) &&
			(info.Capabilities.Supports(CapabilityEmbeddings) || info.Capabilities.Supports(CapabilityImageGeneration) ||
				info.Capabilities.Supports(CapabilityTranscription))
		isGemini3Chat := isGemini3Plus(id) && !isGeminiModalityOnly
		if isGemini3Chat && !info.ThoughtSignature {
			t.Errorf("Model %s (Gemini 3.x chat) should have ThoughtSignature=true", id)
		}
		if info.Provider == "gemini" && !isGemini3Chat && info.ThoughtSignature {
			t.Errorf("Model %s (Gemini non-chat-3.x) should have ThoughtSignature=false", id)
		}
		// Non-Gemini models should never have ThoughtSignature
		if info.Provider != "gemini" && info.ThoughtSignature {
			t.Errorf("Non-Gemini model %s should have ThoughtSignature=false", id)
		}
	}
}
