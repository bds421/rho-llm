package llm

import (
	"slices"
	"strings"
	"testing"
)

func TestCurrentGeminiCatalogExcludesShutdownPreview(t *testing.T) {
	shutdownPreview := "gemini-3.1-flash-lite-" + "preview"
	if _, ok := GetModelInfo(shutdownPreview); ok {
		t.Fatal("shutdown Gemini 3.1 Flash-Lite preview remains registered")
	}
	available := GetAvailableModels("gemini")
	if slices.Contains(available, shutdownPreview) {
		t.Fatal("shutdown Gemini preview remains discoverable")
	}
	for _, model := range []string{
		"gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
	} {
		if _, ok := GetModelInfo(model); !ok || !slices.Contains(available, model) {
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
			(info.Capabilities.Supports(CapabilityEmbeddings) || info.Capabilities.Supports(CapabilityImageGeneration))

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
		if strings.HasPrefix(id, "gemini-3") && !isGeminiModalityOnly {
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
			!strings.HasPrefix(id, "gemini-2.5") && !strings.HasPrefix(id, "gemini-3")
		if isGeminiWithoutThinking || isLegacyHaiku || strings.Contains(id, "non-reasoning") {
			if info.SupportsThinking || info.Thinking {
				t.Errorf("Model %s should not have any thinking flags set", id)
			}
		}
	}
}

// TestThoughtSignatureFlags ensures ThoughtSignature is set correctly across the registry.
// Gemini 3.x models require thought_signature in function call responses; older models do not.
func TestThoughtSignatureFlags(t *testing.T) {
	for id, info := range modelRegistry {
		isGeminiModalityOnly := info.Capabilities != 0 &&
			!info.Capabilities.Supports(CapabilityChat) &&
			(info.Capabilities.Supports(CapabilityEmbeddings) || info.Capabilities.Supports(CapabilityImageGeneration))
		isGemini3Chat := strings.HasPrefix(id, "gemini-3") && !isGeminiModalityOnly
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
