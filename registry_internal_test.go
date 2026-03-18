package llm

import (
	"strings"
	"testing"
)

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

		// 5. Explicit Negatives (Models that should NEVER have either)
		// Haiku 4.5+ supports extended thinking; only Claude 3 Haiku does not.
		isLegacyHaiku := strings.Contains(id, "claude-3-haiku")
		if strings.Contains(id, "gemini-") || isLegacyHaiku || strings.Contains(id, "non-reasoning") {
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
		isGemini3 := strings.HasPrefix(id, "gemini-3")
		if isGemini3 && !info.ThoughtSignature {
			t.Errorf("Model %s (Gemini 3.x) should have ThoughtSignature=true", id)
		}
		if info.Provider == "gemini" && !isGemini3 && info.ThoughtSignature {
			t.Errorf("Model %s (Gemini non-3.x) should have ThoughtSignature=false", id)
		}
		// Non-Gemini models should never have ThoughtSignature
		if info.Provider != "gemini" && info.ThoughtSignature {
			t.Errorf("Non-Gemini model %s should have ThoughtSignature=false", id)
		}
	}
}
