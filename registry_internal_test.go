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
		if strings.Contains(id, "gemini-") || strings.Contains(id, "haiku") || strings.Contains(id, "non-reasoning") {
			if info.SupportsThinking || info.Thinking {
				t.Errorf("Model %s should not have any thinking flags set", id)
			}
		}
	}
}
