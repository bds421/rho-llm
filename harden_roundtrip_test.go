package llm_test

// Hardening pass 13 — adversarial Conversation serialization round-trip.

import (
	"encoding/json"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// The serialized form must round-trip losslessly even for hostile content:
// unicode, emoji, embedded control chars (incl. NUL), and every content kind.
// Invariant: marshal -> load -> marshal is idempotent (no field dropped/mangled).
func TestConversationRoundTripHostileContent(t *testing.T) {
	conv := llm.NewConversation("sys \x00\n\t 你好 🤖 \"quoted\" </script>")
	conv.Append(
		llm.NewTextMessage(llm.RoleUser, "emoji 🤖🔥 ctl \x01\x02 nl\nrtl ‮ wj‍"),
		llm.Message{Role: llm.RoleAssistant, Provider: "anthropic", Model: "claude-sonnet-4-6", StopReason: llm.StopToolUse,
			Content: []llm.ContentPart{
				{Type: llm.ContentThinking, Thinking: "reason 你好", ThinkingSignature: "sig=="},
				{Type: llm.ContentToolUse, ToolUseID: "c1\x00x", ToolName: "f", ToolInput: map[string]any{
					"k": "v\x00", "nested": map[string]any{"a": []any{"x", 1.5, true, nil}}, "uni": "🤖",
				}},
				{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "image/png", Data: "AAAA=="}},
			}},
		llm.NewToolResultMessage("c1\x00x", "result 🤖", true),
	)

	b1, err := json.Marshal(conv)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	loaded, err := llm.LoadConversation(b1)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	b2, err := json.Marshal(loaded)
	if err != nil {
		t.Fatalf("re-marshal: %v", err)
	}
	if string(b1) != string(b2) {
		t.Fatalf("round-trip not idempotent — a field was dropped/mangled:\n  first:  %s\n  second: %s", b1, b2)
	}
}
