package llm_test

import (
	"context"
	"testing"

	llm "github.com/bds421/rho-llm"
)

func thinkingTurn(provider, model string) llm.Message {
	return llm.Message{Role: llm.RoleAssistant, Provider: provider, Model: model, Content: []llm.ContentPart{
		{Type: llm.ContentThinking, Thinking: "old reasoning", ThinkingSignature: "sig"},
		{Type: llm.ContentText, Text: "answer"},
	}}
}

func hasText(m llm.Message, s string) bool {
	for _, p := range m.Content {
		if p.Text == s || p.Thinking == s {
			return true
		}
	}
	return false
}

func TestNormalizeForModelDropsOtherModelsThinking(t *testing.T) {
	for _, tc := range []struct {
		name            string
		fromP, fromM    string
		toP, toM        string
		keepThinking    bool // verbatim thinking block
		keepAsText      bool // degraded to text (provider rules)
		droppedEntirely bool
	}{
		{"same model", "anthropic", "claude-haiku-5-5", "anthropic", "claude-haiku-5-5", true, false, false},
		{"served snapshot ID", "anthropic", "claude-haiku-4-5-20251001", "anthropic", "claude-haiku-4-5", true, false, false},
		{"same provider, other model", "anthropic", "claude-opus-5-5", "anthropic", "claude-haiku-5-5", false, false, true},
		{"other provider and model", "gemini", "gemini-3.8-flash", "anthropic", "claude-haiku-5-5", false, false, true},
		{"family member is not a snapshot", "openai", "gpt-5", "openai", "gpt-5-mini", false, false, true},
		{"minor version is not a snapshot", "anthropic", "claude-opus-4", "anthropic", "claude-opus-4-8", false, false, true},
		{"no recorded model keeps provider rules", "gemini", "", "anthropic", "claude-haiku-5-5", false, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out := llm.NormalizeForModel([]llm.Message{thinkingTurn(tc.fromP, tc.fromM)}, tc.toP, tc.toM)
			if len(out) != 1 || !hasText(out[0], "answer") {
				t.Fatalf("visible answer lost: %+v", out)
			}
			var block, text bool
			for _, p := range out[0].Content {
				if p.Type == llm.ContentThinking {
					block = true
				}
				if p.Type == llm.ContentText && p.Text == "old reasoning" {
					text = true
				}
			}
			if block != tc.keepThinking || text != tc.keepAsText {
				t.Fatalf("thinking block=%v as-text=%v, want %v/%v: %+v", block, text, tc.keepThinking, tc.keepAsText, out[0].Content)
			}
			if tc.droppedEntirely && (block || text) {
				t.Fatalf("other model's reasoning survived: %+v", out[0].Content)
			}
		})
	}
}

func TestNormalizeForModelEmptyModelMatchesNormalizeForProvider(t *testing.T) {
	msgs := []llm.Message{thinkingTurn("gemini", "gemini-3.8-flash")}
	a := llm.NormalizeForModel(msgs, "anthropic", "")
	b := llm.NormalizeForProvider(msgs, "anthropic")
	if len(a) != len(b) || len(a[0].Content) != len(b[0].Content) {
		t.Fatalf("NormalizeForModel(model=\"\") = %+v, NormalizeForProvider = %+v", a, b)
	}
}

func TestSessionModelSwitchDropsPreviousModelsThinking(t *testing.T) {
	opus := &fakeClient{provider: "anthropic", model: "claude-opus-5-5", resp: &llm.Response{Content: "ok", StopReason: llm.StopEndTurn}}
	s := llm.NewSession(opus)
	s.Append(llm.NewTextMessage(llm.RoleUser, "q1"), thinkingTurn("anthropic", "claude-opus-5-5"))

	haiku := &fakeClient{provider: "anthropic", model: "claude-haiku-5-5", resp: &llm.Response{Content: "ok", StopReason: llm.StopEndTurn}}
	s.SwitchProvider(haiku)
	if _, err := s.Send(context.Background(), "q2"); err != nil {
		t.Fatal(err)
	}
	for _, m := range haiku.gotReq.Messages {
		for _, p := range m.Content {
			if p.Type == llm.ContentThinking || p.Text == "old reasoning" {
				t.Fatalf("Opus thinking replayed to Haiku: %+v", haiku.gotReq.Messages)
			}
		}
	}

	// Switching back to the same model keeps its own signed thinking.
	opus2 := &fakeClient{provider: "anthropic", model: "claude-opus-5-5", resp: &llm.Response{Content: "ok", StopReason: llm.StopEndTurn}}
	s2 := llm.NewSession(opus2)
	s2.Append(llm.NewTextMessage(llm.RoleUser, "q1"), thinkingTurn("anthropic", "claude-opus-5-5"))
	if _, err := s2.Send(context.Background(), "q2"); err != nil {
		t.Fatal(err)
	}
	if !hasText(opus2.gotReq.Messages[1], "old reasoning") || opus2.gotReq.Messages[1].Content[0].Type != llm.ContentThinking {
		t.Fatalf("same-model thinking not replayed: %+v", opus2.gotReq.Messages)
	}
}
