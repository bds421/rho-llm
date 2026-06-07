package llm_test

import (
	"encoding/json"
	"testing"

	llm "github.com/bds421/rho-llm"
)

func TestConversationSerializationRoundTrip(t *testing.T) {
	conv := llm.NewConversation("be terse",
		llm.Tool{Name: "get_weather", Description: "weather", InputSchema: map[string]any{"type": "object"}},
	)
	conv.Append(llm.NewTextMessage(llm.RoleUser, "hi"))
	conv.Append(llm.Message{
		Role:     llm.RoleAssistant,
		Provider: "anthropic",
		Model:    "claude-sonnet-4-6",
		Content: []llm.ContentPart{
			{Type: llm.ContentThinking, Thinking: "hmm", ThinkingSignature: "sig123"},
			{Type: llm.ContentText, Text: "hello"},
		},
	})

	data, err := json.Marshal(conv)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	got, err := llm.LoadConversation(data)
	if err != nil {
		t.Fatalf("LoadConversation: %v", err)
	}
	if got.SchemaVersion != llm.ConversationSchemaVersion {
		t.Errorf("SchemaVersion = %d, want %d", got.SchemaVersion, llm.ConversationSchemaVersion)
	}
	if got.System != "be terse" {
		t.Errorf("System = %q", got.System)
	}
	if len(got.Tools) != 1 || got.Tools[0].Name != "get_weather" {
		t.Errorf("Tools not round-tripped: %+v", got.Tools)
	}
	if len(got.Messages) != 2 {
		t.Fatalf("len(Messages) = %d, want 2", len(got.Messages))
	}
	am := got.Messages[1]
	if am.Provider != "anthropic" || am.Model != "claude-sonnet-4-6" {
		t.Errorf("provenance lost: provider=%q model=%q", am.Provider, am.Model)
	}
	if len(am.Content) != 2 || am.Content[0].Type != llm.ContentThinking {
		t.Fatalf("content blocks not round-tripped: %+v", am.Content)
	}
	if am.Content[0].ThinkingSignature != "sig123" {
		t.Errorf("thinking signature lost: %q", am.Content[0].ThinkingSignature)
	}
}

func TestConversationToRequestPreservesBase(t *testing.T) {
	conv := llm.NewConversation("sys")
	conv.Append(llm.NewTextMessage(llm.RoleUser, "q"))

	base := llm.Request{Model: "claude-sonnet-4-6", MaxTokens: 1234, ThinkingLevel: llm.ThinkingLow}
	req := conv.ToRequest(base)

	if req.Model != "claude-sonnet-4-6" || req.MaxTokens != 1234 || req.ThinkingLevel != llm.ThinkingLow {
		t.Errorf("base fields not preserved: %+v", req)
	}
	if req.System != "sys" {
		t.Errorf("System not set from conversation: %q", req.System)
	}
	if len(req.Messages) != 1 {
		t.Errorf("Messages not set from conversation: %d", len(req.Messages))
	}
}

func TestConversationAddResponseAccumulatesUsage(t *testing.T) {
	conv := llm.NewConversation("")
	r1 := &llm.Response{Model: "claude-sonnet-4-6", Content: "a", InputTokens: 10, OutputTokens: 5}
	r2 := &llm.Response{Model: "claude-sonnet-4-6", Content: "b", InputTokens: 20, OutputTokens: 7}

	conv.AddResponse("anthropic", r1)
	conv.AddResponse("anthropic", r2)

	if len(conv.Messages) != 2 {
		t.Fatalf("len(Messages) = %d, want 2", len(conv.Messages))
	}
	if conv.Messages[0].Provider != "anthropic" {
		t.Errorf("provenance not set: %q", conv.Messages[0].Provider)
	}
	if conv.Usage.InputTokens != 30 || conv.Usage.OutputTokens != 12 {
		t.Errorf("usage not accumulated: in=%d out=%d", conv.Usage.InputTokens, conv.Usage.OutputTokens)
	}
	if conv.Usage.Cost < 0 {
		t.Errorf("cost should be non-negative, got %v", conv.Usage.Cost)
	}
}

func TestLoadConversationRejectsNewerSchema(t *testing.T) {
	data := []byte(`{"schema_version": 9999, "messages": []}`)
	if _, err := llm.LoadConversation(data); err == nil {
		t.Error("expected error for newer schema_version, got nil")
	}
}

func TestMarshalStampsSchemaVersionOnZeroValue(t *testing.T) {
	var conv llm.Conversation // zero value, SchemaVersion == 0
	data, err := json.Marshal(conv)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var raw struct {
		SchemaVersion int `json:"schema_version"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if raw.SchemaVersion != llm.ConversationSchemaVersion {
		t.Errorf("schema_version = %d, want %d", raw.SchemaVersion, llm.ConversationSchemaVersion)
	}
}
