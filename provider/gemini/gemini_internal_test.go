package gemini

import (
	"encoding/json"
	"testing"

	llm "gitlab2024.bds421-cloud.com/bds421/rho/llm"
)

// TestMakeToolCallIDResolveToolNameRoundTrip verifies that resolveToolName
// correctly inverts makeToolCallID for various tool name patterns.
func TestMakeToolCallIDResolveToolNameRoundTrip(t *testing.T) {
	tests := []struct {
		index int
		name  string
	}{
		{0, "get_weather"},
		{1, "search"},
		{0, "tool_with_many_underscores"},
		{5, "a"},
		{0, "123_numeric_prefix"},
		{0, "99"},
		{10, "deeply_nested_tool_name"},
	}

	for _, tc := range tests {
		id := makeToolCallID(tc.index, tc.name)
		got := resolveToolName(id)
		if got != tc.name {
			t.Errorf("roundtrip(%d, %q): makeToolCallID=%q, resolveToolName=%q, want %q",
				tc.index, tc.name, id, got, tc.name)
		}
	}
}

// TestResolveToolNameLegacyFormat verifies the legacy "call_<name>" format.
func TestResolveToolNameLegacyFormat(t *testing.T) {
	got := resolveToolName("call_my_tool")
	if got != "my_tool" {
		t.Errorf("resolveToolName(call_my_tool) = %q, want my_tool", got)
	}
}

// TestResolveToolNameNonSynthetic verifies non-call_ IDs pass through.
func TestResolveToolNameNonSynthetic(t *testing.T) {
	got := resolveToolName("toolu_01234abcdef")
	if got != "toolu_01234abcdef" {
		t.Errorf("resolveToolName(toolu_...) = %q, want passthrough", got)
	}
}

// TestBuildRequestEmptyTextPartOmitted verifies that ContentText parts with
// empty strings are not included in the Gemini request. An empty geminiPart{}
// serializes to {} due to omitempty, which violates the Gemini API's oneof
// constraint on the Part.data field (status 400: "required oneof field 'data'
// must have one initialized field").
func TestBuildRequestEmptyTextPartOmitted(t *testing.T) {
	c := &Client{providerName: "gemini"}

	req := llm.Request{
		Messages: []llm.Message{
			{
				Role: llm.RoleUser,
				Content: []llm.ContentPart{
					{Type: llm.ContentText, Text: "hello"},
				},
			},
			// Simulate an assistant message with empty text (e.g., assistant
			// response that had only tool calls, no text content).
			{
				Role: llm.RoleAssistant,
				Content: []llm.ContentPart{
					{Type: llm.ContentText, Text: ""},
				},
			},
			{
				Role: llm.RoleUser,
				Content: []llm.ContentPart{
					{Type: llm.ContentText, Text: "continue"},
				},
			},
		},
	}

	apiReq, err := c.buildRequest(req)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	// Verify no empty parts exist in any content message.
	for i, content := range apiReq.Contents {
		for j, part := range content.Parts {
			b, _ := json.Marshal(part)
			if string(b) == "{}" {
				t.Errorf("contents[%d].parts[%d] serializes to empty {}: %+v", i, j, part)
			}
		}
	}

	// The assistant message with only an empty text part should be dropped
	// entirely (the len(content.Parts) > 0 guard).
	if len(apiReq.Contents) != 2 {
		t.Errorf("expected 2 contents (empty assistant dropped), got %d", len(apiReq.Contents))
	}
}

// TestBuildRequestEmptyTextSystemInstructionOmitted verifies that empty text
// parts in system messages are not included in systemInstruction.
func TestBuildRequestEmptyTextSystemInstructionOmitted(t *testing.T) {
	c := &Client{providerName: "gemini"}

	req := llm.Request{
		Messages: []llm.Message{
			{
				Role: llm.RoleSystem,
				Content: []llm.ContentPart{
					{Type: llm.ContentText, Text: ""},
				},
			},
			{
				Role: llm.RoleUser,
				Content: []llm.ContentPart{
					{Type: llm.ContentText, Text: "hello"},
				},
			},
		},
	}

	apiReq, err := c.buildRequest(req)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	// System instruction should be nil when the only system part was empty.
	if apiReq.SystemInstruction != nil && len(apiReq.SystemInstruction.Parts) > 0 {
		for j, part := range apiReq.SystemInstruction.Parts {
			b, _ := json.Marshal(part)
			if string(b) == "{}" {
				t.Errorf("systemInstruction.parts[%d] serializes to empty {}: %+v", j, part)
			}
		}
	}
}
