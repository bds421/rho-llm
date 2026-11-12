package gemini

import (
	"encoding/json"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

func functionResponseNames(t *testing.T, msgs []llm.Message) []string {
	t.Helper()
	c := &Client{config: llm.Config{Model: "gemini-3.8-flash"}, baseURL: defaultGeminiBase, providerName: "gemini"}
	apiReq, err := c.buildRequest(llm.Request{Messages: msgs})
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, content := range apiReq.Contents {
		for _, p := range content.Parts {
			if p.FunctionResponse != nil {
				names = append(names, p.FunctionResponse.Name)
			}
		}
	}
	return names
}

func toolTurn(provider, id, name, result string) []llm.Message {
	return []llm.Message{
		{Role: llm.RoleAssistant, Provider: provider, Content: []llm.ContentPart{
			{Type: llm.ContentToolUse, ToolUseID: id, ToolName: name, ToolInput: map[string]any{"city": "Graz"}},
		}},
		llm.NewToolResultMessage(id, result, false),
	}
}

// Gemini numbers its synthetic IDs from 0 in every response, so a session that
// calls the same tool first in two turns gets call_0_get_weather twice;
// normalization renames the second. functionResponse.name was parsed back out
// of the (renamed) ID and became "get_weather#1" (B6).
func TestFunctionResponseNameSurvivesDedup(t *testing.T) {
	id := makeToolCallID(0, "get_weather")
	msgs := []llm.Message{llm.NewTextMessage(llm.RoleUser, "weather twice")}
	msgs = append(msgs, toolTurn("gemini", id, "get_weather", "sunny")...)
	msgs = append(msgs, toolTurn("gemini", id, "get_weather", "rainy")...)
	msgs = llm.NormalizeForProvider(msgs, "gemini")

	names := functionResponseNames(t, msgs)
	if len(names) != 2 {
		t.Fatalf("functionResponse names = %v, want two", names)
	}
	for _, n := range names {
		if n != "get_weather" {
			t.Fatalf("functionResponse names = %v, want get_weather for both", names)
		}
	}
}

// After a handoff the call IDs are another provider's (toolu_…, call_abc…);
// they were sent as the function name (B6).
func TestFunctionResponseNameAfterHandoff(t *testing.T) {
	for _, foreignID := range []string{"toolu_01AbCdEf", "call_abc123", "call_7_other_tool"} {
		msgs := []llm.Message{llm.NewTextMessage(llm.RoleUser, "x")}
		msgs = append(msgs, toolTurn("anthropic", foreignID, "get_weather", "sunny")...)
		msgs = llm.NormalizeForProvider(msgs, "gemini")
		names := functionResponseNames(t, msgs)
		if len(names) != 1 || names[0] != "get_weather" {
			t.Fatalf("id %q: functionResponse names = %v, want [get_weather]", foreignID, names)
		}
	}
}

// Parallel calls in one response keep their own names.
func TestFunctionResponseNamesForParallelCalls(t *testing.T) {
	msgs := []llm.Message{
		llm.NewTextMessage(llm.RoleUser, "x"),
		{Role: llm.RoleAssistant, Provider: "gemini", Content: []llm.ContentPart{
			{Type: llm.ContentToolUse, ToolUseID: makeToolCallID(0, "a"), ToolName: "a", ToolInput: map[string]any{}},
			{Type: llm.ContentToolUse, ToolUseID: makeToolCallID(1, "b"), ToolName: "b", ToolInput: map[string]any{}},
		}},
		{Role: llm.RoleUser, Content: []llm.ContentPart{
			{Type: llm.ContentToolResult, ToolResultID: makeToolCallID(1, "b"), ToolResultContent: "rb"},
			{Type: llm.ContentToolResult, ToolResultID: makeToolCallID(0, "a"), ToolResultContent: "ra"},
		}},
	}
	names := functionResponseNames(t, llm.NormalizeForProvider(msgs, "gemini"))
	if strings.Join(names, ",") != "b,a" {
		t.Fatalf("functionResponse names = %v, want [b a]", names)
	}
}

// A result whose call is not in the request (a caller-built transcript) still
// falls back to parsing the synthetic ID.
func TestFunctionResponseNameFallsBackToTheSyntheticID(t *testing.T) {
	msgs := []llm.Message{llm.NewToolResultMessage(makeToolCallID(3, "lookup"), "r", false)}
	names := functionResponseNames(t, msgs)
	if len(names) != 1 || names[0] != "lookup" {
		b, _ := json.Marshal(names)
		t.Fatalf("functionResponse names = %s, want [lookup]", b)
	}
}

// A failed tool's result was sent like a success: IsError was dropped (B30).
func TestFailedToolResultGoesUnderError(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gemini-3.8-flash"}, baseURL: defaultGeminiBase, providerName: "gemini"}
	msgs := []llm.Message{
		llm.NewTextMessage(llm.RoleUser, "x"),
		{Role: llm.RoleAssistant, Provider: "gemini", Content: []llm.ContentPart{
			{Type: llm.ContentToolUse, ToolUseID: "call_0_a", ToolName: "a", ToolInput: map[string]any{}},
			{Type: llm.ContentToolUse, ToolUseID: "call_1_b", ToolName: "b", ToolInput: map[string]any{}},
		}},
		{Role: llm.RoleUser, Content: []llm.ContentPart{
			{Type: llm.ContentToolResult, ToolResultID: "call_0_a", ToolResultContent: "timeout", IsError: true},
			{Type: llm.ContentToolResult, ToolResultID: "call_1_b", ToolResultContent: "42"},
		}},
	}
	apiReq, err := c.buildRequest(llm.Request{Messages: msgs})
	if err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(apiReq.Contents[2])
	if !strings.Contains(string(b), `"name":"a","response":{"error":"timeout"}`) || !strings.Contains(string(b), `"name":"b","response":{"result":"42"}`) {
		t.Fatalf("function responses = %s", b)
	}
}
