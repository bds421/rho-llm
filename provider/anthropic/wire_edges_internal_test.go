package anthropic

import (
	"encoding/json"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// Extended thinking accepts only tool_choice auto or none. The guard covered
// only adaptive models, so a budget-thinking model with a forced tool went out
// and came back as a 400 (B31).
func TestBudgetThinkingRefusesForcedToolChoice(t *testing.T) {
	c := &Client{config: llm.Config{Model: "claude-sonnet-4-6"}}
	tools := []llm.Tool{{Name: "f", InputSchema: map[string]any{"type": "object"}}}
	base := llm.Request{MaxTokens: 32000, Tools: tools, Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "x")}}
	for _, mode := range []llm.ToolChoiceMode{llm.ToolChoiceRequired, llm.ToolChoiceTool} {
		req := base
		req.ThinkingLevel = llm.ThinkingMedium
		req.ToolChoice = &llm.ToolChoice{Mode: mode}
		if mode == llm.ToolChoiceTool {
			req.ToolChoice.Name = "f"
		}
		if _, err := c.buildRequest(req, false); err == nil || !strings.Contains(err.Error(), "tool_choice") {
			t.Errorf("%s with budget thinking: err = %v, want a refusal", mode, err)
		}
		req.ThinkingLevel = llm.ThinkingNone // no thinking: forcing is fine
		if _, err := c.buildRequest(req, false); err != nil {
			t.Errorf("%s without thinking refused: %v", mode, err)
		}
	}
	req := base
	req.ThinkingLevel = llm.ThinkingMedium
	req.ToolChoice = &llm.ToolChoice{Mode: llm.ToolChoiceAuto}
	if _, err := c.buildRequest(req, false); err != nil {
		t.Errorf("auto with thinking refused: %v", err)
	}
}

// Whitespace-only text blocks are rejected by Anthropic, and a rich tool
// result of empty parts serialized "content": null (B31).
func TestEmptyBlocksStayOffTheWire(t *testing.T) {
	c := &Client{config: llm.Config{Model: "claude-sonnet-5-5"}}
	req := llm.Request{MaxTokens: 100, Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.ContentPart{{Type: llm.ContentText, Text: "q"}, {Type: llm.ContentText, Text: " \n\t"}}},
		{Role: llm.RoleAssistant, Content: []llm.ContentPart{{Type: llm.ContentToolUse, ToolUseID: "toolu_1", ToolName: "f", ToolInput: map[string]any{}}}},
		{Role: llm.RoleUser, Content: []llm.ContentPart{{Type: llm.ContentToolResult, ToolResultID: "toolu_1",
			ToolResultParts: []llm.ContentPart{{Type: llm.ContentText, Text: "  "}}}}},
	}}
	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(apiReq.Messages)
	if strings.Contains(string(b), `" \n\t"`) || strings.Contains(string(b), `"text":"  "`) {
		t.Errorf("whitespace-only text sent: %s", b)
	}
	if strings.Contains(string(b), `"content":null`) {
		t.Errorf(`"content":null sent: %s`, b)
	}
	if !strings.Contains(string(b), `"tool_use_id":"toolu_1"`) {
		t.Errorf("tool result lost: %s", b)
	}
}
