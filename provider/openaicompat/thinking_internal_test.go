package openaicompat

import (
	"testing"

	llm "github.com/bds421/rho-llm"
)

// TestBuildRequestSkipsEmptyAssistantMessage verifies that an assistant turn
// which carried only a thinking block (which this protocol does not replay) does
// not produce an invalid {role:assistant, content:null} message — OpenAI rejects
// a message with null content and no tool_calls.
func TestBuildRequestSkipsEmptyAssistantMessage(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gpt-4"}, providerName: "openai"}

	req := llm.Request{
		MaxTokens: 100,
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "hi"),
			{Role: llm.RoleAssistant, Provider: "openai", Content: []llm.ContentPart{
				{Type: llm.ContentThinking, Thinking: "reasoning only"},
			}},
			llm.NewTextMessage(llm.RoleUser, "again"),
		},
	}

	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}
	for _, m := range apiReq.Messages {
		if m.Role == "assistant" && m.Content == nil && len(m.ToolCalls) == 0 {
			t.Errorf("emitted an empty assistant message: %+v", m)
		}
	}
}
