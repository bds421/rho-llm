package anthropic

import (
	"testing"

	llm "github.com/bds421/rho-llm"
)

// TestBuildRequestToolResultImage verifies a rich tool result (text + image)
// serializes as an Anthropic tool_result content-block array.
func TestBuildRequestToolResultImage(t *testing.T) {
	c := &Client{config: llm.Config{Model: "claude-opus-4-5"}, providerName: "anthropic"}
	req := llm.Request{
		MaxTokens: 1024,
		Messages: []llm.Message{
			{Role: llm.RoleAssistant, Content: []llm.ContentPart{{Type: llm.ContentToolUse, ToolUseID: "c1", ToolName: "snap"}}},
			llm.NewToolResultParts("c1", false,
				llm.ContentPart{Type: llm.ContentText, Text: "here you go"},
				llm.ContentPart{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "image/png", Data: "AAAA"}},
			),
		},
	}
	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}
	tr := findToolResult(t, apiReq.Messages[1].Content)
	content, ok := tr["content"].([]any)
	if !ok {
		t.Fatalf("tool_result content should be an array, got %T", tr["content"])
	}
	var hasText, hasImage bool
	for _, raw := range content {
		b, _ := raw.(map[string]any)
		switch b["type"] {
		case "text":
			hasText = true
		case "image":
			hasImage = true
		}
	}
	if !hasText || !hasImage {
		t.Errorf("expected text+image blocks, got %+v", content)
	}
}

// TestBuildRequestToolResultInvalidImage verifies an invalid image in a tool
// result is rejected at build time.
func TestBuildRequestToolResultInvalidImage(t *testing.T) {
	c := &Client{config: llm.Config{Model: "claude-opus-4-5"}, providerName: "anthropic"}
	req := llm.Request{
		MaxTokens: 1024,
		Messages: []llm.Message{
			llm.NewToolResultParts("c1", false,
				llm.ContentPart{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "text/plain", Data: "x"}},
			),
		},
	}
	if _, err := c.buildRequest(req, false); err == nil {
		t.Error("invalid tool-result image should be rejected")
	}
}

func findToolResult(t *testing.T, blocks []any) map[string]any {
	t.Helper()
	for _, raw := range blocks {
		if b, ok := raw.(map[string]any); ok && b["type"] == "tool_result" {
			return b
		}
	}
	t.Fatalf("no tool_result block in %+v", blocks)
	return nil
}
