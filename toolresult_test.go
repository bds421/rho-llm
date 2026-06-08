package llm_test

import (
	"testing"

	llm "github.com/bds421/rho-llm"
)

func TestToolResultTextDegrade(t *testing.T) {
	// Rich tool result: text + image. Providers without image support get text only.
	part := llm.ContentPart{
		Type:         llm.ContentToolResult,
		ToolResultID: "c1",
		ToolResultParts: []llm.ContentPart{
			{Type: llm.ContentText, Text: "line1"},
			{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "image/png", Data: "AAAA"}},
			{Type: llm.ContentText, Text: "line2"},
		},
	}
	if got := part.ToolResultText(); got != "line1\nline2" {
		t.Errorf("ToolResultText() = %q, want %q (image must be dropped)", got, "line1\nline2")
	}

	// Plain ToolResultContent takes precedence / works as before.
	plain := llm.ContentPart{Type: llm.ContentToolResult, ToolResultContent: "plain"}
	if got := plain.ToolResultText(); got != "plain" {
		t.Errorf("ToolResultText() plain = %q", got)
	}

	// Image-only result degrades to empty string (no panic).
	imgOnly := llm.ContentPart{Type: llm.ContentToolResult, ToolResultParts: []llm.ContentPart{
		{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "image/png", Data: "AAAA"}},
	}}
	if got := imgOnly.ToolResultText(); got != "" {
		t.Errorf("image-only ToolResultText() = %q, want empty", got)
	}
}

func TestNewToolResultParts(t *testing.T) {
	msg := llm.NewToolResultParts("call_9", true,
		llm.ContentPart{Type: llm.ContentText, Text: "err"},
	)
	if msg.Role != llm.RoleUser || len(msg.Content) != 1 {
		t.Fatalf("unexpected message: %+v", msg)
	}
	p := msg.Content[0]
	if p.Type != llm.ContentToolResult || p.ToolResultID != "call_9" || !p.IsError {
		t.Errorf("tool result part wrong: %+v", p)
	}
	if len(p.ToolResultParts) != 1 || p.ToolResultParts[0].Text != "err" {
		t.Errorf("parts not set: %+v", p.ToolResultParts)
	}
}
