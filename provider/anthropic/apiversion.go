package anthropic

import (
	"fmt"

	llm "github.com/bds421/rho-llm"
)

// strictVersionContent refuses the message content that API version
// 2026-10-21 rejects with a 400: a document directly in an assistant message,
// and a text block with citations in a user message (also inside a tool
// result). Citations on assistant text, as Claude returned them, stay.
func strictVersionContent(msgs []llm.Message) error {
	for i, msg := range msgs {
		for _, part := range msg.Content {
			switch {
			case msg.Role == llm.RoleAssistant && part.Type == llm.ContentDocument:
				return fmt.Errorf("anthropic: message %d: API version %s rejects a document in an assistant message; send it in a user message", i, llm.AnthropicVersion20261021)
			case msg.Role == llm.RoleUser && hasCitations(part):
				return fmt.Errorf("anthropic: message %d: API version %s rejects citations on user text; send the text without Citations", i, llm.AnthropicVersion20261021)
			}
		}
	}
	return nil
}

func hasCitations(part llm.ContentPart) bool {
	if len(part.Citations) > 0 {
		return true
	}
	for _, sub := range part.ToolResultParts {
		if len(sub.Citations) > 0 {
			return true
		}
	}
	return false
}
