package llm

import "strings"

// SetParts sets the response's content parts and derives the flat fields
// from them (Content, Thinking, its signature and redaction, ToolCalls), so
// the two views of an assistant turn cannot disagree.
func (r *Response) SetParts(parts []ContentPart) {
	r.Parts = parts
	var text, thinking strings.Builder
	r.ThinkingSignature, r.ThinkingRedacted, r.ToolCalls = "", false, nil
	for _, p := range parts {
		switch p.Type {
		case ContentText:
			text.WriteString(p.Text)
		case ContentThinking:
			thinking.WriteString(p.Thinking)
			if p.ThinkingSignature != "" {
				r.ThinkingSignature = p.ThinkingSignature
			}
			if p.Redacted {
				r.ThinkingRedacted = true
			}
		case ContentToolUse:
			r.ToolCalls = append(r.ToolCalls, ToolCall{ID: p.ToolUseID, Name: p.ToolName, Input: p.ToolInput})
		}
	}
	r.Content, r.Thinking = text.String(), thinking.String()
}
