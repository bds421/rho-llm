package anthropic

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/bds421/rho-llm"
)

// anthropicBlock is one response content block. raw keeps the block exactly
// as received, so blocks the neutral model only describes (server tool calls
// and results, unknown types) can go back to the API unchanged.
type anthropicBlock struct {
	Type      string            `json:"type"`
	Text      string            `json:"text,omitempty"`
	ID        string            `json:"id,omitempty"`
	Name      string            `json:"name,omitempty"`
	Input     any               `json:"input,omitempty"`
	Thinking  string            `json:"thinking,omitempty"`
	Signature string            `json:"signature,omitempty"` // thinking-block signature (for replay)
	Data      string            `json:"data,omitempty"`      // redacted_thinking payload
	ToolUseID string            `json:"tool_use_id,omitempty"`
	Content   json.RawMessage   `json:"content,omitempty"`
	Citations []json.RawMessage `json:"citations,omitempty"`

	raw json.RawMessage
}

func (b *anthropicBlock) UnmarshalJSON(data []byte) error {
	type plain anthropicBlock
	var p plain
	if err := json.Unmarshal(data, &p); err != nil {
		return err
	}
	*b = anthropicBlock(p)
	b.raw = compact(data)
	return nil
}

// compact returns data in compact JSON form, the form encoding/json re-emits
// for a json.RawMessage, so a replayed block encodes the same on every turn.
func compact(data []byte) json.RawMessage {
	var buf bytes.Buffer
	if err := json.Compact(&buf, data); err != nil {
		return append(json.RawMessage(nil), data...)
	}
	return json.RawMessage(buf.Bytes())
}

// isServerToolResult reports whether a block type is the result of a
// provider-run tool (web_search_tool_result, web_fetch_tool_result,
// bash_code_execution_tool_result, …), as opposed to a client tool_result.
func isServerToolResult(blockType string) bool {
	return blockType != "tool_result" && strings.HasSuffix(blockType, "_tool_result")
}

// blockPart converts a response block into its neutral part.
func blockPart(b anthropicBlock) llm.ContentPart {
	switch b.Type {
	case "text":
		part := llm.ContentPart{Type: llm.ContentText, Text: b.Text}
		for _, raw := range b.Citations {
			if c, ok := citationOf(raw); ok {
				part.Citations = append(part.Citations, c)
			}
		}
		return part
	case "thinking":
		return llm.ContentPart{Type: llm.ContentThinking, Thinking: b.Thinking, ThinkingSignature: b.Signature, ProviderBlock: b.raw}
	case "redacted_thinking":
		return llm.ContentPart{Type: llm.ContentThinking, Thinking: b.Data, Redacted: true, ProviderBlock: b.raw}
	case "tool_use":
		return llm.ContentPart{Type: llm.ContentToolUse, ToolUseID: b.ID, ToolName: b.Name, ToolInput: b.Input}
	case "server_tool_use":
		return llm.ContentPart{
			Type: llm.ContentServerToolUse, ToolUseID: b.ID, ToolName: b.Name, ToolInput: b.Input,
			ProviderBlock: b.raw,
		}
	}
	if isServerToolResult(b.Type) {
		part := llm.ContentPart{
			Type: llm.ContentServerToolResult, BlockType: b.Type, ToolResultID: b.ToolUseID,
			ProviderBlock: b.raw,
		}
		decodeServerToolContent(&part, b.Content)
		return part
	}
	return llm.ContentPart{Type: llm.ContentProviderBlock, BlockType: b.Type, ProviderBlock: b.raw}
}

// decodeServerToolContent fills the typed result fields: a list of web
// search results, or the error object a failed call returns in place of the
// list. Anything else stays in ProviderBlock only.
func decodeServerToolContent(part *llm.ContentPart, content json.RawMessage) {
	content = bytes.TrimSpace(content)
	if len(content) == 0 {
		return
	}
	switch content[0] {
	case '[':
		if part.BlockType != "web_search_tool_result" {
			return
		}
		var results []llm.WebSearchResult
		if json.Unmarshal(content, &results) == nil {
			part.WebSearchResults = results
		}
	case '{':
		var e llm.ServerToolError
		if json.Unmarshal(content, &e) == nil && e.ErrorCode != "" {
			part.ServerToolError = &e
		}
	}
}

// replayBlock returns the wire block for a server tool or provider part: the
// block as the API sent it, or one rebuilt from the typed fields for a part
// made by hand. ok is false when there is nothing to send.
func replayBlock(part llm.ContentPart) (block any, ok bool, err error) {
	if len(part.ProviderBlock) > 0 {
		if !json.Valid(part.ProviderBlock) || bytes.TrimSpace(part.ProviderBlock)[0] != '{' {
			return nil, false, fmt.Errorf("anthropic: %s part has a ProviderBlock that is not a JSON object", part.Type)
		}
		return part.ProviderBlock, true, nil
	}
	switch part.Type {
	case llm.ContentServerToolUse:
		input := part.ToolInput
		if input == nil {
			input = map[string]any{}
		}
		return map[string]any{"type": "server_tool_use", "id": part.ToolUseID, "name": part.ToolName, "input": input}, true, nil
	case llm.ContentServerToolResult:
		if part.BlockType == "" {
			return nil, false, fmt.Errorf("anthropic: server tool result for %q has no BlockType", part.ToolResultID)
		}
		var content any = []any{}
		switch {
		case part.ServerToolError != nil:
			content = part.ServerToolError
		case len(part.WebSearchResults) > 0:
			results := make([]any, 0, len(part.WebSearchResults))
			for _, r := range part.WebSearchResults {
				results = append(results, webSearchResultWire(r))
			}
			content = results
		}
		return map[string]any{"type": part.BlockType, "tool_use_id": part.ToolResultID, "content": content}, true, nil
	}
	return nil, false, nil
}

func webSearchResultWire(r llm.WebSearchResult) map[string]any {
	block := map[string]any{"type": "web_search_result", "url": r.URL, "title": r.Title, "encrypted_content": r.EncryptedContent}
	if r.PageAge != "" {
		block["page_age"] = r.PageAge
	}
	return block
}

// citationsWire returns the citations of a text part for replay: each one as
// the API sent it, or rebuilt from the typed fields.
func citationsWire(cs []llm.Citation) ([]any, error) {
	out := make([]any, 0, len(cs))
	for _, c := range cs {
		if len(c.Raw) > 0 {
			if !json.Valid(c.Raw) {
				return nil, fmt.Errorf("anthropic: citation %q has invalid Raw JSON", c.Type)
			}
			out = append(out, c.Raw)
			continue
		}
		out = append(out, citationWire(c))
	}
	return out, nil
}

// citationWire rebuilds a citation with the fields its type defines,
// zero values included (document_index 0 is the first document).
func citationWire(c llm.Citation) map[string]any {
	block := map[string]any{"type": c.Type, "cited_text": c.CitedText}
	switch c.Type {
	case llm.CitationWebSearchResultLocation:
		block["url"], block["title"], block["encrypted_index"] = c.URL, c.Title, c.EncryptedIndex
	case llm.CitationCharLocation:
		block["document_index"], block["document_title"] = c.DocumentIndex, c.DocumentTitle
		block["start_char_index"], block["end_char_index"] = c.StartCharIndex, c.EndCharIndex
	case llm.CitationPageLocation:
		block["document_index"], block["document_title"] = c.DocumentIndex, c.DocumentTitle
		block["start_page_number"], block["end_page_number"] = c.StartPageNumber, c.EndPageNumber
	case llm.CitationContentBlockLocation:
		block["document_index"], block["document_title"] = c.DocumentIndex, c.DocumentTitle
		block["start_block_index"], block["end_block_index"] = c.StartBlockIndex, c.EndBlockIndex
	case llm.CitationSearchResultLocation:
		block["source"], block["title"], block["search_result_index"] = c.Source, c.Title, c.SearchResultIndex
		block["start_block_index"], block["end_block_index"] = c.StartBlockIndex, c.EndBlockIndex
	}
	return block
}

// anthropicUsage is the response usage object.
type anthropicUsage struct {
	InputTokens         int `json:"input_tokens"`
	OutputTokens        int `json:"output_tokens"`
	CacheCreationTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadTokens     int `json:"cache_read_input_tokens,omitempty"`
	CacheCreation       struct {
		Ephemeral5m int `json:"ephemeral_5m_input_tokens"`
		Ephemeral1h int `json:"ephemeral_1h_input_tokens"`
	} `json:"cache_creation"`
	ServerToolUse struct {
		WebSearchRequests     int `json:"web_search_requests"`
		WebFetchRequests      int `json:"web_fetch_requests"`
		CodeExecutionRequests int `json:"code_execution_requests"`
	} `json:"server_tool_use"`
	Iterations []anthropicIteration `json:"iterations,omitempty"`
}

// anthropicIteration is one usage.iterations entry; its cache_creation split
// is flattened into the neutral UsageIteration.
type anthropicIteration struct {
	Type                string `json:"type"`
	Model               string `json:"model,omitempty"`
	InputTokens         int    `json:"input_tokens"`
	OutputTokens        int    `json:"output_tokens"`
	CacheCreationTokens int    `json:"cache_creation_input_tokens,omitempty"`
	CacheReadTokens     int    `json:"cache_read_input_tokens,omitempty"`
	CacheCreation       struct {
		Ephemeral1h int `json:"ephemeral_1h_input_tokens"`
	} `json:"cache_creation"`
}

func (u anthropicUsage) iterations() []llm.UsageIteration {
	if len(u.Iterations) == 0 {
		return nil
	}
	out := make([]llm.UsageIteration, len(u.Iterations))
	for i, it := range u.Iterations {
		out[i] = llm.UsageIteration{
			Type: it.Type, Model: it.Model, InputTokens: it.InputTokens, OutputTokens: it.OutputTokens,
			CacheCreationTokens: it.CacheCreationTokens, CacheReadTokens: it.CacheReadTokens,
			CacheCreation1hTokens: it.CacheCreation.Ephemeral1h,
		}
	}
	return out
}

func (u anthropicUsage) serverToolUse() llm.ServerToolUsage {
	return llm.ServerToolUsage{
		WebSearchRequests:     u.ServerToolUse.WebSearchRequests,
		WebFetchRequests:      u.ServerToolUse.WebFetchRequests,
		CodeExecutionRequests: u.ServerToolUse.CodeExecutionRequests,
	}
}

// citationOf decodes a citation object. One whose fields do not decode is
// still kept in Raw, so it goes back to the API as received; anything that is
// not a JSON object is dropped.
func citationOf(raw json.RawMessage) (llm.Citation, bool) {
	if c, err := llm.CitationFromRaw(raw); err == nil {
		return c, true
	}
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '{' || !json.Valid(trimmed) {
		return llm.Citation{}, false
	}
	return llm.Citation{Raw: compact(trimmed)}, true
}
