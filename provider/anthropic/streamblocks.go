package anthropic

import (
	"encoding/json"
	"strings"

	"github.com/bds421/rho-llm"
)

// streamBlock is one content block being assembled from stream events.
type streamBlock struct {
	block      anthropicBlock // the content_block_start payload
	text       strings.Builder
	thinking   strings.Builder
	signature  strings.Builder
	input      strings.Builder
	citations  []llm.Citation
	part       *llm.ContentPart // set once the block is complete
	serverOpen bool             // a server_tool_use whose input is still streaming
}

// streamBlocks assembles a streamed turn into the same ordered parts the
// non-streaming path returns, so a streamed paused turn can be continued.
type streamBlocks struct {
	order   []int
	byIndex map[int]*streamBlock
}

func newStreamBlocks() *streamBlocks {
	return &streamBlocks{byIndex: map[int]*streamBlock{}}
}

// get returns the block at index, creating a block of blockType when a
// spec-violating stream sends a delta without its content_block_start.
func (s *streamBlocks) get(index int, blockType string) *streamBlock {
	if b, ok := s.byIndex[index]; ok {
		return b
	}
	b := &streamBlock{block: anthropicBlock{Type: blockType}}
	s.byIndex[index] = b
	s.order = append(s.order, index)
	return b
}

// start records a content_block_start. Blocks that arrive whole (server tool
// results, redacted thinking, unknown types) are complete at once: done is
// true and part is the block's neutral form.
func (s *streamBlocks) start(index int, block anthropicBlock) (part llm.ContentPart, done bool) {
	b, ok := s.byIndex[index]
	if !ok {
		b = &streamBlock{}
		s.byIndex[index] = b
		s.order = append(s.order, index)
	}
	*b = streamBlock{block: block}
	switch block.Type {
	case "text":
		b.text.WriteString(block.Text)
		return llm.ContentPart{}, false
	case "thinking":
		b.thinking.WriteString(block.Thinking)
		b.signature.WriteString(block.Signature)
		return llm.ContentPart{}, false
	case "tool_use":
		return llm.ContentPart{}, false
	case "server_tool_use":
		b.serverOpen = true
		return llm.ContentPart{}, false
	}
	part = blockPart(block)
	b.part = &part
	return part, true
}

func (s *streamBlocks) appendText(index int, text string) {
	s.get(index, "text").text.WriteString(text)
}

func (s *streamBlocks) appendCitation(index int, c llm.Citation) {
	b := s.get(index, "text")
	b.citations = append(b.citations, c)
}

func (s *streamBlocks) appendThinking(index int, thinking string) {
	s.get(index, "thinking").thinking.WriteString(thinking)
}

func (s *streamBlocks) appendSignature(index int, signature string) {
	s.get(index, "thinking").signature.WriteString(signature)
}

func (s *streamBlocks) isServerToolUse(index int) bool {
	b, ok := s.byIndex[index]
	return ok && b.serverOpen
}

// appendServerInput adds streamed input JSON to a server tool call; false
// when the input would exceed max bytes.
func (s *streamBlocks) appendServerInput(index int, partial string, max int) bool {
	b := s.byIndex[index]
	if b.input.Len()+len(partial) > max {
		return false
	}
	b.input.WriteString(partial)
	return true
}

// openServerToolUse returns the index of a server tool call still streaming
// its input, or -1.
func (s *streamBlocks) openServerToolUse() int {
	for _, index := range s.order {
		if s.byIndex[index].serverOpen {
			return index
		}
	}
	return -1
}

// finishServerToolUse completes the server tool call at index; ok is false
// when no such call is open.
func (s *streamBlocks) finishServerToolUse(index int) (llm.ContentPart, bool) {
	b, ok := s.byIndex[index]
	if !ok || !b.serverOpen {
		return llm.ContentPart{}, false
	}
	b.serverOpen = false
	part := llm.ContentPart{
		Type:      llm.ContentServerToolUse,
		ToolUseID: b.block.ID,
		ToolName:  b.block.Name,
		ToolInput: b.block.Input,
	}
	inputJSON := json.RawMessage(b.input.String())
	if len(inputJSON) > 0 {
		var input any
		if err := json.Unmarshal(inputJSON, &input); err == nil {
			part.ToolInput = input
		} else {
			inputJSON = nil
		}
	}
	part.ProviderBlock = withInput(b.block.raw, inputJSON)
	b.part = &part
	return part, true
}

// withInput returns the started server_tool_use block with its streamed
// input filled in (the start event carries an empty input).
func withInput(raw, input json.RawMessage) json.RawMessage {
	if len(raw) == 0 || len(input) == 0 {
		return raw
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return raw
	}
	fields["input"] = compact(input)
	merged, err := json.Marshal(fields)
	if err != nil {
		return raw
	}
	return compact(merged)
}

// setToolUse records a completed client tool call.
func (s *streamBlocks) setToolUse(index int, tc llm.ToolCall) {
	if index < 0 {
		return
	}
	b := s.get(index, "tool_use")
	b.part = &llm.ContentPart{Type: llm.ContentToolUse, ToolUseID: tc.ID, ToolName: tc.Name, ToolInput: tc.Input}
}

// parts returns the assembled content in block order.
func (s *streamBlocks) parts() []llm.ContentPart {
	var out []llm.ContentPart
	for _, index := range s.order {
		b := s.byIndex[index]
		if b.serverOpen {
			s.finishServerToolUse(index)
		}
		if b.part != nil {
			out = append(out, *b.part)
			continue
		}
		switch b.block.Type {
		case "text":
			out = append(out, llm.ContentPart{Type: llm.ContentText, Text: b.text.String(), Citations: b.citations})
		case "thinking":
			thinking, signature := b.thinking.String(), b.signature.String()
			out = append(out, llm.ContentPart{
				Type: llm.ContentThinking, Thinking: thinking, ThinkingSignature: signature,
				ProviderBlock: withFields(b.block.raw, map[string]string{"thinking": thinking, "signature": signature}),
			})
		}
	}
	return out
}

// withFields returns the started block with string fields set to their
// streamed values (the start event carries them empty).
func withFields(raw json.RawMessage, values map[string]string) json.RawMessage {
	if len(raw) == 0 {
		return nil
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil
	}
	for key, value := range values {
		encoded, err := json.Marshal(value)
		if err != nil {
			return nil
		}
		fields[key] = encoded
	}
	merged, err := json.Marshal(fields)
	if err != nil {
		return nil
	}
	return compact(merged)
}
