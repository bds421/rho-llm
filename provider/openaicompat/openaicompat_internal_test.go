package openaicompat

import (
	"encoding/json"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// TestParseResponseReasoningContent verifies that reasoning_content goes to
// resp.Thinking while content goes to resp.Content.
func TestParseResponseReasoningContent(t *testing.T) {
	raw := `{
		"id": "chatcmpl-123",
		"model": "qwen3:4b",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "The answer is 42.",
				"reasoning_content": "Let me think about this step by step..."
			},
			"finish_reason": "stop"
		}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
	}`

	var apiResp openaiResponse
	if err := json.Unmarshal([]byte(raw), &apiResp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	c := &Client{providerName: "ollama"}
	resp := c.parseResponse(&apiResp)

	if resp.Content != "The answer is 42." {
		t.Errorf("Content = %q, want %q", resp.Content, "The answer is 42.")
	}
	if resp.Thinking != "Let me think about this step by step..." {
		t.Errorf("Thinking = %q, want %q", resp.Thinking, "Let me think about this step by step...")
	}
}

// TestParseResponseNullContent verifies that null content with reasoning_content
// works correctly — content stays empty, thinking is populated.
func TestParseResponseNullContent(t *testing.T) {
	raw := `{
		"id": "chatcmpl-456",
		"model": "qwen3:4b",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": null,
				"reasoning_content": "Thinking only, no visible answer."
			},
			"finish_reason": "stop"
		}],
		"usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
	}`

	var apiResp openaiResponse
	if err := json.Unmarshal([]byte(raw), &apiResp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	c := &Client{providerName: "ollama"}
	resp := c.parseResponse(&apiResp)

	if resp.Content != "" {
		t.Errorf("Content = %q, want empty (null content)", resp.Content)
	}
	if resp.Thinking != "Thinking only, no visible answer." {
		t.Errorf("Thinking = %q, want %q", resp.Thinking, "Thinking only, no visible answer.")
	}
}

// TestParseResponseOllamaReasoningField verifies that Ollama's "reasoning" field
// (as opposed to "reasoning_content") is parsed into resp.Thinking.
func TestParseResponseOllamaReasoningField(t *testing.T) {
	raw := `{
		"id": "chatcmpl-123",
		"model": "qwen3:4b",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "The answer is 42.",
				"reasoning": "Let me think step by step..."
			},
			"finish_reason": "stop"
		}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
	}`

	var apiResp openaiResponse
	if err := json.Unmarshal([]byte(raw), &apiResp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	c := &Client{providerName: "ollama"}
	resp := c.parseResponse(&apiResp)

	if resp.Content != "The answer is 42." {
		t.Errorf("Content = %q, want %q", resp.Content, "The answer is 42.")
	}
	if resp.Thinking != "Let me think step by step..." {
		t.Errorf("Thinking = %q, want %q", resp.Thinking, "Let me think step by step...")
	}
}

// TestParseStreamOllamaReasoningField verifies that Ollama's streaming "reasoning"
// deltas (not "reasoning_content") emit EventThinking events.
func TestParseStreamOllamaReasoningField(t *testing.T) {
	sseData := "data: " + `{"id":"1","choices":[{"index":0,"delta":{"reasoning":"thinking via ollama..."}}]}` + "\n\n" +
		"data: " + `{"id":"1","choices":[{"index":0,"delta":{"content":"answer"}}]}` + "\n\n" +
		"data: " + `{"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n" +
		"data: " + `{"id":"1","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20}}` + "\n\n" +
		"data: [DONE]\n\n"

	c := &Client{providerName: "ollama"}
	var events []llm.StreamEvent
	c.parseStream(strings.NewReader(sseData), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		events = append(events, ev)
		return true
	})

	if len(events) != 3 {
		t.Fatalf("got %d events, want 3 (thinking, content, done)", len(events))
	}

	if events[0].Type != llm.EventThinking || events[0].Thinking != "thinking via ollama..." {
		t.Errorf("event[0] = %+v, want EventThinking with 'thinking via ollama...'", events[0])
	}
	if events[1].Type != llm.EventContent || events[1].Text != "answer" {
		t.Errorf("event[1] = %+v, want EventContent with 'answer'", events[1])
	}
}

// TestParseStreamReasoningContent verifies that streaming reasoning_content
// deltas emit EventThinking events.
func TestParseStreamReasoningContent(t *testing.T) {
	sseData := "data: " + `{"id":"1","choices":[{"index":0,"delta":{"reasoning_content":"thinking..."}}]}` + "\n\n" +
		"data: " + `{"id":"1","choices":[{"index":0,"delta":{"content":"answer"}}]}` + "\n\n" +
		"data: " + `{"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n" +
		"data: " + `{"id":"1","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20}}` + "\n\n" +
		"data: [DONE]\n\n"

	c := &Client{providerName: "ollama"}
	var events []llm.StreamEvent
	c.parseStream(strings.NewReader(sseData), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		events = append(events, ev)
		return true
	})

	if len(events) != 3 {
		t.Fatalf("got %d events, want 3 (thinking, content, done)", len(events))
	}

	if events[0].Type != llm.EventThinking || events[0].Thinking != "thinking..." {
		t.Errorf("event[0] = %+v, want EventThinking with 'thinking...'", events[0])
	}
	if events[1].Type != llm.EventContent || events[1].Text != "answer" {
		t.Errorf("event[1] = %+v, want EventContent with 'answer'", events[1])
	}
	if events[2].Type != llm.EventDone {
		t.Errorf("event[2].Type = %v, want EventDone", events[2].Type)
	}
}
