package openairesponses

import (
	"encoding/json"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// A function_call with "arguments":"" was stored as the string "" and replayed
// as "arguments":"\"\"" (B8).
func TestEmptyFunctionCallArgumentsBecomeAnEmptyObject(t *testing.T) {
	c := &Client{providerName: "openai", config: llm.Config{Model: "gpt-6.1-sol"}}
	var r responsesResponse
	if err := json.Unmarshal([]byte(`{"status":"completed","output":[{"type":"function_call","call_id":"call_1","name":"now","arguments":""}]}`), &r); err != nil {
		t.Fatal(err)
	}
	resp := c.parseResponse(&r)
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("tool calls = %+v", resp.ToolCalls)
	}
	if m, ok := resp.ToolCalls[0].Input.(map[string]any); !ok || len(m) != 0 {
		t.Fatalf("Input = %#v, want an empty object", resp.ToolCalls[0].Input)
	}

	var streamed *llm.ToolCall
	sse := "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_2\",\"name\":\"now\"}}\n\n" +
		"data: {\"type\":\"response.function_call_arguments.done\",\"output_index\":0,\"item_id\":\"fc_1\",\"arguments\":\"\"}\n\n" +
		"data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n"
	c.parseStream(strings.NewReader(sse), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			t.Fatal(err)
		}
		if ev.Type == llm.EventToolUse {
			streamed = ev.ToolCall
		}
		return true
	})
	if streamed == nil {
		t.Fatal("stream yielded no tool call")
	}
	if m, ok := streamed.Input.(map[string]any); !ok || len(m) != 0 {
		t.Fatalf("streamed Input = %#v, want an empty object", streamed.Input)
	}

	req := llm.Request{Messages: []llm.Message{
		llm.NewTextMessage(llm.RoleUser, "x"),
		llm.NewAssistantMessage(resp),
		llm.NewToolResultMessage("call_1", "12", false),
	}}
	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(apiReq.Input)
	if !strings.Contains(string(b), `"arguments":"{}"`) {
		t.Fatalf("replay = %s, want \"arguments\":\"{}\"", b)
	}
}
