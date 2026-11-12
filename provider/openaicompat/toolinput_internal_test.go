package openaicompat

import (
	"encoding/json"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// "arguments":"" (a tool without parameters) was stored as the string "" and
// replayed as "arguments":"\"\"", a JSON string instead of an object (B8).
func TestEmptyToolArgumentsBecomeAnEmptyObject(t *testing.T) {
	c := &Client{providerName: "together", config: llm.Config{Model: "m"}}

	var r openaiResponse
	if err := json.Unmarshal([]byte(`{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"now","arguments":""}}]},"finish_reason":"tool_calls"}]}`), &r); err != nil {
		t.Fatal(err)
	}
	resp := c.parseResponse(&r)

	var streamed *llm.ToolCall
	evs, errs := streamAll(c, "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_2\",\"type\":\"function\",\"function\":{\"name\":\"now\",\"arguments\":\"\"}}]}}]}\n\n"+
		"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n")
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	for _, ev := range evs {
		if ev.Type == llm.EventToolUse {
			streamed = ev.ToolCall
		}
	}
	if streamed == nil {
		t.Fatal("stream yielded no tool call")
	}

	for name, in := range map[string]any{"complete": resp.ToolCalls[0].Input, "stream": streamed.Input} {
		if m, ok := in.(map[string]any); !ok || len(m) != 0 {
			t.Fatalf("%s: Input = %#v, want an empty object", name, in)
		}
	}

	req := llm.Request{Model: "m", Messages: []llm.Message{
		llm.NewTextMessage(llm.RoleUser, "x"),
		llm.NewAssistantMessage(resp),
		llm.NewToolResultMessage("call_1", "12", false),
		// a transcript persisted before the fix
		{Role: llm.RoleAssistant, Content: []llm.ContentPart{{Type: llm.ContentToolUse, ToolUseID: "call_3", ToolName: "now", ToolInput: ""}}},
		llm.NewToolResultMessage("call_3", "13", false),
	}}
	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(apiReq.Messages)
	if strings.Contains(string(b), `"arguments":"\"\""`) || strings.Count(string(b), `"arguments":"{}"`) != 2 {
		t.Fatalf("replay = %s, want both calls with \"arguments\":\"{}\"", b)
	}
}
