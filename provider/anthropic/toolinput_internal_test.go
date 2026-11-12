package anthropic

import (
	"encoding/json"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// A tool without parameters streams no partial_json, or an empty one. The
// input was stored as the string "" and replayed as "input":"", which
// Anthropic rejects: tool_use.input must be an object (B8).
func TestNoArgumentToolCallBecomesAnEmptyObject(t *testing.T) {
	start := `data: {"type":"message_start","message":{"model":"claude-opus-5-5","usage":{"input_tokens":5,"output_tokens":1}}}

data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"now","input":{}}}

`
	end := `data: {"type":"content_block_stop","index":0}

data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}

data: {"type":"message_stop"}

`
	for name, deltas := range map[string]string{
		"no delta":    "",
		"empty delta": `data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":""}}` + "\n\n",
		"whitespace":  `data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":" "}}` + "\n\n",
	} {
		c := &Client{config: llm.Config{Model: "claude-opus-5-5"}}
		var done llm.StreamEvent
		var call *llm.ToolCall
		c.parseStream(strings.NewReader(start+deltas+end), func(ev llm.StreamEvent, err error) bool {
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			switch ev.Type {
			case llm.EventToolUse:
				call = ev.ToolCall
			case llm.EventDone:
				done = ev
			}
			return true
		})
		if call == nil {
			t.Fatalf("%s: no tool call", name)
		}
		if in, ok := call.Input.(map[string]any); !ok || len(in) != 0 {
			t.Fatalf("%s: ToolCall.Input = %#v, want an empty object", name, call.Input)
		}
		req := llm.Request{Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "time?"),
			{Role: llm.RoleAssistant, Content: done.Parts},
			llm.NewToolResultMessage("toolu_1", "12:00", false),
		}}
		apiReq, err := c.buildRequest(req, false)
		if err != nil {
			t.Fatal(err)
		}
		b, _ := json.Marshal(apiReq.Messages)
		if !strings.Contains(string(b), `"input":{}`) {
			t.Fatalf("%s: replay = %s, want \"input\":{}", name, b)
		}
	}
}

// Transcripts persisted before the fix hold "" (or a JSON object as text);
// replaying them must still send an object.
func TestPersistedStringToolInputIsReplayedAsAnObject(t *testing.T) {
	c := &Client{config: llm.Config{Model: "claude-opus-5-5"}}
	for in, want := range map[any]string{
		"":                `"input":{}`,
		"null":            `"input":{}`,
		`{"city":"Graz"}`: `"input":{"city":"Graz"}`,
	} {
		req := llm.Request{Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "x"),
			{Role: llm.RoleAssistant, Content: []llm.ContentPart{{Type: llm.ContentToolUse, ToolUseID: "toolu_1", ToolName: "f", ToolInput: in}}},
			llm.NewToolResultMessage("toolu_1", "r", false),
		}}
		apiReq, err := c.buildRequest(req, false)
		if err != nil {
			t.Fatal(err)
		}
		b, _ := json.Marshal(apiReq.Messages)
		if !strings.Contains(string(b), want) {
			t.Errorf("input %#v: replay = %s, want %s", in, b, want)
		}
	}
}
