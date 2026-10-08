package openaicompat

import (
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

func collectStream(t *testing.T, c *Client, sse string) ([]llm.StreamEvent, error) {
	t.Helper()
	var events []llm.StreamEvent
	var streamErr error
	c.parseStream(strings.NewReader(sse), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			streamErr = err
			return false
		}
		events = append(events, ev)
		return true
	})
	return events, streamErr
}

func toolCalls(events []llm.StreamEvent) []*llm.ToolCall {
	var out []*llm.ToolCall
	for _, ev := range events {
		if ev.Type == llm.EventToolUse {
			out = append(out, ev.ToolCall)
		}
	}
	return out
}

func sse(lines ...string) string {
	var b strings.Builder
	for _, l := range lines {
		b.WriteString("data: " + l + "\n\n")
	}
	return b.String()
}

// M7: two parallel tool calls whose argument deltas interleave. Only the
// first delta of each call carries the id; continuations carry only the
// index. Keying on id spliced call B's arguments into call A.
func TestStreamParallelToolCallsInterleavedByIndex(t *testing.T) {
	c := &Client{providerName: "openai"}
	events, err := collectStream(t, c, sse(
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"weather","arguments":""}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"time","arguments":""}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"tz\":"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Vienna\"}"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"CET\"}"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`[DONE]`,
	))
	if err != nil {
		t.Fatal(err)
	}
	calls := toolCalls(events)
	if len(calls) != 2 {
		t.Fatalf("calls = %d, want 2", len(calls))
	}
	a, _ := calls[0].Input.(map[string]any)
	b, _ := calls[1].Input.(map[string]any)
	if calls[0].ID != "call_a" || calls[0].Name != "weather" || a["city"] != "Vienna" {
		t.Fatalf("call A corrupted: %+v", calls[0])
	}
	if calls[1].ID != "call_b" || calls[1].Name != "time" || b["tz"] != "CET" {
		t.Fatalf("call B corrupted: %+v", calls[1])
	}
	if events[len(events)-1].StopReason != llm.StopToolUse {
		t.Fatalf("stop = %q", events[len(events)-1].StopReason)
	}
}

// Servers that number every call index 0 but give each a fresh id: a new id
// at a used index starts a new call instead of merging.
func TestStreamToolCallsIndexReusedWithNewID(t *testing.T) {
	c := &Client{providerName: "gemini-compat"}
	events, err := collectStream(t, c, sse(
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":"{\"n\":1}"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c2","function":{"name":"g","arguments":"{\"n\":2}"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`[DONE]`,
	))
	if err != nil {
		t.Fatal(err)
	}
	calls := toolCalls(events)
	if len(calls) != 2 || calls[0].ID != "c1" || calls[1].ID != "c2" {
		t.Fatalf("calls = %+v", calls)
	}
	if calls[0].Input.(map[string]any)["n"] != float64(1) || calls[1].Input.(map[string]any)["n"] != float64(2) {
		t.Fatalf("arguments crossed: %v / %v", calls[0].Input, calls[1].Input)
	}
}

// Index-less servers keep the pre-v0.9.5 behaviour.
func TestStreamToolCallsWithoutIndexStillAssemble(t *testing.T) {
	c := &Client{providerName: "ollama"}
	events, err := collectStream(t, c, sse(
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"id":"x1","function":{"name":"f","arguments":"{\"a\":"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"function":{"arguments":"1}"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"id":"x2","function":{"name":"g","arguments":"{}"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
	))
	if err != nil {
		t.Fatal(err)
	}
	calls := toolCalls(events)
	if len(calls) != 2 || calls[0].Input.(map[string]any)["a"] != float64(1) || calls[1].Name != "g" {
		t.Fatalf("calls = %+v", calls)
	}
}

// The tool-input cap covers ALL in-flight calls together: interleaving many
// calls must not multiply the memory bound.
func TestStreamToolInputCapAcrossParallelCalls(t *testing.T) {
	c := &Client{providerName: "openai", config: llm.Config{MaxToolInputBytes: 64}}
	chunk := strings.Repeat("x", 40)
	_, err := collectStream(t, c, sse(
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"f","arguments":"`+chunk+`"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"b","function":{"name":"f","arguments":"`+chunk+`"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
	))
	if err == nil || !strings.Contains(err.Error(), "tool input exceeded") {
		t.Fatalf("want the shared cap to trip, got %v", err)
	}
}
