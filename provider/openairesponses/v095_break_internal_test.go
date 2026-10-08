package openairesponses

import (
	"math"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

func runStream(t *testing.T, c *Client, lines ...string) ([]llm.StreamEvent, error) {
	t.Helper()
	var b strings.Builder
	for _, l := range lines {
		b.WriteString("data: " + l + "\n\n")
	}
	var events []llm.StreamEvent
	var streamErr error
	c.parseStream(strings.NewReader(b.String()), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			streamErr = err
			return false
		}
		events = append(events, ev)
		return true
	})
	return events, streamErr
}

// M6: reasoning_tokens lives in output_tokens_details and is a SUBSET of
// output_tokens. Reading it (it used to be read from a top-level field the
// API never sends) must not bill reasoning twice.
func TestResponsesReasoningTokensFromDetailsBilledOnce(t *testing.T) {
	c := &Client{providerName: "openai", config: llm.Config{Model: "gpt-5"}}
	events, err := runStream(t, c,
		`{"type":"response.output_text.delta","delta":"42"}`,
		`{"type":"response.completed","response":{"id":"r","status":"completed","usage":{"input_tokens":1000,"output_tokens":5000,"output_tokens_details":{"reasoning_tokens":4800}}}}`)
	if err != nil {
		t.Fatal(err)
	}
	done := events[len(events)-1]
	if done.ThinkingTokens != 4800 || done.OutputTokens != 200 {
		t.Fatalf("thinking/output = %d/%d, want 4800/200", done.ThinkingTokens, done.OutputTokens)
	}
	got := llm.EstimateCost(llm.CostInput{Model: "gpt-5", InputTokens: done.InputTokens, OutputTokens: done.OutputTokens, ThinkingTokens: done.ThinkingTokens})
	info, _ := llm.GetModelInfo("gpt-5")
	want := (1000*info.InputPricePer1M + 5000*info.OutputPricePer1M) / 1e6
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("cost = %v, want %v (reasoning double-billed?)", got, want)
	}

	// Non-stream path, and hostile counts.
	for _, tc := range []struct {
		out, reasoning, wantOut, wantThink int
	}{{5000, 4800, 200, 4800}, {10, 500, 0, 10}, {10, -5, 10, 0}, {-3, 2, 0, 0}} {
		u := responsesUsage{OutputTokens: tc.out}
		u.OutputTokensDetails.ReasoningTokens = tc.reasoning
		resp := c.parseResponse(&responsesResponse{Status: "completed", Usage: u})
		if resp.OutputTokens != tc.wantOut || resp.ThinkingTokens != tc.wantThink {
			t.Errorf("out=%d reasoning=%d: got %d/%d, want %d/%d", tc.out, tc.reasoning, resp.OutputTokens, resp.ThinkingTokens, tc.wantOut, tc.wantThink)
		}
	}
}

// M7: interleaved argument deltas of two parallel function calls are kept
// apart by item_id; name/call_id come from response.output_item.added when
// the done event omits them (the real API's done event carries neither).
func TestResponsesParallelFunctionCallsInterleaved(t *testing.T) {
	c := &Client{providerName: "openai"}
	events, err := runStream(t, c,
		`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_a","call_id":"call_a","name":"weather","arguments":""}}`,
		`{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_b","call_id":"call_b","name":"time","arguments":""}}`,
		`{"type":"response.function_call_arguments.delta","item_id":"fc_a","output_index":0,"delta":"{\"city\":"}`,
		`{"type":"response.function_call_arguments.delta","item_id":"fc_b","output_index":1,"delta":"{\"tz\":"}`,
		`{"type":"response.function_call_arguments.delta","item_id":"fc_a","output_index":0,"delta":"\"Vienna\"}"}`,
		`{"type":"response.function_call_arguments.delta","item_id":"fc_b","output_index":1,"delta":"\"CET\"}"}`,
		`{"type":"response.function_call_arguments.done","item_id":"fc_a","output_index":0}`,
		`{"type":"response.function_call_arguments.done","item_id":"fc_b","output_index":1}`,
		`{"type":"response.completed","response":{"id":"r","status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}`)
	if err != nil {
		t.Fatal(err)
	}
	var calls []*llm.ToolCall
	for _, ev := range events {
		if ev.Type == llm.EventToolUse {
			calls = append(calls, ev.ToolCall)
		}
	}
	if len(calls) != 2 {
		t.Fatalf("calls = %d", len(calls))
	}
	if calls[0].ID != "call_a" || calls[0].Name != "weather" || calls[0].Input.(map[string]any)["city"] != "Vienna" {
		t.Fatalf("call A = %+v", calls[0])
	}
	if calls[1].ID != "call_b" || calls[1].Name != "time" || calls[1].Input.(map[string]any)["tz"] != "CET" {
		t.Fatalf("call B = %+v", calls[1])
	}
}

// L: an in-stream error event is provider text that reaches the caller's
// error and logs — it must be scrubbed of the key and bounded like an HTTP
// error body.
func TestResponsesStreamErrorEventRedactedAndTruncated(t *testing.T) {
	key := "sk-test-0123456789abcdef"
	c := &Client{providerName: "openai", config: llm.Config{APIKey: key, MaxErrorMessageLen: 128}}
	huge := strings.Repeat("A", 10000)
	_, err := runStream(t, c, `{"type":"error","error":{"code":"invalid_api_key","message":"Incorrect API key provided: `+key+` `+huge+`"}}`)
	if err == nil {
		t.Fatal("want an error")
	}
	msg := err.Error()
	if strings.Contains(msg, key) {
		t.Fatalf("key leaked: %q", msg)
	}
	if len(msg) > 300 || !strings.Contains(msg, "[truncated]") || !strings.Contains(msg, "REDACTED") {
		t.Fatalf("message not bounded/redacted (len %d): %.200q", len(msg), msg)
	}
}
