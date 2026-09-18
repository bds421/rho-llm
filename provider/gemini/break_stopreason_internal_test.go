package gemini

import (
	"encoding/json"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// Attack 1: malformed / adversarial candidates must not panic and must not
// invent a tool_use turn the caller would loop on forever.
func TestBreakParseResponseHostileCandidates(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gemini-2.5-flash"}, providerName: "gemini"}
	cases := []struct{ name, raw string }{
		{"no candidates", `{"candidates":[]}`},
		{"null candidates", `{"candidates":null}`},
		{"null parts", `{"candidates":[{"content":{"parts":null},"finishReason":"STOP"}]}`},
		{"empty finishReason", `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{}}}]},"finishReason":""}]}`},
		{"unknown finishReason", `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{}}}]},"finishReason":"SAFETY"}]}`},
		{"empty fn name", `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"","args":{}}}]},"finishReason":"STOP"}]}`},
		{"null args", `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":null}}]},"finishReason":"STOP"}]}`},
		{"RECITATION", `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{}}}]},"finishReason":"RECITATION"}]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var api geminiResponse
			if err := json.Unmarshal([]byte(tc.raw), &api); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			resp := c.parseResponse(&api, "gemini-2.5-flash") // must not panic
			if resp == nil {
				t.Fatal("nil response: caller would nil-deref")
			}
			// A terminal reason that is NOT a normal completion must never be
			// rewritten to tool_use — the caller would loop on a turn the
			// provider refused or truncated.
			if resp.RawStopReason != "" && resp.RawStopReason != "STOP" && resp.StopReason == "tool_use" {
				t.Errorf("finishReason %q rewritten to tool_use: caller loops on a non-completed turn",
					resp.RawStopReason)
			}
			// tool_use must never be claimed without a tool call to execute.
			if resp.StopReason == "tool_use" && len(resp.ToolCalls) == 0 {
				t.Error("StopReason=tool_use with zero ToolCalls: agentic loop spins with nothing to run")
			}
		})
	}
}

// Attack 2: a stream that emits a tool call then dies without finishReason
// must surface an error, not a clean tool_use turn the caller would act on.
func TestBreakParseStreamToolCallThenTruncated(t *testing.T) {
	sse := "data: " + `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{}}}]}}]}` + "\n\n"
	c := &Client{providerName: "gemini"}
	var sawErr bool
	var done *llm.StreamEvent
	c.parseStream(strings.NewReader(sse), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			sawErr = true
			return false
		}
		if ev.Type == llm.EventDone {
			cp := ev
			done = &cp
		}
		return true
	})
	if done != nil {
		t.Error("truncated stream produced a clean EventDone: indistinguishable from a completed turn")
	}
	if !sawErr {
		t.Error("truncated stream after a tool call ended silently: caller cannot tell the turn was cut off")
	}
}

// Attack 3: MAX_TOKENS mid-tool-call must stay max_tokens on the stream path.
func TestBreakParseStreamMaxTokensNotRewritten(t *testing.T) {
	sse := "data: " + `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{}}}]}}]}` + "\n\n" +
		"data: " + `{"candidates":[{"content":{"parts":[]},"finishReason":"MAX_TOKENS"}]}` + "\n\n"
	c := &Client{providerName: "gemini"}
	var done llm.StreamEvent
	c.parseStream(strings.NewReader(sse), func(ev llm.StreamEvent, err error) bool {
		if err == nil && ev.Type == llm.EventDone {
			done = ev
		}
		return true
	})
	if done.StopReason == "tool_use" {
		t.Error("MAX_TOKENS rewritten to tool_use: caller executes a truncated, possibly invalid tool call")
	}
	if done.StopReason != "max_tokens" {
		t.Errorf("StopReason = %q, want max_tokens: truncation must stay visible", done.StopReason)
	}
}

// Attack 4: the property that actually matters — a candidate carrying a
// function call must be REPORTED as tool_use across every shape Gemini
// produces. The earlier assertions here only checked that tool_use is never
// *invented*; they pass even with the inference removed, so they do not pin
// the bug this file exists for. This one fails the moment it regresses.
func TestBreakToolCallMustBeReportedAsToolUse(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gemini-2.5-flash"}, providerName: "gemini"}
	cases := []struct{ name, raw string }{
		{"single call", `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{"a":1}}}]},"finishReason":"STOP"}]}`},
		{"parallel calls", `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{}}},{"functionCall":{"name":"g","args":{}}}]},"finishReason":"STOP"}]}`},
		{"text then call", `{"candidates":[{"content":{"parts":[{"text":"let me check"},{"functionCall":{"name":"f","args":{}}}]},"finishReason":"STOP"}]}`},
		{"thought then call", `{"candidates":[{"content":{"parts":[{"text":"hmm","thought":true},{"functionCall":{"name":"f","args":{}}}]},"finishReason":"STOP"}]}`},
		{"empty args", `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{}}}]},"finishReason":"STOP"}]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var api geminiResponse
			if err := json.Unmarshal([]byte(tc.raw), &api); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			resp := c.parseResponse(&api, "gemini-2.5-flash")
			if len(resp.ToolCalls) == 0 {
				t.Fatal("tool call not parsed at all")
			}
			if resp.StopReason != "tool_use" {
				t.Errorf("StopReason = %q with %d tool call(s): the documented agentic loop "+
					"(`for resp.StopReason == \"tool_use\"`) never runs, so the call is silently dropped",
					resp.StopReason, len(resp.ToolCalls))
			}
		})
	}
}

// Attack 5: the same property on the streaming path, where the terminal event
// is the only signal the caller sees.
func TestBreakStreamToolCallMustBeReportedAsToolUse(t *testing.T) {
	cases := []struct{ name, sse string }{
		{"call then finish", "data: " + `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{}}}]}}]}` + "\n\n" +
			"data: " + `{"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}]}` + "\n\n"},
		{"call and finish same chunk", "data: " + `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{}}}]},"finishReason":"STOP"}]}` + "\n\n"},
		{"text then call then finish", "data: " + `{"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}` + "\n\n" +
			"data: " + `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{}}}]}}]}` + "\n\n" +
			"data: " + `{"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}]}` + "\n\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := &Client{providerName: "gemini"}
			var done llm.StreamEvent
			var toolCalls int
			c.parseStream(strings.NewReader(tc.sse), func(ev llm.StreamEvent, err error) bool {
				if err != nil {
					t.Fatalf("unexpected stream error: %v", err)
				}
				if ev.Type == llm.EventToolUse {
					toolCalls++
				}
				if ev.Type == llm.EventDone {
					done = ev
				}
				return true
			})
			if toolCalls == 0 {
				t.Fatal("no EventToolUse emitted")
			}
			if done.StopReason != "tool_use" {
				t.Errorf("EventDone.StopReason = %q after %d tool call(s): caller ends the turn "+
					"and the tool call is discarded", done.StopReason, toolCalls)
			}
		})
	}
}

// Attack 6: a response with no usageMetadata must report the
// TokensNotReported sentinel, never 0.
//
// Found by mutation audit: initializing InputTokens/OutputTokens to 0 instead
// of the sentinel left the suite green. CLAUDE.md calls this out explicitly —
// callers must be able to tell "the provider omitted usage" from "this turn
// cost zero tokens". Collapsing the two silently under-reports spend and
// corrupts accumulated Usage across a conversation.
func TestBreakMissingUsageReportsSentinelNotZero(t *testing.T) {
	cases := []struct{ name, raw string }{
		{"no usageMetadata", `{"candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}]}`},
		{"null usageMetadata", `{"candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}],"usageMetadata":null}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var api geminiResponse
			if err := json.Unmarshal([]byte(tc.raw), &api); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			c := &Client{config: llm.Config{Model: "gemini-2.5-flash"}, providerName: "gemini"}
			resp := c.parseResponse(&api, "gemini-2.5-flash")

			if resp.InputTokens != llm.TokensNotReported {
				t.Errorf("InputTokens = %d, want TokensNotReported (%d): a missing usage block is "+
					"indistinguishable from a zero-cost turn, so accumulated Usage under-reports spend",
					resp.InputTokens, llm.TokensNotReported)
			}
			if resp.OutputTokens != llm.TokensNotReported {
				t.Errorf("OutputTokens = %d, want TokensNotReported (%d)",
					resp.OutputTokens, llm.TokensNotReported)
			}
		})
	}
}
