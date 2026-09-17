package openairesponses

import (
	"fmt"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

func TestTerminalEventsAndRawReasons(t *testing.T) {
	for _, tc := range []struct{ status, detail, want, raw string }{
		{"completed", "", llm.StopEndTurn, "completed"},
		{"incomplete", "max_output_tokens", llm.StopMaxTokens, "max_output_tokens"},
		{"incomplete", "content_filter", "content_filter", "content_filter"},
		{"incomplete", "", llm.StopMaxTokens, "incomplete"},
		{"failed", "", "error", "failed"},
	} {
		t.Run(tc.status+"/"+tc.detail, func(t *testing.T) {
			c := &Client{providerName: "openai"}
			response := &responsesResponse{Status: tc.status}
			if tc.detail != "" {
				response.IncompleteDetails = &responsesIncomplete{Reason: tc.detail}
			}
			parsed := c.parseResponse(response)
			if parsed.StopReason != tc.want || parsed.RawStopReason != tc.raw {
				t.Errorf("Complete reasons = %q/%q, want %q/%q", parsed.StopReason, parsed.RawStopReason, tc.want, tc.raw)
			}
			wire := fmt.Sprintf("data: {\"type\":\"response.%s\",\"response\":{\"status\":%q,\"incomplete_details\":{\"reason\":%q},\"usage\":{\"input_tokens\":7,\"output_tokens\":9}}}\n\ndata: invalid trailing bytes\n\n", tc.status, tc.status, tc.detail)
			count := 0
			c.parseStream(strings.NewReader(wire), func(ev llm.StreamEvent, err error) bool {
				if err != nil {
					t.Errorf("Stream: %v", err)
					return false
				}
				count++
				if ev.Type != llm.EventDone || ev.StopReason != tc.want || ev.RawStopReason != tc.raw || ev.InputTokens != 7 || ev.OutputTokens != 9 {
					t.Errorf("Done = %+v", ev)
				}
				return true
			})
			if count != 1 {
				t.Errorf("got %d events, want one terminal event", count)
			}
		})
	}
}

func TestTerminalToolUsePreservesRawReason(t *testing.T) {
	c := &Client{providerName: "openai"}
	response := &responsesResponse{
		Status: "incomplete", IncompleteDetails: &responsesIncomplete{Reason: "max_output_tokens"},
		Output: []responsesOutputItem{{Type: "function_call", CallID: "call_1", Name: "lookup", Arguments: "{}"}},
	}
	parsed := c.parseResponse(response)
	if parsed.StopReason != llm.StopToolUse || parsed.RawStopReason != "max_output_tokens" {
		t.Fatalf("Complete reasons = %q/%q", parsed.StopReason, parsed.RawStopReason)
	}
	wire := "data: " + `{"type":"response.function_call_arguments.done","call_id":"call_1","name":"lookup","arguments":"{}"}` + "\n\n" +
		"data: " + `{"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}` + "\n\n"
	var done bool
	c.parseStream(strings.NewReader(wire), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			t.Fatal(err)
		}
		if ev.Type == llm.EventDone {
			done = true
			if ev.StopReason != llm.StopToolUse || ev.RawStopReason != "max_output_tokens" {
				t.Errorf("Done = %+v", ev)
			}
		}
		return true
	})
	if !done {
		t.Fatal("missing terminal event")
	}
}
