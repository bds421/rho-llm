package llm_test

// Regression tests for the v0.4.0 streaming-contract fixes (architecture
// review findings R-H2 silent truncation, R-M8 Gemini token sentinel,
// R-M4 stop-reason normalization). Break-the-system tests: each fails
// without its fix.

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/provider/anthropic"
	"github.com/bds421/rho-llm/provider/gemini"
	"github.com/bds421/rho-llm/provider/openaicompat"
	"github.com/bds421/rho-llm/provider/openairesponses"
)

// sseServer returns an httptest server that writes the given SSE lines and
// then closes the connection — simulating a server-side truncation when the
// protocol-final event is missing.
func sseServer(lines ...string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, l := range lines {
			fmt.Fprint(w, l+"\n\n")
		}
	}))
}

func collectStream(t *testing.T, client llm.Client) (events []llm.StreamEvent, errs []error) {
	t.Helper()
	for ev, err := range client.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
	}) {
		if err != nil {
			errs = append(errs, err)
		} else {
			events = append(events, ev)
		}
	}
	return events, errs
}

func hasDone(events []llm.StreamEvent) bool {
	for _, ev := range events {
		if ev.Type == llm.EventDone {
			return true
		}
	}
	return false
}

// requireTruncationError asserts the contract for a truncated stream: an
// explicit error wrapping io.ErrUnexpectedEOF, and no EventDone — a truncated
// turn must never look like a complete one OR end silently.
func requireTruncationError(t *testing.T, events []llm.StreamEvent, errs []error) {
	t.Helper()
	if hasDone(events) {
		t.Fatal("truncated stream produced an EventDone — looks like a complete turn")
	}
	if len(errs) == 0 {
		t.Fatal("truncated stream ended silently: no error, no EventDone")
	}
	if !errors.Is(errs[len(errs)-1], io.ErrUnexpectedEOF) {
		t.Fatalf("truncation error should wrap io.ErrUnexpectedEOF (so the pool can classify it), got: %v", errs[len(errs)-1])
	}
}

// R-H2: a stream that ends cleanly without its protocol-final event is
// truncated — every adapter must yield an explicit error, not end silently.
func TestStreamTruncationYieldsError(t *testing.T) {
	t.Run("anthropic", func(t *testing.T) {
		srv := sseServer(
			`data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}`,
			`data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"partial"}}`,
			// no message_delta — turn truncated
		)
		defer srv.Close()
		client, err := anthropic.New(llm.Config{Provider: "anthropic", Model: "claude-sonnet-4-6", APIKey: "test-key", BaseURL: srv.URL, Timeout: 10 * time.Second})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		events, errs := collectStream(t, client)
		requireTruncationError(t, events, errs)
		if len(events) == 0 || events[0].Type != llm.EventContent {
			t.Fatal("content yielded before truncation should still be delivered")
		}
	})

	t.Run("anthropic mid-tool-call", func(t *testing.T) {
		srv := sseServer(
			`data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}`,
			`data: {"type":"content_block_start","content_block":{"type":"tool_use","id":"call_1","name":"search"}}`,
			`data: {"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{\"query\":\"go"}}`,
			// no content_block_stop, no message_delta — the half-accumulated
			// tool call must NOT be silently dropped
		)
		defer srv.Close()
		client, err := anthropic.New(llm.Config{Provider: "anthropic", Model: "claude-sonnet-4-6", APIKey: "test-key", BaseURL: srv.URL, Timeout: 10 * time.Second})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		events, errs := collectStream(t, client)
		requireTruncationError(t, events, errs)
	})

	t.Run("gemini", func(t *testing.T) {
		srv := sseServer(
			`data: {"candidates":[{"content":{"parts":[{"text":"partial"}],"role":"model"}}]}`,
			// no finishReason — turn truncated
		)
		defer srv.Close()
		client, err := gemini.New(llm.Config{Provider: "gemini", Model: "gemini-2.5-flash", APIKey: "test-key", BaseURL: srv.URL, Timeout: 10 * time.Second})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		events, errs := collectStream(t, client)
		requireTruncationError(t, events, errs)
	})

	t.Run("openaicompat", func(t *testing.T) {
		srv := sseServer(
			`data: {"choices":[{"delta":{"content":"partial"}}]}`,
			// no finish_reason, no [DONE] — turn truncated
		)
		defer srv.Close()
		client, err := openaicompat.New(llm.Config{Provider: "openai", Model: "gpt-4.1", APIKey: "test-key", BaseURL: srv.URL, Timeout: 10 * time.Second})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		events, errs := collectStream(t, client)
		requireTruncationError(t, events, errs)
	})

	t.Run("openairesponses", func(t *testing.T) {
		srv := sseServer(
			`data: {"type":"response.output_text.delta","delta":"partial"}`,
			// no response.completed — turn truncated
		)
		defer srv.Close()
		client, err := openairesponses.New(llm.Config{Provider: "openai_responses", Model: "gpt-5.2", APIKey: "test-key", BaseURL: srv.URL, Timeout: 10 * time.Second})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		events, errs := collectStream(t, client)
		requireTruncationError(t, events, errs)
	})
}

// R-H2 (tolerance guard): an explicit [DONE] without finish_reason is a known
// spec violation of some local servers — the server DID signal completion, so
// the adapter synthesizes a stop reason instead of erroring.
func TestOpenAICompatDoneWithoutFinishReasonSynthesizesStop(t *testing.T) {
	srv := sseServer(
		`data: {"choices":[{"delta":{"content":"hello"}}]}`,
		`data: [DONE]`,
	)
	defer srv.Close()
	client, err := openaicompat.New(llm.Config{Provider: "openai", Model: "gpt-4.1", APIKey: "test-key", BaseURL: srv.URL, Timeout: 10 * time.Second})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	events, errs := collectStream(t, client)
	if len(errs) != 0 {
		t.Fatalf("[DONE] without finish_reason must not error, got: %v", errs)
	}
	var done *llm.StreamEvent
	for i := range events {
		if events[i].Type == llm.EventDone {
			done = &events[i]
		}
	}
	if done == nil {
		t.Fatal("no EventDone after explicit [DONE]")
	}
	if done.StopReason != llm.StopEndTurn {
		t.Fatalf("synthesized stop reason = %q, want %q", done.StopReason, llm.StopEndTurn)
	}
}

// R-M8: a Gemini stream whose final chunk carries no usageMetadata must report
// the TokensNotReported sentinel (-1), not a fake "0 tokens" — same convention
// as the other adapters.
func TestGeminiMissingUsageReportsSentinel(t *testing.T) {
	srv := sseServer(
		`data: {"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},"finishReason":"STOP"}]}`,
	)
	defer srv.Close()
	client, err := gemini.New(llm.Config{Provider: "gemini", Model: "gemini-2.5-flash", APIKey: "test-key", BaseURL: srv.URL, Timeout: 10 * time.Second})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	events, errs := collectStream(t, client)
	if len(errs) != 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	var done *llm.StreamEvent
	for i := range events {
		if events[i].Type == llm.EventDone {
			done = &events[i]
		}
	}
	if done == nil {
		t.Fatal("no EventDone")
	}
	if done.InputTokens != llm.TokensNotReported || done.OutputTokens != llm.TokensNotReported {
		t.Fatalf("missing usage reported as in=%d out=%d, want TokensNotReported (-1) for both",
			done.InputTokens, done.OutputTokens)
	}
}

// R-M8 guard: when usage IS reported (possibly on an earlier chunk than the
// finishReason), it must be delivered — not the sentinel.
func TestGeminiUsageOnEarlierChunkIsKept(t *testing.T) {
	srv := sseServer(
		`data: {"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"}}],"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":3}}`,
		`data: {"candidates":[{"content":{"parts":[{"text":"!"}],"role":"model"},"finishReason":"STOP"}]}`,
	)
	defer srv.Close()
	client, err := gemini.New(llm.Config{Provider: "gemini", Model: "gemini-2.5-flash", APIKey: "test-key", BaseURL: srv.URL, Timeout: 10 * time.Second})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	events, errs := collectStream(t, client)
	if len(errs) != 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	for _, ev := range events {
		if ev.Type == llm.EventDone {
			if ev.InputTokens != 7 || ev.OutputTokens != 3 {
				t.Fatalf("usage from earlier chunk lost: in=%d out=%d, want 7/3", ev.InputTokens, ev.OutputTokens)
			}
			return
		}
	}
	t.Fatal("no EventDone")
}

// R-M4: Anthropic must normalize stop reasons into the unified vocabulary like
// every other adapter — a configured stop sequence is a normal end of turn.
func TestAnthropicStopSequenceNormalized(t *testing.T) {
	t.Run("complete", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"id":"msg_1","model":"claude-sonnet-4-6","content":[{"type":"text","text":"hi"}],"stop_reason":"stop_sequence","usage":{"input_tokens":1,"output_tokens":1}}`)
		}))
		defer srv.Close()
		client, err := anthropic.New(llm.Config{Provider: "anthropic", Model: "claude-sonnet-4-6", APIKey: "test-key", BaseURL: srv.URL, Timeout: 10 * time.Second})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		resp, err := client.Complete(context.Background(), llm.Request{
			Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
		})
		if err != nil {
			t.Fatalf("Complete: %v", err)
		}
		if resp.StopReason != llm.StopEndTurn {
			t.Fatalf("stop_sequence not normalized: got %q, want %q", resp.StopReason, llm.StopEndTurn)
		}
	})

	t.Run("stream", func(t *testing.T) {
		srv := sseServer(
			`data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}`,
			`data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}`,
			`data: {"type":"message_delta","delta":{"stop_reason":"stop_sequence"},"usage":{"output_tokens":1}}`,
		)
		defer srv.Close()
		client, err := anthropic.New(llm.Config{Provider: "anthropic", Model: "claude-sonnet-4-6", APIKey: "test-key", BaseURL: srv.URL, Timeout: 10 * time.Second})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		events, errs := collectStream(t, client)
		if len(errs) != 0 {
			t.Fatalf("unexpected errors: %v", errs)
		}
		for _, ev := range events {
			if ev.Type == llm.EventDone {
				if ev.StopReason != llm.StopEndTurn {
					t.Fatalf("stop_sequence not normalized in stream: got %q, want %q", ev.StopReason, llm.StopEndTurn)
				}
				return
			}
		}
		t.Fatal("no EventDone")
	})
}

// A streamed openai_responses turn that emitted a function call must report
// StopReason == tool_use on EventDone — matching the non-streaming parseResponse
// path, which sets "tool_use" for any function_call output item *regardless of
// status* (completed OR incomplete/max_tokens). Otherwise an agent loop keying
// on resp.StopReason==StopToolUse stops after a streamed tool call even though
// ToolCalls is populated, and the two code paths disagree for the same turn.
func TestOpenAIResponsesStreamToolCallStopReason(t *testing.T) {
	cases := []struct {
		name           string
		completedEvent string
	}{
		{
			name:           "completed",
			completedEvent: `data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":5,"output_tokens":3}}}`,
		},
		{
			// A truncated (max_output_tokens) turn that still emitted a function
			// call: tool_use must win over max_tokens, mirroring parseResponse.
			name:           "incomplete-max-tokens",
			completedEvent: `data: {"type":"response.completed","response":{"id":"resp_1","status":"incomplete","usage":{"input_tokens":5,"output_tokens":3}}}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := sseServer(
				`data: {"type":"response.output_item.added","item":{"type":"function_call","name":"search","call_id":"call_1"}}`,
				`data: {"type":"response.function_call_arguments.delta","delta":"{\"q\":\"go\"}"}`,
				`data: {"type":"response.function_call_arguments.done","name":"search","call_id":"call_1","arguments":"{\"q\":\"go\"}"}`,
				tc.completedEvent,
			)
			defer srv.Close()
			client, err := openairesponses.New(llm.Config{Provider: "openai_responses", Model: "gpt-5.2", APIKey: "test-key", BaseURL: srv.URL, Timeout: 10 * time.Second})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			events, errs := collectStream(t, client)
			if len(errs) != 0 {
				t.Fatalf("unexpected errors: %v", errs)
			}
			var sawTool bool
			for _, ev := range events {
				if ev.Type == llm.EventToolUse {
					sawTool = true
				}
				if ev.Type == llm.EventDone {
					if !sawTool {
						t.Fatal("EventDone arrived before any EventToolUse")
					}
					if ev.StopReason != llm.StopToolUse {
						t.Fatalf("streamed tool call reported StopReason %q, want %q (diverges from Complete)", ev.StopReason, llm.StopToolUse)
					}
					return
				}
			}
			t.Fatal("no EventDone")
		})
	}
}

// A complete Anthropic turn whose FINAL tool_use block is missing its
// content_block_stop (spec violation) must still deliver the tool call with its
// accumulated input — the terminal message_delta must flush it before Done, not
// drop it while reporting a clean tool_use turn.
func TestAnthropicTerminalToolCallFlushedOnMissingStop(t *testing.T) {
	srv := sseServer(
		`data: {"type":"message_start","message":{"usage":{"input_tokens":5}}}`,
		`data: {"type":"content_block_start","content_block":{"type":"tool_use","id":"call_1","name":"search"}}`,
		`data: {"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{\"q\":\"go\"}"}}`,
		// NO content_block_stop — straight to the terminal event
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}`,
	)
	defer srv.Close()
	client, err := anthropic.New(llm.Config{Provider: "anthropic", Model: "claude-sonnet-4-6", APIKey: "test-key", BaseURL: srv.URL, Timeout: 10 * time.Second})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	events, errs := collectStream(t, client)
	if len(errs) != 0 {
		t.Fatalf("a complete (non-truncated) turn must not error: %v", errs)
	}
	if !hasDone(events) {
		t.Fatal("no EventDone")
	}
	var tool *llm.ToolCall
	for i := range events {
		if events[i].Type == llm.EventToolUse && events[i].ToolCall != nil {
			tool = events[i].ToolCall
		}
	}
	if tool == nil {
		t.Fatal("terminal tool call dropped on missing content_block_stop before message_delta")
	}
	if tool.ID != "call_1" {
		t.Fatalf("tool call ID = %q, want call_1", tool.ID)
	}
	m, ok := tool.Input.(map[string]any)
	if !ok || m["q"] != "go" {
		t.Fatalf("accumulated tool input not preserved: %#v", tool.Input)
	}
}

// StreamWithBoundaries must honor a consumer that stops during the injected
// block-End event on the error path: it must NOT call yield again with the
// error. Calling yield after it returned false violates the iter.Seq2 contract
// and panics the range-over-func runtime — so a regression makes this test
// panic (and fail) rather than pass.
func TestStreamWithBoundariesErrorPathHonorsConsumerStop(t *testing.T) {
	seq := func(yield func(llm.StreamEvent, error) bool) {
		// Open a text block, then error mid-block (forces closeBlock to emit a
		// TextEnd before the error propagates).
		if !yield(llm.StreamEvent{Type: llm.EventContent, Text: "hi"}, nil) {
			return
		}
		yield(llm.StreamEvent{}, errors.New("boom"))
	}
	for ev, err := range llm.StreamWithBoundaries(seq) {
		if err == nil && ev.Type == llm.EventTextEnd {
			break // stop exactly on the injected End — the wrapper must not yield again
		}
	}
}
