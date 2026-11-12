package llm_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider"
	"github.com/bds421/rho-llm/provider/anthropic"
	"github.com/bds421/rho-llm/provider/gemini"
	"github.com/bds421/rho-llm/provider/openaicompat"
	"github.com/bds421/rho-llm/provider/openairesponses"
)

// Anthropic stream with two tool_use blocks and NO content_block_stop between
// them (malformed framing): the first tool call's accumulated input must NOT be
// silently lost — both calls should reach the caller.
func TestAnthropicTwoToolBlocksWithoutStopBothEmitted(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, `data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}`+"\n\n")
		fmt.Fprint(w, `data: {"type":"content_block_start","content_block":{"type":"tool_use","id":"call_A","name":"alpha"}}`+"\n\n")
		fmt.Fprint(w, `data: {"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{\"a\":1}"}}`+"\n\n")
		// NO content_block_stop — straight into the next tool block
		fmt.Fprint(w, `data: {"type":"content_block_start","content_block":{"type":"tool_use","id":"call_B","name":"beta"}}`+"\n\n")
		fmt.Fprint(w, `data: {"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{\"b\":2}"}}`+"\n\n")
		fmt.Fprint(w, `data: {"type":"content_block_stop"}`+"\n\n")
		fmt.Fprint(w, `data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":1}}`+"\n\n")
	}))
	defer srv.Close()

	c, _ := anthropic.New(llm.Config{Provider: "anthropic", Model: "claude-sonnet-4-6", APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
	inputs := map[string]any{}
	var ids []string
	for ev, err := range c.Stream(context.Background(), llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "x")}}) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if ev.Type == llm.EventToolUse && ev.ToolCall != nil {
			ids = append(ids, ev.ToolCall.ID)
			inputs[ev.ToolCall.ID] = ev.ToolCall.Input
		}
	}
	if len(ids) != 2 {
		t.Fatalf("got tool calls %v, want both call_A and call_B — first lost on missing content_block_stop", ids)
	}
	// The whole point of the flush-on-new-block fix is that call_A's ACCUMULATED
	// input survives — not just its ID. Pin the parsed input for both calls so a
	// regression that emitted call_A with empty/garbage input would still fail.
	wantA := map[string]any{"a": float64(1)}
	wantB := map[string]any{"b": float64(2)}
	if got, ok := inputs["call_A"].(map[string]any); !ok || !reflect.DeepEqual(got, wantA) {
		t.Fatalf("call_A input = %#v, want %#v — accumulated input lost on missing stop", inputs["call_A"], wantA)
	}
	if got, ok := inputs["call_B"].(map[string]any); !ok || !reflect.DeepEqual(got, wantB) {
		t.Fatalf("call_B input = %#v, want %#v", inputs["call_B"], wantB)
	}
}

// StreamWithBoundaries: when the underlying stream errors mid-thinking-block,
// the wrapper must still propagate the error (and not hang or panic). Also
// verify a normal run brackets blocks correctly.
func TestStreamWithBoundariesErrorMidBlock(t *testing.T) {
	boom := errors.New("mid-block boom")
	seq := func(yield func(llm.StreamEvent, error) bool) {
		if !yield(llm.StreamEvent{Type: llm.EventThinking, Thinking: "reasoning"}, nil) {
			return
		}
		yield(llm.StreamEvent{}, boom) // error while a thinking block is open
	}
	var sawErr bool
	var thinkingStart, thinkingEnd int
	for ev, err := range llm.StreamWithBoundaries(seq) {
		if err != nil {
			if !errors.Is(err, boom) {
				t.Fatalf("wrong error: %v", err)
			}
			sawErr = true
			break
		}
		switch ev.Type {
		case llm.EventThinkingStart:
			thinkingStart++
		case llm.EventThinkingEnd:
			thinkingEnd++
		}
	}
	if !sawErr {
		t.Fatal("error during an open block was swallowed by StreamWithBoundaries")
	}
	if thinkingStart != 1 {
		t.Fatalf("expected exactly one ThinkingStart, got %d", thinkingStart)
	}
	// The open thinking block must be closed before the error — every Start
	// needs a matching End or block-based UIs leak an open panel.
	if thinkingEnd != 1 {
		t.Fatalf("open thinking block not closed before the error: ThinkingEnd=%d", thinkingEnd)
	}
}

func sseSrv(lines ...string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, l := range lines {
			fmt.Fprint(w, l+"\n\n")
		}
	}))
}

func drain(t *testing.T, c llm.Client) (done bool, errs int) {
	t.Helper()
	for ev, err := range c.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
	}) {
		if err != nil {
			errs++
		} else if ev.Type == llm.EventDone {
			done = true
		}
	}
	return done, errs
}

// A garbage SSE line AFTER the protocol-final event must be ignored: the turn
// already completed, so a trailing bad byte must not surface as an error (which
// would make a Session drop an otherwise-complete turn).
func TestStreamGarbageAfterDoneIgnored(t *testing.T) {
	t.Run("anthropic", func(t *testing.T) {
		srv := sseSrv(
			`data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}`,
			`data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}`,
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
			`data: {GARBAGE not json}`,
		)
		defer srv.Close()
		c, _ := anthropic.New(llm.Config{Provider: "anthropic", Model: "claude-sonnet-4-6", APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
		done, errs := drain(t, c)
		if !done || errs != 0 {
			t.Fatalf("done=%v errs=%d — trailing garbage after Done not ignored", done, errs)
		}
	})

	t.Run("gemini", func(t *testing.T) {
		srv := sseSrv(
			`data: {"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}`,
			`data: {GARBAGE not json}`,
		)
		defer srv.Close()
		c, _ := gemini.New(llm.Config{Provider: "gemini", Model: "gemini-2.5-flash", APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
		done, errs := drain(t, c)
		if !done || errs != 0 {
			t.Fatalf("done=%v errs=%d — trailing garbage after Done not ignored", done, errs)
		}
	})

	t.Run("openairesponses", func(t *testing.T) {
		srv := sseSrv(
			`data: {"type":"response.output_text.delta","delta":"hi"}`,
			`data: {"type":"response.completed","response":{"id":"r","status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}`,
			`data: {GARBAGE not json}`,
		)
		defer srv.Close()
		c, _ := openairesponses.New(llm.Config{Provider: "openai_responses", Model: "gpt-5.2", APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
		done, errs := drain(t, c)
		if !done || errs != 0 {
			t.Fatalf("done=%v errs=%d — trailing garbage after Done not ignored", done, errs)
		}
	})

	// openai_compat structurally CANNOT return right after finish_reason: usage
	// arrives in a separate trailing chunk, so the scan continues. A garbage line
	// after finish_reason+usage but with NO [DONE] (what loose local servers like
	// Ollama/vLLM/LM Studio emit) must be ignored, not surfaced as an error that
	// masks the completed turn.
	t.Run("openaicompat", func(t *testing.T) {
		srv := sseSrv(
			`data: {"choices":[{"delta":{"content":"hi"}}]}`,
			`data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			`data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}`,
			`data: {GARBAGE not json}`, // no [DONE] — trailing noise
		)
		defer srv.Close()
		c, _ := openaicompat.New(llm.Config{Provider: "openai", Model: "gpt-4.1", APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
		done, errs := drain(t, c)
		if !done || errs != 0 {
			t.Fatalf("done=%v errs=%d — trailing garbage after finish_reason not ignored", done, errs)
		}
	})
}

// TestAnthropicStreaming tests streaming with Anthropic.
func TestAnthropicStreaming(t *testing.T) {
	apiKey := envKey("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEYS")
	requireLiveAPI(t, apiKey, "ANTHROPIC_API_KEY(S) not set")

	cfg := llm.Config{
		Provider:    "anthropic",
		Model:       "claude-haiku-4-5-20251001",
		APIKey:      apiKey,
		MaxTokens:   100,
		Temperature: floatPtr(0.0),
		Timeout:     30 * time.Second,
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req := llm.Request{
		Messages:    []llm.Message{llm.NewTextMessage("user", "Count from 1 to 5")},
		MaxTokens:   100,
		Temperature: floatPtr(0.0),
	}

	var chunks []string
	var done bool

	for event, err := range client.Stream(ctx, req) {
		if err != nil {
			t.Fatalf("Streaming failed: %v", err)
		}
		switch event.Type {
		case llm.EventContent:
			chunks = append(chunks, event.Text)
		case llm.EventDone:
			done = true
			t.Logf("Stream done: tokens in=%d, out=%d", event.InputTokens, event.OutputTokens)
		}
	}

	if !done {
		t.Error("Stream did not complete")
	}

	if len(chunks) == 0 {
		t.Error("No content chunks received")
	}

	fullResponse := strings.Join(chunks, "")
	t.Logf("Streamed response (%d chunks): %s", len(chunks), fullResponse)
}

// TestXAIStreaming tests streaming with xAI Grok via OpenAI-compat adapter.
func TestXAIStreaming(t *testing.T) {
	apiKey := envKey("XAI_API_KEY", "XAI_API_KEYS")
	requireLiveAPI(t, apiKey, "XAI_API_KEY(S) not set")

	cfg := llm.Config{
		Provider:    "xai",
		Model:       "grok-4.20-0309-non-reasoning",
		APIKey:      apiKey,
		MaxTokens:   100,
		Temperature: floatPtr(0.0),
		Timeout:     30 * time.Second,
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req := llm.Request{
		Messages:    []llm.Message{llm.NewTextMessage("user", "Count from 1 to 5")},
		MaxTokens:   100,
		Temperature: floatPtr(0.0),
	}

	var chunks []string
	var done bool

	for event, err := range client.Stream(ctx, req) {
		if err != nil {
			t.Fatalf("Streaming failed: %v", err)
		}
		switch event.Type {
		case llm.EventContent:
			chunks = append(chunks, event.Text)
		case llm.EventDone:
			done = true
			t.Logf("Stream done: tokens in=%d, out=%d", event.InputTokens, event.OutputTokens)
		}
	}

	if !done {
		t.Error("Stream did not complete")
	}

	fullResponse := strings.Join(chunks, "")
	t.Logf("Streamed response (%d chunks): %s", len(chunks), fullResponse)
}

// TestGeminiStreaming tests streaming with Google Gemini.
func TestGeminiStreaming(t *testing.T) {
	apiKey := envKey("GEMINI_API_KEY", "GEMINI_API_KEYS")
	requireLiveAPI(t, apiKey, "GEMINI_API_KEY(S) not set")

	cfg := llm.Config{
		Provider:    "gemini",
		Model:       "gemini-2.5-flash-lite",
		APIKey:      apiKey,
		MaxTokens:   100,
		Temperature: floatPtr(0.0),
		Timeout:     30 * time.Second,
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req := llm.Request{
		Messages:    []llm.Message{llm.NewTextMessage("user", "Count from 1 to 5")},
		MaxTokens:   100,
		Temperature: floatPtr(0.0),
	}

	var chunks []string
	var done bool

	for event, err := range client.Stream(ctx, req) {
		if err != nil {
			t.Fatalf("Streaming failed: %v", err)
		}
		switch event.Type {
		case llm.EventContent:
			chunks = append(chunks, event.Text)
		case llm.EventDone:
			done = true
			t.Logf("Stream done: tokens in=%d, out=%d", event.InputTokens, event.OutputTokens)
		}
	}

	if !done {
		t.Error("Stream did not complete")
	}

	fullResponse := strings.Join(chunks, "")
	t.Logf("Streamed response (%d chunks): %s", len(chunks), fullResponse)
}

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

// BUG 6/2: a no-arg tool (nil InputSchema) and a replayed tool_use with nil
// ToolInput must not serialize to `"input_schema":null` / `"input":null` —
// Anthropic requires those fields to be JSON objects, so null gets rejected.
func TestAnthropicNilToolSchemaAndInputNotNull(t *testing.T) {
	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"m","model":"claude-sonnet-4-6","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer srv.Close()
	client, err := anthropic.New(llm.Config{Provider: "anthropic", Model: "claude-sonnet-4-6", APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = client.Complete(context.Background(), llm.Request{
		MaxTokens: 16,
		Tools:     []llm.Tool{{Name: "now", Description: "current time", InputSchema: nil}}, // no-arg tool
		Messages: []llm.Message{
			{Role: llm.RoleAssistant, Content: []llm.ContentPart{
				{Type: llm.ContentToolUse, ToolUseID: "c1", ToolName: "now", ToolInput: nil}, // nil input
			}},
			llm.NewToolResultMessage("c1", "12:00", false),
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if bytes.Contains(body, []byte(`"input_schema":null`)) {
		t.Fatalf("nil tool schema sent as null: %s", body)
	}
	if bytes.Contains(body, []byte(`"input":null`)) {
		t.Fatalf("nil tool input sent as null: %s", body)
	}
}

// A mid-stream empty SSE data line ("data: " with no value — keep-alive/padding
// some servers emit) must be skipped, not turned into a spurious malformed-event
// error that aborts an otherwise-complete turn.
func TestStreamEmptyDataLineIgnored(t *testing.T) {
	srv := sseServer(
		`data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}`,
		`data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}`,
		"data: ", // empty data value mid-stream
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
	)
	defer srv.Close()
	client, err := anthropic.New(llm.Config{Provider: "anthropic", Model: "claude-sonnet-4-6", APIKey: "test-key", BaseURL: srv.URL, Timeout: 10 * time.Second})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	events, errs := collectStream(t, client)
	if len(errs) != 0 {
		t.Fatalf("empty SSE data line produced a spurious error: %v", errs)
	}
	if !hasDone(events) {
		t.Fatal("no EventDone — empty data line aborted the turn")
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

// seqOf builds an iter.Seq2 that yields the given events (then optionally an error).
func seqOf(events []llm.StreamEvent, tail error) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		for _, ev := range events {
			if !yield(ev, nil) {
				return
			}
		}
		if tail != nil {
			yield(llm.StreamEvent{}, tail)
		}
	}
}

func TestStreamWithBoundariesOrder(t *testing.T) {
	in := []llm.StreamEvent{
		{Type: llm.EventThinking, Thinking: "a"},
		{Type: llm.EventThinking, Thinking: "b"},
		{Type: llm.EventContent, Text: "x"},
		{Type: llm.EventContent, Text: "y"},
		{Type: llm.EventToolUse, ToolCall: &llm.ToolCall{ID: "c1", Name: "f"}},
		{Type: llm.EventDone, StopReason: "tool_use"},
	}
	var got []llm.EventType
	for ev, err := range llm.StreamWithBoundaries(seqOf(in, nil)) {
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, ev.Type)
	}
	want := []llm.EventType{
		llm.EventThinkingStart, llm.EventThinking, llm.EventThinking, llm.EventThinkingEnd,
		llm.EventTextStart, llm.EventContent, llm.EventContent, llm.EventTextEnd,
		llm.EventToolStart, llm.EventToolUse, llm.EventToolEnd,
		llm.EventDone,
	}
	if len(got) != len(want) {
		t.Fatalf("event sequence:\n got=%v\nwant=%v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("event[%d] = %q, want %q (full: %v)", i, got[i], want[i], got)
		}
	}
}

func TestStreamWithBoundariesErrorPropagates(t *testing.T) {
	boom := errors.New("boom")
	in := []llm.StreamEvent{{Type: llm.EventContent, Text: "partial"}}
	var gotErr error
	for _, err := range llm.StreamWithBoundaries(seqOf(in, boom)) {
		if err != nil {
			gotErr = err
		}
	}
	if !errors.Is(gotErr, boom) {
		t.Errorf("error not propagated: %v", gotErr)
	}
}

func TestStreamWithBoundariesNoDoneClosesBlock(t *testing.T) {
	// Stream ends without EventDone — an open text block must still be closed.
	in := []llm.StreamEvent{{Type: llm.EventContent, Text: "x"}}
	var got []llm.EventType
	for ev := range llm.StreamWithBoundaries(seqOf(in, nil)) {
		got = append(got, ev.Type)
	}
	if len(got) == 0 || got[len(got)-1] != llm.EventTextEnd {
		t.Errorf("open block not closed on stream end: %v", got)
	}
}

func TestStreamWithBoundariesBreakEarly(t *testing.T) {
	in := []llm.StreamEvent{
		{Type: llm.EventContent, Text: "x"},
		{Type: llm.EventContent, Text: "y"},
		{Type: llm.EventDone},
	}
	count := 0
	for range llm.StreamWithBoundaries(seqOf(in, nil)) {
		count++
		if count == 2 {
			break // must not panic / must stop cleanly
		}
	}
	if count != 2 {
		t.Errorf("break early did not stop iteration: %d", count)
	}
}

func TestCollectStream(t *testing.T) {
	b := &brokenClient{}
	resp, err := llm.CollectStream(b.Stream(context.Background(), llm.Request{}))
	if !errors.Is(err, errBroken) || resp.Content != "Part" || resp.InputTokens != 50 || resp.CacheReadTokens != 900 {
		t.Errorf("broken: %+v, %v", resp, err)
	}

	f := &fakeClient{events: []llm.StreamEvent{{Type: llm.EventContent, Text: "x"}}}
	if _, err := llm.CollectStream(f.Stream(context.Background(), llm.Request{})); !errors.Is(err, llm.ErrStreamIncomplete) {
		t.Errorf("no EventDone: %v", err)
	}

	f.events = append(f.events, llm.StreamEvent{Type: llm.EventDone, Model: "claude-haiku-5-5", StopReason: llm.StopEndTurn, OutputTokens: 3})
	resp, err = llm.CollectStream(f.Stream(context.Background(), llm.Request{}))
	if err != nil || resp.Model != "claude-haiku-5-5" || resp.Content != "x" || resp.OutputTokens != 3 {
		t.Errorf("complete: %+v, %v", resp, err)
	}
}

// Logging and the auth pool hand the partial-usage event on with the error.
func TestWrappersPassTheErrorEvent(t *testing.T) {
	broken := &brokenClient{fakeClient{provider: "anthropic", model: "claude-opus-5-5"}}
	pooled, err := llm.NewPooledClient(llm.Config{Provider: "anthropic", Model: "claude-opus-5-5", DisableRetries: true},
		[]string{"k"}, func(llm.AuthProfile) (llm.Client, error) { return broken, nil })
	if err != nil {
		t.Fatal(err)
	}
	for name, c := range map[string]llm.Client{"logging": llm.WithLogging(broken), "pool": pooled} {
		resp, err := llm.CollectStream(c.Stream(context.Background(), llm.Request{}))
		if !errors.Is(err, errBroken) || resp.InputTokens != 50 || resp.CacheReadTokens != 900 {
			t.Errorf("%s: %+v, %v", name, resp, err)
		}
	}
}

type streamFixture struct {
	name   string
	cfg    func(url string) llm.Config
	chunks []string // SSE data payloads; the last completes the turn
}

func streamFixtures() []streamFixture {
	const n = 8
	repeat := func(chunk string) []string {
		out := make([]string, n)
		for i := range out {
			out[i] = chunk
		}
		return out
	}
	return []streamFixture{
		{
			name: "openai_compat",
			cfg: func(url string) llm.Config {
				return llm.Config{Provider: "openai", Model: "gpt-4.1", APIKey: "k", BaseURL: url}
			},
			chunks: append(repeat(`{"choices":[{"index":0,"delta":{"content":"a"}}]}`),
				`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`, `[DONE]`),
		},
		{
			name: "anthropic",
			cfg: func(url string) llm.Config {
				return llm.Config{Provider: "anthropic", Model: "claude-sonnet-5", APIKey: "k", BaseURL: url}
			},
			chunks: append(append([]string{
				`{"type":"message_start","message":{"usage":{"input_tokens":1}}}`,
				`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			}, repeat(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"a"}}`)...),
				`{"type":"content_block_stop","index":0}`,
				`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
				`{"type":"message_stop"}`),
		},
		{
			name: "gemini",
			cfg: func(url string) llm.Config {
				return llm.Config{Provider: "gemini", Model: "gemini-2.5-flash", APIKey: "k", BaseURL: url}
			},
			chunks: append(repeat(`{"candidates":[{"content":{"role":"model","parts":[{"text":"a"}]}}]}`),
				`{"candidates":[{"content":{"role":"model","parts":[{"text":"."}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}`),
		},
		{
			name: "openai_responses",
			cfg: func(url string) llm.Config {
				return llm.Config{Provider: "openai_responses", Model: "gpt-5", APIKey: "k", BaseURL: url}
			},
			chunks: append(repeat(`{"type":"response.output_text.delta","delta":"a"}`),
				`{"type":"response.completed","response":{"id":"r","status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}`),
		},
	}
}

// pacedSSEServer writes chunks with a gap between them; stallAfter >= 0 makes it
// go silent (without closing) after that many chunks.
func pacedSSEServer(t *testing.T, chunks []string, gap time.Duration, stallAfter int) *httptest.Server {
	t.Helper()
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		flusher.Flush()
		for i, chunk := range chunks {
			if stallAfter >= 0 && i == stallAfter {
				select {
				case <-release:
				case <-r.Context().Done():
				}
				return
			}
			time.Sleep(gap)
			_, _ = fmt.Fprintf(w, "data: %s\n\n", chunk)
			flusher.Flush()
		}
	}))
	t.Cleanup(func() { close(release); srv.Close() })
	return srv
}

func newStreamClient(t *testing.T, cfg llm.Config, timeout time.Duration) llm.Client {
	t.Helper()
	cfg.Timeout = timeout
	cfg.DisableProxy = true
	cfg.DisableRetries = true
	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() { _ = client.Close() })
	return client
}

// A stream whose total duration (~9 × 60ms ≈ 540ms) far exceeds a 200ms
// Config.Timeout completes, because no single gap exceeds it.
func TestStreamOutlivesClientTimeout(t *testing.T) {
	for _, fx := range streamFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			srv := pacedSSEServer(t, fx.chunks, 60*time.Millisecond, -1)
			client := newStreamClient(t, fx.cfg(srv.URL), 200*time.Millisecond)
			start := time.Now()
			var text strings.Builder
			var done bool
			for ev, err := range client.Stream(context.Background(), llm.Request{
				Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
			}) {
				if err != nil {
					t.Fatalf("stream died after %v (Timeout=200ms): %v", time.Since(start), err)
				}
				text.WriteString(ev.Text)
				done = done || ev.Type == llm.EventDone
			}
			if !done || !strings.HasPrefix(text.String(), "aaaaaaaa") {
				t.Fatalf("incomplete stream: done=%v text=%q", done, text.String())
			}
			if elapsed := time.Since(start); elapsed < 400*time.Millisecond {
				t.Fatalf("stream took only %v; fixture did not outlive Timeout", elapsed)
			}
		})
	}
}

// A stream that goes silent mid-response fails after ~Timeout instead of
// hanging until the context (here: 5s) runs out.
func TestStreamIdleGapFailsAfterTimeout(t *testing.T) {
	for _, fx := range streamFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			srv := pacedSSEServer(t, fx.chunks, 0, 3)
			client := newStreamClient(t, fx.cfg(srv.URL), 200*time.Millisecond)
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			start := time.Now()
			var streamErr error
			for _, err := range client.Stream(ctx, llm.Request{
				Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
			}) {
				if err != nil {
					streamErr = err
					break
				}
			}
			elapsed := time.Since(start)
			if streamErr == nil {
				t.Fatal("stalled stream ended without an error")
			}
			if elapsed > 2*time.Second {
				t.Fatalf("stalled stream took %v to fail; idle timeout not applied", elapsed)
			}
			var idleErr *llm.StreamIdleTimeoutError
			var netErr net.Error
			if !errors.As(streamErr, &idleErr) || !errors.As(streamErr, &netErr) || !netErr.Timeout() {
				t.Fatalf("want a StreamIdleTimeoutError (net.Error timeout), got %T: %v", streamErr, streamErr)
			}
		})
	}
}

// The watchdog measures server silence, not the caller: a loop body slower
// than Timeout (here a 300 ms pause per event, Timeout 200 ms) killed a stream
// whose server never paused (B17).
func TestStreamIdleTimeoutIgnoresASlowConsumer(t *testing.T) {
	for _, fx := range streamFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			srv := pacedSSEServer(t, fx.chunks, 0, -1)
			client := newStreamClient(t, fx.cfg(srv.URL), 200*time.Millisecond)
			var done bool
			for ev, err := range client.Stream(context.Background(), llm.Request{
				Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
			}) {
				if err != nil {
					t.Fatalf("slow consumer killed the stream: %v", err)
				}
				done = done || ev.Type == llm.EventDone
				time.Sleep(300 * time.Millisecond)
			}
			if !done {
				t.Fatal("stream ended without EventDone")
			}
		})
	}
}

// Non-stream calls keep the whole-request Timeout.
func TestCompleteStillBoundedByTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		select { // headers sent, body never finishes
		case <-time.After(3 * time.Second):
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()
	client := newStreamClient(t, llm.Config{Provider: "openai", Model: "gpt-4.1", APIKey: "k", BaseURL: srv.URL}, 200*time.Millisecond)
	start := time.Now()
	_, err := client.Complete(context.Background(), llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")}})
	if err == nil || time.Since(start) > 2*time.Second {
		t.Fatalf("Complete err=%v after %v, want a timeout near 200ms", err, time.Since(start))
	}
}

// A stream whose headers never arrive fails via ResponseHeaderTimeout.
func TestStreamHeaderTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-time.After(3 * time.Second):
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()
	client := newStreamClient(t, llm.Config{Provider: "openai", Model: "gpt-4.1", APIKey: "k", BaseURL: srv.URL}, 200*time.Millisecond)
	start := time.Now()
	var streamErr error
	for _, err := range client.Stream(context.Background(), llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")}}) {
		if err != nil {
			streamErr = err
			break
		}
	}
	if streamErr == nil || time.Since(start) > 2*time.Second {
		t.Fatalf("err=%v after %v, want a header timeout near 200ms", streamErr, time.Since(start))
	}
}
