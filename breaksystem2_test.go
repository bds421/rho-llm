package llm_test

// Break-the-system pass, round 2. Attacks surfaces the first round didn't:
// misbehaving Client (nil response), ValidateToolCall on hostile schemas,
// adapter tool-call accumulation on malformed framing, NormalizeForProvider
// ID-collision corner cases, and StreamWithBoundaries on a mid-block error.

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"math"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/provider/anthropic"
)

// nilClient is a hostile Client: Complete returns (nil, nil) — no response and
// no error, which a buggy custom adapter could do.
type nilClient struct{}

func (nilClient) Complete(context.Context, llm.Request) (*llm.Response, error) {
	return nil, nil
}
func (nilClient) Stream(_ context.Context, _ llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {}
}
func (nilClient) Provider() string { return "nilprov" }
func (nilClient) Model() string    { return "nilmodel" }
func (nilClient) Close() error     { return nil }

// A Client returning (nil, nil) must NOT panic the Session — it must surface as
// an error, and the transcript must stay unchanged.
func TestSessionNilResponseDoesNotPanic(t *testing.T) {
	sess := llm.NewSession(nilClient{})
	resp, err := sess.Send(context.Background(), "hi")
	if err == nil {
		t.Fatal("want an error when the client returns (nil, nil), got nil")
	}
	if resp != nil {
		t.Fatalf("want nil response, got %+v", resp)
	}
	if n := len(sess.Conversation().Messages); n != 0 {
		t.Fatalf("failed turn must leave the transcript empty, got %d messages", n)
	}
}

// A streaming turn that ends WITHOUT an EventDone (the nilClient yields nothing)
// must not be recorded and must not panic.
func TestSessionNilStreamDoesNotPanicOrCommit(t *testing.T) {
	sess := llm.NewSession(nilClient{})
	for range sess.Stream(context.Background(), "hi") {
	}
	if n := len(sess.Conversation().Messages); n != 0 {
		t.Fatalf("stream with no Done must record nothing, got %d messages", n)
	}
}

// ValidateToolCall must never PANIC, even on a hostile schema. An `enum`
// containing an uncomparable value (a slice) would panic with a naive `==`.
func TestValidateToolCallEnumUncomparableNoPanic(t *testing.T) {
	tool := llm.Tool{
		Name: "t",
		InputSchema: map[string]any{
			"properties": map[string]any{
				// enum entries are slices (uncomparable with ==)
				"p": map[string]any{"enum": []any{[]any{"x"}, []any{"y"}}},
			},
		},
	}
	// arg value is also a slice — same dynamic type as the enum entries.
	call := llm.ToolCall{Name: "t", Input: map[string]any{"p": []any{"x"}}}

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("ValidateToolCall panicked on uncomparable enum: %v", r)
		}
	}()
	_ = llm.ValidateToolCall(tool, call) // must return (nil or error), never panic
}

// ValidateToolCall with a `required` list holding non-strings must not panic.
func TestValidateToolCallHostileRequiredNoPanic(t *testing.T) {
	tool := llm.Tool{
		Name: "t",
		InputSchema: map[string]any{
			"required": []any{123, []any{"nested"}, nil, "realfield"},
			"properties": map[string]any{
				"realfield": map[string]any{"type": "string"},
			},
		},
	}
	call := llm.ToolCall{Name: "t", Input: map[string]any{"realfield": "v"}}
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("ValidateToolCall panicked on hostile required: %v", r)
		}
	}()
	_ = llm.ValidateToolCall(tool, call)
}

// ValidateToolCall must reject non-finite floats for an integer field — +Inf
// would otherwise pass (math.Trunc(Inf)==Inf) and serialize to invalid JSON.
func TestValidateToolCallRejectsNonFiniteInteger(t *testing.T) {
	tool := llm.Tool{Name: "t", InputSchema: map[string]any{
		"properties": map[string]any{"n": map[string]any{"type": "integer"}},
	}}
	if err := llm.ValidateToolCall(tool, llm.ToolCall{Name: "t", Input: map[string]any{"n": math.Inf(1)}}); err == nil {
		t.Error("ValidateToolCall accepted +Inf as a valid integer")
	}
	if err := llm.ValidateToolCall(tool, llm.ToolCall{Name: "t", Input: map[string]any{"n": math.NaN()}}); err == nil {
		t.Error("ValidateToolCall accepted NaN as a valid integer")
	}
}

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
	var ids []string
	for ev, err := range c.Stream(context.Background(), llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "x")}}) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if ev.Type == llm.EventToolUse && ev.ToolCall != nil {
			ids = append(ids, ev.ToolCall.ID)
		}
	}
	if len(ids) != 2 {
		t.Fatalf("got tool calls %v, want both call_A and call_B — first lost on missing content_block_stop", ids)
	}
}

// NormalizeForProvider must produce unique tool_use IDs and correctly paired
// results even when a literal "dup#1" coexists with a duplicated "dup" (whose
// dedup target is also "dup#1").
func TestNormalizeIDCollisionWithLiteralSuffix(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleAssistant, Provider: "anthropic", Content: []llm.ContentPart{
			{Type: llm.ContentToolUse, ToolUseID: "dup", ToolName: "f", ToolInput: map[string]any{}},
		}},
		llm.NewToolResultMessage("dup", "r0", false),
		{Role: llm.RoleAssistant, Provider: "anthropic", Content: []llm.ContentPart{
			{Type: llm.ContentToolUse, ToolUseID: "dup#1", ToolName: "f", ToolInput: map[string]any{}}, // literal collision target
		}},
		llm.NewToolResultMessage("dup#1", "r1", false),
		{Role: llm.RoleAssistant, Provider: "anthropic", Content: []llm.ContentPart{
			{Type: llm.ContentToolUse, ToolUseID: "dup", ToolName: "f", ToolInput: map[string]any{}}, // duplicate of first "dup"
		}},
		llm.NewToolResultMessage("dup", "r2", false),
	}
	out := llm.NormalizeForProvider(msgs, "anthropic")

	// Collect tool_use IDs and tool_result IDs.
	seen := map[string]int{}
	var resultIDs []string
	for _, m := range out {
		for _, p := range m.Content {
			switch p.Type {
			case llm.ContentToolUse:
				seen[p.ToolUseID]++
			case llm.ContentToolResult:
				resultIDs = append(resultIDs, p.ToolResultID)
			}
		}
	}
	for id, n := range seen {
		if n != 1 {
			t.Fatalf("tool_use ID %q appears %d times — dedup produced a collision", id, n)
		}
	}
	// Every tool_result must reference an existing tool_use ID (no dangling pairing).
	for _, rid := range resultIDs {
		if _, ok := seen[rid]; !ok {
			t.Fatalf("tool_result references %q which is not an emitted tool_use ID", rid)
		}
	}
	// Normalize must be idempotent — a second pass must not stack suffixes.
	out2 := llm.NormalizeForProvider(out, "anthropic")
	if len(out2) != len(out) {
		t.Fatalf("normalize not idempotent: len %d then %d", len(out), len(out2))
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

// Concurrency: drive a Session while concurrently snapshotting and persisting
// it to a Store. Conversation() must hand back independent copies so a Save
// running alongside a Send never races or serializes a torn transcript.
func TestSessionStorePersistenceUnderConcurrency(t *testing.T) {
	mock := llm.NewMockClient("anthropic", "m")
	mock.SetResponseFunc(func(llm.Request) (*llm.Response, error) {
		return &llm.Response{Content: "r", StopReason: llm.StopEndTurn, InputTokens: 1, OutputTokens: 1}, nil
	})
	sess := llm.NewSession(mock)
	store := llm.NewMemoryStore()
	ctx := context.Background()

	var wg sync.WaitGroup
	for i := 0; i < 12; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			for j := 0; j < 10; j++ {
				switch n % 3 {
				case 0:
					_, _ = sess.Send(ctx, "x")
				case 1:
					if err := store.Save(ctx, "s", sess.Conversation()); err != nil {
						t.Errorf("save: %v", err)
					}
				case 2:
					if c, err := store.Load(ctx, "s"); err != nil && !errors.Is(err, llm.ErrConversationNotFound) {
						t.Errorf("load saw corruption: %v", err)
					} else if c != nil && len(c.Messages)%2 != 0 {
						t.Errorf("persisted a torn transcript: %d messages", len(c.Messages))
					}
				}
			}
		}(i)
	}
	wg.Wait()
}
