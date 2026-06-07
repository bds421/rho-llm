package llm_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	llm "github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/provider/anthropic"
)

// =============================================================================
// ADVERSARIAL / BREAK-THE-SYSTEM TESTS
//
// These deliberately feed nil, empty, malformed, corrupt, contradictory, and
// concurrent inputs into the conversation/session/handoff machinery. They assert
// it degrades safely (no panic, no data race, no transcript corruption) rather
// than testing the happy path.
// =============================================================================

// ---- nil / empty handling ---------------------------------------------------

func TestNewAssistantMessagePanicsOnNil(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("NewAssistantMessage(nil) did not panic")
		}
	}()
	_ = llm.NewAssistantMessage(nil)
}

func TestUsageAddResponseNilSafe(t *testing.T) {
	var u llm.Usage
	u.AddResponse(nil) // must not panic
	if u.InputTokens != 0 || u.Cost != 0 {
		t.Errorf("nil response mutated usage: %+v", u)
	}
}

func TestNormalizeNilAndEmpty(t *testing.T) {
	if got := llm.NormalizeForProvider(nil, ""); len(got) != 0 {
		t.Errorf("nil input -> %v", got)
	}
	if got := llm.NormalizeForProvider([]llm.Message{}, "anthropic"); len(got) != 0 {
		t.Errorf("empty input -> %v", got)
	}
}

func TestConversationToRequestEmpty(t *testing.T) {
	c := llm.NewConversation("")
	req := c.ToRequest(llm.Request{Model: "m"})
	if len(req.Messages) != 0 || req.Model != "m" {
		t.Errorf("unexpected request from empty conversation: %+v", req)
	}
}

// ---- corrupt / malformed serialization --------------------------------------

func TestLoadConversationMalformed(t *testing.T) {
	cases := []struct {
		name    string
		data    string
		wantErr bool
	}{
		{"empty", ``, true},
		{"garbage", `not json at all`, true},
		{"array not object", `[]`, true},
		{"truncated", `{"messages":[{"role":"user"`, true},
		{"null", `null`, false},                              // -> empty conversation, version normalized
		{"empty object", `{}`, false},                        // -> version normalized to 1
		{"negative version", `{"schema_version":-7}`, false}, // -> normalized to 1
		{"null messages", `{"schema_version":1,"messages":null}`, false},
		{"future version", `{"schema_version":999999}`, true}, // newer than build -> reject
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c, err := llm.LoadConversation([]byte(tc.data))
			if tc.wantErr && err == nil {
				t.Errorf("expected error, got conv %+v", c)
			}
			if !tc.wantErr {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				} else if c.SchemaVersion < 1 {
					t.Errorf("schema_version not normalized: %d", c.SchemaVersion)
				}
			}
		})
	}
}

func TestConversationRoundTripAllContentTypes(t *testing.T) {
	conv := llm.NewConversation("sys",
		llm.Tool{Name: "t", Description: "d", InputSchema: map[string]any{"type": "object"}})
	conv.Append(
		llm.Message{Role: llm.RoleUser, Content: []llm.ContentPart{
			{Type: llm.ContentText, Text: "look"},
			{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "image/png", Data: "AAAA"}},
			{Type: llm.ContentDocument, Document: &llm.DocumentSource{Type: "base64", MediaType: "application/pdf", Data: "JVBE"}},
		}},
		llm.Message{Role: llm.RoleAssistant, Provider: "anthropic", Model: "claude-sonnet-4-6", StopReason: "tool_use", Content: []llm.ContentPart{
			{Type: llm.ContentThinking, Thinking: "reason", ThinkingSignature: "sig"},
			{Type: llm.ContentThinking, Thinking: "enc", Redacted: true},
			{Type: llm.ContentText, Text: "calling"},
			{Type: llm.ContentToolUse, ToolUseID: "c1", ToolName: "t", ToolInput: map[string]any{"x": 1.0}, ThoughtSignature: "ts"},
		}},
		llm.NewToolResultMessage("c1", "42", false),
	)

	blob, err := json.Marshal(conv)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := llm.LoadConversation(blob)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if !reflect.DeepEqual(conv.Messages, got.Messages) {
		t.Errorf("messages changed across round-trip:\n in: %+v\nout: %+v", conv.Messages, got.Messages)
	}
}

// ---- NormalizeForProvider: hostile transcripts ------------------------------

func assistantTool(provider, id string) llm.Message {
	return llm.Message{Role: llm.RoleAssistant, Provider: provider, StopReason: "tool_use", Content: []llm.ContentPart{
		{Type: llm.ContentToolUse, ToolUseID: id, ToolName: "f", ToolInput: map[string]any{}},
	}}
}

func TestNormalizeDropsDanglingToolResult(t *testing.T) {
	// A tool result whose id matches NO tool call must be dropped.
	msgs := []llm.Message{
		llm.NewTextMessage(llm.RoleUser, "hi"),
		llm.NewToolResultMessage("ghost", "stale", false),
	}
	out := llm.NormalizeForProvider(msgs, "anthropic")
	for _, m := range out {
		for _, p := range m.Content {
			if p.Type == llm.ContentToolResult {
				t.Errorf("dangling tool result survived: %+v", p)
			}
		}
	}
}

func TestNormalizeMultipleOrphanCalls(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleAssistant, Provider: "anthropic", StopReason: "tool_use", Content: []llm.ContentPart{
			{Type: llm.ContentToolUse, ToolUseID: "a", ToolName: "f"},
			{Type: llm.ContentToolUse, ToolUseID: "b", ToolName: "g"},
		}},
	}
	out := llm.NormalizeForProvider(msgs, "anthropic")
	got := map[string]bool{}
	for _, m := range out {
		for _, p := range m.Content {
			if p.Type == llm.ContentToolResult && p.IsError {
				got[p.ToolResultID] = true
			}
		}
	}
	if !got["a"] || !got["b"] {
		t.Errorf("expected synthetic results for both orphan calls, got %v", got)
	}
}

func TestNormalizeErroredTurnDropsItsResult(t *testing.T) {
	msgs := []llm.Message{
		llm.NewTextMessage(llm.RoleUser, "q"),
		{Role: llm.RoleAssistant, Provider: "anthropic", StopReason: "error", Content: []llm.ContentPart{
			{Type: llm.ContentToolUse, ToolUseID: "dead", ToolName: "f"},
		}},
		llm.NewToolResultMessage("dead", "result-for-dead-call", false),
	}
	out := llm.NormalizeForProvider(msgs, "anthropic")
	for _, m := range out {
		if m.Role == llm.RoleAssistant {
			t.Errorf("errored turn survived: %+v", m)
		}
		for _, p := range m.Content {
			if p.Type == llm.ContentToolResult && p.ToolResultID == "dead" {
				t.Errorf("result for dropped errored call survived: %+v", p)
			}
		}
	}
}

func TestNormalizeIsIdempotent(t *testing.T) {
	msgs := []llm.Message{
		llm.NewTextMessage(llm.RoleUser, "q"),
		assistantTool("anthropic", "x"), // orphan -> gets synthetic result on first pass
		{Role: llm.RoleAssistant, Provider: "anthropic", StopReason: "aborted", Content: []llm.ContentPart{
			{Type: llm.ContentText, Text: "partial"},
		}},
	}
	once := llm.NormalizeForProvider(msgs, "anthropic")
	twice := llm.NormalizeForProvider(once, "anthropic")
	if !reflect.DeepEqual(once, twice) {
		t.Errorf("normalize not idempotent:\nonce:  %+v\ntwice: %+v", once, twice)
	}
}

func TestNormalizeDoesNotMutateInput(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleAssistant, Provider: "anthropic", Content: []llm.ContentPart{
			{Type: llm.ContentThinking, Thinking: "keep me", ThinkingSignature: "sig"},
		}},
	}
	before, _ := json.Marshal(msgs)
	_ = llm.NormalizeForProvider(msgs, "openai") // would degrade thinking -> text in the COPY
	after, _ := json.Marshal(msgs)
	if string(before) != string(after) {
		t.Errorf("input mutated by NormalizeForProvider:\nbefore: %s\nafter:  %s", before, after)
	}
}

func TestNormalizeUnsignedThinkingDegradesEvenSameProvider(t *testing.T) {
	// Same provider but NO signature -> cannot be replayed validly -> degrade to text.
	msgs := []llm.Message{
		{Role: llm.RoleAssistant, Provider: "anthropic", Content: []llm.ContentPart{
			{Type: llm.ContentThinking, Thinking: "unsigned"},
		}},
	}
	out := llm.NormalizeForProvider(msgs, "anthropic")
	for _, p := range out[0].Content {
		if p.Type == llm.ContentThinking {
			t.Errorf("unsigned thinking kept as thinking block: %+v", out[0].Content)
		}
	}
}

func TestNormalizePreservesToolCallThoughtSignature(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleAssistant, Provider: "gemini", Content: []llm.ContentPart{
			{Type: llm.ContentToolUse, ToolUseID: "c", ToolName: "f", ThoughtSignature: "gemini-ts"},
		}},
		llm.NewToolResultMessage("c", "ok", false),
	}
	out := llm.NormalizeForProvider(msgs, "gemini")
	var found bool
	for _, m := range out {
		for _, p := range m.Content {
			if p.Type == llm.ContentToolUse && p.ThoughtSignature == "gemini-ts" {
				found = true
			}
		}
	}
	if !found {
		t.Errorf("tool-call ThoughtSignature lost through normalize: %+v", out)
	}
}

func TestNormalizeLargeTranscriptNoPanic(t *testing.T) {
	var msgs []llm.Message
	for i := 0; i < 5000; i++ {
		msgs = append(msgs, llm.NewTextMessage(llm.RoleUser, "x"))
		msgs = append(msgs, llm.Message{Role: llm.RoleAssistant, Provider: "anthropic", Content: []llm.ContentPart{
			{Type: llm.ContentText, Text: "y"},
		}})
	}
	if got := llm.NormalizeForProvider(msgs, "anthropic"); len(got) != len(msgs) {
		t.Errorf("large transcript length changed unexpectedly: %d -> %d", len(msgs), len(got))
	}
}

// ---- Session: failure rollback ----------------------------------------------

func TestSessionSendFailureRollsBack(t *testing.T) {
	fc := &fakeClient{provider: "anthropic", model: "m", completeErr: errors.New("boom")}
	s := llm.NewSession(fc)

	if _, err := s.Send(context.Background(), "q1"); err == nil {
		t.Fatal("expected error")
	}
	if n := len(s.Conversation().Messages); n != 0 {
		t.Fatalf("failed Send left %d messages; want 0 (rolled back)", n)
	}

	// Recover and succeed: must not stack two user turns.
	fc.completeErr = nil
	fc.resp = &llm.Response{Content: "ok", StopReason: "end_turn"}
	if _, err := s.Send(context.Background(), "q2"); err != nil {
		t.Fatalf("recovery send: %v", err)
	}
	msgs := s.Conversation().Messages
	if len(msgs) != 2 || msgs[0].Role != llm.RoleUser || msgs[1].Role != llm.RoleAssistant {
		t.Fatalf("after recovery want [user, assistant], got %+v", msgs)
	}
}

func TestSessionStreamErrorRollsBack(t *testing.T) {
	fc := &fakeClient{
		provider: "anthropic", model: "m",
		events:    []llm.StreamEvent{{Type: llm.EventContent, Text: "par"}},
		streamErr: errors.New("mid-stream boom"),
	}
	s := llm.NewSession(fc)

	var gotErr bool
	for _, err := range s.Stream(context.Background(), "q") {
		if err != nil {
			gotErr = true
		}
	}
	if !gotErr {
		t.Fatal("expected a stream error")
	}
	if n := len(s.Conversation().Messages); n != 0 {
		t.Errorf("errored stream left %d messages; want 0 (rolled back)", n)
	}
}

func TestSessionStreamBreakEarlyRollsBack(t *testing.T) {
	fc := &fakeClient{
		provider: "anthropic", model: "m",
		events: []llm.StreamEvent{
			{Type: llm.EventContent, Text: "a"},
			{Type: llm.EventContent, Text: "b"},
			{Type: llm.EventDone, StopReason: "end_turn"},
		},
	}
	s := llm.NewSession(fc)

	for _, err := range s.Stream(context.Background(), "q") {
		if err != nil {
			t.Fatalf("unexpected err: %v", err)
		}
		break // bail after the first event
	}
	if n := len(s.Conversation().Messages); n != 0 {
		t.Errorf("abandoned stream left %d messages; want 0 (rolled back)", n)
	}
}

func TestSessionStreamNoDoneRollsBack(t *testing.T) {
	// Stream ends cleanly but never sends EventDone (spec-violating provider).
	fc := &fakeClient{
		provider: "anthropic", model: "m",
		events: []llm.StreamEvent{{Type: llm.EventContent, Text: "incomplete"}},
	}
	s := llm.NewSession(fc)
	for _, err := range s.Stream(context.Background(), "q") {
		if err != nil {
			t.Fatalf("unexpected err: %v", err)
		}
	}
	if n := len(s.Conversation().Messages); n != 0 {
		t.Errorf("no-EventDone stream left %d messages; want 0 (rolled back)", n)
	}
}

// ---- Session: handoff integrity ---------------------------------------------

func TestSessionHandoffThereAndBackPreservesThinking(t *testing.T) {
	a := &fakeClient{provider: "anthropic", model: "claude",
		resp: &llm.Response{Content: "a1", Thinking: "deep", ThinkingSignature: "SIG", StopReason: "end_turn"}}
	s := llm.NewSession(a)
	if _, err := s.Send(context.Background(), "q1"); err != nil {
		t.Fatal(err)
	}

	// Detour to OpenAI (thinking should degrade to text for it)...
	b := &fakeClient{provider: "openai", model: "gpt", resp: &llm.Response{Content: "b1", StopReason: "end_turn"}}
	s.SwitchProvider(b)
	if _, err := s.Send(context.Background(), "q2"); err != nil {
		t.Fatal(err)
	}

	// ...and back to Anthropic. The ORIGINAL signed thinking block must still be
	// in storage and replayed verbatim — the openai detour must not corrupt it.
	s.SwitchProvider(a)
	if _, err := s.Send(context.Background(), "q3"); err != nil {
		t.Fatal(err)
	}
	var replayed bool
	for _, m := range a.gotReq.Messages {
		for _, p := range m.Content {
			if p.Type == llm.ContentThinking && p.ThinkingSignature == "SIG" {
				replayed = true
			}
		}
	}
	if !replayed {
		t.Error("signed thinking block was not preserved across A->B->A handoff")
	}
}

func TestNormalizeDropsToolResultBeforeItsCall(t *testing.T) {
	// Out-of-order transcript: a tool_result appears BEFORE the tool_use it
	// answers. Every provider rejects a result with no preceding call.
	msgs := []llm.Message{
		llm.NewToolResultMessage("c1", "premature", false), // result first
		assistantTool("anthropic", "c1"),                   // its call comes later
	}
	out := llm.NormalizeForProvider(msgs, "anthropic")

	// Invariant: every tool_result must appear AFTER a tool_use with the same id.
	seen := map[string]bool{}
	for _, m := range out {
		for _, p := range m.Content {
			if p.Type == llm.ContentToolUse {
				seen[p.ToolUseID] = true
			}
			if p.Type == llm.ContentToolResult && !seen[p.ToolResultID] {
				t.Errorf("tool_result %q appears before its call: %+v", p.ToolResultID, out)
			}
		}
	}
}

func TestNormalizeDeduplicatesToolUseIDs(t *testing.T) {
	// Two turns each emit a tool call with the SAME id (e.g. Gemini's per-turn
	// synthetic ids colliding across turns), each answered. A handoff to a
	// provider that requires unique ids must not see duplicates.
	msgs := []llm.Message{
		assistantTool("gemini", "call_0_f"),
		llm.NewToolResultMessage("call_0_f", "r1", false),
		assistantTool("gemini", "call_0_f"),
		llm.NewToolResultMessage("call_0_f", "r2", false),
	}
	out := llm.NormalizeForProvider(msgs, "anthropic")

	var callIDs []string
	resultCounts := map[string]int{}
	for _, m := range out {
		for _, p := range m.Content {
			if p.Type == llm.ContentToolUse {
				callIDs = append(callIDs, p.ToolUseID)
			}
			if p.Type == llm.ContentToolResult {
				resultCounts[p.ToolResultID]++
			}
		}
	}
	if len(callIDs) != 2 {
		t.Fatalf("want 2 tool calls, got %v", callIDs)
	}
	if callIDs[0] == callIDs[1] {
		t.Errorf("duplicate tool_use ids not deduplicated: %v", callIDs)
	}
	for _, id := range callIDs {
		if resultCounts[id] != 1 {
			t.Errorf("call %q not paired with exactly one result (got %d): %+v", id, resultCounts[id], out)
		}
	}
}

func TestToolInputNumberDriftIsWireSafe(t *testing.T) {
	// Known Go-JSON behavior: integers in ToolInput (typed any) deserialize as
	// float64. This is a documented limitation for callers that type-assert the
	// in-memory value — but the WIRE bytes are identical, so a resend (the thing
	// that actually matters) is unaffected.
	conv := llm.NewConversation("")
	conv.Append(llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentPart{
		{Type: llm.ContentToolUse, ToolUseID: "c", ToolName: "f", ToolInput: map[string]any{"page": 3}},
	}})

	blob, err := json.Marshal(conv)
	if err != nil {
		t.Fatal(err)
	}
	got, err := llm.LoadConversation(blob)
	if err != nil {
		t.Fatal(err)
	}

	// In-memory type drifts int -> float64 (documented)...
	in := got.Messages[0].Content[0].ToolInput.(map[string]any)
	if _, isFloat := in["page"].(float64); !isFloat {
		t.Errorf("expected float64 after reload (Go JSON), got %T", in["page"])
	}
	// ...but the re-serialized bytes are byte-identical, so resend is safe.
	reblob, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if string(blob) != string(reblob) {
		t.Errorf("wire form changed across reload:\n in: %s\nout: %s", blob, reblob)
	}
}

// ---- Usage: sentinel / negative pollution -----------------------------------

func TestUsageIgnoresTokensNotReportedSentinel(t *testing.T) {
	var u llm.Usage
	u.AddResponse(&llm.Response{
		Model:        "claude-sonnet-4-6",
		InputTokens:  llm.TokensNotReported, // -1 sentinel from a stream without usage
		OutputTokens: llm.TokensNotReported,
	})
	if u.InputTokens != 0 || u.OutputTokens != 0 {
		t.Errorf("TokensNotReported polluted usage: %+v", u)
	}
	if u.Cost < 0 {
		t.Errorf("negative cost from sentinel tokens: %v", u.Cost)
	}
}

func TestSessionStreamWithoutUsageNoNegativeTotals(t *testing.T) {
	fc := &fakeClient{
		provider: "anthropic", model: "claude-sonnet-4-6",
		events: []llm.StreamEvent{
			{Type: llm.EventContent, Text: "hi"},
			{Type: llm.EventDone, StopReason: "end_turn", InputTokens: llm.TokensNotReported, OutputTokens: llm.TokensNotReported},
		},
	}
	s := llm.NewSession(fc)
	for _, err := range s.Stream(context.Background(), "q") {
		if err != nil {
			t.Fatal(err)
		}
	}
	if u := s.Usage(); u.InputTokens < 0 || u.OutputTokens < 0 || u.Cost < 0 {
		t.Errorf("stream without usage produced negative totals: %+v", u)
	}
}

// ---- Integration: full Session -> real Anthropic adapter (httptest) ----------

// TestSessionAnthropicThinkingRoundTripIntegration drives the ENTIRE new stack
// through the real Anthropic adapter against a mock server: capture the thinking
// signature, store it, normalize, and replay it on the next turn's wire request.
func TestSessionAnthropicThinkingRoundTripIntegration(t *testing.T) {
	var secondBody []byte
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		calls++
		w.Header().Set("Content-Type", "application/json")
		if calls == 1 {
			// First turn: a thinking block (+signature) followed by a tool call.
			fmt.Fprint(w, `{"id":"m1","type":"message","role":"assistant","model":"claude-sonnet-4-6",
				"content":[
					{"type":"thinking","thinking":"let me compute","signature":"SIG123"},
					{"type":"tool_use","id":"tool_1","name":"calc","input":{"x":1}}
				],"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":5}}`)
			return
		}
		secondBody = body
		fmt.Fprint(w, `{"id":"m2","type":"message","role":"assistant","model":"claude-sonnet-4-6",
			"content":[{"type":"text","text":"done"}],"stop_reason":"end_turn","usage":{"input_tokens":12,"output_tokens":3}}`)
	}))
	defer srv.Close()

	client, err := anthropic.New(llm.Config{
		Provider: "anthropic", Model: "claude-sonnet-4-6", APIKey: "k",
		BaseURL: srv.URL, MaxTokens: 1024, ThinkingLevel: llm.ThinkingLow,
	})
	if err != nil {
		t.Fatalf("anthropic.New: %v", err)
	}
	sess := llm.NewSession(client, llm.WithBaseRequest(llm.Request{ThinkingLevel: llm.ThinkingLow, MaxTokens: 1024}))

	resp, err := sess.Send(context.Background(), "compute 1")
	if err != nil {
		t.Fatalf("turn 1: %v", err)
	}
	if resp.ThinkingSignature != "SIG123" {
		t.Errorf("thinking signature not captured from response: %q", resp.ThinkingSignature)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("want 1 tool call, got %+v", resp.ToolCalls)
	}

	// Continue the tool loop — this builds the SECOND wire request.
	if _, err := sess.SendMessages(context.Background(), llm.NewToolResultMessage(resp.ToolCalls[0].ID, "2", false)); err != nil {
		t.Fatalf("turn 2: %v", err)
	}

	// The replayed transcript must carry the thinking block + its signature.
	if !strings.Contains(string(secondBody), `"type":"thinking"`) {
		t.Errorf("second request missing thinking block:\n%s", secondBody)
	}
	if !strings.Contains(string(secondBody), `"signature":"SIG123"`) {
		t.Errorf("second request missing replayed thinking signature:\n%s", secondBody)
	}
	if !strings.Contains(string(secondBody), `"tool_use_id":"tool_1"`) {
		t.Errorf("second request missing the tool result pairing:\n%s", secondBody)
	}
}

// ---- Session: concurrency (run under -race) ---------------------------------

func TestSessionConcurrentAccessRaceFree(t *testing.T) {
	a := &fakeClient{provider: "anthropic", model: "m1", resp: &llm.Response{Content: "x", StopReason: "end_turn"}}
	b := &fakeClient{provider: "openai", model: "m2", resp: &llm.Response{Content: "y", StopReason: "end_turn"}}
	s := llm.NewSession(a)

	const goroutines = 16
	const perG = 8
	var sends int64
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for g := 0; g < goroutines; g++ {
		go func(g int) {
			defer wg.Done()
			for i := 0; i < perG; i++ {
				switch (g + i) % 4 {
				case 0:
					if _, err := s.Send(context.Background(), "hammer"); err == nil {
						atomic.AddInt64(&sends, 1)
					}
				case 1:
					_ = s.Conversation() // snapshot read while others write
				case 2:
					_ = s.Usage()
				case 3:
					if g%2 == 0 {
						s.SwitchProvider(b)
					} else {
						s.SwitchProvider(a)
					}
				}
			}
		}(g)
	}
	wg.Wait()

	// Every successful Send appends exactly a (user, assistant) pair.
	if n := len(s.Conversation().Messages); int64(n) != 2*sends {
		t.Errorf("message count %d != 2*sends %d — concurrent corruption", n, 2*sends)
	}
}
