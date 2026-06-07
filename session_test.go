package llm_test

import (
	"context"
	"iter"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// fakeClient is a scripted in-memory llm.Client for testing Session/handoff
// without real API calls. It records the last Request it received.
type fakeClient struct {
	provider    string
	model       string
	resp        *llm.Response
	events      []llm.StreamEvent
	gotReq      llm.Request
	completeErr error // if set, Complete returns this error
	streamErr   error // if set, Stream yields this error after its events
}

func (f *fakeClient) Complete(_ context.Context, req llm.Request) (*llm.Response, error) {
	f.gotReq = req
	if f.completeErr != nil {
		return nil, f.completeErr
	}
	r := *f.resp
	if r.Model == "" {
		r.Model = f.model
	}
	return &r, nil
}

func (f *fakeClient) Stream(_ context.Context, req llm.Request) iter.Seq2[llm.StreamEvent, error] {
	f.gotReq = req
	return func(yield func(llm.StreamEvent, error) bool) {
		for _, ev := range f.events {
			if !yield(ev, nil) {
				return
			}
		}
		if f.streamErr != nil {
			yield(llm.StreamEvent{}, f.streamErr)
		}
	}
}

func (f *fakeClient) Provider() string { return f.provider }
func (f *fakeClient) Model() string    { return f.model }
func (f *fakeClient) Close() error     { return nil }

func TestSessionSendAppendsAndTracks(t *testing.T) {
	fc := &fakeClient{
		provider: "anthropic", model: "claude-sonnet-4-6",
		resp: &llm.Response{Content: "hello back", InputTokens: 10, OutputTokens: 4, StopReason: "end_turn"},
	}
	s := llm.NewSession(fc, llm.WithSystem("be nice"))

	resp, err := s.Send(context.Background(), "hi")
	if err != nil {
		t.Fatalf("Send: %v", err)
	}
	if resp.Content != "hello back" {
		t.Errorf("content = %q", resp.Content)
	}
	// The request the client saw must include the system prompt + the user turn.
	if fc.gotReq.System != "be nice" {
		t.Errorf("system not threaded: %q", fc.gotReq.System)
	}
	if len(fc.gotReq.Messages) != 1 || fc.gotReq.Messages[0].Role != llm.RoleUser {
		t.Fatalf("expected 1 user message, got %+v", fc.gotReq.Messages)
	}

	conv := s.Conversation()
	if len(conv.Messages) != 2 {
		t.Fatalf("conv has %d messages, want 2 (user+assistant)", len(conv.Messages))
	}
	am := conv.Messages[1]
	if am.Role != llm.RoleAssistant || am.Provider != "anthropic" || am.Model != "claude-sonnet-4-6" {
		t.Errorf("assistant provenance wrong: %+v", am)
	}
	if u := s.Usage(); u.InputTokens != 10 || u.OutputTokens != 4 {
		t.Errorf("usage not tracked: %+v", u)
	}
}

func TestSessionProviderHandoffDegradesThinking(t *testing.T) {
	// Turn 1 on Anthropic: response carries a signed thinking block.
	a := &fakeClient{
		provider: "anthropic", model: "claude-sonnet-4-6",
		resp: &llm.Response{Content: "ok", Thinking: "let me think", ThinkingSignature: "sig-abc", StopReason: "end_turn"},
	}
	s := llm.NewSession(a)
	if _, err := s.Send(context.Background(), "q1"); err != nil {
		t.Fatalf("send1: %v", err)
	}

	// Hand off to OpenAI and send again.
	b := &fakeClient{provider: "openai", model: "gpt-5.5", resp: &llm.Response{Content: "ok2", StopReason: "end_turn"}}
	s.SwitchProvider(b)
	if _, err := s.Send(context.Background(), "q2"); err != nil {
		t.Fatalf("send2: %v", err)
	}

	// The OpenAI client must NOT receive a raw thinking block (foreign signature);
	// it must have been degraded to a text part.
	for _, m := range b.gotReq.Messages {
		for _, p := range m.Content {
			if p.Type == llm.ContentThinking {
				t.Errorf("thinking block leaked to openai after handoff: %+v", p)
			}
		}
	}
	// The reasoning text should survive as a text part somewhere in the history.
	found := false
	for _, m := range b.gotReq.Messages {
		for _, p := range m.Content {
			if p.Type == llm.ContentText && p.Text == "let me think" {
				found = true
			}
		}
	}
	if !found {
		t.Error("degraded thinking text not preserved after handoff")
	}
}

func TestSessionStreamAssemblesTurn(t *testing.T) {
	fc := &fakeClient{
		provider: "anthropic", model: "claude-sonnet-4-6",
		events: []llm.StreamEvent{
			{Type: llm.EventThinking, Thinking: "pondering"},
			{Type: llm.EventContent, Text: "hel"},
			{Type: llm.EventContent, Text: "lo"},
			{Type: llm.EventDone, StopReason: "end_turn", InputTokens: 3, OutputTokens: 2, ThinkingSignature: "sig-x"},
		},
	}
	s := llm.NewSession(fc)

	var streamed string
	for ev, err := range s.Stream(context.Background(), "hi") {
		if err != nil {
			t.Fatalf("stream err: %v", err)
		}
		if ev.Type == llm.EventContent {
			streamed += ev.Text
		}
	}
	if streamed != "hello" {
		t.Errorf("streamed text = %q", streamed)
	}

	conv := s.Conversation()
	if len(conv.Messages) != 2 {
		t.Fatalf("conv has %d messages, want 2", len(conv.Messages))
	}
	am := conv.Messages[1]
	// Assembled assistant message should carry thinking (with signature) + text.
	var gotThinking, gotText bool
	for _, p := range am.Content {
		if p.Type == llm.ContentThinking && p.Thinking == "pondering" && p.ThinkingSignature == "sig-x" {
			gotThinking = true
		}
		if p.Type == llm.ContentText && p.Text == "hello" {
			gotText = true
		}
	}
	if !gotThinking || !gotText {
		t.Errorf("assembled assistant message wrong: %+v", am.Content)
	}
}

func TestNormalizeForProviderRepairsOrphanToolCall(t *testing.T) {
	msgs := []llm.Message{
		llm.NewTextMessage(llm.RoleUser, "do it"),
		{Role: llm.RoleAssistant, Provider: "anthropic", Content: []llm.ContentPart{
			{Type: llm.ContentToolUse, ToolUseID: "call_1", ToolName: "do", ToolInput: map[string]any{}},
		}},
		// no tool result for call_1
	}
	out := llm.NormalizeForProvider(msgs, "anthropic")

	var found bool
	for _, m := range out {
		for _, p := range m.Content {
			if p.Type == llm.ContentToolResult && p.ToolResultID == "call_1" && p.IsError {
				found = true
			}
		}
	}
	if !found {
		t.Errorf("orphan tool call not repaired with synthetic result; got %+v", out)
	}
}

func TestNormalizeForProviderDropsErroredTurn(t *testing.T) {
	msgs := []llm.Message{
		llm.NewTextMessage(llm.RoleUser, "q"),
		{Role: llm.RoleAssistant, Provider: "anthropic", StopReason: "error", Content: []llm.ContentPart{
			{Type: llm.ContentText, Text: "partial"},
		}},
	}
	out := llm.NormalizeForProvider(msgs, "anthropic")
	for _, m := range out {
		if m.Role == llm.RoleAssistant {
			t.Errorf("errored assistant turn was not dropped: %+v", m)
		}
	}
}

func TestNormalizeForProviderKeepsSameProviderThinking(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleAssistant, Provider: "anthropic", Content: []llm.ContentPart{
			{Type: llm.ContentThinking, Thinking: "t", ThinkingSignature: "sig"},
			{Type: llm.ContentText, Text: "answer"},
		}},
	}
	// Same provider → thinking block kept.
	same := llm.NormalizeForProvider(msgs, "anthropic")
	if same[0].Content[0].Type != llm.ContentThinking {
		t.Errorf("same-provider thinking should be kept, got %+v", same[0].Content)
	}
	// Different provider → degraded to text.
	diff := llm.NormalizeForProvider(msgs, "openai")
	for _, p := range diff[0].Content {
		if p.Type == llm.ContentThinking {
			t.Errorf("cross-provider thinking should be degraded, got %+v", diff[0].Content)
		}
	}
}
