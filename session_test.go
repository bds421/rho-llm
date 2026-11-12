package llm_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/provider/anthropic"
)

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

// Breaking out of the range on EventDone is the natural way to consume a
// stream. The turn is complete at that point, but it was discarded: the user
// message, the reply and the usage vanished without an error (B19).
func TestSessionStreamBreakOnDoneKeepsTheTurn(t *testing.T) {
	fc := &fakeClient{
		provider: "anthropic", model: "m",
		events: []llm.StreamEvent{
			{Type: llm.EventContent, Text: "an"},
			{Type: llm.EventContent, Text: "swer"},
			{Type: llm.EventDone, StopReason: "end_turn", InputTokens: 11, OutputTokens: 7},
		},
	}
	s := llm.NewSession(fc)

	for ev, err := range s.Stream(context.Background(), "q") {
		if err != nil {
			t.Fatalf("unexpected err: %v", err)
		}
		if ev.Type == llm.EventDone {
			break
		}
	}
	msgs := s.Conversation().Messages
	if len(msgs) != 2 || msgs[0].Role != llm.RoleUser || msgs[1].Role != llm.RoleAssistant {
		t.Fatalf("after breaking on EventDone: %+v, want [user, assistant]", msgs)
	}
	if got := msgs[1].Content[0].Text; got != "answer" {
		t.Fatalf("assistant text = %q, want %q", got, "answer")
	}
	if u := s.Usage(); u.InputTokens != 11 || u.OutputTokens != 7 {
		t.Fatalf("usage = %+v, want 11 in / 7 out", u)
	}

	// The next turn carries the recorded history.
	for range s.Stream(context.Background(), "again") {
	}
	if n := len(fc.gotReq.Messages); n != 3 {
		t.Fatalf("second turn sent %d messages, want 3 (history + new prompt)", n)
	}
}

// A Model set through WithBaseRequest named the first provider's model and was
// sent to the client SwitchProvider installed, which refuses it (B21).
func TestSwitchProviderDropsTheBaseModel(t *testing.T) {
	first := &fakeClient{provider: "anthropic", model: "claude-sonnet-5-5", resp: &llm.Response{Content: "a", StopReason: "end_turn"}}
	second := &fakeClient{provider: "gemini", model: "gemini-3.8-flash", resp: &llm.Response{Content: "b", StopReason: "end_turn"}}
	s := llm.NewSession(first, llm.WithBaseRequest(llm.Request{Model: "claude-sonnet-5-5", MaxTokens: 321}))

	if _, err := s.Send(context.Background(), "one"); err != nil {
		t.Fatal(err)
	}
	if first.gotReq.Model != "claude-sonnet-5-5" {
		t.Fatalf("first turn model = %q", first.gotReq.Model)
	}
	s.SwitchProvider(second)
	if _, err := s.Send(context.Background(), "two"); err != nil {
		t.Fatal(err)
	}
	if second.gotReq.Model != "gemini-3.8-flash" {
		t.Fatalf("after SwitchProvider the new client got model %q, want its own gemini-3.8-flash", second.gotReq.Model)
	}
	if second.gotReq.MaxTokens != 321 {
		t.Fatalf("other base settings must carry over: MaxTokens = %d", second.gotReq.MaxTokens)
	}
}

// A session that read a text document on Anthropic must keep working after a
// switch to a provider without text documents (B20): the Gemini adapter
// refused the document on every later turn.
func TestSwitchProviderAfterATextDocument(t *testing.T) {
	var gotParts int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Contents []struct {
				Parts []json.RawMessage `json:"parts"`
			} `json:"contents"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		for _, c := range body.Contents {
			gotParts += len(c.Parts)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`))
	}))
	defer srv.Close()

	first := &fakeClient{provider: "anthropic", model: "claude-sonnet-5-5", resp: &llm.Response{Content: "read it", StopReason: "end_turn"}}
	s := llm.NewSession(first)
	if _, err := s.SendMessages(context.Background(), llm.Message{Role: llm.RoleUser, Content: []llm.ContentPart{
		llm.NewTextDocument("notes.txt", "Graz is in Styria."), {Type: llm.ContentText, Text: "read this"},
	}}); err != nil {
		t.Fatal(err)
	}
	gem, err := llm.NewClient(llm.Config{Provider: "gemini", Model: "gemini-3.8-flash", APIKey: "k", BaseURL: srv.URL, DisableRetries: true})
	if err != nil {
		t.Fatal(err)
	}
	defer gem.Close()
	s.SwitchProvider(gem)
	if _, err := s.Send(context.Background(), "where is Graz?"); err != nil {
		t.Fatalf("turn after the handoff failed: %v", err)
	}
	if gotParts != 4 {
		t.Fatalf("Gemini got %d parts, want 4 (document as text, prompt, reply, new prompt)", gotParts)
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

// A Session must COMMIT a turn whose stream completed (saw Done) even if the
// server appended trailing garbage — the complete turn must not be dropped.
func TestSessionCommitsDespiteTrailingGarbage(t *testing.T) {
	srv := sseSrv(
		`data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}`,
		`data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"answer"}}`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
		`data: {trailing garbage}`,
	)
	defer srv.Close()
	c, _ := anthropic.New(llm.Config{Provider: "anthropic", Model: "claude-sonnet-4-6", APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
	sess := llm.NewSession(c)
	for range sess.Stream(context.Background(), "q") {
	}
	conv := sess.Conversation()
	if len(conv.Messages) != 2 {
		t.Fatalf("got %d messages, want 2 (user+assistant) — complete turn dropped by trailing garbage", len(conv.Messages))
	}
	if conv.Messages[1].Content[0].Text != "answer" {
		t.Fatalf("assistant content = %q, want %q", conv.Messages[1].Content[0].Text, "answer")
	}
}

// Mixed concurrent Session ops: turns serialize, reads never block, no race,
// no torn state. Asserts the committed transcript stays well-formed (every
// assistant turn carries provider provenance, message count is even).
func TestSessionConcurrentMixedOpsRaceFree(t *testing.T) {
	mk := func(name string) *llm.MockClient {
		m := llm.NewMockClient(name, name+"-model")
		m.SetResponseFunc(func(llm.Request) (*llm.Response, error) {
			return &llm.Response{Content: "r", StopReason: llm.StopEndTurn, InputTokens: 1, OutputTokens: 1}, nil
		})
		return m
	}
	sess := llm.NewSession(mk("a"))
	alt := mk("b")

	var wg sync.WaitGroup
	for i := 0; i < 24; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			switch n % 6 {
			case 0:
				_, _ = sess.Send(context.Background(), "x")
			case 1:
				for range sess.Stream(context.Background(), "y") {
				}
			case 2:
				_ = sess.Conversation()
			case 3:
				_ = sess.Usage()
			case 4:
				sess.SwitchProvider(alt)
			case 5:
				_ = sess.Provider()
			}
		}(i)
	}
	wg.Wait()

	conv := sess.Conversation()
	if len(conv.Messages)%2 != 0 {
		t.Fatalf("odd message count %d — a turn interleaved or was partially committed", len(conv.Messages))
	}
	for i := 1; i < len(conv.Messages); i += 2 {
		if conv.Messages[i].Role != llm.RoleAssistant {
			t.Fatalf("message %d is %q, expected assistant — turns interleaved", i, conv.Messages[i].Role)
		}
		if conv.Messages[i].Provider == "" {
			t.Fatalf("assistant message %d has no provider provenance", i)
		}
	}
}

// emptyModelClient is a (minimal/misbehaving) custom Client that returns a
// Response with an empty Model verbatim — unlike MockClient, it does NOT
// auto-fill it. Model() reports the model the turn actually runs against.
type emptyModelClient struct{ id string }

func (c *emptyModelClient) Complete(context.Context, llm.Request) (*llm.Response, error) {
	return &llm.Response{Model: "", Content: "ok", StopReason: llm.StopEndTurn, InputTokens: 1_000_000, OutputTokens: 1_000_000}, nil
}

func (c *emptyModelClient) Stream(context.Context, llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		yield(llm.StreamEvent{Type: llm.EventDone, StopReason: llm.StopEndTurn}, nil)
	}
}

func (c *emptyModelClient) Provider() string { return "custom" }

func (c *emptyModelClient) Model() string { return c.id }

func (c *emptyModelClient) Close() error { return nil }

// A Client that returns a Response with an empty Model must not silently zero
// out the cost: the turn ran against a known model (Model() / the request's), so
// usage must be priced against that model, not $0. EstimateCost("") returns 0 —
// a wrong, under-reported cost.
func TestSessionCostFallsBackToRequestModelWhenResponseModelEmpty(t *testing.T) {
	id := uniqueModelID("test-only-session-cost")
	if err := llm.RegisterModel(llm.ModelInfo{ID: id, Provider: "custom", InputPricePer1M: 10, OutputPricePer1M: 30}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}
	sess := llm.NewSession(&emptyModelClient{id: id})
	if _, err := sess.Send(context.Background(), "x"); err != nil {
		t.Fatalf("Send: %v", err)
	}
	if got := sess.Usage().Cost; got != 40.0 {
		t.Fatalf("session cost = %v, want 40.0 — empty Response.Model silently zeroed the cost", got)
	}
}

func TestSessionModelSwitchDropsPreviousModelsThinking(t *testing.T) {
	opus := &fakeClient{provider: "anthropic", model: "claude-opus-5-5", resp: &llm.Response{Content: "ok", StopReason: llm.StopEndTurn}}
	s := llm.NewSession(opus)
	s.Append(llm.NewTextMessage(llm.RoleUser, "q1"), thinkingTurn("anthropic", "claude-opus-5-5"))

	haiku := &fakeClient{provider: "anthropic", model: "claude-haiku-5-5", resp: &llm.Response{Content: "ok", StopReason: llm.StopEndTurn}}
	s.SwitchProvider(haiku)
	if _, err := s.Send(context.Background(), "q2"); err != nil {
		t.Fatal(err)
	}
	for _, m := range haiku.gotReq.Messages {
		for _, p := range m.Content {
			if p.Type == llm.ContentThinking || p.Text == "old reasoning" {
				t.Fatalf("Opus thinking replayed to Haiku: %+v", haiku.gotReq.Messages)
			}
		}
	}

	// Switching back to the same model keeps its own signed thinking.
	opus2 := &fakeClient{provider: "anthropic", model: "claude-opus-5-5", resp: &llm.Response{Content: "ok", StopReason: llm.StopEndTurn}}
	s2 := llm.NewSession(opus2)
	s2.Append(llm.NewTextMessage(llm.RoleUser, "q1"), thinkingTurn("anthropic", "claude-opus-5-5"))
	if _, err := s2.Send(context.Background(), "q2"); err != nil {
		t.Fatal(err)
	}
	if !hasText(opus2.gotReq.Messages[1], "old reasoning") || opus2.gotReq.Messages[1].Content[0].Type != llm.ContentThinking {
		t.Fatalf("same-model thinking not replayed: %+v", opus2.gotReq.Messages)
	}
}

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

func (f *fakeClient) Model() string { return f.model }

func (f *fakeClient) Close() error { return nil }

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

func TestSessionProviderHandoffDropsThinking(t *testing.T) {
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

	// Another model's reasoning is dropped (since v0.11.0 — before, it was
	// degraded to text), neither as a thinking block nor as text; the visible
	// answer survives.
	answer := false
	for _, m := range b.gotReq.Messages {
		for _, p := range m.Content {
			if p.Type == llm.ContentThinking || p.Text == "let me think" {
				t.Errorf("Anthropic thinking reached openai after handoff: %+v", p)
			}
			if p.Text == "ok" {
				answer = true
			}
		}
	}
	if !answer {
		t.Error("visible answer lost after handoff")
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

// R-M1: the snapshot returned by Session.Conversation() must not share any
// mutable state with the live transcript — mutating nested content of the
// snapshot must never corrupt the session.
func TestSessionConversationSnapshotIsDeep(t *testing.T) {
	mock := llm.NewMockClient("mock", "m1").PushText("hello")
	sess := llm.NewSession(mock,
		llm.WithTools(llm.Tool{
			Name:        "search",
			InputSchema: map[string]any{"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}}},
		}),
	)
	if _, err := sess.Send(context.Background(), "hi"); err != nil {
		t.Fatalf("Send: %v", err)
	}

	snap := sess.Conversation()
	snap.Messages[0].Content[0].Text = "TAMPERED"
	snap.Tools[0].InputSchema["type"] = "TAMPERED"

	live := sess.Conversation()
	if got := live.Messages[0].Content[0].Text; got != "hi" {
		t.Fatalf("mutating the snapshot corrupted the live transcript: %q", got)
	}
	if got := live.Tools[0].InputSchema["type"]; got != "object" {
		t.Fatalf("mutating the snapshot corrupted the live tool schema: %v", got)
	}
}

// R-M2: snapshot reads (Conversation/Usage/Provider) must not block while a
// turn is streaming — a long generation must not freeze every observer.
func TestSessionReadsDoNotBlockDuringStream(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	mock := llm.NewMockClient("mock", "m1").SetResponseFunc(func(llm.Request) (*llm.Response, error) {
		close(started)
		<-release
		return llm.FauxResponse("done"), nil
	})
	sess := llm.NewSession(mock)

	streamDone := make(chan struct{})
	go func() {
		defer close(streamDone)
		for ev, err := range sess.Stream(context.Background(), "hi") {
			_ = ev
			if err != nil {
				t.Errorf("stream error: %v", err)
			}
		}
	}()
	<-started // the turn is now in flight and blocked inside the client

	reads := make(chan int, 1)
	go func() {
		_ = sess.Usage()
		_ = sess.Provider()
		conv := sess.Conversation()
		reads <- len(conv.Messages)
	}()

	select {
	case n := <-reads:
		// The uncommitted in-flight turn must be invisible to snapshots.
		if n != 0 {
			t.Fatalf("in-flight (uncommitted) turn visible in snapshot: %d messages", n)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Session reads blocked while a stream is in flight")
	}

	close(release)
	<-streamDone

	if got := len(sess.Conversation().Messages); got != 2 {
		t.Fatalf("after the turn completes the transcript must hold user+assistant, got %d", got)
	}
}

// R-M2 guard: turns stay strictly serialized — two concurrent Sends must not
// interleave or fork the history.
func TestSessionTurnsStillSerialized(t *testing.T) {
	mock := llm.NewMockClient("mock", "m1")
	mock.SetResponseFunc(func(llm.Request) (*llm.Response, error) {
		time.Sleep(5 * time.Millisecond)
		return llm.FauxResponse("r"), nil
	})
	sess := llm.NewSession(mock)

	const turns = 8
	var wg sync.WaitGroup
	for i := 0; i < turns; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := sess.Send(context.Background(), "x"); err != nil {
				t.Errorf("Send: %v", err)
			}
		}()
	}
	wg.Wait()

	conv := sess.Conversation()
	if got := len(conv.Messages); got != 2*turns {
		t.Fatalf("got %d messages, want %d (user+assistant per turn)", got, 2*turns)
	}
	for i, m := range conv.Messages {
		wantRole := llm.RoleUser
		if i%2 == 1 {
			wantRole = llm.RoleAssistant
		}
		if m.Role != wantRole {
			t.Fatalf("message %d role %q — turns interleaved", i, m.Role)
		}
	}
}

// brokenClient streams some text, then fails with an EventError carrying the
// usage reported so far, as the Anthropic adapter does.
type brokenClient struct{ fakeClient }

var errBroken = errors.New("connection reset")

func (b *brokenClient) Stream(context.Context, llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		if !yield(llm.StreamEvent{Type: llm.EventContent, Text: "Part"}, nil) {
			return
		}
		yield(llm.StreamEvent{Type: llm.EventError, Error: errBroken.Error(), Model: "claude-opus-5-5",
			InputTokens: 50, CacheReadTokens: 900}, errBroken)
	}
}

func TestSessionStreamPassesPartialUsage(t *testing.T) {
	s := llm.NewSession(&brokenClient{fakeClient{provider: "anthropic", model: "claude-opus-5-5"}})
	var got llm.StreamEvent
	for ev, err := range llm.StreamWithBoundaries(s.Stream(context.Background(), "q")) {
		if err != nil {
			got = ev
			if !errors.Is(err, errBroken) {
				t.Fatalf("err = %v", err)
			}
		}
	}
	if got.Type != llm.EventError || got.InputTokens != 50 || got.CacheReadTokens != 900 || got.Model != "claude-opus-5-5" {
		t.Errorf("error event lost its usage: %+v", got)
	}
	if n := len(s.Conversation().Messages); n != 0 {
		t.Errorf("broken turn recorded: %d messages", n)
	}
}

func TestSessionRecordsTheAnsweringModel(t *testing.T) {
	f := &fakeClient{provider: "anthropic", model: "claude-opus-5-5", events: []llm.StreamEvent{
		{Type: llm.EventContent, Text: "a"},
		{Type: llm.EventDone, Model: "claude-opus-5", StopReason: llm.StopEndTurn},
	}}
	s := llm.NewSession(f)
	for _, err := range s.Stream(context.Background(), "q") {
		if err != nil {
			t.Fatal(err)
		}
	}
	msgs := s.Conversation().Messages
	if len(msgs) != 2 || msgs[1].Model != "claude-opus-5" {
		t.Errorf("provenance model: %+v", msgs)
	}
}

// The session kept the caller's *Conversation and wrote to it on every turn,
// racing with a caller that still read or saved it (B23). Run with -race.
func TestWithConversationDoesNotShareTheCallersPointer(t *testing.T) {
	conv := llm.NewConversation("sys")
	conv.Append(llm.NewTextMessage(llm.RoleUser, "earlier"))
	fc := &fakeClient{provider: "anthropic", model: "m", resp: &llm.Response{Content: "ok", StopReason: "end_turn"}}
	s := llm.NewSession(fc, llm.WithConversation(conv))
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 50; i++ {
			_, _ = json.Marshal(conv) // the caller saving its copy
		}
	}()
	for i := 0; i < 5; i++ {
		if _, err := s.Send(context.Background(), "q"); err != nil {
			t.Fatal(err)
		}
	}
	<-done
	if len(conv.Messages) != 1 {
		t.Fatalf("caller's conversation changed: %d messages", len(conv.Messages))
	}
	if n := len(s.Conversation().Messages); n != 11 {
		t.Fatalf("session has %d messages, want 11", n)
	}
}
