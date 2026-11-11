package llm_test

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider"
)

func TestMockNoScriptErrorsNotPanics(t *testing.T) {
	m := llm.NewMockClient("anthropic", "claude")
	if _, err := m.Complete(context.Background(), llm.Request{}); err == nil {
		t.Error("Complete with empty queue should error")
	}
	var gotErr bool
	for _, err := range m.Stream(context.Background(), llm.Request{}) {
		if err != nil {
			gotErr = true
		}
	}
	if !gotErr {
		t.Error("Stream with empty queue should yield an error")
	}
}

func TestMockSymmetricResponseToStream(t *testing.T) {
	m := llm.NewMockClient("anthropic", "claude")
	m.PushResponse(&llm.Response{
		Thinking: "ponder", Content: "hi",
		ToolCalls:  []llm.ToolCall{{ID: "c1", Name: "f", Input: map[string]any{}}},
		StopReason: "tool_use",
	})
	var types []llm.EventType
	for ev, err := range m.Stream(context.Background(), llm.Request{}) {
		if err != nil {
			t.Fatal(err)
		}
		types = append(types, ev.Type)
	}
	// thinking, content, tool_use, done — in order
	want := []llm.EventType{llm.EventThinking, llm.EventContent, llm.EventToolUse, llm.EventDone}
	if len(types) != len(want) {
		t.Fatalf("event types = %v, want %v", types, want)
	}
	for i := range want {
		if types[i] != want[i] {
			t.Errorf("event[%d] = %v, want %v", i, types[i], want[i])
		}
	}
}

func TestMockSymmetricStreamToResponse(t *testing.T) {
	m := llm.NewMockClient("anthropic", "claude")
	m.PushStream(
		llm.StreamEvent{Type: llm.EventContent, Text: "a"},
		llm.StreamEvent{Type: llm.EventContent, Text: "b"},
		llm.StreamEvent{Type: llm.EventToolUse, ToolCall: &llm.ToolCall{ID: "c1", Name: "f"}},
		llm.StreamEvent{Type: llm.EventDone, StopReason: "tool_use", InputTokens: 5, OutputTokens: 3},
	)
	resp, err := m.Complete(context.Background(), llm.Request{})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "ab" || resp.StopReason != "tool_use" || resp.InputTokens != 5 {
		t.Errorf("assembled response wrong: %+v", resp)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].ID != "c1" {
		t.Errorf("tool call not assembled: %+v", resp.ToolCalls)
	}
}

func TestMockPushErrorPropagates(t *testing.T) {
	boom := errors.New("boom")
	m := llm.NewMockClient("x", "y").PushError(boom)
	if _, err := m.Complete(context.Background(), llm.Request{}); !errors.Is(err, boom) {
		t.Errorf("Complete err = %v, want boom", err)
	}
	m2 := llm.NewMockClient("x", "y").PushError(boom)
	var got error
	for _, err := range m2.Stream(context.Background(), llm.Request{}) {
		got = err
	}
	if !errors.Is(got, boom) {
		t.Errorf("Stream err = %v, want boom", got)
	}
}

func TestMockRecordsRequests(t *testing.T) {
	m := llm.NewMockClient("x", "y").PushText("a").PushText("b")
	_, _ = m.Complete(context.Background(), llm.Request{Model: "m1"})
	_, _ = m.Complete(context.Background(), llm.Request{Model: "m2"})
	if reqs := m.Requests(); len(reqs) != 2 || reqs[0].Model != "m1" || reqs[1].Model != "m2" {
		t.Errorf("requests not recorded: %+v", reqs)
	}
	if last, ok := m.LastRequest(); !ok || last.Model != "m2" {
		t.Errorf("LastRequest = %+v, %v", last, ok)
	}
}

func TestMockResponseFuncPrecedence(t *testing.T) {
	m := llm.NewMockClient("x", "y").PushText("queued")
	m.SetResponseFunc(func(req llm.Request) (*llm.Response, error) {
		return &llm.Response{Content: "dynamic", StopReason: "end_turn"}, nil
	})
	resp, _ := m.Complete(context.Background(), llm.Request{})
	if resp.Content != "dynamic" {
		t.Errorf("respFunc should take precedence over queue, got %q", resp.Content)
	}
}

func TestMockDrivesSessionWithHandoff(t *testing.T) {
	a := llm.NewMockClient("anthropic", "claude").PushText("from A")
	b := llm.NewMockClient("openai", "gpt").PushText("from B")
	s := llm.NewSession(a)
	r1, err := s.Send(context.Background(), "q1")
	if err != nil || r1.Content != "from A" {
		t.Fatalf("send A: %v %+v", err, r1)
	}
	s.SwitchProvider(b)
	r2, err := s.Send(context.Background(), "q2")
	if err != nil || r2.Content != "from B" {
		t.Fatalf("send B: %v %+v", err, r2)
	}
	// B must have received the prior history (q1 + assistant + q2).
	last, _ := b.LastRequest()
	if len(last.Messages) < 3 {
		t.Errorf("handoff did not carry history to B: %d messages", len(last.Messages))
	}
}

func TestMockConcurrentRaceFree(t *testing.T) {
	m := llm.NewMockClient("x", "y").SetResponseFunc(func(req llm.Request) (*llm.Response, error) {
		return &llm.Response{Content: "ok", StopReason: "end_turn"}, nil
	})
	var n int64
	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 10; i++ {
				switch (g + i) % 3 {
				case 0:
					if _, err := m.Complete(context.Background(), llm.Request{}); err == nil {
						atomic.AddInt64(&n, 1)
					}
				case 1:
					for _, err := range m.Stream(context.Background(), llm.Request{}) {
						_ = err
					}
					atomic.AddInt64(&n, 1)
				case 2:
					_ = m.Requests()
				}
			}
		}(g)
	}
	wg.Wait()
	if got := int64(len(m.Requests())); got != n {
		t.Errorf("recorded %d requests, expected %d", got, n)
	}
}
