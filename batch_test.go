package llm_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider" // register adapters for NewClient
)

// =============================================================================
// P6 MockClient — adversarial
// =============================================================================

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

// =============================================================================
// P9 model discovery — adversarial
// =============================================================================

func TestModelsNonEmptySortedUnique(t *testing.T) {
	models := llm.Models()
	if len(models) == 0 {
		t.Fatal("Models() is empty")
	}
	seen := map[string]bool{}
	for i, m := range models {
		if m.ID == "" {
			t.Errorf("model[%d] has empty ID", i)
		}
		if seen[m.ID] {
			t.Errorf("duplicate model ID %q", m.ID)
		}
		seen[m.ID] = true
		if i > 0 && models[i-1].ID > m.ID {
			t.Errorf("Models() not sorted at %d: %q > %q", i, models[i-1].ID, m.ID)
		}
	}
}

func TestModelsByProviderUnknownIsEmptyNotNilPanic(t *testing.T) {
	got := llm.ModelsByProvider("does-not-exist")
	if got == nil {
		t.Error("ModelsByProvider(unknown) returned nil, want empty slice")
	}
	if len(got) != 0 {
		t.Errorf("ModelsByProvider(unknown) = %d models, want 0", len(got))
	}
}

func TestModelsByProviderConsistent(t *testing.T) {
	for _, m := range llm.ModelsByProvider("anthropic") {
		if m.Provider != "anthropic" {
			t.Errorf("ModelsByProvider(anthropic) returned %q model %q", m.Provider, m.ID)
		}
	}
	if len(llm.ModelsByProvider("anthropic")) == 0 {
		t.Error("expected anthropic models")
	}
}

func TestProvidersIncludesKnownSorted(t *testing.T) {
	ps := llm.Providers()
	want := map[string]bool{"anthropic": false, "openai": false, "gemini": false, "ollama": false}
	for i, p := range ps {
		if _, ok := want[p]; ok {
			want[p] = true
		}
		if i > 0 && ps[i-1] > p {
			t.Errorf("Providers() not sorted at %d: %q > %q", i, ps[i-1], p)
		}
	}
	for p, found := range want {
		if !found {
			t.Errorf("Providers() missing %q", p)
		}
	}
}

// =============================================================================
// P10 simple wrappers — adversarial
// =============================================================================

func TestCompleteSimpleSendsExactlyOneUserMessage(t *testing.T) {
	m := llm.NewMockClient("x", "y").PushText("ok")
	if _, err := llm.CompleteSimple(context.Background(), m, "hello"); err != nil {
		t.Fatal(err)
	}
	last, _ := m.LastRequest()
	if len(last.Messages) != 1 || last.Messages[0].Role != llm.RoleUser {
		t.Fatalf("want 1 user message, got %+v", last.Messages)
	}
	if last.Messages[0].Content[0].Text != "hello" {
		t.Errorf("prompt not threaded: %+v", last.Messages[0].Content)
	}
}

func TestCompleteSimpleEmptyPrompt(t *testing.T) {
	m := llm.NewMockClient("x", "y").PushText("ok")
	if _, err := llm.CompleteSimple(context.Background(), m, ""); err != nil {
		t.Fatalf("empty prompt should not error: %v", err)
	}
	last, _ := m.LastRequest()
	if len(last.Messages) != 1 {
		t.Errorf("empty prompt should still send one message, got %d", len(last.Messages))
	}
}

func TestStreamSimpleStreams(t *testing.T) {
	m := llm.NewMockClient("x", "y").PushStream(
		llm.StreamEvent{Type: llm.EventContent, Text: "hi"},
		llm.StreamEvent{Type: llm.EventDone, StopReason: "end_turn"},
	)
	var text string
	for ev, err := range llm.StreamSimple(context.Background(), m, "q") {
		if err != nil {
			t.Fatal(err)
		}
		if ev.Type == llm.EventContent {
			text += ev.Text
		}
	}
	if text != "hi" {
		t.Errorf("streamed text = %q", text)
	}
}

// =============================================================================
// P7 provider breadth — adversarial
// =============================================================================

func TestNewProviderPresets(t *testing.T) {
	newOnes := []string{"together", "fireworks", "nvidia", "perplexity", "deepinfra"}
	for _, p := range newOnes {
		preset, ok := llm.PresetFor(p)
		if !ok {
			t.Errorf("preset %q missing", p)
			continue
		}
		if preset.Protocol != "openai_compat" {
			t.Errorf("%q protocol = %q, want openai_compat", p, preset.Protocol)
		}
		if !strings.HasPrefix(preset.BaseURL, "https://") {
			t.Errorf("%q baseURL = %q, want https://...", p, preset.BaseURL)
		}
		// Must construct without any network call.
		c, err := llm.NewClient(llm.Config{Provider: p, Model: "some-model", APIKey: "k", MaxTokens: 10})
		if err != nil {
			t.Errorf("NewClient(%q): %v", p, err)
			continue
		}
		if c.Provider() != p {
			t.Errorf("Provider() = %q, want %q", c.Provider(), p)
		}
		_ = c.Close()
	}
	// Discovery must include them.
	got := map[string]bool{}
	for _, p := range llm.Providers() {
		got[p] = true
	}
	for _, p := range newOnes {
		if !got[p] {
			t.Errorf("Providers() missing %q", p)
		}
	}
}

func TestNewProvidersHaveRegistryEntries(t *testing.T) {
	for _, p := range []string{"perplexity", "fireworks", "together", "deepinfra", "nvidia"} {
		models := llm.ModelsByProvider(p)
		if len(models) == 0 {
			t.Errorf("ModelsByProvider(%q) is empty — provider has no registry entries", p)
		}
		for _, m := range models {
			if m.Provider != p {
				t.Errorf("%q query returned a %q model: %s", p, m.Provider, m.ID)
			}
		}
	}
	// Cost estimation now works for a priced model (sonar-pro = $3/$15 per 1M) — was 0 before.
	if cost := llm.EstimateCost(llm.CostInput{Model: "sonar-pro", InputTokens: 1_000_000, OutputTokens: 1_000_000}); cost <= 0 {
		t.Errorf("EstimateCost(sonar-pro, 1M/1M) = %v, want > 0", cost)
	}
	// They appear via GetModelInfo too.
	if _, ok := llm.GetModelInfo("accounts/fireworks/models/fireworks/kimi-k2p6"); !ok {
		t.Error("fireworks model not found via GetModelInfo")
	}
}
