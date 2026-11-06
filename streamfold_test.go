package llm_test

import (
	"context"
	"errors"
	"iter"
	"testing"

	llm "github.com/bds421/rho-llm"
)

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

func TestOverloadedStatuses(t *testing.T) {
	for _, status := range []int{503, 529} {
		err := llm.NewAPIErrorFromStatus("anthropic", status, "overloaded")
		if !llm.IsRetryable(err) || !llm.IsOverloaded(err) || err.StatusCode != status {
			t.Errorf("%d: %+v", status, err)
		}
	}
	if !llm.IsRetryable(llm.NewAPIErrorFromStatus("anthropic", 504, "timeout")) {
		t.Error("504 not retryable")
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

// errorFirstClient fails before any content, after the provider reported
// usage: a retryable overloaded error carrying the billed input.
type errorFirstClient struct {
	fakeClient
	calls int
}

func (e *errorFirstClient) Stream(context.Context, llm.Request) iter.Seq2[llm.StreamEvent, error] {
	e.calls++
	return func(yield func(llm.StreamEvent, error) bool) {
		err := llm.NewAPIErrorFromStatus("anthropic", 529, "overloaded")
		yield(llm.StreamEvent{Type: llm.EventError, Error: err.Error(), InputTokens: 70}, err)
	}
}

func TestPoolDoesNotHideABilledAttempt(t *testing.T) {
	inner := &errorFirstClient{fakeClient: fakeClient{provider: "anthropic", model: "claude-opus-5-5"}}
	pooled, err := llm.NewPooledClient(llm.Config{Provider: "anthropic", Model: "claude-opus-5-5"},
		[]string{"k"}, func(llm.AuthProfile) (llm.Client, error) { return inner, nil })
	if err != nil {
		t.Fatal(err)
	}
	resp, err := llm.CollectStream(pooled.Stream(context.Background(), llm.Request{}))
	if err == nil || resp.InputTokens != 70 || inner.calls != 1 {
		t.Errorf("usage %d, calls %d, err %v", resp.InputTokens, inner.calls, err)
	}
}
