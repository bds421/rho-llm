package llm_test

import (
	"errors"
	"iter"
	"testing"

	llm "github.com/bds421/rho-llm"
)

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
