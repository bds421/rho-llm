package llm_test

import (
	"context"
	"testing"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider"
)

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
