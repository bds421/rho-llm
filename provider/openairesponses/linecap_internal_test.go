package openairesponses

import (
	"encoding/json"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// response.completed repeats the whole output on one line. An answer past the
// 256 KB SSE line cap streamed fine and then failed at its very end (B26).
func TestLongAnswerSurvivesTheCompletedEvent(t *testing.T) {
	text := strings.Repeat("word ", 80_000) // 400 KB
	delta, _ := json.Marshal(map[string]any{"type": "response.output_text.delta", "delta": text})
	done, _ := json.Marshal(map[string]any{"type": "response.completed", "response": map[string]any{
		"status": "completed",
		"output": []any{map[string]any{"type": "message", "content": []any{map[string]any{"type": "output_text", "text": text}}}},
		"usage":  map[string]any{"input_tokens": 3, "output_tokens": 80000},
	}})
	sse := "data: " + string(delta) + "\n\ndata: " + string(done) + "\n\n"

	c := &Client{providerName: "openai"}
	var got strings.Builder
	var stop string
	c.parseStream(strings.NewReader(sse), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			t.Fatalf("stream failed: %v", err)
		}
		if ev.Type == llm.EventContent {
			got.WriteString(ev.Text)
		}
		if ev.Type == llm.EventDone {
			stop = ev.StopReason
		}
		return true
	})
	if got.Len() != len(text) || stop != llm.StopEndTurn {
		t.Fatalf("got %d bytes, stop %q; want %d bytes and end_turn", got.Len(), stop, len(text))
	}

	// An explicit cap is still honoured.
	c = &Client{providerName: "openai", config: llm.Config{MaxSSELineBytes: 1024}}
	var failed bool
	c.parseStream(strings.NewReader(sse), func(_ llm.StreamEvent, err error) bool {
		failed = failed || err != nil
		return true
	})
	if !failed {
		t.Fatal("a line past an explicit MaxSSELineBytes was accepted")
	}
}
