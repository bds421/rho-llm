package llm_test

import (
	"math"
	"testing"

	llm "github.com/bds421/rho-llm"
)

func TestSetPartsDerivesFlatFields(t *testing.T) {
	var r llm.Response
	r.SetParts([]llm.ContentPart{
		{Type: llm.ContentThinking, Thinking: "", ThinkingSignature: "sig"},
		{Type: llm.ContentText, Text: "a"},
		{Type: llm.ContentServerToolUse, ToolUseID: "srv_1", ToolName: "web_search"},
		{Type: llm.ContentText, Text: "b"},
		{Type: llm.ContentToolUse, ToolUseID: "t1", ToolName: "lookup", ToolInput: map[string]any{"q": "x"}},
		{Type: llm.ContentThinking, Thinking: "r", Redacted: true},
	})
	if r.Content != "ab" || r.Thinking != "r" || r.ThinkingSignature != "sig" || !r.ThinkingRedacted ||
		len(r.ToolCalls) != 1 || r.ToolCalls[0].ID != "t1" || len(r.Parts) != 6 {
		t.Errorf("%+v", r)
	}
}

func TestIterationPricing(t *testing.T) {
	near := func(a, b float64) bool { return math.Abs(a-b) < 1e-9 }
	// Haiku 5.5: input $0.10, output $0.50, 5m write $0.125, 1h write $0.20, read $0.01, search $10/1k.
	it := []llm.UsageIteration{{Type: "message", Model: "claude-haiku-5-5", InputTokens: 1_000_000,
		CacheCreationTokens: 1_000_000, CacheCreation1hTokens: 400_000}}
	got := llm.EstimateCost(llm.CostInput{Model: "claude-haiku-5-5", Iterations: it})
	if want := 0.10 + 0.6*0.125 + 0.4*0.20; !near(got, want) {
		t.Errorf("1h share per iteration: %v, want %v", got, want)
	}
	// An unregistered top-level model: searches priced at the iteration's model.
	got = llm.EstimateCost(llm.CostInput{Model: "claude-example-1", Iterations: it[:1], WebSearchRequests: 2})
	if want := 0.10 + 0.6*0.125 + 0.4*0.20 + 0.02; !near(got, want) {
		t.Errorf("searches with unregistered top model: %v, want %v", got, want)
	}
	if got := llm.EstimateCost(llm.CostInput{Model: "claude-example-1", WebSearchRequests: 2}); got != 0 {
		t.Errorf("nothing registered: %v, want 0", got)
	}
}
