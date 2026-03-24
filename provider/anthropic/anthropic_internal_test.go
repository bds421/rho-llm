package anthropic

import (
	"testing"

	llm "github.com/bds421/rho-llm"
)

// TestBuildRequestThinkingBudgetClampedToModelMax verifies that a thinking
// budget exceeding the model's MaxTokens is clamped.
func TestBuildRequestThinkingBudgetClampedToModelMax(t *testing.T) {
	// claude-opus-4-5 has MaxTokens=64000 — ThinkingXHigh (128000) exceeds it.
	c := &Client{
		config:       llm.Config{Model: "claude-opus-4-5"},
		providerName: "anthropic",
	}

	req := llm.Request{
		MaxTokens:     8192,
		ThinkingLevel: llm.ThinkingXHigh,
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "think hard"),
		},
	}

	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	info, ok := llm.GetModelInfo("claude-opus-4-5")
	if !ok {
		t.Fatal("claude-opus-4-5 not in registry")
	}

	if apiReq.Thinking == nil {
		t.Fatal("Thinking is nil, want non-nil")
	}
	if apiReq.Thinking.BudgetTokens > info.MaxTokens {
		t.Errorf("BudgetTokens = %d, want <= %d (model MaxTokens)",
			apiReq.Thinking.BudgetTokens, info.MaxTokens)
	}
	if apiReq.Thinking.BudgetTokens != info.MaxTokens {
		t.Errorf("BudgetTokens = %d, want %d (should clamp to model MaxTokens)",
			apiReq.Thinking.BudgetTokens, info.MaxTokens)
	}
}

// TestBuildRequestThinkingBudgetNotClampedWhenWithinLimit verifies that a
// budget within the model's MaxTokens is not modified.
func TestBuildRequestThinkingBudgetNotClampedWhenWithinLimit(t *testing.T) {
	// claude-opus-4-6 has MaxTokens=128000 — ThinkingXHigh (128000) fits exactly.
	c := &Client{
		config:       llm.Config{Model: "claude-opus-4-6"},
		providerName: "anthropic",
	}

	req := llm.Request{
		MaxTokens:     8192,
		ThinkingLevel: llm.ThinkingXHigh,
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "think hard"),
		},
	}

	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	if apiReq.Thinking == nil {
		t.Fatal("Thinking is nil, want non-nil")
	}
	want := llm.ThinkingBudgetTokens(llm.ThinkingXHigh, 0)
	if apiReq.Thinking.BudgetTokens != want {
		t.Errorf("BudgetTokens = %d, want %d (should not clamp when within limit)",
			apiReq.Thinking.BudgetTokens, want)
	}
}
