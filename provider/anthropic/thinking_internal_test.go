package anthropic

import (
	"testing"

	llm "github.com/bds421/rho-llm"
)

// TestParseResponseCapturesThinkingSignature verifies the non-streaming path
// extracts the thinking block's signature so it can be replayed on the next turn.
func TestParseResponseCapturesThinkingSignature(t *testing.T) {
	c := &Client{config: llm.Config{Model: "claude-opus-4-5"}, providerName: "anthropic"}

	apiResp := &anthropicResponse{Model: "claude-opus-4-5", StopReason: "end_turn"}
	apiResp.Content = append(apiResp.Content, struct {
		Type      string `json:"type"`
		Text      string `json:"text,omitempty"`
		ID        string `json:"id,omitempty"`
		Name      string `json:"name,omitempty"`
		Input     any    `json:"input,omitempty"`
		Thinking  string `json:"thinking,omitempty"`
		Signature string `json:"signature,omitempty"`
		Data      string `json:"data,omitempty"`
	}{Type: "thinking", Thinking: "step by step", Signature: "sig-xyz"})

	resp := c.parseResponse(apiResp)
	if resp.Thinking != "step by step" {
		t.Errorf("Thinking = %q", resp.Thinking)
	}
	if resp.ThinkingSignature != "sig-xyz" {
		t.Errorf("ThinkingSignature = %q, want sig-xyz", resp.ThinkingSignature)
	}
}

// TestBuildRequestReplaysThinkingBlock verifies a stored thinking part with a
// signature is re-emitted as an Anthropic "thinking" block (required for
// multi-turn extended thinking), while an unsigned one is skipped.
func TestBuildRequestReplaysThinkingBlock(t *testing.T) {
	c := &Client{config: llm.Config{Model: "claude-opus-4-5"}, providerName: "anthropic"}

	req := llm.Request{
		MaxTokens:     1024,
		ThinkingLevel: llm.ThinkingHigh, // thinking must be enabled for blocks to be valid
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "q"),
			{
				Role:     llm.RoleAssistant,
				Provider: "anthropic",
				Content: []llm.ContentPart{
					{Type: llm.ContentThinking, Thinking: "reasoning", ThinkingSignature: "sig-1"},
					{Type: llm.ContentText, Text: "answer"},
				},
			},
		},
	}

	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	// The assistant message (index 1) must lead with a thinking block carrying the signature.
	blocks := apiReq.Messages[1].Content
	if len(blocks) == 0 {
		t.Fatal("assistant message has no content blocks")
	}
	first, ok := blocks[0].(map[string]any)
	if !ok || first["type"] != "thinking" {
		t.Fatalf("first block = %+v, want a thinking block", blocks[0])
	}
	if first["signature"] != "sig-1" || first["thinking"] != "reasoning" {
		t.Errorf("thinking block = %+v, want thinking=reasoning signature=sig-1", first)
	}

	// An unsigned thinking block must be skipped (cannot be replayed validly).
	req.Messages[1].Content[0] = llm.ContentPart{Type: llm.ContentThinking, Thinking: "no sig"}
	apiReq2, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatalf("buildRequest (unsigned): %v", err)
	}
	for _, raw := range apiReq2.Messages[1].Content {
		if b, ok := raw.(map[string]any); ok && b["type"] == "thinking" {
			t.Errorf("unsigned thinking block should have been skipped, got %+v", b)
		}
	}
}

// TestBuildRequestDropsThinkingWhenDisabled verifies that a signed thinking block
// is NOT replayed when extended thinking is off for the request — Anthropic
// rejects thinking blocks unless thinking is enabled.
func TestBuildRequestDropsThinkingWhenDisabled(t *testing.T) {
	c := &Client{config: llm.Config{Model: "claude-opus-4-5"}, providerName: "anthropic"}

	req := llm.Request{
		MaxTokens: 1024, // ThinkingLevel intentionally unset (none)
		Messages: []llm.Message{
			{
				Role:     llm.RoleAssistant,
				Provider: "anthropic",
				Content: []llm.ContentPart{
					{Type: llm.ContentThinking, Thinking: "reasoning", ThinkingSignature: "sig-1"},
					{Type: llm.ContentText, Text: "answer"},
				},
			},
		},
	}

	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}
	for _, raw := range apiReq.Messages[0].Content {
		if b, ok := raw.(map[string]any); ok && (b["type"] == "thinking" || b["type"] == "redacted_thinking") {
			t.Errorf("thinking block emitted with thinking disabled: %+v", b)
		}
	}
	// The text block must still be present.
	var hasText bool
	for _, raw := range apiReq.Messages[0].Content {
		if b, ok := raw.(map[string]any); ok && b["type"] == "text" {
			hasText = true
		}
	}
	if !hasText {
		t.Error("text block missing after dropping thinking")
	}
}

// TestThinkingBudgetClampedToRequestMaxTokens verifies that the thinking
// budget is clamped to the request's effective max_tokens, not merely to the
// model's registry ceiling.
//
// Anthropic requires max_tokens > thinking.budget_tokens. ThinkingHigh
// resolves to 65536, so a caller asking for MaxTokens 16000 on a model whose
// registry ceiling is 64000 previously clamped to 64000 — still far above the
// request's own 16000 — and the API rejected the call with HTTP 400
// "`max_tokens` must be greater than `thinking.budget_tokens`".
func TestThinkingBudgetClampedToRequestMaxTokens(t *testing.T) {
	c := &Client{
		config:       llm.Config{Model: "claude-sonnet-4-6", MaxTokens: 16000},
		providerName: "anthropic",
	}

	apiReq, err := c.buildRequest(llm.Request{
		Messages:      []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
		ThinkingLevel: llm.ThinkingHigh,
	}, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}
	if apiReq.Thinking == nil {
		t.Fatal("Thinking = nil, want enabled")
	}
	if apiReq.Thinking.BudgetTokens >= apiReq.MaxTokens {
		t.Errorf("budget_tokens = %d, max_tokens = %d; Anthropic requires budget < max",
			apiReq.Thinking.BudgetTokens, apiReq.MaxTokens)
	}
}

// TestThinkingBudgetRespectsPerRequestMaxTokens covers the same invariant when
// max_tokens arrives on the Request rather than the Config.
func TestThinkingBudgetRespectsPerRequestMaxTokens(t *testing.T) {
	c := &Client{config: llm.Config{Model: "claude-sonnet-4-6"}, providerName: "anthropic"}

	apiReq, err := c.buildRequest(llm.Request{
		Messages:      []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
		MaxTokens:     8000,
		ThinkingLevel: llm.ThinkingHigh,
	}, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}
	if apiReq.Thinking.BudgetTokens >= apiReq.MaxTokens {
		t.Errorf("budget_tokens = %d, max_tokens = %d; want budget < max",
			apiReq.Thinking.BudgetTokens, apiReq.MaxTokens)
	}
}

// TestThinkingBudgetRespectsAnthropicMinimum verifies the other bound:
// Anthropic rejects budget_tokens below 1024, so clamping down to fit
// max_tokens must not push the budget under that floor. When max_tokens is too
// small to carry any valid budget, thinking is disabled instead of sending a
// request the API would reject from the other direction.
func TestThinkingBudgetRespectsAnthropicMinimum(t *testing.T) {
	for _, maxTokens := range []int{512, 1024, 1100, 1400, 2000, 6000, 16000} {
		c := &Client{
			config:       llm.Config{Model: "claude-sonnet-4-6", MaxTokens: maxTokens},
			providerName: "anthropic",
		}
		apiReq, err := c.buildRequest(llm.Request{
			Messages:      []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
			ThinkingLevel: llm.ThinkingHigh,
		}, false)
		if err != nil {
			t.Fatalf("max_tokens=%d: buildRequest: %v", maxTokens, err)
		}
		if apiReq.Thinking == nil {
			// Disabling thinking is the valid outcome only when no budget
			// could satisfy both bounds at once.
			if maxTokens > minThinkingBudgetTokens {
				t.Errorf("max_tokens=%d: thinking disabled, but a valid budget exists", maxTokens)
			}
			continue
		}
		if apiReq.Thinking.BudgetTokens < minThinkingBudgetTokens {
			t.Errorf("max_tokens=%d: budget_tokens = %d, want >= %d (Anthropic minimum)",
				maxTokens, apiReq.Thinking.BudgetTokens, minThinkingBudgetTokens)
		}
		if apiReq.Thinking.BudgetTokens >= apiReq.MaxTokens {
			t.Errorf("max_tokens=%d: budget_tokens = %d, want < max_tokens",
				maxTokens, apiReq.Thinking.BudgetTokens)
		}
	}
}
