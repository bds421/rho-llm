package anthropic

import (
	"encoding/json"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// TestBuildMessageBatchParamsHasNonZeroMaxTokens verifies the batch encoder
// emits a usable max_tokens for a Config built as a struct literal.
//
// The batch path reuses the live buildRequest, which falls back to
// Config.MaxTokens. NewBatchClient applied the Timeout floor but not the
// MaxTokens floor, so every batch entry went out as "max_tokens": 0 — rejected
// by Anthropic exactly like the single-request path was.
func TestBuildMessageBatchParamsHasNonZeroMaxTokens(t *testing.T) {
	// Mirror what NewBatchClient hands the translator: a config that has been
	// through the shared floors. Passing an un-floored literal here would only
	// re-test the constructor, not the encoder's own contract.
	c, err := BatchTranslator(llm.Config{
		Provider: "anthropic", Model: "claude-sonnet-4-6", APIKey: "test-key",
		MaxTokens: llm.DefaultMaxTokens,
	})
	if err != nil {
		t.Fatalf("BatchTranslator: %v", err)
	}

	raw, err := c.BuildMessageBatchParams(llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
	})
	if err != nil {
		t.Fatalf("BuildMessageBatchParams: %v", err)
	}

	var wire struct {
		MaxTokens int `json:"max_tokens"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatalf("unmarshal batch params: %v", err)
	}
	if wire.MaxTokens <= 0 {
		t.Errorf("batch wire max_tokens = %d, want > 0 (Anthropic rejects 0)", wire.MaxTokens)
	}
}
