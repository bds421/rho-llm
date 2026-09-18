package anthropic

import (
	"testing"

	llm "github.com/bds421/rho-llm"
)

// Attack: sweep max_tokens across every boundary around the 1024 floor and the
// 3/4 reserve. The invariant Anthropic enforces is absolute:
// either thinking is off, or 1024 <= budget_tokens < max_tokens.
func TestBreakThinkingBudgetInvariantSweep(t *testing.T) {
	levels := []llm.ThinkingLevel{llm.ThinkingMinimal, llm.ThinkingLow, llm.ThinkingMedium, llm.ThinkingHigh}
	maxTokens := []int{1, 2, 1023, 1024, 1025, 1364, 1365, 1366, 1400, 2047, 2048, 4096, 8192, 16000, 64000, 64001, 1 << 20}
	for _, lvl := range levels {
		for _, mt := range maxTokens {
			c := &Client{config: llm.Config{Model: "claude-sonnet-4-6", MaxTokens: mt}, providerName: "anthropic"}
			apiReq, err := c.buildRequest(llm.Request{
				Messages:      []llm.Message{llm.NewTextMessage(llm.RoleUser, "x")},
				ThinkingLevel: lvl,
			}, false)
			if err != nil {
				t.Fatalf("level=%s max_tokens=%d: buildRequest: %v", lvl, mt, err)
			}
			if apiReq.Thinking == nil {
				continue // thinking disabled is always a safe outcome
			}
			b := apiReq.Thinking.BudgetTokens
			if b < minThinkingBudgetTokens {
				t.Errorf("level=%s max_tokens=%d: budget=%d < %d — API rejects (below minimum)",
					lvl, mt, b, minThinkingBudgetTokens)
			}
			if b >= apiReq.MaxTokens {
				t.Errorf("level=%s max_tokens=%d: budget=%d >= max=%d — API rejects (budget must be less)",
					lvl, mt, b, apiReq.MaxTokens)
			}
		}
	}
}

// Attack: a caller-supplied ThinkingBudget must not bypass the clamps.
func TestBreakExplicitThinkingBudgetCannotBypassClamps(t *testing.T) {
	for _, budget := range []int{1, 1023, 100000, 1 << 30} {
		c := &Client{config: llm.Config{Model: "claude-sonnet-4-6", MaxTokens: 8192}, providerName: "anthropic"}
		apiReq, err := c.buildRequest(llm.Request{
			Messages:       []llm.Message{llm.NewTextMessage(llm.RoleUser, "x")},
			ThinkingLevel:  llm.ThinkingHigh,
			ThinkingBudget: budget,
		}, false)
		if err != nil {
			t.Fatalf("budget=%d: %v", budget, err)
		}
		if apiReq.Thinking == nil {
			continue
		}
		b := apiReq.Thinking.BudgetTokens
		if b >= apiReq.MaxTokens || b < minThinkingBudgetTokens {
			t.Errorf("explicit budget %d escaped clamping: got %d with max_tokens=%d",
				budget, b, apiReq.MaxTokens)
		}
	}
}

// Attack: an unregistered model has no registry ceiling. The request-level
// clamp must still hold, or unknown/custom deployments send invalid requests.
func TestBreakUnregisteredModelStillClamped(t *testing.T) {
	c := &Client{config: llm.Config{Model: "some-private-deployment", MaxTokens: 4096}, providerName: "anthropic"}
	apiReq, err := c.buildRequest(llm.Request{
		Messages:      []llm.Message{llm.NewTextMessage(llm.RoleUser, "x")},
		ThinkingLevel: llm.ThinkingHigh,
	}, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}
	if apiReq.Thinking == nil {
		return
	}
	if apiReq.Thinking.BudgetTokens >= apiReq.MaxTokens {
		t.Errorf("unregistered model: budget=%d >= max_tokens=%d — no registry entry means no ceiling, "+
			"so the request-level clamp is the only guard",
			apiReq.Thinking.BudgetTokens, apiReq.MaxTokens)
	}
}
