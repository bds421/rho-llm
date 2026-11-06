package anthropic

import (
	"encoding/json"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// Claude 4.7 and later reject thinking.type "enabled" + budget_tokens with a
// 400 (platform.claude.com extended-thinking docs). Before this table existed
// only claude-sonnet-5 was routed to adaptive thinking, so ThinkingLevel on
// Opus 4.7/4.8/5 and Fable 5/5.1 produced a request the API rejects.

// budgetThinkingModels still take thinking.type "enabled" + budget_tokens.
var budgetThinkingModels = map[string]struct{}{
	"claude-opus-4-6":            {},
	"claude-opus-4-5":            {},
	"claude-opus-4-5-20251101":   {},
	"claude-sonnet-4-6":          {},
	"claude-sonnet-4-5":          {},
	"claude-sonnet-4-5-20250929": {},
	"claude-haiku-4-5":           {},
	"claude-haiku-4-5-20251001":  {},
}

func wireOf(t *testing.T, req anthropicRequest) map[string]any {
	t.Helper()
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	var wire map[string]any
	if err := json.Unmarshal(data, &wire); err != nil {
		t.Fatal(err)
	}
	return wire
}

// A Claude model registered without a thinking decision silently falls back
// to the budget path — the exact bug this file exists for.
func TestEveryAnthropicModelHasAThinkingDecision(t *testing.T) {
	seen := 0
	for _, info := range llm.ModelsByProvider("anthropic") {
		seen++
		_, adaptive := adaptiveThinkingModels[info.ID]
		_, budget := budgetThinkingModels[info.ID]
		if adaptive == budget {
			t.Errorf("%s: adaptive=%t budget=%t — classify it in exactly one of adaptiveThinkingModels / budgetThinkingModels", info.ID, adaptive, budget)
		}
	}
	if seen < 15 {
		t.Fatalf("only %d anthropic models inspected; the registry walk is probably broken", seen)
	}
	for id := range adaptiveThinkingModels {
		if _, ok := llm.GetModelInfo(id); !ok {
			t.Errorf("adaptiveThinkingModels names %q, which is not registered", id)
		}
	}
}

func TestAdaptiveModelsNeverSendBudgetTokens(t *testing.T) {
	levels := []llm.ThinkingLevel{llm.ThinkingNone, llm.ThinkingMinimal, llm.ThinkingLow, llm.ThinkingMedium, llm.ThinkingHigh, llm.ThinkingXHigh, llm.ThinkingMax}
	for id, profile := range adaptiveThinkingModels {
		for _, level := range levels {
			for _, maxTokens := range []int{0, 1, 1024, 16000, 128000} {
				c := &Client{config: llm.Config{Model: id, MaxTokens: maxTokens}, providerName: "anthropic"}
				req, err := c.buildRequest(llm.Request{ThinkingLevel: level, Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "x")}}, false)
				if err != nil {
					t.Fatalf("%s/%q/max=%d: %v", id, level, maxTokens, err)
				}
				wire := wireOf(t, req)
				thinking, hasThinking := wire["thinking"].(map[string]any)
				if hasThinking {
					if _, ok := thinking["budget_tokens"]; ok {
						t.Fatalf("%s/%q: budget_tokens on the wire (API 400s): %v", id, level, thinking)
					}
					if thinking["type"] == "enabled" {
						t.Fatalf("%s/%q: thinking.type enabled (API 400s)", id, level)
					}
				}
				if level == llm.ThinkingNone {
					if _, ok := wire["output_config"]; ok {
						t.Fatalf("%s: ThinkingNone sent output_config %v", id, wire["output_config"])
					}
					switch profile.off {
					case thinkingOffOmit:
						if hasThinking {
							t.Fatalf("%s: always-on thinking model got explicit thinking %v (400)", id, thinking)
						}
					case thinkingOffBetweenTools:
						if !hasThinking || thinking["type"] != "between_tools" || len(thinking) != 1 {
							t.Fatalf("%s: want exactly {type:between_tools}, got %v", id, wire["thinking"])
						}
					case thinkingOffDisabled:
						if !hasThinking || thinking["type"] != "disabled" || len(thinking) != 1 {
							t.Fatalf("%s: want exactly {type:disabled}, got %v", id, wire["thinking"])
						}
					}
					continue
				}
				if !hasThinking || thinking["type"] != "adaptive" {
					t.Fatalf("%s/%q: want adaptive thinking, got %v", id, level, wire["thinking"])
				}
				want := string(level)
				if level == llm.ThinkingMinimal {
					want = "low"
				}
				out, _ := wire["output_config"].(map[string]any)
				if out["effort"] != want {
					t.Fatalf("%s/%q: effort = %v, want %s", id, level, out["effort"], want)
				}
			}
		}
	}
}

func TestAdaptiveModelsRejectManualBudget(t *testing.T) {
	for id := range adaptiveThinkingModels {
		c := &Client{config: llm.Config{Model: id}, providerName: "anthropic"}
		for _, budget := range []int{1, 1024, 1 << 30} {
			_, err := c.buildRequest(llm.Request{ThinkingLevel: llm.ThinkingHigh, ThinkingBudget: budget}, false)
			if err == nil || !strings.Contains(err.Error(), id) {
				t.Fatalf("%s budget %d: want error naming the model, got %v", id, budget, err)
			}
		}
		// A budget with thinking off is irrelevant, not an error.
		if _, err := c.buildRequest(llm.Request{ThinkingBudget: 4096}, false); err != nil {
			t.Fatalf("%s: budget with ThinkingNone must be ignored, got %v", id, err)
		}
	}
}

// Aliases and client-level ThinkingLevel must reach the same path: "opus" and
// "sonnet" now resolve to adaptive-only models.
func TestAdaptiveThinkingThroughAliasesAndConfig(t *testing.T) {
	for _, alias := range []string{"opus", "sonnet", "haiku", "fable", "claude"} {
		resolved := llm.ResolveModelAlias(alias)
		if _, ok := adaptiveThinkingModels[resolved]; !ok {
			t.Fatalf("alias %q → %q is not adaptive; test premise broken", alias, resolved)
		}
		c := &Client{config: llm.Config{Model: alias, ThinkingLevel: llm.ThinkingHigh}, providerName: "anthropic"}
		req, err := c.buildRequest(llm.Request{}, false)
		if err != nil {
			t.Fatal(err)
		}
		if req.Thinking == nil || req.Thinking.Type != "adaptive" || req.Thinking.BudgetTokens != 0 {
			t.Fatalf("alias %q: thinking = %+v", alias, req.Thinking)
		}
	}
	// Batch encoding shares buildRequest.
	c := &Client{config: llm.Config{Model: "claude-opus-4-8", ThinkingLevel: llm.ThinkingLow}, providerName: "anthropic"}
	data, err := c.BuildMessageBatchParams(llm.Request{})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "budget_tokens") || !strings.Contains(string(data), `"adaptive"`) {
		t.Fatalf("batch params for opus-4-8: %s", data)
	}
}

// Budget models keep the legacy path: enabled + budget_tokens.
func TestBudgetModelsKeepManualThinking(t *testing.T) {
	for id := range budgetThinkingModels {
		c := &Client{config: llm.Config{Model: id, MaxTokens: 16000}, providerName: "anthropic"}
		req, err := c.buildRequest(llm.Request{ThinkingLevel: llm.ThinkingMedium}, false)
		if err != nil {
			t.Fatal(err)
		}
		if req.Thinking == nil || req.Thinking.Type != "enabled" || req.Thinking.BudgetTokens < minThinkingBudgetTokens {
			t.Fatalf("%s: thinking = %+v", id, req.Thinking)
		}
	}
}

func TestForcedToolChoiceRejectedWhereTheAPIRejectsIt(t *testing.T) {
	tools := []llm.Tool{{Name: "lookup", InputSchema: map[string]any{"type": "object"}}}
	forced := []*llm.ToolChoice{{Mode: llm.ToolChoiceRequired}, {Mode: llm.ToolChoiceTool, Name: "lookup"}}
	allowed := []*llm.ToolChoice{nil, {Mode: llm.ToolChoiceAuto}, {Mode: llm.ToolChoiceNone}}
	for id, profile := range adaptiveThinkingModels {
		c := &Client{config: llm.Config{Model: id}, providerName: "anthropic"}
		for _, tc := range forced {
			_, err := c.buildRequest(llm.Request{Tools: tools, ToolChoice: tc}, false)
			if profile.noForcedToolChoice && err == nil {
				t.Fatalf("%s: forced tool_choice %q must fail before the wire", id, tc.Mode)
			}
			if !profile.noForcedToolChoice && err != nil {
				t.Fatalf("%s: forced tool_choice %q is accepted by the API, got %v", id, tc.Mode, err)
			}
			// Without tools no tool_choice is sent, so nothing can 400.
			if _, err := c.buildRequest(llm.Request{ToolChoice: tc}, false); err != nil {
				t.Fatalf("%s: tool_choice without tools is dropped, got %v", id, err)
			}
		}
		for _, tc := range allowed {
			if _, err := c.buildRequest(llm.Request{Tools: tools, ToolChoice: tc}, false); err != nil {
				t.Fatalf("%s: tool_choice %+v must pass, got %v", id, tc, err)
			}
		}
	}
}

// The llm package's sampling deny list and this adapter's table describe the
// same models; if they drift, a model either advertises a temperature the API
// rejects or refuses one it accepts.
func TestAdaptiveModelsRejectTemperatureAndBudgetModelsAccept(t *testing.T) {
	temperature := 0.2
	for id := range adaptiveThinkingModels {
		cfg := llm.Config{Provider: "anthropic", Model: id, AnthropicVersion: llm.AnthropicVersion20230601}
		if err := llm.ValidateRequestCapabilities(cfg, llm.Request{Temperature: &temperature}, false); err == nil {
			t.Errorf("%s: explicit temperature must be rejected", id)
		}
	}
	for id := range budgetThinkingModels {
		cfg := llm.Config{Provider: "anthropic", Model: id, AnthropicVersion: llm.AnthropicVersion20230601}
		if err := llm.ValidateRequestCapabilities(cfg, llm.Request{Temperature: &temperature}, false); err != nil {
			t.Errorf("%s: temperature is supported, got %v", id, err)
		}
		// API version 2026-10-21 (the default) rejects it on every model.
		if err := llm.ValidateRequestCapabilities(llm.Config{Provider: "anthropic", Model: id}, llm.Request{Temperature: &temperature}, false); err == nil {
			t.Errorf("%s: temperature accepted on 2026-10-21", id)
		}
	}
}
