package anthropic

import (
	"encoding/json"
	"testing"

	llm "github.com/bds421/rho-llm"
)

func TestSonnet5ThinkingWire(t *testing.T) {
	for _, tc := range []struct {
		level  llm.ThinkingLevel
		effort string
	}{
		{llm.ThinkingNone, ""}, {llm.ThinkingMinimal, "low"}, {llm.ThinkingLow, "low"},
		{llm.ThinkingMedium, "medium"}, {llm.ThinkingHigh, "high"}, {llm.ThinkingXHigh, "xhigh"},
	} {
		t.Run(string(tc.level), func(t *testing.T) {
			c := &Client{config: llm.Config{Model: "claude-sonnet-5"}}
			req, err := c.buildRequest(llm.Request{ThinkingLevel: tc.level}, false)
			if err != nil {
				t.Fatal(err)
			}
			data, err := json.Marshal(req)
			if err != nil {
				t.Fatal(err)
			}
			var wire map[string]any
			if err := json.Unmarshal(data, &wire); err != nil {
				t.Fatal(err)
			}
			thinking, ok := wire["thinking"].(map[string]any)
			if !ok {
				t.Fatalf("missing thinking config: %s", data)
			}
			wantType := "adaptive"
			if tc.level == llm.ThinkingNone {
				wantType = "disabled"
			}
			if thinking["type"] != wantType {
				t.Fatalf("thinking = %v, want %s", thinking, wantType)
			}
			if _, ok := thinking["budget_tokens"]; ok {
				t.Fatal("manual thinking budget on Sonnet 5")
			}
			if tc.effort != "" {
				if thinking["display"] != "summarized" {
					t.Fatalf("display = %v", thinking["display"])
				}
				output, ok := wire["output_config"].(map[string]any)
				if !ok || output["effort"] != tc.effort {
					t.Fatalf("output_config = %v, want effort %s", output, tc.effort)
				}
			}
		})
	}
}

func TestSonnet5RejectsManualBudget(t *testing.T) {
	c := &Client{config: llm.Config{Model: "claude-sonnet-5"}}
	if _, err := c.buildRequest(llm.Request{ThinkingLevel: llm.ThinkingHigh, ThinkingBudget: 4096}, false); err == nil {
		t.Fatal("expected unsupported manual budget error")
	}
}

func TestSonnet5RejectsTemperatureCapability(t *testing.T) {
	temperature := 0.2
	for _, cfg := range []llm.Config{
		{Provider: "anthropic", Model: "claude-sonnet-5"},
		{Provider: "anthropic", Model: "claude-sonnet-5", ModelCapabilities: llm.Capabilities(llm.CapabilityChat, llm.CapabilityTemperature)},
	} {
		if err := llm.ValidateRequestCapabilities(cfg, llm.Request{Temperature: &temperature}, false); err == nil {
			t.Fatal("Sonnet 5 must reject sampling temperature")
		}
	}
}

func TestSonnet5BatchInheritsThinking(t *testing.T) {
	c := &Client{config: llm.Config{Model: "claude-sonnet-5", ThinkingLevel: llm.ThinkingMedium}}
	data, err := c.BuildMessageBatchParams(llm.Request{})
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Thinking struct{ Type string }   `json:"thinking"`
		Output   struct{ Effort string } `json:"output_config"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.Thinking.Type != "adaptive" || wire.Output.Effort != "medium" {
		t.Fatalf("batch did not inherit configured thinking: %s", data)
	}
}
