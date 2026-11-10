package anthropic

import (
	"encoding/json"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

var personSchema = map[string]any{
	"type":                 "object",
	"properties":           map[string]any{"name": map[string]any{"type": "string"}},
	"required":             []any{"name"},
	"additionalProperties": false,
}

func wireOutputConfig(t *testing.T, model string, req llm.Request) (map[string]any, map[string]any) {
	t.Helper()
	c := &Client{config: llm.Config{Model: model, MaxTokens: 2048}, providerName: "anthropic"}
	req.Messages = []llm.Message{llm.NewTextMessage(llm.RoleUser, "q")}
	data, err := c.BuildMessageBatchParams(req)
	if err != nil {
		t.Fatalf("%s: %v", model, err)
	}
	var body map[string]any
	if err := json.Unmarshal(data, &body); err != nil {
		t.Fatal(err)
	}
	oc, _ := body["output_config"].(map[string]any)
	th, _ := body["thinking"].(map[string]any)
	return oc, th
}

func TestResponseFormatOnTheWire(t *testing.T) {
	format := &llm.ResponseFormat{Type: llm.ResponseFormatJSONSchema, Name: "person", Schema: personSchema}
	wantFormat := map[string]any{"type": "json_schema", "schema": personSchema}
	sameFormat := func(got any) bool {
		a, _ := json.Marshal(got)
		b, _ := json.Marshal(wantFormat)
		return string(a) == string(b)
	}

	// Adaptive thinking: effort and format share output_config.
	for _, model := range []string{"claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-2", "claude-fable-5-1"} {
		oc, th := wireOutputConfig(t, model, llm.Request{ThinkingLevel: llm.ThinkingHigh, ResponseFormat: format})
		if oc["effort"] != "high" || !sameFormat(oc["format"]) || th["type"] != "adaptive" {
			t.Errorf("%s adaptive: output_config %v, thinking %v", model, oc, th)
		}
	}

	// No ThinkingLevel: format alone, no effort, thinking as the model takes "off".
	for model, thinking := range map[string]any{
		"claude-opus-5-5":   nil,
		"claude-sonnet-5-5": "between_tools",
		"claude-haiku-5-5":  "disabled",
		"claude-opus-4-8":   "disabled",
	} {
		oc, th := wireOutputConfig(t, model, llm.Request{ResponseFormat: format})
		if _, ok := oc["effort"]; ok || !sameFormat(oc["format"]) || th["type"] != thinking {
			t.Errorf("%s no thinking: output_config %v, thinking %v", model, oc, th)
		}
	}

	// Budget-thinking models (the 4.5/4.6 line) keep budget_tokens and get the format.
	for _, model := range []string{"claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-6"} {
		oc, th := wireOutputConfig(t, model, llm.Request{ThinkingLevel: llm.ThinkingLow, ResponseFormat: format})
		if _, ok := oc["effort"]; ok || !sameFormat(oc["format"]) || th["type"] != "enabled" {
			t.Errorf("%s budget: output_config %v, thinking %v", model, oc, th)
		}
		if oc, _ := wireOutputConfig(t, model, llm.Request{ResponseFormat: format}); !sameFormat(oc["format"]) {
			t.Errorf("%s plain: output_config %v", model, oc)
		}
	}

	// The server-side refusal fallback re-serves the same body, format included.
	c := &Client{config: llm.Config{Model: "claude-opus-5-5", MaxTokens: 2048}, providerName: "anthropic"}
	data, err := c.BuildMessageBatchParams(llm.Request{ResponseFormat: format, ThinkingLevel: llm.ThinkingMedium,
		SamplingParams: map[string]any{"fallbacks": "default"},
		Messages:       []llm.Message{llm.NewTextMessage(llm.RoleUser, "q")}})
	if err != nil || !strings.Contains(string(data), `"fallbacks":"default"`) || !strings.Contains(string(data), `"format":{"type":"json_schema"`) {
		t.Errorf("fallbacks + format: %s, %v", data, err)
	}

	// Without ResponseFormat nothing changes: no format, effort only with a level.
	if oc, _ := wireOutputConfig(t, "claude-opus-5-5", llm.Request{}); oc != nil {
		t.Errorf("no format, no level: output_config %v", oc)
	}
}

func TestResponseFormatRefusals(t *testing.T) {
	c := &Client{config: llm.Config{Model: "claude-haiku-5-5", MaxTokens: 1024}, providerName: "anthropic"}
	msgs := []llm.Message{llm.NewTextMessage(llm.RoleUser, "q")}
	for name, rf := range map[string]*llm.ResponseFormat{
		"json_object": {Type: llm.ResponseFormatJSONObject},
		"schema-less": {Type: llm.ResponseFormatJSONSchema},
		"unknown":     {Type: "xml"},
	} {
		if _, err := c.buildRequest(llm.Request{ResponseFormat: rf, Messages: msgs}, false); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
	// output_config through SamplingParams collides with the typed field.
	_, err := c.BuildMessageBatchParams(llm.Request{Messages: msgs,
		ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSONSchema, Schema: personSchema},
		SamplingParams: map[string]any{"output_config": map[string]any{"effort": "low"}}})
	if err == nil {
		t.Error("output_config in SamplingParams next to ResponseFormat accepted")
	}
	if _, err := c.buildRequest(llm.Request{CachedContent: "cachedContents/x", Messages: msgs}, false); err == nil || !strings.Contains(err.Error(), "CachedContent") {
		t.Errorf("CachedContent: %v", err)
	}
}

// A model without reviewed structured-output support is refused before
// dispatch, not sent without its schema.
func TestResponseFormatNeedsTheCapability(t *testing.T) {
	format := &llm.ResponseFormat{Type: llm.ResponseFormatJSONSchema, Schema: personSchema}
	req := llm.Request{ResponseFormat: format, Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "q")}}
	if err := llm.ValidateRequestCapabilities(llm.Config{Provider: "anthropic", Model: "claude-haiku-5-5"}, req, false); err != nil {
		t.Errorf("registered model refused: %v", err)
	}
	custom := llm.Config{Provider: "anthropic", Model: "claude-example-1",
		ModelCapabilities: llm.Capabilities(llm.CapabilityChat, llm.CapabilityStream)}
	if err := llm.ValidateRequestCapabilities(custom, req, false); err == nil {
		t.Error("model without structured output accepted a ResponseFormat")
	}
}

func TestThinkingDisplayChoice(t *testing.T) {
	for summary, want := range map[llm.ReasoningSummary]any{
		llm.ReasoningSummaryNone:     "summarized",
		llm.ReasoningSummaryDetailed: "summarized",
		llm.ReasoningSummaryConcise:  "summarized",
		llm.ReasoningSummaryAuto:     nil,
		llm.ReasoningSummaryOmitted:  "omitted",
	} {
		for _, model := range []string{"claude-opus-5-5", "claude-haiku-5-5", "claude-opus-4-7"} {
			_, th := wireOutputConfig(t, model, llm.Request{ThinkingLevel: llm.ThinkingLow, ReasoningSummary: summary})
			if th["display"] != want {
				t.Errorf("%s %q: display %v, want %v", model, summary, th["display"], want)
			}
		}
	}
	// between_tools takes no other field, whatever the summary setting.
	_, th := wireOutputConfig(t, "claude-sonnet-5-5", llm.Request{ReasoningSummary: llm.ReasoningSummaryDetailed})
	if len(th) != 1 || th["type"] != "between_tools" {
		t.Errorf("between_tools: %v", th)
	}
}

func TestTypedEffort(t *testing.T) {
	// Effort without thinking: output_config.effort, thinking as the model takes "off".
	for model, thinking := range map[string]any{
		"claude-opus-5-5":   nil,
		"claude-sonnet-5-5": "between_tools",
		"claude-haiku-5-5":  "disabled",
		"claude-opus-4-8":   "disabled",
	} {
		oc, th := wireOutputConfig(t, model, llm.Request{Effort: llm.EffortLow})
		if oc["effort"] != "low" || th["type"] != thinking {
			t.Errorf("%s: output_config %v thinking %v", model, oc, th)
		}
	}
	// With a matching ThinkingLevel, and next to a format.
	oc, _ := wireOutputConfig(t, "claude-opus-5-5", llm.Request{Effort: llm.EffortHigh, ThinkingLevel: llm.ThinkingHigh,
		ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSONSchema, Schema: personSchema}})
	if oc["effort"] != "high" || oc["format"] == nil {
		t.Errorf("effort + level + format: %v", oc)
	}
	// Budget models take effort where documented.
	if oc, th := wireOutputConfig(t, "claude-opus-4-6", llm.Request{Effort: llm.EffortMax, ThinkingLevel: llm.ThinkingLow}); oc["effort"] != "max" || th["type"] != "enabled" {
		t.Errorf("opus 4.6: %v %v", oc, th)
	}

	c := &Client{config: llm.Config{MaxTokens: 1024}, providerName: "anthropic"}
	msgs := []llm.Message{llm.NewTextMessage(llm.RoleUser, "q")}
	for name, req := range map[string]llm.Request{
		"conflict":              {Model: "claude-opus-5-5", Effort: llm.EffortLow, ThinkingLevel: llm.ThinkingHigh},
		"unknown":               {Model: "claude-opus-5-5", Effort: "extreme"},
		"between_tools at max":  {Model: "claude-sonnet-5-5", Effort: llm.EffortMax},
		"disabled at xhigh":     {Model: "claude-haiku-5-5", Effort: llm.EffortXHigh},
		"haiku 4.5 has none":    {Model: "claude-haiku-4-5", Effort: llm.EffortLow},
		"opus 4.6 has no xhigh": {Model: "claude-opus-4-6", Effort: llm.EffortXHigh},
		"opus 4.5 has no max":   {Model: "claude-opus-4-5", Effort: llm.EffortMax},
	} {
		req.Messages = msgs
		if _, err := c.buildRequest(req, false); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
}
