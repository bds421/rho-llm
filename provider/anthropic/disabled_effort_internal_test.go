package anthropic

import (
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// Claude Haiku 5.5 and Claude Opus 5 accept {"type":"disabled"} only at
// effort low, medium or high; xhigh and max return a 400 (the docs: lower the
// effort, or use adaptive thinking). Without a ThinkingLevel the adapter
// sends "disabled" and no effort (the model default), so the only way to hit
// the 400 is an effort passed through SamplingParams. That is refused up
// front.
func TestDisabledThinkingEffortLimit(t *testing.T) {
	build := func(model string, level llm.ThinkingLevel, params map[string]any) (anthropicRequest, error) {
		c := &Client{config: llm.Config{Model: model, MaxTokens: 1024}, providerName: "anthropic"}
		return c.buildRequest(llm.Request{ThinkingLevel: level, SamplingParams: params,
			Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "q")}}, false)
	}
	effort := func(e any) map[string]any { return map[string]any{"output_config": map[string]any{"effort": e}} }

	for _, model := range []string{"claude-haiku-5-5", "claude-opus-5"} {
		for _, e := range []string{"xhigh", "max"} {
			_, err := build(model, llm.ThinkingNone, effort(e))
			if err == nil || !strings.Contains(err.Error(), e) || !strings.Contains(err.Error(), "ThinkingLevel") {
				t.Errorf("%s disabled + %s: err = %v", model, e, err)
			}
		}
		for _, e := range []string{"low", "medium", "high", ""} {
			req, err := build(model, llm.ThinkingNone, effort(e))
			if err != nil || req.Thinking == nil || req.Thinking.Type != "disabled" {
				t.Errorf("%s disabled + %q: req.Thinking = %+v, err = %v", model, e, req.Thinking, err)
			}
		}
		if req, err := build(model, llm.ThinkingNone, nil); err != nil || req.Thinking.Type != "disabled" || req.OutputConfig != nil {
			t.Errorf("%s default: thinking %+v, output_config %+v, err %v", model, req.Thinking, req.OutputConfig, err)
		}
		for _, level := range []llm.ThinkingLevel{llm.ThinkingXHigh, llm.ThinkingMax} {
			req, err := build(model, level, nil)
			if err != nil || req.Thinking.Type != "adaptive" {
				t.Errorf("%s %s: thinking %+v, err %v", model, level, req.Thinking, err)
			}
		}
	}

	// A typed struct and odd shapes are read the same way, never panic.
	type outputConfig struct {
		Effort string `json:"effort"`
	}
	if _, err := build("claude-haiku-5-5", llm.ThinkingNone, map[string]any{"output_config": outputConfig{Effort: "max"}}); err == nil {
		t.Error("typed output_config with max effort accepted")
	}
	for _, odd := range []any{"max", 7, nil, []any{"max"}, map[string]any{"effort": 3}, make(chan int)} {
		if _, err := build("claude-haiku-5-5", llm.ThinkingNone, map[string]any{"output_config": odd}); err != nil && strings.Contains(err.Error(), "rejects thinking disabled") {
			t.Errorf("odd output_config %v refused as an effort: %v", odd, err)
		}
	}

	// Models without the documented limit are left alone.
	if _, err := build("claude-opus-4-8", llm.ThinkingNone, effort("max")); err != nil {
		t.Errorf("claude-opus-4-8 disabled + max: %v", err)
	}
}
