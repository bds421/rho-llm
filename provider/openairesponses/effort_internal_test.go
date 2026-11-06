package openairesponses

import (
	"testing"

	llm "github.com/bds421/rho-llm"
)

func TestEffortMapsToReasoningEffort(t *testing.T) {
	c := &Client{providerName: "openai_responses", config: llm.Config{Model: "gpt-5.5"}}
	msgs := []llm.Message{llm.NewTextMessage(llm.RoleUser, "q")}
	for _, tc := range []struct {
		req           llm.Request
		effort, summ  string
		wantReasoning bool
	}{
		{llm.Request{Effort: llm.EffortLow}, "low", "", true},
		{llm.Request{Effort: llm.EffortMax}, "xhigh", "", true},
		{llm.Request{Effort: llm.EffortHigh, ThinkingLevel: llm.ThinkingHigh, ReasoningSummary: llm.ReasoningSummaryAuto}, "high", "auto", true},
		{llm.Request{ReasoningSummary: llm.ReasoningSummaryOmitted}, "", "", false},
		{llm.Request{ThinkingLevel: llm.ThinkingMedium, ReasoningSummary: llm.ReasoningSummaryOmitted}, "medium", "", true},
	} {
		tc.req.Messages = msgs
		got, err := c.buildRequest(tc.req, false)
		if err != nil {
			t.Fatalf("%+v: %v", tc.req, err)
		}
		if (got.Reasoning != nil) != tc.wantReasoning || (got.Reasoning != nil && (got.Reasoning.Effort != tc.effort || got.Reasoning.Summary != tc.summ)) {
			t.Errorf("%+v: reasoning %+v", tc.req, got.Reasoning)
		}
	}
	for name, req := range map[string]llm.Request{
		"conflict":  {Effort: llm.EffortLow, ThinkingLevel: llm.ThinkingHigh},
		"unknown":   {Effort: "extreme"},
		"container": {ContainerID: "container_1"},
	} {
		req.Messages = msgs
		if _, err := c.buildRequest(req, false); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
}
