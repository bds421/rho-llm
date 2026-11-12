package gemini

import (
	"encoding/json"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// Without includeThoughts Gemini returns no thought parts, so a caller that
// set ThinkingLevel never saw EventThinking or Response.Thinking (B25).
func TestThinkingRequestAsksForThoughtSummaries(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gemini-3.8-flash"}, providerName: "gemini"}
	for _, level := range []llm.ThinkingLevel{llm.ThinkingLow, llm.ThinkingHigh, llm.ThinkingMinimal, llm.ThinkingMax} {
		apiReq, err := c.buildRequest(llm.Request{ThinkingLevel: level, Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "x")}})
		if err != nil {
			t.Fatal(err)
		}
		b, _ := json.Marshal(apiReq.GenerationConfig)
		if !strings.Contains(string(b), `"includeThoughts":true`) {
			t.Errorf("level %s: generationConfig = %s, want includeThoughts", level, b)
		}
	}
	// No ThinkingLevel: no thinkingConfig at all, as before.
	apiReq, err := c.buildRequest(llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "x")}})
	if err != nil {
		t.Fatal(err)
	}
	if b, _ := json.Marshal(apiReq.GenerationConfig); strings.Contains(string(b), "thinkingConfig") {
		t.Errorf("no ThinkingLevel, yet generationConfig = %s", b)
	}
}
