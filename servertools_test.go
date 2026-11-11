package llm_test

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/provider/gemini"
	"github.com/bds421/rho-llm/provider/openaicompat"
	"github.com/bds421/rho-llm/provider/openairesponses"
)

func TestEstimateCostServerToolsAndIterations(t *testing.T) {
	base := llm.EstimateCost(llm.CostInput{Model: "claude-opus-5-5", InputTokens: 1000, OutputTokens: 100})
	withSearch := llm.EstimateCost(llm.CostInput{Model: "claude-opus-5-5", InputTokens: 1000, OutputTokens: 100, WebSearchRequests: 3})
	if math.Abs(withSearch-base-0.03) > 1e-12 {
		t.Errorf("3 searches added %.6f, want 0.03", withSearch-base)
	}
	batch := llm.EstimateCost(llm.CostInput{Model: "claude-opus-5-5", InputTokens: 1000, OutputTokens: 100, WebSearchRequests: 3, Batch: true})
	if math.Abs(batch-(base/2+0.03)) > 1e-12 {
		t.Errorf("batch = %.6f: searches must not get the batch discount", batch)
	}
	if got := llm.EstimateCost(llm.CostInput{Model: "unknown-model", WebSearchRequests: 10}); got != 0 {
		t.Errorf("unknown model priced searches: %v", got)
	}
	if got := llm.EstimateCost(llm.CostInput{Model: "gemini-2.5-flash", WebSearchRequests: 10}); got != 0 {
		t.Errorf("model without a search price priced searches: %v", got)
	}
	if got := llm.EstimateCost(llm.CostInput{Model: "claude-opus-5-5", WebSearchRequests: -5}); got != 0 {
		t.Errorf("negative searches = %v", got)
	}

	iter := llm.EstimateCost(llm.CostInput{
		Model: "claude-opus-5-5", InputTokens: 999999, OutputTokens: 999999, // ignored when iterations are set
		Iterations: []llm.UsageIteration{
			{Type: "compaction", InputTokens: 1000, CacheReadTokens: 2000},
			{Type: "fallback_message", Model: "claude-opus-4-8", InputTokens: 100, OutputTokens: 10, CacheCreationTokens: 50},
			{Type: "message", Model: "no-such-model", InputTokens: 1 << 30},
			{Type: "message", InputTokens: -7, OutputTokens: llm.TokensNotReported},
		},
	})
	want := (1000*4.00+2000*0.20)/1e6 + (100*5.00+10*25.00+50*6.25)/1e6
	if math.Abs(iter-want) > 1e-12 {
		t.Errorf("iterations = %.10f, want %.10f", iter, want)
	}
	huge := llm.EstimateCost(llm.CostInput{Model: "claude-opus-5-5", WebSearchRequests: math.MaxInt, Iterations: []llm.UsageIteration{
		{InputTokens: math.MaxInt, OutputTokens: math.MaxInt, CacheReadTokens: math.MaxInt, CacheCreationTokens: math.MaxInt},
		{InputTokens: math.MaxInt, OutputTokens: math.MaxInt, CacheReadTokens: math.MaxInt, CacheCreationTokens: math.MaxInt},
	}})
	if math.IsInf(huge, 0) || math.IsNaN(huge) || huge <= 0 {
		t.Errorf("saturated cost = %v", huge)
	}
}

func TestNormalizeKeepsServerToolsWithTheirProvider(t *testing.T) {
	assistant := llm.Message{Role: llm.RoleAssistant, Provider: "anthropic", Content: []llm.ContentPart{
		{Type: llm.ContentServerToolUse, ToolUseID: "srvtoolu_1", ToolName: "web_search", ProviderBlock: json.RawMessage(`{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}`)},
		{Type: llm.ContentServerToolResult, ToolResultID: "srvtoolu_1", BlockType: "web_search_tool_result", ProviderBlock: json.RawMessage(`{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[]}`)},
		{Type: llm.ContentProviderBlock, BlockType: "fallback", ProviderBlock: json.RawMessage(`{"type":"fallback"}`)},
		{Type: llm.ContentText, Text: "answer", Citations: []llm.Citation{{Type: llm.CitationWebSearchResultLocation, URL: "https://example.com"}}},
	}}
	msgs := []llm.Message{llm.NewTextMessage(llm.RoleUser, "q"), assistant}

	same := llm.NormalizeForProvider(msgs, "anthropic")
	if len(same[1].Content) != 4 || len(same[1].Content[3].Citations) != 1 {
		t.Fatalf("same provider lost content: %+v", same[1].Content)
	}
	other := llm.NormalizeForProvider(msgs, "gemini")
	if len(other[1].Content) != 1 || other[1].Content[0].Text != "answer" || other[1].Content[0].Citations != nil {
		t.Fatalf("handoff content = %+v", other[1].Content)
	}
	if len(assistant.Content[3].Citations) != 1 {
		t.Fatal("normalization mutated the input")
	}
}

func TestOtherAdaptersRejectServerTools(t *testing.T) {
	cfg := func(provider, model string) llm.Config {
		return llm.Config{Provider: provider, Model: model, APIKey: "k", BaseURL: "http://127.0.0.1:1", Timeout: time.Second}
	}
	clients := map[string]func() (llm.Client, error){
		"gemini":           func() (llm.Client, error) { return gemini.New(cfg("gemini", "gemini-2.5-flash")) },
		"openai_compat":    func() (llm.Client, error) { return openaicompat.New(cfg("openai", "gpt-4o")) },
		"openai_responses": func() (llm.Client, error) { return openairesponses.New(cfg("openai", "gpt-5")) },
	}
	for name, newClient := range clients {
		t.Run(name, func(t *testing.T) {
			c, err := newClient()
			if err != nil {
				t.Fatal(err)
			}
			req := llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "q")}, ServerTools: searchTools}
			if _, err := c.Complete(context.Background(), req); err == nil || !bytes.Contains([]byte(err.Error()), []byte("server tools")) {
				t.Fatalf("Complete err = %v", err)
			}
			var streamErr error
			for _, err := range c.Stream(context.Background(), req) {
				streamErr = err
			}
			if streamErr == nil {
				t.Fatal("Stream accepted server tools")
			}
		})
	}
}

func TestServerToolsRequireToolCapability(t *testing.T) {
	caps := llm.RequestCapabilities(llm.Request{ServerTools: searchTools}, false)
	for _, c := range caps {
		if c == llm.CapabilityTools {
			return
		}
	}
	t.Fatalf("capabilities = %v, want CapabilityTools for a server tool", caps)
}
