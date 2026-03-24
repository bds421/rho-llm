package gemini

import (
	"encoding/json"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// TestMakeToolCallIDResolveToolNameRoundTrip verifies that resolveToolName
// correctly inverts makeToolCallID for various tool name patterns.
func TestMakeToolCallIDResolveToolNameRoundTrip(t *testing.T) {
	tests := []struct {
		index int
		name  string
	}{
		{0, "get_weather"},
		{1, "search"},
		{0, "tool_with_many_underscores"},
		{5, "a"},
		{0, "123_numeric_prefix"},
		{0, "99"},
		{10, "deeply_nested_tool_name"},
	}

	for _, tc := range tests {
		id := makeToolCallID(tc.index, tc.name)
		got := resolveToolName(id)
		if got != tc.name {
			t.Errorf("roundtrip(%d, %q): makeToolCallID=%q, resolveToolName=%q, want %q",
				tc.index, tc.name, id, got, tc.name)
		}
	}
}

// TestResolveToolNameLegacyFormat verifies the legacy "call_<name>" format.
func TestResolveToolNameLegacyFormat(t *testing.T) {
	got := resolveToolName("call_my_tool")
	if got != "my_tool" {
		t.Errorf("resolveToolName(call_my_tool) = %q, want my_tool", got)
	}
}

// TestResolveToolNameNonSynthetic verifies non-call_ IDs pass through.
func TestResolveToolNameNonSynthetic(t *testing.T) {
	got := resolveToolName("toolu_01234abcdef")
	if got != "toolu_01234abcdef" {
		t.Errorf("resolveToolName(toolu_...) = %q, want passthrough", got)
	}
}

// TestBuildRequestEmptyTextPartOmitted verifies that ContentText parts with
// empty strings are not included in the Gemini request. An empty geminiPart{}
// serializes to {} due to omitempty, which violates the Gemini API's oneof
// constraint on the Part.data field (status 400: "required oneof field 'data'
// must have one initialized field").
func TestBuildRequestEmptyTextPartOmitted(t *testing.T) {
	c := &Client{providerName: "gemini"}

	req := llm.Request{
		Messages: []llm.Message{
			{
				Role: llm.RoleUser,
				Content: []llm.ContentPart{
					{Type: llm.ContentText, Text: "hello"},
				},
			},
			// Simulate an assistant message with empty text (e.g., assistant
			// response that had only tool calls, no text content).
			{
				Role: llm.RoleAssistant,
				Content: []llm.ContentPart{
					{Type: llm.ContentText, Text: ""},
				},
			},
			{
				Role: llm.RoleUser,
				Content: []llm.ContentPart{
					{Type: llm.ContentText, Text: "continue"},
				},
			},
		},
	}

	apiReq, err := c.buildRequest(req)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	// Verify no empty parts exist in any content message.
	for i, content := range apiReq.Contents {
		for j, part := range content.Parts {
			b, err := json.Marshal(part)
			if err != nil {
				t.Fatalf("json.Marshal(part[%d][%d]): %v", i, j, err)
			}
			if string(b) == "{}" {
				t.Errorf("contents[%d].parts[%d] serializes to empty {}: %+v", i, j, part)
			}
		}
	}

	// The assistant message with only an empty text part should be dropped
	// entirely (the len(content.Parts) > 0 guard).
	if len(apiReq.Contents) != 2 {
		t.Errorf("expected 2 contents (empty assistant dropped), got %d", len(apiReq.Contents))
	}
}

// TestBuildRequestEmptyTextSystemInstructionOmitted verifies that empty text
// parts in system messages are not included in systemInstruction.
func TestBuildRequestEmptyTextSystemInstructionOmitted(t *testing.T) {
	c := &Client{providerName: "gemini"}

	req := llm.Request{
		Messages: []llm.Message{
			{
				Role: llm.RoleSystem,
				Content: []llm.ContentPart{
					{Type: llm.ContentText, Text: ""},
				},
			},
			{
				Role: llm.RoleUser,
				Content: []llm.ContentPart{
					{Type: llm.ContentText, Text: "hello"},
				},
			},
		},
	}

	apiReq, err := c.buildRequest(req)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	// System instruction should be nil when the only system part was empty.
	if apiReq.SystemInstruction != nil && len(apiReq.SystemInstruction.Parts) > 0 {
		for j, part := range apiReq.SystemInstruction.Parts {
			b, err := json.Marshal(part)
			if err != nil {
				t.Fatalf("json.Marshal(systemInstruction.parts[%d]): %v", j, err)
			}
			if string(b) == "{}" {
				t.Errorf("systemInstruction.parts[%d] serializes to empty {}: %+v", j, part)
			}
		}
	}
}

// TestParseResponseThinkingParts verifies that parts with thought:true go to
// resp.Thinking while non-thought parts go to resp.Content.
func TestParseResponseThinkingParts(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gemini-2.5-flash"}, providerName: "gemini"}

	apiResp := &geminiResponse{
		Candidates: []struct {
			Content       geminiContent `json:"content"`
			FinishReason  string        `json:"finishReason"`
			SafetyRatings []struct {
				Category    string `json:"category"`
				Probability string `json:"probability"`
			} `json:"safetyRatings"`
		}{
			{
				Content: geminiContent{
					Parts: []geminiPart{
						{Text: "Let me think...", Thought: true},
						{Text: "More reasoning.", Thought: true},
						{Text: "The answer is 42."},
					},
				},
				FinishReason: "STOP",
			},
		},
	}

	resp := c.parseResponse(apiResp, "gemini-2.5-flash")

	if resp.Content != "The answer is 42." {
		t.Errorf("Content = %q, want %q", resp.Content, "The answer is 42.")
	}
	if resp.Thinking != "Let me think...More reasoning." {
		t.Errorf("Thinking = %q, want %q", resp.Thinking, "Let me think...More reasoning.")
	}
}

// TestParseResponseThoughtsTokenCount verifies that thoughtsTokenCount is
// exposed as ThinkingTokens on the response, separate from OutputTokens
// (which maps to CandidatesTokenCount only).
func TestParseResponseThoughtsTokenCount(t *testing.T) {
	raw := `{
		"candidates": [{
			"content": {"parts": [{"text": "thinking", "thought": true}, {"text": "answer"}]},
			"finishReason": "STOP"
		}],
		"usageMetadata": {
			"promptTokenCount": 10,
			"candidatesTokenCount": 5,
			"totalTokenCount": 115,
			"thoughtsTokenCount": 100
		}
	}`

	var apiResp geminiResponse
	if err := json.Unmarshal([]byte(raw), &apiResp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if apiResp.UsageMetadata.ThoughtsTokenCount != 100 {
		t.Errorf("ThoughtsTokenCount = %d, want 100", apiResp.UsageMetadata.ThoughtsTokenCount)
	}

	c := &Client{config: llm.Config{Model: "gemini-2.5-flash"}, providerName: "gemini"}
	resp := c.parseResponse(&apiResp, "gemini-2.5-flash")

	// OutputTokens maps to CandidatesTokenCount (excludes thinking)
	if resp.OutputTokens != 5 {
		t.Errorf("OutputTokens = %d, want 5 (CandidatesTokenCount)", resp.OutputTokens)
	}
	// ThinkingTokens exposes the separate thoughtsTokenCount
	if resp.ThinkingTokens != 100 {
		t.Errorf("ThinkingTokens = %d, want 100", resp.ThinkingTokens)
	}
}

// TestParseStreamThinkingParts verifies that streaming thought parts emit
// EventThinking while non-thought parts emit EventContent.
func TestParseStreamThinkingParts(t *testing.T) {
	// Simulate two SSE chunks: one with a thought part, one with content + usage
	sseData := "data: " + `{"candidates":[{"content":{"parts":[{"text":"reasoning...","thought":true}]}}]}` + "\n\n" +
		"data: " + `{"candidates":[{"content":{"parts":[{"text":"answer"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"thoughtsTokenCount":80}}` + "\n\n"

	c := &Client{providerName: "gemini"}
	var events []llm.StreamEvent
	c.parseStream(strings.NewReader(sseData), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		events = append(events, ev)
		return true
	})

	if len(events) != 3 {
		t.Fatalf("got %d events, want 3 (thinking, content, done)", len(events))
	}

	if events[0].Type != llm.EventThinking || events[0].Thinking != "reasoning..." {
		t.Errorf("event[0] = %+v, want EventThinking with 'reasoning...'", events[0])
	}
	if events[1].Type != llm.EventContent || events[1].Text != "answer" {
		t.Errorf("event[1] = %+v, want EventContent with 'answer'", events[1])
	}
	if events[2].Type != llm.EventDone {
		t.Errorf("event[2].Type = %v, want EventDone", events[2].Type)
	}
	if events[2].ThinkingTokens != 80 {
		t.Errorf("EventDone.ThinkingTokens = %d, want 80", events[2].ThinkingTokens)
	}
}

// TestBuildRequestMaxOutputTokensPaddedForThinkingModel verifies that the
// Gemini adapter pads maxOutputTokens for models that think by default
// (e.g. gemini-2.5-flash). Gemini 2.5 models do NOT support thinkingConfig —
// thinking tokens silently consume maxOutputTokens. The adapter must increase
// it so the caller's intended output budget isn't starved.
func TestBuildRequestMaxOutputTokensPaddedForThinkingModel(t *testing.T) {
	c := &Client{
		config:       llm.Config{Model: "gemini-2.5-flash"},
		providerName: "gemini",
	}

	req := llm.Request{
		MaxTokens: 50,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentPart{{Type: llm.ContentText, Text: "What is 2+2?"}}},
		},
	}

	apiReq, err := c.buildRequest(req)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	// With MaxTokens: 50, thinking would consume the budget leaving ~5 output
	// tokens. The adapter should pad maxOutputTokens with a thinking overhead.
	if apiReq.GenerationConfig.MaxOutputTokens <= 50 {
		t.Errorf("MaxOutputTokens = %d, want > 50 (should be padded for thinking model)",
			apiReq.GenerationConfig.MaxOutputTokens)
	}

	// thinkingConfig should NOT be set — Gemini 2.5 API rejects it
	if apiReq.GenerationConfig.ThinkingConfig != nil {
		t.Error("ThinkingConfig should be nil for gemini-2.5 (API rejects it)")
	}
}

// TestBuildRequestThinkingConfigSkippedForNativeThinkingModel verifies that
// setting ThinkingLevel on a Gemini 2.5 model (Thinking: true) does NOT
// produce thinkingConfig — the API rejects it.
func TestBuildRequestThinkingConfigSkippedForNativeThinkingModel(t *testing.T) {
	c := &Client{
		config:       llm.Config{Model: "gemini-2.5-flash"},
		providerName: "gemini",
	}

	req := llm.Request{
		MaxTokens:     100,
		ThinkingLevel: llm.ThinkingHigh,
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "think"),
		},
	}

	apiReq, err := c.buildRequest(req)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	if apiReq.GenerationConfig.ThinkingConfig != nil {
		t.Error("ThinkingConfig should be nil for gemini-2.5-flash (native thinking model)")
	}

	// MaxOutputTokens should still be padded
	if apiReq.GenerationConfig.MaxOutputTokens <= 100 {
		t.Errorf("MaxOutputTokens = %d, want > 100 (should be padded)",
			apiReq.GenerationConfig.MaxOutputTokens)
	}
}

// TestBuildRequestMaxOutputTokensNotPaddedForNonThinkingModel verifies that
// models that do NOT think by default keep maxOutputTokens as-is.
func TestBuildRequestMaxOutputTokensNotPaddedForNonThinkingModel(t *testing.T) {
	c := &Client{
		config:       llm.Config{Model: "gemini-2.0-flash"},
		providerName: "gemini",
	}

	req := llm.Request{
		MaxTokens: 50,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentPart{{Type: llm.ContentText, Text: "hi"}}},
		},
	}

	apiReq, err := c.buildRequest(req)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	if apiReq.GenerationConfig.MaxOutputTokens != 50 {
		t.Errorf("MaxOutputTokens = %d, want 50 (should not pad for non-thinking model)",
			apiReq.GenerationConfig.MaxOutputTokens)
	}
}

// TestBuildRequestThinkingConfigInsideGenerationConfig verifies that
// thinkingConfig is nested inside generationConfig (not top-level).
func TestBuildRequestThinkingConfigInsideGenerationConfig(t *testing.T) {
	c := &Client{
		config:       llm.Config{Model: "gemini-3-pro-preview"},
		providerName: "gemini",
	}

	req := llm.Request{
		MaxTokens:     1024,
		ThinkingLevel: llm.ThinkingMedium,
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "reason about this"),
		},
	}

	apiReq, err := c.buildRequest(req)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	tc := apiReq.GenerationConfig.ThinkingConfig
	if tc == nil {
		t.Fatal("GenerationConfig.ThinkingConfig is nil, want non-nil")
	}

	// Standard level → string-based thinkingLevel, not thinkingBudget
	if tc.ThinkingLevel != "medium" {
		t.Errorf("ThinkingLevel = %q, want %q", tc.ThinkingLevel, "medium")
	}
	if tc.ThinkingBudget != 0 {
		t.Errorf("ThinkingBudget = %d, want 0 (should use ThinkingLevel string)", tc.ThinkingBudget)
	}

	// Verify JSON nesting via marshal
	data, err := json.Marshal(apiReq)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if _, ok := raw["thinkingConfig"]; ok {
		t.Error("thinkingConfig found at top level — must be nested inside generationConfig")
	}
	var gc map[string]json.RawMessage
	if err := json.Unmarshal(raw["generationConfig"], &gc); err != nil {
		t.Fatalf("Unmarshal generationConfig: %v", err)
	}
	if _, ok := gc["thinkingConfig"]; !ok {
		t.Error("thinkingConfig not found inside generationConfig")
	}
}

// TestBuildRequestThinkingLevelStringMapping verifies the ThinkingLevel →
// Gemini thinkingLevel/thinkingBudget mapping.
func TestBuildRequestThinkingLevelStringMapping(t *testing.T) {
	tests := []struct {
		name       string
		level      llm.ThinkingLevel
		wantLevel  string // expected thinkingLevel string (empty = use budget)
		wantBudget bool   // true if thinkingBudget should be set instead
	}{
		{"low", llm.ThinkingLow, "low", false},
		{"medium", llm.ThinkingMedium, "medium", false},
		{"high", llm.ThinkingHigh, "high", false},
		{"minimal_fallback", llm.ThinkingMinimal, "", true},
		{"xhigh_fallback", llm.ThinkingXHigh, "", true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c := &Client{
				config:       llm.Config{Model: "gemini-3-pro-preview"},
				providerName: "gemini",
			}
			apiReq, err := c.buildRequest(llm.Request{
				MaxTokens:     1024,
				ThinkingLevel: tc.level,
				Messages:      []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
			})
			if err != nil {
				t.Fatalf("buildRequest: %v", err)
			}
			cfg := apiReq.GenerationConfig.ThinkingConfig
			if cfg == nil {
				t.Fatal("ThinkingConfig is nil")
			}
			if tc.wantLevel != "" {
				if cfg.ThinkingLevel != tc.wantLevel {
					t.Errorf("ThinkingLevel = %q, want %q", cfg.ThinkingLevel, tc.wantLevel)
				}
				if cfg.ThinkingBudget != 0 {
					t.Errorf("ThinkingBudget = %d, want 0", cfg.ThinkingBudget)
				}
			} else {
				if cfg.ThinkingBudget == 0 {
					t.Error("ThinkingBudget = 0, want > 0 (fallback to token budget)")
				}
				if cfg.ThinkingLevel != "" {
					t.Errorf("ThinkingLevel = %q, want empty", cfg.ThinkingLevel)
				}
			}
		})
	}
}

// TestBuildRequestThinkingBudgetClampedToModelMax verifies that a thinking
// budget exceeding the model's MaxTokens is clamped.
func TestBuildRequestThinkingBudgetClampedToModelMax(t *testing.T) {
	// gemini-3-pro-preview has MaxTokens=65536 — ThinkingXHigh (128000) exceeds it.
	c := &Client{
		config:       llm.Config{Model: "gemini-3-pro-preview"},
		providerName: "gemini",
	}

	req := llm.Request{
		MaxTokens:     100,
		ThinkingLevel: llm.ThinkingXHigh,
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "think hard"),
		},
	}

	apiReq, err := c.buildRequest(req)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	info, ok := llm.GetModelInfo("gemini-3-pro-preview")
	if !ok {
		t.Fatal("gemini-3-pro-preview not in registry")
	}

	tc := apiReq.GenerationConfig.ThinkingConfig
	if tc == nil {
		t.Fatal("ThinkingConfig is nil, want non-nil")
	}
	if tc.ThinkingBudget > info.MaxTokens {
		t.Errorf("ThinkingBudget = %d, want <= %d (model MaxTokens)",
			tc.ThinkingBudget, info.MaxTokens)
	}
}

// TestBuildRequestCustomThinkingBudget verifies that a custom ThinkingBudget
// overrides the level and uses thinkingBudget (not thinkingLevel).
func TestBuildRequestCustomThinkingBudget(t *testing.T) {
	c := &Client{
		config:       llm.Config{Model: "gemini-3-pro-preview"},
		providerName: "gemini",
	}

	req := llm.Request{
		MaxTokens:      1024,
		ThinkingLevel:  llm.ThinkingMedium,
		ThinkingBudget: 8192,
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "hi"),
		},
	}

	apiReq, err := c.buildRequest(req)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	tc := apiReq.GenerationConfig.ThinkingConfig
	if tc == nil {
		t.Fatal("ThinkingConfig is nil")
	}
	if tc.ThinkingBudget != 8192 {
		t.Errorf("ThinkingBudget = %d, want 8192", tc.ThinkingBudget)
	}
	if tc.ThinkingLevel != "" {
		t.Errorf("ThinkingLevel = %q, want empty (custom budget overrides)", tc.ThinkingLevel)
	}
}
