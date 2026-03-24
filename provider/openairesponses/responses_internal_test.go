package openairesponses

import (
	"encoding/json"
	"log/slog"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// =============================================================================
// buildRequest TESTS
// =============================================================================

// TestBuildRequestBasicText verifies a simple text-only request.
func TestBuildRequestBasicText(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gpt-5"}, providerName: "openai_responses"}

	req := llm.Request{
		System:    "You are helpful.",
		MaxTokens: 100,
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "Hello"),
		},
	}

	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	if apiReq.Model != "gpt-5" {
		t.Errorf("Model = %q, want %q", apiReq.Model, "gpt-5")
	}
	if apiReq.MaxOutputTokens != 100 {
		t.Errorf("MaxOutputTokens = %d, want 100", apiReq.MaxOutputTokens)
	}
	if apiReq.Stream {
		t.Error("Stream should be false")
	}
	if apiReq.Store {
		t.Error("Store should be false")
	}
	if apiReq.Temperature != nil {
		t.Error("Temperature should be nil (omitted for reasoning models)")
	}

	// Should have 2 input items: system + user
	if len(apiReq.Input) != 2 {
		t.Fatalf("len(Input) = %d, want 2", len(apiReq.Input))
	}

	// System message
	sysMsg, ok := apiReq.Input[0].(responsesInputMsg)
	if !ok {
		t.Fatalf("Input[0] is %T, want responsesInputMsg", apiReq.Input[0])
	}
	if sysMsg.Role != "system" || sysMsg.Content != "You are helpful." {
		t.Errorf("system msg = %+v", sysMsg)
	}

	// User message
	userMsg, ok := apiReq.Input[1].(responsesInputMsg)
	if !ok {
		t.Fatalf("Input[1] is %T, want responsesInputMsg", apiReq.Input[1])
	}
	if userMsg.Role != "user" || userMsg.Content != "Hello" {
		t.Errorf("user msg = %+v", userMsg)
	}
}

// TestBuildRequestStreaming verifies stream flag is set.
func TestBuildRequestStreaming(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gpt-5"}, providerName: "openai_responses"}

	apiReq, err := c.buildRequest(llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
	}, true)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}
	if !apiReq.Stream {
		t.Error("Stream should be true")
	}
}

// TestBuildRequestReasoningEffort verifies reasoning effort and summary.
func TestBuildRequestReasoningEffort(t *testing.T) {
	tests := []struct {
		name           string
		level          llm.ThinkingLevel
		summary        llm.ReasoningSummary
		wantEffort     string
		wantSummary    string
		wantReasoning  bool
	}{
		{"none", llm.ThinkingNone, llm.ReasoningSummaryNone, "", "", false},
		{"low", llm.ThinkingLow, llm.ReasoningSummaryNone, "low", "", true},
		{"medium+auto", llm.ThinkingMedium, llm.ReasoningSummaryAuto, "medium", "auto", true},
		{"high+detailed", llm.ThinkingHigh, llm.ReasoningSummaryDetailed, "high", "detailed", true},
		{"minimal", llm.ThinkingMinimal, llm.ReasoningSummaryNone, "minimal", "", true},
		{"xhigh+concise", llm.ThinkingXHigh, llm.ReasoningSummaryConcise, "xhigh", "concise", true},
		{"summary only", llm.ThinkingNone, llm.ReasoningSummaryAuto, "", "auto", true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c := &Client{config: llm.Config{Model: "gpt-5"}, providerName: "openai_responses"}
			apiReq, err := c.buildRequest(llm.Request{
				Messages:         []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
				ThinkingLevel:    tc.level,
				ReasoningSummary: tc.summary,
			}, false)
			if err != nil {
				t.Fatalf("buildRequest: %v", err)
			}

			if tc.wantReasoning {
				if apiReq.Reasoning == nil {
					t.Fatal("Reasoning is nil, want non-nil")
				}
				if apiReq.Reasoning.Effort != tc.wantEffort {
					t.Errorf("Reasoning.Effort = %q, want %q", apiReq.Reasoning.Effort, tc.wantEffort)
				}
				if apiReq.Reasoning.Summary != tc.wantSummary {
					t.Errorf("Reasoning.Summary = %q, want %q", apiReq.Reasoning.Summary, tc.wantSummary)
				}
			} else {
				if apiReq.Reasoning != nil {
					t.Errorf("Reasoning = %+v, want nil", apiReq.Reasoning)
				}
			}
		})
	}
}

// TestBuildRequestTemperatureWarning verifies Temperature is not set on the
// wire format even when provided in the request, and that a warning is logged.
func TestBuildRequestTemperatureWarning(t *testing.T) {
	// Capture slog output to verify warning
	var buf strings.Builder
	old := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, nil)))
	defer slog.SetDefault(old)

	c := &Client{config: llm.Config{Model: "gpt-5"}, providerName: "openai_responses"}
	temp := 0.7

	apiReq, err := c.buildRequest(llm.Request{
		Messages:    []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
		Temperature: &temp,
	}, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	if apiReq.Temperature != nil {
		t.Errorf("Temperature = %v, want nil (should be omitted for reasoning models)", *apiReq.Temperature)
	}

	if !strings.Contains(buf.String(), "ignoring Temperature") {
		t.Errorf("expected slog.Warn about ignoring Temperature, got: %q", buf.String())
	}
}

// TestBuildRequestConfigFallback verifies config defaults are used when
// request fields are zero.
func TestBuildRequestConfigFallback(t *testing.T) {
	c := &Client{
		config: llm.Config{
			Model:         "gpt-5-pro",
			MaxTokens:     4096,
			ThinkingLevel: llm.ThinkingHigh,
		},
		providerName: "openai_responses",
	}

	apiReq, err := c.buildRequest(llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
	}, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	if apiReq.Model != "gpt-5-pro" {
		t.Errorf("Model = %q, want %q", apiReq.Model, "gpt-5-pro")
	}
	if apiReq.MaxOutputTokens != 4096 {
		t.Errorf("MaxOutputTokens = %d, want 4096", apiReq.MaxOutputTokens)
	}
	if apiReq.Reasoning == nil || apiReq.Reasoning.Effort != "high" {
		t.Errorf("Reasoning = %+v, want effort=high", apiReq.Reasoning)
	}
}

// TestBuildRequestTools verifies tool serialization.
func TestBuildRequestTools(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gpt-5"}, providerName: "openai_responses"}

	apiReq, err := c.buildRequest(llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "What's the weather?")},
		Tools: []llm.Tool{
			{
				Name:        "get_weather",
				Description: "Get weather for a location",
				InputSchema: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]interface{}{"type": "string"},
					},
				},
			},
		},
	}, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	if len(apiReq.Tools) != 1 {
		t.Fatalf("len(Tools) = %d, want 1", len(apiReq.Tools))
	}
	if apiReq.Tools[0].Type != "function" {
		t.Errorf("Tools[0].Type = %q, want %q", apiReq.Tools[0].Type, "function")
	}
	if apiReq.Tools[0].Function.Name != "get_weather" {
		t.Errorf("Tools[0].Function.Name = %q, want %q", apiReq.Tools[0].Function.Name, "get_weather")
	}
}

// TestBuildRequestAssistantToolUseRoundTrip verifies that assistant messages
// with ContentToolUse parts are serialized as function_call items, and that
// tool results are serialized as function_call_output items.
func TestBuildRequestAssistantToolUseRoundTrip(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gpt-5"}, providerName: "openai_responses"}

	// Simulate a multi-turn tool-use conversation:
	// 1. User asks a question
	// 2. Assistant responds with a tool call (from NewAssistantMessage)
	// 3. User provides the tool result (from NewToolResultMessage)
	// 4. User asks a follow-up
	toolInput := map[string]interface{}{"location": "Paris"}
	assistantResp := &llm.Response{
		Content: "Let me check the weather.",
		ToolCalls: []llm.ToolCall{
			{ID: "call_abc123", Name: "get_weather", Input: toolInput},
		},
	}

	req := llm.Request{
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "What's the weather in Paris?"),
			llm.NewAssistantMessage(assistantResp),
			llm.NewToolResultMessage("call_abc123", `{"temp": 22, "condition": "sunny"}`, false),
			llm.NewTextMessage(llm.RoleUser, "Thanks! How about tomorrow?"),
		},
	}

	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	// Expected input items:
	// [0] user text "What's the weather in Paris?"
	// [1] assistant text "Let me check the weather."
	// [2] function_call {call_id: "call_abc123", name: "get_weather", arguments: ...}
	// [3] function_call_output {call_id: "call_abc123", output: ...}
	// [4] user text "Thanks! How about tomorrow?"
	if len(apiReq.Input) != 5 {
		t.Fatalf("len(Input) = %d, want 5", len(apiReq.Input))
	}

	// [0] user message
	userMsg, ok := apiReq.Input[0].(responsesInputMsg)
	if !ok || userMsg.Role != "user" {
		t.Errorf("Input[0]: got %T %+v, want user message", apiReq.Input[0], apiReq.Input[0])
	}

	// [1] assistant text
	assistantMsg, ok := apiReq.Input[1].(responsesInputMsg)
	if !ok || assistantMsg.Role != "assistant" {
		t.Errorf("Input[1]: got %T %+v, want assistant message", apiReq.Input[1], apiReq.Input[1])
	}
	if assistantMsg.Content != "Let me check the weather." {
		t.Errorf("Input[1].Content = %v, want %q", assistantMsg.Content, "Let me check the weather.")
	}

	// [2] function_call
	funcCall, ok := apiReq.Input[2].(responsesFunctionCall)
	if !ok {
		t.Fatalf("Input[2] is %T, want responsesFunctionCall", apiReq.Input[2])
	}
	if funcCall.Type != "function_call" {
		t.Errorf("function_call.Type = %q, want %q", funcCall.Type, "function_call")
	}
	if funcCall.CallID != "call_abc123" {
		t.Errorf("function_call.CallID = %q, want %q", funcCall.CallID, "call_abc123")
	}
	if funcCall.Name != "get_weather" {
		t.Errorf("function_call.Name = %q, want %q", funcCall.Name, "get_weather")
	}
	// Verify arguments contain the tool input
	var parsedArgs map[string]interface{}
	if err := json.Unmarshal([]byte(funcCall.Arguments), &parsedArgs); err != nil {
		t.Fatalf("failed to parse function_call.Arguments: %v", err)
	}
	if parsedArgs["location"] != "Paris" {
		t.Errorf("Arguments.location = %v, want %q", parsedArgs["location"], "Paris")
	}

	// [3] function_call_output
	funcOutput, ok := apiReq.Input[3].(responsesFunctionCallOutput)
	if !ok {
		t.Fatalf("Input[3] is %T, want responsesFunctionCallOutput", apiReq.Input[3])
	}
	if funcOutput.Type != "function_call_output" {
		t.Errorf("function_call_output.Type = %q, want %q", funcOutput.Type, "function_call_output")
	}
	if funcOutput.CallID != "call_abc123" {
		t.Errorf("function_call_output.CallID = %q, want %q", funcOutput.CallID, "call_abc123")
	}
	if funcOutput.Output != `{"temp": 22, "condition": "sunny"}` {
		t.Errorf("function_call_output.Output = %q", funcOutput.Output)
	}

	// [4] follow-up user message
	followUp, ok := apiReq.Input[4].(responsesInputMsg)
	if !ok || followUp.Role != "user" {
		t.Errorf("Input[4]: got %T %+v, want user message", apiReq.Input[4], apiReq.Input[4])
	}
}

// TestBuildRequestAssistantToolUseOnly verifies an assistant message that has
// only tool calls (no text content).
func TestBuildRequestAssistantToolUseOnly(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gpt-5"}, providerName: "openai_responses"}

	assistantResp := &llm.Response{
		ToolCalls: []llm.ToolCall{
			{ID: "call_1", Name: "search", Input: map[string]interface{}{"q": "test"}},
			{ID: "call_2", Name: "fetch", Input: map[string]interface{}{"url": "https://example.com"}},
		},
	}

	req := llm.Request{
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "Find and fetch"),
			llm.NewAssistantMessage(assistantResp),
		},
	}

	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	// [0] user text, [1] function_call "search", [2] function_call "fetch"
	if len(apiReq.Input) != 3 {
		t.Fatalf("len(Input) = %d, want 3", len(apiReq.Input))
	}

	fc1, ok := apiReq.Input[1].(responsesFunctionCall)
	if !ok {
		t.Fatalf("Input[1] is %T, want responsesFunctionCall", apiReq.Input[1])
	}
	if fc1.Name != "search" || fc1.CallID != "call_1" {
		t.Errorf("Input[1] = %+v, want search/call_1", fc1)
	}

	fc2, ok := apiReq.Input[2].(responsesFunctionCall)
	if !ok {
		t.Fatalf("Input[2] is %T, want responsesFunctionCall", apiReq.Input[2])
	}
	if fc2.Name != "fetch" || fc2.CallID != "call_2" {
		t.Errorf("Input[2] = %+v, want fetch/call_2", fc2)
	}
}

// TestBuildRequestImageContent verifies image messages produce the correct
// multipart content array.
func TestBuildRequestImageContent(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gpt-5"}, providerName: "openai_responses"}

	req := llm.Request{
		Messages: []llm.Message{
			{
				Role: llm.RoleUser,
				Content: []llm.ContentPart{
					{Type: llm.ContentText, Text: "What is this?"},
					{
						Type: llm.ContentImage,
						Source: &llm.ImageSource{
							Type:      "base64",
							MediaType: "image/png",
							Data:      "iVBORw0KGgo=",
						},
					},
				},
			},
		},
	}

	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	if len(apiReq.Input) != 1 {
		t.Fatalf("len(Input) = %d, want 1", len(apiReq.Input))
	}

	msg, ok := apiReq.Input[0].(responsesInputMsg)
	if !ok {
		t.Fatalf("Input[0] is %T, want responsesInputMsg", apiReq.Input[0])
	}
	if msg.Role != "user" {
		t.Errorf("Role = %q, want %q", msg.Role, "user")
	}

	// Content should be an array with input_text and input_image
	contentArray, ok := msg.Content.([]interface{})
	if !ok {
		t.Fatalf("Content is %T, want []interface{}", msg.Content)
	}
	if len(contentArray) != 2 {
		t.Fatalf("len(contentArray) = %d, want 2", len(contentArray))
	}

	textPart, ok := contentArray[0].(map[string]interface{})
	if !ok || textPart["type"] != "input_text" {
		t.Errorf("contentArray[0] = %+v, want input_text", contentArray[0])
	}
	imgPart, ok := contentArray[1].(map[string]interface{})
	if !ok || imgPart["type"] != "input_image" {
		t.Errorf("contentArray[1] = %+v, want input_image", contentArray[1])
	}
}

// TestBuildRequestInvalidImage verifies that invalid image sources return errors.
func TestBuildRequestInvalidImage(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gpt-5"}, providerName: "openai_responses"}

	req := llm.Request{
		Messages: []llm.Message{
			{
				Role: llm.RoleUser,
				Content: []llm.ContentPart{
					{Type: llm.ContentImage, Source: nil},
				},
			},
		},
	}

	_, err := c.buildRequest(req, false)
	if err == nil {
		t.Fatal("expected error for nil image source")
	}
}

// TestBuildRequestWireFormat verifies the full JSON wire format of a request
// with tools, reasoning, and messages.
func TestBuildRequestWireFormat(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gpt-5"}, providerName: "openai_responses"}

	req := llm.Request{
		System:           "Be concise.",
		MaxTokens:        500,
		ThinkingLevel:    llm.ThinkingMedium,
		ReasoningSummary: llm.ReasoningSummaryAuto,
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "Explain Go generics."),
		},
	}

	apiReq, err := c.buildRequest(req, true)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	b, err := json.Marshal(apiReq)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}

	// Verify key fields in the JSON
	var wire map[string]interface{}
	json.Unmarshal(b, &wire)

	if wire["model"] != "gpt-5" {
		t.Errorf("wire model = %v", wire["model"])
	}
	if wire["stream"] != true {
		t.Errorf("wire stream = %v", wire["stream"])
	}
	if wire["store"] != false {
		t.Errorf("wire store = %v", wire["store"])
	}
	if wire["temperature"] != nil {
		t.Errorf("wire temperature should be omitted, got %v", wire["temperature"])
	}

	reasoning, ok := wire["reasoning"].(map[string]interface{})
	if !ok {
		t.Fatalf("reasoning is %T", wire["reasoning"])
	}
	if reasoning["effort"] != "medium" {
		t.Errorf("reasoning.effort = %v", reasoning["effort"])
	}
	if reasoning["summary"] != "auto" {
		t.Errorf("reasoning.summary = %v", reasoning["summary"])
	}
}

// TestBuildRequestToolUseWireFormat verifies the JSON wire format for
// function_call and function_call_output items.
func TestBuildRequestToolUseWireFormat(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gpt-5"}, providerName: "openai_responses"}

	assistantResp := &llm.Response{
		ToolCalls: []llm.ToolCall{
			{ID: "call_xyz", Name: "calc", Input: map[string]interface{}{"expr": "2+2"}},
		},
	}

	req := llm.Request{
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "Calculate 2+2"),
			llm.NewAssistantMessage(assistantResp),
			llm.NewToolResultMessage("call_xyz", "4", false),
		},
	}

	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	b, err := json.Marshal(apiReq)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}

	// Verify wire format contains expected types
	s := string(b)
	if !strings.Contains(s, `"type":"function_call"`) {
		t.Error("wire format missing function_call type")
	}
	if !strings.Contains(s, `"type":"function_call_output"`) {
		t.Error("wire format missing function_call_output type")
	}
	if !strings.Contains(s, `"call_id":"call_xyz"`) {
		t.Error("wire format missing call_id")
	}

	// Should NOT contain role="tool" (old format)
	if strings.Contains(s, `"role":"tool"`) {
		t.Error("wire format should not contain role=tool (uses function_call_output instead)")
	}
}

// =============================================================================
// parseResponse TESTS
// =============================================================================

// TestParseResponseTextOnly verifies parsing a simple text response.
func TestParseResponseTextOnly(t *testing.T) {
	c := &Client{providerName: "openai_responses"}

	apiResp := &responsesResponse{
		ID:     "resp_001",
		Model:  "gpt-5",
		Status: "completed",
		Output: []responsesOutputItem{
			{
				Type: "message",
				Content: []responsesContentBlock{
					{Type: "output_text", Text: "Hello, world!"},
				},
			},
		},
		Usage: responsesUsage{
			InputTokens:  10,
			OutputTokens: 5,
		},
	}

	resp := c.parseResponse(apiResp)

	if resp.ID != "resp_001" {
		t.Errorf("ID = %q, want %q", resp.ID, "resp_001")
	}
	if resp.Model != "gpt-5" {
		t.Errorf("Model = %q, want %q", resp.Model, "gpt-5")
	}
	if resp.Content != "Hello, world!" {
		t.Errorf("Content = %q, want %q", resp.Content, "Hello, world!")
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, "end_turn")
	}
	if resp.InputTokens != 10 {
		t.Errorf("InputTokens = %d, want 10", resp.InputTokens)
	}
	if resp.OutputTokens != 5 {
		t.Errorf("OutputTokens = %d, want 5", resp.OutputTokens)
	}
}

// TestParseResponseToolCalls verifies parsing a response with function calls.
func TestParseResponseToolCalls(t *testing.T) {
	c := &Client{providerName: "openai_responses"}

	apiResp := &responsesResponse{
		ID:     "resp_002",
		Model:  "gpt-5",
		Status: "completed",
		Output: []responsesOutputItem{
			{
				Type:      "function_call",
				CallID:    "call_abc",
				Name:      "get_weather",
				Arguments: `{"location":"Paris"}`,
			},
		},
		Usage: responsesUsage{InputTokens: 20, OutputTokens: 10},
	}

	resp := c.parseResponse(apiResp)

	if resp.StopReason != "tool_use" {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, "tool_use")
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls) = %d, want 1", len(resp.ToolCalls))
	}
	tc := resp.ToolCalls[0]
	if tc.ID != "call_abc" {
		t.Errorf("ToolCall.ID = %q, want %q", tc.ID, "call_abc")
	}
	if tc.Name != "get_weather" {
		t.Errorf("ToolCall.Name = %q, want %q", tc.Name, "get_weather")
	}
	inputMap, ok := tc.Input.(map[string]interface{})
	if !ok {
		t.Fatalf("ToolCall.Input is %T, want map", tc.Input)
	}
	if inputMap["location"] != "Paris" {
		t.Errorf("ToolCall.Input.location = %v", inputMap["location"])
	}
}

// TestParseResponseMultipleToolCalls verifies parsing multiple function calls.
func TestParseResponseMultipleToolCalls(t *testing.T) {
	c := &Client{providerName: "openai_responses"}

	apiResp := &responsesResponse{
		ID:     "resp_003",
		Model:  "gpt-5",
		Status: "completed",
		Output: []responsesOutputItem{
			{Type: "function_call", CallID: "call_1", Name: "search", Arguments: `{"q":"go"}`},
			{Type: "function_call", CallID: "call_2", Name: "fetch", Arguments: `{"url":"https://go.dev"}`},
		},
		Usage: responsesUsage{InputTokens: 30, OutputTokens: 15},
	}

	resp := c.parseResponse(apiResp)

	if len(resp.ToolCalls) != 2 {
		t.Fatalf("len(ToolCalls) = %d, want 2", len(resp.ToolCalls))
	}
	if resp.ToolCalls[0].Name != "search" {
		t.Errorf("ToolCalls[0].Name = %q", resp.ToolCalls[0].Name)
	}
	if resp.ToolCalls[1].Name != "fetch" {
		t.Errorf("ToolCalls[1].Name = %q", resp.ToolCalls[1].Name)
	}
}

// TestParseResponseReasoning verifies parsing reasoning summary output.
func TestParseResponseReasoning(t *testing.T) {
	c := &Client{providerName: "openai_responses"}

	apiResp := &responsesResponse{
		ID:     "resp_004",
		Model:  "gpt-5",
		Status: "completed",
		Output: []responsesOutputItem{
			{
				Type: "reasoning",
				Summary: []responsesSummaryText{
					{Type: "summary_text", Text: "Step 1: analyze the problem."},
					{Type: "summary_text", Text: "Step 2: solve it."},
				},
			},
			{
				Type:    "message",
				Content: []responsesContentBlock{{Type: "output_text", Text: "42"}},
			},
		},
		Usage: responsesUsage{InputTokens: 15, OutputTokens: 8, ReasoningTokens: 200},
	}

	resp := c.parseResponse(apiResp)

	if resp.Content != "42" {
		t.Errorf("Content = %q, want %q", resp.Content, "42")
	}
	if resp.Thinking != "Step 1: analyze the problem.\nStep 2: solve it." {
		t.Errorf("Thinking = %q", resp.Thinking)
	}
	if resp.ThinkingTokens != 200 {
		t.Errorf("ThinkingTokens = %d, want 200", resp.ThinkingTokens)
	}
}

// TestParseResponseStopReasons verifies mapping of status to stop reasons.
func TestParseResponseStopReasons(t *testing.T) {
	tests := []struct {
		status           string
		incompleteReason string
		wantStop         string
	}{
		{"completed", "", "end_turn"},
		{"incomplete", "max_output_tokens", "max_tokens"},
		{"incomplete", "content_filter", "content_filter"},
		{"incomplete", "", "max_tokens"},
		{"failed", "", "error"},
	}

	c := &Client{providerName: "openai_responses"}

	for _, tc := range tests {
		t.Run(tc.status+"_"+tc.incompleteReason, func(t *testing.T) {
			apiResp := &responsesResponse{Status: tc.status}
			if tc.incompleteReason != "" {
				apiResp.IncompleteDetails = &responsesIncomplete{Reason: tc.incompleteReason}
			}

			resp := c.parseResponse(apiResp)
			if resp.StopReason != tc.wantStop {
				t.Errorf("StopReason = %q, want %q", resp.StopReason, tc.wantStop)
			}
		})
	}
}

// TestParseResponseInvalidToolJSON verifies graceful handling of malformed
// function call arguments.
func TestParseResponseInvalidToolJSON(t *testing.T) {
	c := &Client{providerName: "openai_responses"}

	apiResp := &responsesResponse{
		Status: "completed",
		Output: []responsesOutputItem{
			{Type: "function_call", CallID: "call_bad", Name: "broken", Arguments: "not json{{{"},
		},
	}

	resp := c.parseResponse(apiResp)

	if len(resp.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls) = %d, want 1", len(resp.ToolCalls))
	}
	// Input should fall back to raw string
	if resp.ToolCalls[0].Input != "not json{{{" {
		t.Errorf("Input = %v, want raw string fallback", resp.ToolCalls[0].Input)
	}
}

// =============================================================================
// parseStream TESTS
// =============================================================================

// TestParseStreamContentOnly verifies streaming a simple text response.
func TestParseStreamContentOnly(t *testing.T) {
	sseData := "data: " + `{"type":"response.output_text.delta","delta":"Hello"}` + "\n\n" +
		"data: " + `{"type":"response.output_text.delta","delta":", world!"}` + "\n\n" +
		"data: " + `{"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":10,"output_tokens":5}}}` + "\n\n"

	c := &Client{providerName: "openai_responses"}
	var events []llm.StreamEvent
	c.parseStream(strings.NewReader(sseData), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		events = append(events, ev)
		return true
	})

	if len(events) != 3 {
		t.Fatalf("got %d events, want 3 (content, content, done)", len(events))
	}
	if events[0].Type != llm.EventContent || events[0].Text != "Hello" {
		t.Errorf("event[0] = %+v", events[0])
	}
	if events[1].Type != llm.EventContent || events[1].Text != ", world!" {
		t.Errorf("event[1] = %+v", events[1])
	}
	if events[2].Type != llm.EventDone {
		t.Errorf("event[2].Type = %v, want EventDone", events[2].Type)
	}
	if events[2].StopReason != "end_turn" {
		t.Errorf("StopReason = %q, want %q", events[2].StopReason, "end_turn")
	}
	if events[2].InputTokens != 10 {
		t.Errorf("InputTokens = %d, want 10", events[2].InputTokens)
	}
	if events[2].OutputTokens != 5 {
		t.Errorf("OutputTokens = %d, want 5", events[2].OutputTokens)
	}
}

// TestParseStreamToolCall verifies streaming a function call with argument deltas.
func TestParseStreamToolCall(t *testing.T) {
	sseData := "data: " + `{"type":"response.function_call_arguments.delta","delta":"{\"loc"}` + "\n\n" +
		"data: " + `{"type":"response.function_call_arguments.delta","delta":"ation\":\"NYC\"}"}` + "\n\n" +
		"data: " + `{"type":"response.function_call_arguments.done","name":"get_weather","call_id":"call_123","arguments":"{\"location\":\"NYC\"}"}` + "\n\n" +
		"data: " + `{"type":"response.completed","response":{"id":"resp_2","status":"completed","usage":{"input_tokens":20,"output_tokens":10}}}` + "\n\n"

	c := &Client{providerName: "openai_responses"}
	var events []llm.StreamEvent
	c.parseStream(strings.NewReader(sseData), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		events = append(events, ev)
		return true
	})

	if len(events) != 2 {
		t.Fatalf("got %d events, want 2 (tool_use, done)", len(events))
	}

	if events[0].Type != llm.EventToolUse {
		t.Fatalf("event[0].Type = %v, want EventToolUse", events[0].Type)
	}
	tc := events[0].ToolCall
	if tc == nil {
		t.Fatal("ToolCall is nil")
	}
	if tc.ID != "call_123" {
		t.Errorf("ToolCall.ID = %q, want %q", tc.ID, "call_123")
	}
	if tc.Name != "get_weather" {
		t.Errorf("ToolCall.Name = %q, want %q", tc.Name, "get_weather")
	}
	inputMap, ok := tc.Input.(map[string]interface{})
	if !ok {
		t.Fatalf("ToolCall.Input is %T", tc.Input)
	}
	if inputMap["location"] != "NYC" {
		t.Errorf("Input.location = %v", inputMap["location"])
	}
}

// TestParseStreamToolCallFallbackToBuffer verifies that if the done event has
// empty arguments, the accumulated buffer is used.
func TestParseStreamToolCallFallbackToBuffer(t *testing.T) {
	sseData := "data: " + `{"type":"response.function_call_arguments.delta","delta":"{\"x\":1}"}` + "\n\n" +
		"data: " + `{"type":"response.function_call_arguments.done","name":"calc","call_id":"call_buf","arguments":""}` + "\n\n" +
		"data: " + `{"type":"response.completed","response":{"id":"resp_buf","status":"completed","usage":{"input_tokens":5,"output_tokens":3}}}` + "\n\n"

	c := &Client{providerName: "openai_responses"}
	var events []llm.StreamEvent
	c.parseStream(strings.NewReader(sseData), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		events = append(events, ev)
		return true
	})

	if len(events) != 2 {
		t.Fatalf("got %d events, want 2", len(events))
	}

	tc := events[0].ToolCall
	if tc == nil {
		t.Fatal("ToolCall is nil")
	}
	inputMap, ok := tc.Input.(map[string]interface{})
	if !ok {
		t.Fatalf("ToolCall.Input is %T", tc.Input)
	}
	if inputMap["x"] != float64(1) {
		t.Errorf("Input.x = %v, want 1", inputMap["x"])
	}
}

// TestParseStreamReasoning verifies streaming reasoning summary deltas.
func TestParseStreamReasoning(t *testing.T) {
	sseData := "data: " + `{"type":"response.reasoning_summary_text.delta","delta":"thinking..."}` + "\n\n" +
		"data: " + `{"type":"response.output_text.delta","delta":"answer"}` + "\n\n" +
		"data: " + `{"type":"response.completed","response":{"id":"resp_r","status":"completed","usage":{"input_tokens":10,"output_tokens":5,"reasoning_tokens":100}}}` + "\n\n"

	c := &Client{providerName: "openai_responses"}
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
	if events[0].Type != llm.EventThinking || events[0].Thinking != "thinking..." {
		t.Errorf("event[0] = %+v, want EventThinking", events[0])
	}
	if events[1].Type != llm.EventContent || events[1].Text != "answer" {
		t.Errorf("event[1] = %+v, want EventContent", events[1])
	}
	if events[2].ThinkingTokens != 100 {
		t.Errorf("ThinkingTokens = %d, want 100", events[2].ThinkingTokens)
	}
}

// TestParseStreamIncompleteStatus verifies that incomplete status maps to max_tokens.
func TestParseStreamIncompleteStatus(t *testing.T) {
	sseData := "data: " + `{"type":"response.completed","response":{"id":"resp_i","status":"incomplete","usage":{"input_tokens":100,"output_tokens":4096}}}` + "\n\n"

	c := &Client{providerName: "openai_responses"}
	var events []llm.StreamEvent
	c.parseStream(strings.NewReader(sseData), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		events = append(events, ev)
		return true
	})

	if len(events) != 1 {
		t.Fatalf("got %d events, want 1", len(events))
	}
	if events[0].StopReason != "max_tokens" {
		t.Errorf("StopReason = %q, want %q", events[0].StopReason, "max_tokens")
	}
}

// TestParseStreamError verifies that error events yield errors.
func TestParseStreamError(t *testing.T) {
	sseData := "data: " + `{"type":"error","error":{"message":"rate limit exceeded","code":"rate_limit"}}` + "\n\n"

	c := &Client{providerName: "openai_responses"}
	var gotErr error
	c.parseStream(strings.NewReader(sseData), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			gotErr = err
			return false
		}
		return true
	})

	if gotErr == nil {
		t.Fatal("expected error from stream")
	}
	if !strings.Contains(gotErr.Error(), "rate_limit") {
		t.Errorf("error = %q, want to contain 'rate_limit'", gotErr.Error())
	}
	if !strings.Contains(gotErr.Error(), "rate limit exceeded") {
		t.Errorf("error = %q, want to contain 'rate limit exceeded'", gotErr.Error())
	}
}

// TestParseStreamDoneSignal verifies that [DONE] terminates parsing.
func TestParseStreamDoneSignal(t *testing.T) {
	sseData := "data: " + `{"type":"response.output_text.delta","delta":"hello"}` + "\n\n" +
		"data: [DONE]\n\n" +
		"data: " + `{"type":"response.output_text.delta","delta":"should not appear"}` + "\n\n"

	c := &Client{providerName: "openai_responses"}
	var events []llm.StreamEvent
	c.parseStream(strings.NewReader(sseData), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		events = append(events, ev)
		return true
	})

	// Only the first content event should appear; [DONE] stops parsing
	if len(events) != 1 {
		t.Fatalf("got %d events, want 1 (after [DONE])", len(events))
	}
}

// TestParseStreamIgnoresNonDataLines verifies that comment and event lines are skipped.
func TestParseStreamIgnoresNonDataLines(t *testing.T) {
	sseData := ": this is a comment\n" +
		"event: response.output_text.delta\n" +
		"data: " + `{"type":"response.output_text.delta","delta":"text"}` + "\n\n" +
		"data: " + `{"type":"response.completed","response":{"id":"resp","status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}` + "\n\n"

	c := &Client{providerName: "openai_responses"}
	var events []llm.StreamEvent
	c.parseStream(strings.NewReader(sseData), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		events = append(events, ev)
		return true
	})

	if len(events) != 2 {
		t.Fatalf("got %d events, want 2", len(events))
	}
	if events[0].Text != "text" {
		t.Errorf("event[0].Text = %q", events[0].Text)
	}
}

// TestParseStreamToolInputOverflow verifies that oversized tool input terminates the stream.
func TestParseStreamToolInputOverflow(t *testing.T) {
	// Build multiple deltas that accumulate past MaxToolInputBytes.
	// Each delta must fit within MaxSSELineBytes (256KB), so use 128KB chunks.
	chunkSize := 128 * 1024
	chunk := strings.Repeat("x", chunkSize)
	numChunks := (llm.MaxToolInputBytes / chunkSize) + 2 // enough to exceed the limit

	var sb strings.Builder
	for i := 0; i < numChunks; i++ {
		sb.WriteString("data: " + `{"type":"response.function_call_arguments.delta","delta":"` + chunk + `"}` + "\n\n")
	}

	c := &Client{providerName: "openai_responses"}
	var gotErr error
	c.parseStream(strings.NewReader(sb.String()), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			gotErr = err
			return false
		}
		return true
	})

	if gotErr == nil {
		t.Fatal("expected overflow error")
	}
	if !strings.Contains(gotErr.Error(), "tool input exceeded") {
		t.Errorf("error = %q, want 'tool input exceeded'", gotErr.Error())
	}
}

// TestParseStreamCallerBreaksEarly verifies that the stream stops when
// yield returns false.
func TestParseStreamCallerBreaksEarly(t *testing.T) {
	sseData := "data: " + `{"type":"response.output_text.delta","delta":"first"}` + "\n\n" +
		"data: " + `{"type":"response.output_text.delta","delta":"second"}` + "\n\n" +
		"data: " + `{"type":"response.output_text.delta","delta":"third"}` + "\n\n"

	c := &Client{providerName: "openai_responses"}
	var count int
	c.parseStream(strings.NewReader(sseData), func(ev llm.StreamEvent, err error) bool {
		count++
		return count < 2 // break after first event
	})

	if count != 2 {
		t.Errorf("count = %d, want 2 (yield returns false after first)", count)
	}
}

// TestParseStreamMalformedJSON verifies that malformed SSE data yields errors
// but does not stop parsing.
func TestParseStreamMalformedJSON(t *testing.T) {
	sseData := "data: not-json\n\n" +
		"data: " + `{"type":"response.output_text.delta","delta":"ok"}` + "\n\n" +
		"data: " + `{"type":"response.completed","response":{"id":"r","status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}` + "\n\n"

	c := &Client{providerName: "openai_responses"}
	var events []llm.StreamEvent
	var errors []error
	c.parseStream(strings.NewReader(sseData), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			errors = append(errors, err)
		} else {
			events = append(events, ev)
		}
		return true
	})

	if len(errors) != 1 {
		t.Errorf("got %d errors, want 1 (malformed JSON)", len(errors))
	}
	if len(events) != 2 {
		t.Errorf("got %d events, want 2 (content + done)", len(events))
	}
}

// TestParseStreamUnknownEventType verifies that unknown event types are silently skipped.
func TestParseStreamUnknownEventType(t *testing.T) {
	sseData := "data: " + `{"type":"response.created","id":"resp_new"}` + "\n\n" +
		"data: " + `{"type":"response.output_item.added","output_index":0}` + "\n\n" +
		"data: " + `{"type":"response.output_text.delta","delta":"hello"}` + "\n\n" +
		"data: " + `{"type":"response.output_item.done"}` + "\n\n" +
		"data: " + `{"type":"response.completed","response":{"id":"r","status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}` + "\n\n"

	c := &Client{providerName: "openai_responses"}
	var events []llm.StreamEvent
	c.parseStream(strings.NewReader(sseData), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		events = append(events, ev)
		return true
	})

	// Only content + done should come through; unknown types are skipped
	if len(events) != 2 {
		t.Fatalf("got %d events, want 2", len(events))
	}
}

// TestParseStreamMultipleToolCalls verifies that the inputBuffer is properly
// reset between sequential tool calls in a single stream.
func TestParseStreamMultipleToolCalls(t *testing.T) {
	sseData := // First tool call
		"data: " + `{"type":"response.function_call_arguments.delta","delta":"{\"a\":1}"}` + "\n\n" +
		"data: " + `{"type":"response.function_call_arguments.done","name":"tool1","call_id":"call_1","arguments":"{\"a\":1}"}` + "\n\n" +
		// Second tool call — buffer should be reset
		"data: " + `{"type":"response.function_call_arguments.delta","delta":"{\"b\":2}"}` + "\n\n" +
		"data: " + `{"type":"response.function_call_arguments.done","name":"tool2","call_id":"call_2","arguments":"{\"b\":2}"}` + "\n\n" +
		"data: " + `{"type":"response.completed","response":{"id":"r","status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}` + "\n\n"

	c := &Client{providerName: "openai_responses"}
	var events []llm.StreamEvent
	c.parseStream(strings.NewReader(sseData), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		events = append(events, ev)
		return true
	})

	// Should get: tool_use, tool_use, done
	if len(events) != 3 {
		t.Fatalf("got %d events, want 3", len(events))
	}

	if events[0].ToolCall.Name != "tool1" || events[0].ToolCall.ID != "call_1" {
		t.Errorf("event[0] = %+v, want tool1/call_1", events[0].ToolCall)
	}
	if events[1].ToolCall.Name != "tool2" || events[1].ToolCall.ID != "call_2" {
		t.Errorf("event[1] = %+v, want tool2/call_2", events[1].ToolCall)
	}

	// Verify second tool call didn't get contaminated by first
	input2, ok := events[1].ToolCall.Input.(map[string]interface{})
	if !ok {
		t.Fatalf("tool2 Input is %T", events[1].ToolCall.Input)
	}
	if _, hasA := input2["a"]; hasA {
		t.Error("tool2 input should not contain key 'a' from tool1")
	}
	if input2["b"] != float64(2) {
		t.Errorf("tool2 input.b = %v, want 2", input2["b"])
	}
}

// TestBuildRequestEmptyAssistantMessage verifies that an assistant message
// with no text and no tool calls produces no input items.
func TestBuildRequestEmptyAssistantMessage(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gpt-5"}, providerName: "openai_responses"}

	req := llm.Request{
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "hello"),
			{Role: llm.RoleAssistant, Content: []llm.ContentPart{}},
			llm.NewTextMessage(llm.RoleUser, "world"),
		},
	}

	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	// Empty assistant message should be skipped: just user, user
	if len(apiReq.Input) != 2 {
		t.Fatalf("len(Input) = %d, want 2 (empty assistant skipped)", len(apiReq.Input))
	}
}

// TestBuildRequestMultipleToolResults verifies that multiple tool results
// in a single user message each become a function_call_output item.
func TestBuildRequestMultipleToolResults(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gpt-5"}, providerName: "openai_responses"}

	req := llm.Request{
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "question"),
			// Simulate assistant calling two tools (already tested above)
			// Now user provides both results in one message
			{
				Role: llm.RoleUser,
				Content: []llm.ContentPart{
					{
						Type:              llm.ContentToolResult,
						ToolResultID:      "call_1",
						ToolResultContent: "result 1",
					},
					{
						Type:              llm.ContentToolResult,
						ToolResultID:      "call_2",
						ToolResultContent: "result 2",
					},
				},
			},
		},
	}

	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	// [0] user text, [1] function_call_output call_1, [2] function_call_output call_2
	if len(apiReq.Input) != 3 {
		t.Fatalf("len(Input) = %d, want 3", len(apiReq.Input))
	}

	out1, ok := apiReq.Input[1].(responsesFunctionCallOutput)
	if !ok {
		t.Fatalf("Input[1] is %T, want responsesFunctionCallOutput", apiReq.Input[1])
	}
	if out1.CallID != "call_1" || out1.Output != "result 1" {
		t.Errorf("Input[1] = %+v", out1)
	}

	out2, ok := apiReq.Input[2].(responsesFunctionCallOutput)
	if !ok {
		t.Fatalf("Input[2] is %T, want responsesFunctionCallOutput", apiReq.Input[2])
	}
	if out2.CallID != "call_2" || out2.Output != "result 2" {
		t.Errorf("Input[2] = %+v", out2)
	}
}

// TestBuildRequestToolResultWithText verifies that a user message with both
// tool results and text emits function_call_output items AND a user text message.
func TestBuildRequestToolResultWithText(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gpt-5"}, providerName: "openai_responses"}

	req := llm.Request{
		Messages: []llm.Message{
			{
				Role: llm.RoleUser,
				Content: []llm.ContentPart{
					{
						Type:              llm.ContentToolResult,
						ToolResultID:      "call_1",
						ToolResultContent: "result",
					},
					{Type: llm.ContentText, Text: "Here's the result, now continue."},
				},
			},
		},
	}

	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	// [0] function_call_output, [1] user text
	if len(apiReq.Input) != 2 {
		t.Fatalf("len(Input) = %d, want 2", len(apiReq.Input))
	}

	out, ok := apiReq.Input[0].(responsesFunctionCallOutput)
	if !ok {
		t.Fatalf("Input[0] is %T, want responsesFunctionCallOutput", apiReq.Input[0])
	}
	if out.CallID != "call_1" {
		t.Errorf("Input[0].CallID = %q", out.CallID)
	}

	msg, ok := apiReq.Input[1].(responsesInputMsg)
	if !ok {
		t.Fatalf("Input[1] is %T, want responsesInputMsg", apiReq.Input[1])
	}
	if msg.Role != "user" || msg.Content != "Here's the result, now continue." {
		t.Errorf("Input[1] = %+v", msg)
	}
}

// TestParseResponseTextAndToolCalls verifies a response with both text content
// and function calls (text + tool_use in same response).
func TestParseResponseTextAndToolCalls(t *testing.T) {
	c := &Client{providerName: "openai_responses"}

	apiResp := &responsesResponse{
		Status: "completed",
		Output: []responsesOutputItem{
			{
				Type:    "message",
				Content: []responsesContentBlock{{Type: "output_text", Text: "I'll search for that."}},
			},
			{
				Type:      "function_call",
				CallID:    "call_mixed",
				Name:      "search",
				Arguments: `{"q":"test"}`,
			},
		},
		Usage: responsesUsage{InputTokens: 10, OutputTokens: 5},
	}

	resp := c.parseResponse(apiResp)

	if resp.Content != "I'll search for that." {
		t.Errorf("Content = %q", resp.Content)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls) = %d, want 1", len(resp.ToolCalls))
	}
	// StopReason should be tool_use (function_call overrides end_turn)
	if resp.StopReason != "tool_use" {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, "tool_use")
	}
}

// TestParseResponseEmptyOutput verifies a response with no output items.
func TestParseResponseEmptyOutput(t *testing.T) {
	c := &Client{providerName: "openai_responses"}

	apiResp := &responsesResponse{
		Status: "completed",
		Output: []responsesOutputItem{},
		Usage:  responsesUsage{InputTokens: 5, OutputTokens: 0},
	}

	resp := c.parseResponse(apiResp)
	if resp.Content != "" {
		t.Errorf("Content = %q, want empty", resp.Content)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, "end_turn")
	}
}

// TestParseStreamFailedStatus verifies that failed status maps to error stop reason.
func TestParseStreamFailedStatus(t *testing.T) {
	sseData := "data: " + `{"type":"response.completed","response":{"id":"r","status":"failed","usage":{"input_tokens":1,"output_tokens":0}}}` + "\n\n"

	c := &Client{providerName: "openai_responses"}
	var events []llm.StreamEvent
	c.parseStream(strings.NewReader(sseData), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		events = append(events, ev)
		return true
	})

	if len(events) != 1 {
		t.Fatalf("got %d events, want 1", len(events))
	}
	if events[0].StopReason != "error" {
		t.Errorf("StopReason = %q, want %q", events[0].StopReason, "error")
	}
}

// TestParseStreamErrorNoCode verifies error events with no code.
func TestParseStreamErrorNoCode(t *testing.T) {
	sseData := "data: " + `{"type":"error","error":{"message":"internal server error"}}` + "\n\n"

	c := &Client{providerName: "openai_responses"}
	var gotErr error
	c.parseStream(strings.NewReader(sseData), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			gotErr = err
		}
		return err == nil
	})

	if gotErr == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(gotErr.Error(), "internal server error") {
		t.Errorf("error = %q", gotErr.Error())
	}
	// Should NOT contain ":" prefix from empty code
	if strings.Contains(gotErr.Error(), ": internal") && strings.Contains(gotErr.Error(), "openai_responses: : internal") {
		t.Errorf("error has double colon from empty code: %q", gotErr.Error())
	}
}

// TestParseStreamEmptyBody verifies that an empty stream body doesn't panic.
func TestParseStreamEmptyBody(t *testing.T) {
	c := &Client{providerName: "openai_responses"}
	var events []llm.StreamEvent
	c.parseStream(strings.NewReader(""), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		events = append(events, ev)
		return true
	})

	// Should produce no events and not panic
	if len(events) != 0 {
		t.Errorf("got %d events from empty stream", len(events))
	}
}

// =============================================================================
// ThinkingBudgetTokens TESTS (types.go)
// =============================================================================

// TestThinkingBudgetTokensAllLevels verifies that all ThinkingLevel values
// return non-zero budgets (except ThinkingNone).
func TestThinkingBudgetTokensAllLevels(t *testing.T) {
	tests := []struct {
		level llm.ThinkingLevel
		want  int
	}{
		{llm.ThinkingNone, 0},
		{llm.ThinkingMinimal, 1024},
		{llm.ThinkingLow, 4096},
		{llm.ThinkingMedium, 16384},
		{llm.ThinkingHigh, 65536},
		{llm.ThinkingXHigh, 128000},
	}

	for _, tc := range tests {
		t.Run(string(tc.level), func(t *testing.T) {
			got := llm.ThinkingBudgetTokens(tc.level, 0)
			if got != tc.want {
				t.Errorf("ThinkingBudgetTokens(%q, 0) = %d, want %d", tc.level, got, tc.want)
			}
		})
	}
}

// TestThinkingBudgetTokensCustomOverride verifies that custom budget overrides
// the level default.
func TestThinkingBudgetTokensCustomOverride(t *testing.T) {
	got := llm.ThinkingBudgetTokens(llm.ThinkingLow, 8192)
	if got != 8192 {
		t.Errorf("ThinkingBudgetTokens(low, 8192) = %d, want 8192", got)
	}
}
