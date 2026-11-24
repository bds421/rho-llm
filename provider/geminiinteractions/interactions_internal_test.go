package geminiinteractions

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// Fixtures are shortened copies of live responses (gemini-3.8-flash,
// 2026-11-18).
const (
	textBody = `{"status":"completed","usage":{"total_tokens":1093,"total_input_tokens":14,"total_cached_tokens":0,"total_output_tokens":270,"total_thought_tokens":809},"steps":[{"signature":"SIG1","summary":[{"text":"Thinking A. ","type":"text"},{"text":"Thinking B.","type":"text"}],"type":"thought"},{"content":[{"text":"391","type":"text"}],"type":"model_output"}],"object":"interaction","model":"gemini-3.8-flash"}`
	toolBody = `{"status":"requires_action","usage":{"total_input_tokens":46,"total_output_tokens":16},"steps":[{"signature":"SIGT","type":"thought"},{"id":"call_111","type":"function_call","name":"get_weather","arguments":{"city":"Paris"}}],"object":"interaction","model":"gemini-3.8-flash"}`
)

type captured struct {
	mu     sync.Mutex
	paths  []string
	keys   []string
	bodies []map[string]any
}

func (c *captured) last() map[string]any {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.bodies[len(c.bodies)-1]
}

// serve answers every request with status and body and records what arrived.
func serve(t *testing.T, status int, body string, contentType string) (*Client, *captured) {
	t.Helper()
	cap := &captured{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var m map[string]any
		_ = json.Unmarshal(raw, &m)
		cap.mu.Lock()
		cap.paths = append(cap.paths, r.URL.Path)
		cap.keys = append(cap.keys, r.Header.Get("x-goog-api-key"))
		cap.bodies = append(cap.bodies, m)
		cap.mu.Unlock()
		if contentType != "" {
			w.Header().Set("Content-Type", contentType)
		}
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	c, err := New(llm.Config{Provider: "gemini-interactions", Model: "gemini-3.8-flash", APIKey: "sekrit-key", BaseURL: srv.URL + "/v1beta", DisableRetries: true})
	if err != nil {
		t.Fatal(err)
	}
	return c, cap
}

func user(text string) llm.Request {
	return llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, text)}}
}

func TestCompleteMapsWireAndResponse(t *testing.T) {
	c, cap := serve(t, 200, textBody, "application/json")
	req := user("17*23?")
	req.System = "Be brief."
	req.ThinkingLevel = llm.ThinkingHigh
	req.MaxTokens = 100
	resp, err := c.Complete(t.Context(), req)
	if err != nil {
		t.Fatal(err)
	}
	if cap.paths[0] != "/v1beta/interactions" || cap.keys[0] != "sekrit-key" {
		t.Errorf("path %q key %q", cap.paths[0], cap.keys[0])
	}
	b := cap.last()
	if b["store"] != false {
		t.Errorf("store = %v, want false (stateless default)", b["store"])
	}
	if b["system_instruction"] != "Be brief." || b["model"] != "gemini-3.8-flash" {
		t.Errorf("body %v", b)
	}
	gc := b["generation_config"].(map[string]any)
	if gc["thinking_level"] != "high" || gc["thinking_summaries"] != "auto" {
		t.Errorf("generation_config %v", gc)
	}
	// 100 + 1024-class thinking pad, below the model cap.
	if mo := gc["max_output_tokens"].(float64); mo <= 100 {
		t.Errorf("max_output_tokens %v not padded for thinking", mo)
	}
	in := b["input"].([]any)
	if len(in) != 1 || in[0].(map[string]any)["type"] != "user_input" {
		t.Errorf("input %v", in)
	}
	if resp.Content != "391" || resp.Thinking != "Thinking A. Thinking B." || resp.ThinkingSignature != "SIG1" {
		t.Errorf("resp %+v", resp)
	}
	if resp.InputTokens != 14 || resp.OutputTokens != 270 || resp.ThinkingTokens != 809 {
		t.Errorf("usage %d/%d/%d", resp.InputTokens, resp.OutputTokens, resp.ThinkingTokens)
	}
	if resp.StopReason != llm.StopEndTurn || resp.RawStopReason != "completed" {
		t.Errorf("stop %q raw %q", resp.StopReason, resp.RawStopReason)
	}
}

func TestCompleteToolCall(t *testing.T) {
	c, _ := serve(t, 200, toolBody, "application/json")
	resp, err := c.Complete(t.Context(), user("weather"))
	if err != nil {
		t.Fatal(err)
	}
	if resp.StopReason != llm.StopToolUse || len(resp.ToolCalls) != 1 {
		t.Fatalf("%+v", resp)
	}
	tc := resp.ToolCalls[0]
	if tc.ID != "call_111" || tc.Name != "get_weather" || tc.Input.(map[string]any)["city"] != "Paris" {
		t.Errorf("call %+v", tc)
	}
	if resp.ThinkingSignature != "SIGT" {
		t.Errorf("signature %q", resp.ThinkingSignature)
	}
}

// The replayed transcript: the signed thought goes back as a thought step in
// front of the call, results are function_result steps carrying the call's
// name, and the text turn after them is a user_input.
func TestReplayTranscriptSteps(t *testing.T) {
	c, cap := serve(t, 200, textBody, "application/json")
	req := llm.Request{Messages: []llm.Message{
		llm.NewTextMessage(llm.RoleUser, "weather in Paris and Rome"),
		{Role: llm.RoleAssistant, Content: []llm.ContentPart{
			{Type: llm.ContentThinking, ThinkingSignature: "SIGT"},
			{Type: llm.ContentText, Text: "checking"},
			{Type: llm.ContentToolUse, ToolUseID: "call_1", ToolName: "get_weather", ToolInput: map[string]any{"city": "Paris"}},
			{Type: llm.ContentToolUse, ToolUseID: "call_2", ToolName: "get_weather", ToolInput: map[string]any{"city": "Rome"}},
		}},
		{Role: llm.RoleUser, Content: []llm.ContentPart{
			{Type: llm.ContentToolResult, ToolResultID: "call_1", ToolResultContent: "sunny"},
			{Type: llm.ContentToolResult, ToolResultID: "call_2", ToolResultContent: "boom", IsError: true},
			{Type: llm.ContentText, Text: "thanks"},
		}},
	}}
	if _, err := c.Complete(t.Context(), req); err != nil {
		t.Fatal(err)
	}
	got, _ := json.Marshal(cap.last()["input"])
	want := `[{"type":"user_input","content":[{"type":"text","text":"weather in Paris and Rome"}]},` +
		`{"type":"thought","signature":"SIGT"},` +
		`{"type":"model_output","content":[{"type":"text","text":"checking"}]},` +
		`{"type":"function_call","id":"call_1","name":"get_weather","arguments":{"city":"Paris"}},` +
		`{"type":"function_call","id":"call_2","name":"get_weather","arguments":{"city":"Rome"}},` +
		`{"type":"function_result","call_id":"call_1","name":"get_weather","result":"sunny"},` +
		`{"type":"function_result","call_id":"call_2","name":"get_weather","result":"boom","is_error":true},` +
		`{"type":"user_input","content":[{"type":"text","text":"thanks"}]}]`
	var g, w any
	_ = json.Unmarshal(got, &g)
	_ = json.Unmarshal([]byte(want), &w)
	if !reflect.DeepEqual(g, w) {
		t.Errorf("input\n got %s\nwant %s", got, want)
	}
}

// A call made by another provider has no signature; Gemini 3 answers 400
// without a thought step, so exactly one placeholder goes in front of the
// first call (live: accepted).
func TestUnsignedToolCallGetsPlaceholderThought(t *testing.T) {
	c, cap := serve(t, 200, textBody, "application/json")
	req := llm.Request{Messages: []llm.Message{
		llm.NewTextMessage(llm.RoleUser, "x"),
		{Role: llm.RoleAssistant, Content: []llm.ContentPart{
			{Type: llm.ContentToolUse, ToolUseID: "a", ToolName: "f"},
			{Type: llm.ContentToolUse, ToolUseID: "b", ToolName: "f"},
		}},
		{Role: llm.RoleUser, Content: []llm.ContentPart{
			{Type: llm.ContentToolResult, ToolResultID: "a", ToolResultContent: "1"},
			{Type: llm.ContentToolResult, ToolResultID: "b", ToolResultContent: "2"},
		}},
	}}
	if _, err := c.Complete(t.Context(), req); err != nil {
		t.Fatal(err)
	}
	in := cap.last()["input"].([]any)
	var types []string
	for _, s := range in {
		types = append(types, s.(map[string]any)["type"].(string))
	}
	if strings.Join(types, ",") != "user_input,thought,function_call,function_call,function_result,function_result" {
		t.Errorf("steps %v", types)
	}
	if in[1].(map[string]any)["signature"] != skipThoughtSignature {
		t.Errorf("placeholder %v", in[1])
	}
	// no-argument call still sends an object
	if _, ok := in[2].(map[string]any)["arguments"].(map[string]any); !ok {
		t.Errorf("arguments %v", in[2])
	}
}

// Hostile transcripts fail loudly before any request.
func TestBuildRequestRejects(t *testing.T) {
	c, cap := serve(t, 200, textBody, "application/json")
	tr := func(m ...llm.Message) llm.Request { return llm.Request{Messages: m} }
	cases := map[string]llm.Request{
		"orphan result": tr(llm.Message{Role: llm.RoleUser, Content: []llm.ContentPart{{Type: llm.ContentToolResult, ToolResultID: "nope", ToolResultContent: "x"}}}),
		"no messages":   {},
		"empty message": tr(llm.Message{Role: llm.RoleUser}),
		"bad role":      tr(llm.Message{Role: "tool", Content: []llm.ContentPart{{Type: llm.ContentText, Text: "x"}}}),
		"budget":        {Messages: user("x").Messages, ThinkingBudget: 1000},
		"effort":        {Messages: user("x").Messages, Effort: "high"},
		"cache":         {Messages: user("x").Messages, CachedContent: "cachedContents/1"},
		"format":        {Messages: user("x").Messages, ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSONObject}},
		"bad level":     {Messages: user("x").Messages, ThinkingLevel: "ludicrous"},
		"bad store":     {Messages: user("x").Messages, SamplingParams: map[string]any{"store": "yes"}},
		"reserved key":  {Messages: user("x").Messages, SamplingParams: map[string]any{"input": []any{}}},
		"bad image":     tr(llm.Message{Role: llm.RoleUser, Content: []llm.ContentPart{{Type: llm.ContentImage}}}),
	}
	for name, req := range cases {
		if _, err := c.Complete(t.Context(), req); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
	if len(cap.paths) != 0 {
		t.Errorf("%d requests reached the server, want 0", len(cap.paths))
	}
}

func TestStoreOptIn(t *testing.T) {
	c, cap := serve(t, 200, textBody, "application/json")
	req := user("x")
	req.SamplingParams = map[string]any{"store": true, "service_tier": "flex"}
	if _, err := c.Complete(t.Context(), req); err != nil {
		t.Fatal(err)
	}
	b := cap.last()
	if b["store"] != true || b["service_tier"] != "flex" {
		t.Errorf("body %v", b)
	}
	// The caller's map is not mutated.
	if _, ok := req.SamplingParams["store"]; !ok {
		t.Error("SamplingParams mutated")
	}
}

func TestThinkingLevelsAndToolChoice(t *testing.T) {
	cases := []struct {
		model string
		level llm.ThinkingLevel
		want  string
	}{
		{"gemini-3.8-flash", llm.ThinkingMinimal, "low"},
		{"gemini-4-carbon", llm.ThinkingMinimal, "low"},
		{"gemini-3.5-flash", llm.ThinkingMinimal, "minimal"},
		{"gemini-3.8-flash", llm.ThinkingXHigh, "high"},
		{"gemini-4-hydrogen", llm.ThinkingXHigh, "xhigh"},
		{"gemini-4-neon", llm.ThinkingXHigh, "xhigh"},
		{"gemini-4-hydrogen", llm.ThinkingMax, "xhigh"},
		{"gemini-3.8-flash", llm.ThinkingMedium, "medium"},
	}
	for _, tc := range cases {
		c := &Client{providerName: "gemini-interactions", config: llm.Config{Model: tc.model}}
		r, err := c.buildRequest(llm.Request{Messages: user("x").Messages, ThinkingLevel: tc.level})
		if err != nil || r.GenerationConfig.ThinkingLevel != tc.want {
			t.Errorf("%s/%s = %q, %v; want %q", tc.model, tc.level, r.GenerationConfig.ThinkingLevel, err, tc.want)
		}
	}
	c := &Client{providerName: "gemini-interactions", config: llm.Config{Model: "gemini-3.8-flash"}}
	tools := []llm.Tool{{Name: "f", Description: "d"}}
	wire := func(tc *llm.ToolChoice) string {
		r, err := c.buildRequest(llm.Request{Messages: user("x").Messages, Tools: tools, ToolChoice: tc})
		if err != nil {
			t.Fatal(err)
		}
		b, _ := json.Marshal(r.GenerationConfig.ToolChoice)
		return string(b)
	}
	if got := wire(nil); got != "null" {
		t.Errorf("default %s", got)
	}
	if got := wire(llm.NewToolChoice(llm.ToolChoiceRequired)); got != `"any"` {
		t.Errorf("required %s", got)
	}
	if got := wire(llm.NewToolChoice(llm.ToolChoiceNone)); got != `"none"` {
		t.Errorf("none %s", got)
	}
	if got := wire(llm.ForceTool("f")); got != `{"allowed_tools":{"mode":"any","tools":["f"]}}` {
		t.Errorf("forced %s", got)
	}
	// Tool with no schema still sends an object.
	r, _ := c.buildRequest(llm.Request{Messages: user("x").Messages, Tools: tools})
	if b, _ := json.Marshal(r.Tools); !strings.Contains(string(b), `"parameters":{"type":"object"}`) {
		t.Errorf("tools %s", b)
	}
	// Config-level thinking applies when the request sets none.
	c.config.ThinkingLevel = llm.ThinkingLow
	r, _ = c.buildRequest(user("x"))
	if r.GenerationConfig.ThinkingLevel != "low" {
		t.Error("config thinking level not applied")
	}
}

func TestStatusAndUsageMapping(t *testing.T) {
	c := &Client{providerName: "p", config: llm.Config{Model: "m"}}
	stops := map[string]string{
		"completed": llm.StopEndTurn, "incomplete": llm.StopMaxTokens, "failed": llm.StopError,
		"cancelled": llm.StopError, "in_progress": llm.StopError, "weird": llm.StopError, "": "",
		"requires_action": llm.StopError, // no call attached
	}
	for status, want := range stops {
		r := c.parseInteraction(&interaction{Status: status}, "m")
		if r.StopReason != want || r.RawStopReason != status {
			t.Errorf("%q -> %q (raw %q), want %q", status, r.StopReason, r.RawStopReason, want)
		}
	}
	// No usage object: not reported, never a fake zero.
	r := c.parseInteraction(&interaction{Status: "completed"}, "m")
	if r.InputTokens != llm.TokensNotReported || r.OutputTokens != llm.TokensNotReported {
		t.Errorf("tokens %d/%d", r.InputTokens, r.OutputTokens)
	}
	// Negative and inconsistent counts are clamped.
	r = c.parseInteraction(&interaction{Usage: &usage{TotalInputTokens: 10, TotalCachedTokens: 50, TotalOutputTokens: -1, TotalThoughtTokens: -9}}, "m")
	if r.InputTokens != 0 || r.CacheReadTokens != 10 || r.OutputTokens != 0 || r.ThinkingTokens != 0 {
		t.Errorf("clamp %+v", r)
	}
	r = c.parseInteraction(&interaction{Usage: &usage{TotalInputTokens: -5, TotalCachedTokens: -5}}, "m")
	if r.InputTokens != 0 || r.CacheReadTokens != 0 {
		t.Errorf("negative input %+v", r)
	}
	// A 200 carrying completed + a function call is a tool turn.
	r = c.parseInteraction(&interaction{Status: "completed", Steps: []outStep{{Type: "function_call", Name: "f"}}}, "m")
	if r.StopReason != llm.StopToolUse || r.ToolCalls[0].ID == "" {
		t.Errorf("%+v", r)
	}
}

// Unknown step kinds, empty steps, null arguments and non-text output blocks
// must not break or leak into the answer.
func TestParseHostileInteraction(t *testing.T) {
	body := `{"status":"completed","steps":[null,{},{"type":"future_kind","x":1},{"type":"model_output","content":[{"type":"image","data":"AAAA"},{"type":"text","text":"ok"},null]},{"type":"thought"},{"type":"function_call","name":"f","arguments":null},{"type":"function_call","id":"z","name":"g","arguments":"{\"oops"}]}`
	c, _ := serve(t, 200, body, "application/json")
	resp, err := c.Complete(t.Context(), user("x"))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "ok" || len(resp.ToolCalls) != 2 {
		t.Fatalf("%+v", resp)
	}
	if m, ok := resp.ToolCalls[0].Input.(map[string]any); !ok || len(m) != 0 {
		t.Errorf("null arguments -> %#v", resp.ToolCalls[0].Input)
	}
}

func TestCompleteErrors(t *testing.T) {
	cases := []struct {
		status int
		body   string
		check  func(*llm.APIError) bool
	}{
		{400, `{"error":{"message":"Provide a 'agent', or 'model' parameter.","code":"invalid_request"}}`, func(e *llm.APIError) bool { return e.StatusCode == 400 }},
		{404, `{"error":{"message":"Model 'nope' not found.","code":"not_found"}}`, func(e *llm.APIError) bool { return e.StatusCode == 404 }},
		{429, `{"error":{"message":"quota","code":"resource_exhausted"}}`, func(e *llm.APIError) bool { return llm.IsRateLimited(e) }},
		{500, `not json at all`, func(e *llm.APIError) bool { return e.StatusCode == 500 }},
		{401, `{"error":{"message":"bad key sekrit-key echoed"}}`, func(e *llm.APIError) bool { return !strings.Contains(e.Error(), "sekrit-key") }},
	}
	for _, tc := range cases {
		c, _ := serve(t, tc.status, tc.body, "application/json")
		_, err := c.Complete(t.Context(), user("x"))
		var apiErr *llm.APIError
		if !errors.As(err, &apiErr) || !tc.check(apiErr) {
			t.Errorf("%d: %v", tc.status, err)
		}
	}
	// A 200 whose body is an error object, and garbage JSON.
	c, _ := serve(t, 200, `{"error":{"message":"overloaded","code":"unavailable"}}`, "application/json")
	var apiErr *llm.APIError
	if _, err := c.Complete(t.Context(), user("x")); !errors.As(err, &apiErr) || apiErr.StatusCode != 503 {
		t.Errorf("in-band error: %v", err)
	}
	c, _ = serve(t, 200, `{"status":`, "application/json")
	if _, err := c.Complete(t.Context(), user("x")); err == nil {
		t.Error("truncated JSON accepted")
	}
	c, _ = serve(t, 200, ``, "application/json")
	if _, err := c.Complete(t.Context(), user("x")); err == nil {
		t.Error("empty body accepted")
	}
}
