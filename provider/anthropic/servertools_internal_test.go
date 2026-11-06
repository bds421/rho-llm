package anthropic

import (
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

func newTestClient() *Client {
	return &Client{config: llm.Config{Model: "claude-opus-5-5", MaxTokens: 1024}, providerName: "anthropic"}
}

func loadFixture(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func parseFixture(t *testing.T, c *Client, name string) (*llm.Response, []any) {
	t.Helper()
	data := loadFixture(t, name)
	var apiResp anthropicResponse
	if err := json.Unmarshal(data, &apiResp); err != nil {
		t.Fatal(err)
	}
	var generic struct {
		Content []any `json:"content"`
	}
	if err := json.Unmarshal(data, &generic); err != nil {
		t.Fatal(err)
	}
	return c.parseResponse(&apiResp), generic.Content
}

// asJSON re-decodes v through JSON so wire values compare by meaning.
func asJSON(t *testing.T, v any) any {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var out any
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func toolsJSON(t *testing.T, c *Client, req llm.Request) string {
	t.Helper()
	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}
	data, err := json.Marshal(apiReq.Tools)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func TestWebSearchToolWire(t *testing.T) {
	c := newTestClient()
	req := llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "q")},
		Tools:    []llm.Tool{{Name: "lookup", Description: "d", InputSchema: map[string]any{"type": "object"}}},
		ServerTools: []llm.ServerTool{{
			Type:           llm.WebSearch20260209,
			Name:           llm.ServerToolWebSearch,
			MaxUses:        3,
			AllowedDomains: []string{"example.com", "example.org/docs"},
			UserLocation:   &llm.UserLocation{City: "Vienna", Country: "AT", Timezone: "Europe/Vienna"},
			AllowedCallers: []string{llm.CallerDirect},
			CacheControl:   true,
			Params:         map[string]any{"response_inclusion": "full"},
		}},
	}
	got := toolsJSON(t, c, req)
	want := `[{"name":"lookup","description":"d","input_schema":{"type":"object"}},` +
		`{"allowed_callers":["direct"],"allowed_domains":["example.com","example.org/docs"],"cache_control":{"type":"ephemeral"},` +
		`"max_uses":3,"name":"web_search","response_inclusion":"full","type":"web_search_20260209",` +
		`"user_location":{"city":"Vienna","country":"AT","timezone":"Europe/Vienna","type":"approximate"}}]`
	if got != want {
		t.Fatalf("tools =\n%s\nwant\n%s", got, want)
	}
}

func TestServerToolsOnlyStillSendsToolChoice(t *testing.T) {
	c := newTestClient()
	req := llm.Request{
		Messages:    []llm.Message{llm.NewTextMessage(llm.RoleUser, "q")},
		ServerTools: []llm.ServerTool{llm.NewWebSearchTool(llm.WebSearch20250305, 0)},
		ToolChoice:  &llm.ToolChoice{Mode: llm.ToolChoiceNone},
	}
	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatal(err)
	}
	if apiReq.ToolChoice == nil {
		t.Fatal("tool_choice dropped although a server tool is defined")
	}
	if got := toolsJSON(t, c, req); got != `[{"name":"web_search","type":"web_search_20250305"}]` {
		t.Fatalf("tools = %s", got)
	}
}

func TestServerToolValidationBreaks(t *testing.T) {
	cases := map[string]llm.ServerTool{
		"no type":         {Name: "web_search"},
		"no name":         {Type: llm.WebSearch20250305},
		"negative uses":   {Type: llm.WebSearch20250305, Name: "web_search", MaxUses: -1},
		"both lists":      {Type: llm.WebSearch20250305, Name: "web_search", AllowedDomains: []string{"a.example"}, BlockedDomains: []string{"b.example"}},
		"empty domain":    {Type: llm.WebSearch20250305, Name: "web_search", BlockedDomains: []string{""}},
		"empty location":  {Type: llm.WebSearch20250305, Name: "web_search", UserLocation: &llm.UserLocation{Type: "approximate"}},
		"params max_uses": {Type: llm.WebSearch20250305, Name: "web_search", Params: map[string]any{"max_uses": 99}},
		"params type":     {Type: llm.WebSearch20250305, Name: "web_search", Params: map[string]any{"type": "web_fetch_20250910"}},
	}
	for name, tool := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := newTestClient().buildRequest(llm.Request{ServerTools: []llm.ServerTool{tool}}, false)
			if err == nil {
				t.Fatal("want an error")
			}
		})
	}
}

// TestServerToolsEncodeIdentically guards prompt caching: the same tool
// definitions must encode to the same bytes on every request, map iteration
// order notwithstanding.
func TestServerToolsEncodeIdentically(t *testing.T) {
	c := newTestClient()
	tool := llm.ServerTool{
		Type: llm.WebSearch20260209, Name: llm.ServerToolWebSearch, MaxUses: 5,
		UserLocation: &llm.UserLocation{City: "c", Region: "r", Country: "US", Timezone: "America/New_York"},
		Params:       map[string]any{"a": 1, "b": []any{"x", "y"}, "c": map[string]any{"z": true, "y": false}, "d": "e"},
	}
	req := llm.Request{ServerTools: []llm.ServerTool{tool}}
	first := toolsJSON(t, c, req)
	for i := 0; i < 200; i++ {
		if got := toolsJSON(t, c, req); got != first {
			t.Fatalf("encoding %d differs:\n%s\n%s", i, got, first)
		}
	}
}

func TestParseWebSearchResponse(t *testing.T) {
	resp, _ := parseFixture(t, newTestClient(), "web_search_paused.json")
	if resp.StopReason != llm.StopPauseTurn {
		t.Fatalf("StopReason = %q", resp.StopReason)
	}
	if resp.ContainerID != "container_01" {
		t.Errorf("ContainerID = %q", resp.ContainerID)
	}
	if resp.ServerToolUse.WebSearchRequests != 1 {
		t.Errorf("WebSearchRequests = %d", resp.ServerToolUse.WebSearchRequests)
	}
	if want := "I'll search for that.Based on the search results, the widget shipped on April 30, 2026"; resp.Content != want {
		t.Errorf("Content = %q", resp.Content)
	}
	types := make([]llm.ContentType, len(resp.Parts))
	for i, p := range resp.Parts {
		types[i] = p.Type
	}
	wantTypes := []llm.ContentType{llm.ContentText, llm.ContentServerToolUse, llm.ContentServerToolResult, llm.ContentText, llm.ContentText, llm.ContentServerToolUse}
	if !reflect.DeepEqual(types, wantTypes) {
		t.Fatalf("part types = %v", types)
	}
	use := resp.Parts[1]
	if use.ToolUseID != "srvtoolu_01" || use.ToolName != "web_search" || use.ToolInput.(map[string]any)["query"] != "example widget release date" {
		t.Errorf("server_tool_use = %+v", use)
	}
	result := resp.Parts[2]
	if result.BlockType != "web_search_tool_result" || result.ToolResultID != "srvtoolu_01" || len(result.WebSearchResults) != 2 {
		t.Fatalf("result = %+v", result)
	}
	if r := result.WebSearchResults[0]; r.URL != "https://example.com/widgets" || r.EncryptedContent != "EqgfCioIARgBIiQ3YTAw" || r.PageAge != "April 30, 2026" {
		t.Errorf("result[0] = %+v", r)
	}
	cited := resp.Parts[4]
	if len(cited.Citations) != 1 {
		t.Fatalf("citations = %+v", cited.Citations)
	}
	if c := cited.Citations[0]; c.Type != llm.CitationWebSearchResultLocation || c.EncryptedIndex != "Eo8BCioIAhgBIiQy" || c.URL != "https://example.com/widgets" || len(c.Raw) == 0 {
		t.Errorf("citation = %+v", c)
	}
}

func TestParseServerToolErrorAndIterations(t *testing.T) {
	resp, _ := parseFixture(t, newTestClient(), "web_search_final.json")
	result := resp.Parts[0]
	if result.ServerToolError == nil || result.ServerToolError.ErrorCode != "max_uses_exceeded" || len(result.WebSearchResults) != 0 {
		t.Fatalf("error result = %+v", result)
	}
	if len(resp.Iterations) != 2 || resp.Iterations[1].Type != "fallback_message" || resp.Iterations[1].Model != "claude-opus-4-8" || resp.Iterations[0].InputTokens != 535 {
		t.Fatalf("iterations = %+v", resp.Iterations)
	}
}

// TestPausedTurnReplaysUnchanged is the core round trip: the assistant
// content of a paused response, sent back through NewAssistantMessage — and
// through a persisted Conversation — must reach the API exactly as received.
func TestPausedTurnReplaysUnchanged(t *testing.T) {
	c := newTestClient()
	resp, original := parseFixture(t, c, "web_search_paused.json")

	conv := llm.NewConversation("")
	conv.Append(llm.NewTextMessage(llm.RoleUser, "When did the example widget ship?"))
	conv.AddResponse("anthropic", resp)
	data, err := json.Marshal(conv)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := llm.LoadConversation(data)
	if err != nil {
		t.Fatal(err)
	}

	for name, msgs := range map[string][]llm.Message{"live": conv.Messages, "restored": restored.Messages} {
		t.Run(name, func(t *testing.T) {
			apiReq, err := c.buildRequest(llm.Request{Messages: llm.NormalizeForModel(msgs, "anthropic", "claude-opus-5-5")}, false)
			if err != nil {
				t.Fatal(err)
			}
			if len(apiReq.Messages) != 2 {
				t.Fatalf("messages = %d", len(apiReq.Messages))
			}
			got := asJSON(t, apiReq.Messages[1].Content)
			if want := asJSON(t, original); !reflect.DeepEqual(got, want) {
				t.Fatalf("replayed content differs\n got %v\nwant %v", got, want)
			}
		})
	}
}

func TestStreamAssemblesSameParts(t *testing.T) {
	c := newTestClient()
	var events []llm.StreamEvent
	c.parseStream(strings.NewReader(string(loadFixture(t, "web_search_stream.sse"))), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		events = append(events, ev)
		return true
	})
	done := events[len(events)-1]
	if done.Type != llm.EventDone || done.StopReason != llm.StopPauseTurn || done.ContainerID != "container_01" || done.ServerToolUse.WebSearchRequests != 1 {
		t.Fatalf("done = %+v", done)
	}
	var serverEvents, citationEvents int
	for _, ev := range events {
		switch ev.Type {
		case llm.EventServerTool:
			serverEvents++
		case llm.EventCitation:
			citationEvents++
		}
	}
	if serverEvents != 3 || citationEvents != 1 {
		t.Errorf("server tool events = %d, citation events = %d", serverEvents, citationEvents)
	}

	_, original := parseFixture(t, c, "web_search_paused.json")
	msg := llm.NewAssistantMessage(&llm.Response{Parts: done.Parts, StopReason: done.StopReason})
	msg.Provider = "anthropic"
	apiReq, err := c.buildRequest(llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "q"), msg}}, false)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := asJSON(t, apiReq.Messages[1].Content), asJSON(t, original); !reflect.DeepEqual(got, want) {
		t.Fatalf("streamed replay differs\n got %v\nwant %v", got, want)
	}
}

func TestUnknownBlocksAndFieldsSurvive(t *testing.T) {
	c := newTestClient()
	raw := `{"id":"m","model":"claude-opus-5-5","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1},"content":[` +
		`{"type":"fallback","from":{"model":"claude-opus-5-5"},"to":{"model":"claude-opus-4-8"}},` +
		`{"type":"server_tool_use","id":"srvtoolu_9","name":"web_fetch","input":{"url":"https://example.com/a"},"x_future":{"k":[1,2]}},` +
		`{"type":"web_fetch_tool_result","tool_use_id":"srvtoolu_9","content":{"type":"web_fetch_result","url":"https://example.com/a","content":{"type":"document","source":{"type":"text","media_type":"text/plain","data":"<b>&amp;</b>"}}}},` +
		`{"type":"text","text":"ok","citations":[{"type":"x_future_location","cited_text":"t","x":1},{"type":"char_location","document_index":"zero"}]}]}`
	var apiResp anthropicResponse
	if err := json.Unmarshal([]byte(raw), &apiResp); err != nil {
		t.Fatal(err)
	}
	resp := c.parseResponse(&apiResp)
	if resp.Parts[0].Type != llm.ContentProviderBlock || resp.Parts[0].BlockType != "fallback" {
		t.Fatalf("part 0 = %+v", resp.Parts[0])
	}
	if resp.Parts[2].Type != llm.ContentServerToolResult || resp.Parts[2].ServerToolError != nil {
		t.Fatalf("part 2 = %+v", resp.Parts[2])
	}
	msg := llm.NewAssistantMessage(resp)
	apiReq, err := c.buildRequest(llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "q"), msg}}, false)
	if err != nil {
		t.Fatal(err)
	}
	var original struct {
		Content []any `json:"content"`
	}
	_ = json.Unmarshal([]byte(raw), &original)
	if got, want := asJSON(t, apiReq.Messages[1].Content), asJSON(t, original.Content); !reflect.DeepEqual(got, want) {
		t.Fatalf("replay differs\n got %v\nwant %v", got, want)
	}
}

func TestHandBuiltServerPartsEncode(t *testing.T) {
	c := newTestClient()
	msg := llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentPart{
		{Type: llm.ContentServerToolUse, ToolUseID: "srvtoolu_1", ToolName: "web_search"},
		{Type: llm.ContentServerToolResult, BlockType: "web_search_tool_result", ToolResultID: "srvtoolu_1",
			WebSearchResults: []llm.WebSearchResult{{URL: "https://example.com", Title: "t", EncryptedContent: "e"}}},
		{Type: llm.ContentText, Text: "x", Citations: []llm.Citation{{Type: llm.CitationCharLocation, CitedText: "c", DocumentIndex: 0, StartCharIndex: 0, EndCharIndex: 4}}},
		{Type: llm.ContentProviderBlock, BlockType: "nothing-to-send"},
	}}
	apiReq, err := c.buildRequest(llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "q"), msg}}, false)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := json.Marshal(apiReq.Messages[1].Content)
	want := `[{"id":"srvtoolu_1","input":{},"name":"web_search","type":"server_tool_use"},` +
		`{"content":[{"encrypted_content":"e","title":"t","type":"web_search_result","url":"https://example.com"}],"tool_use_id":"srvtoolu_1","type":"web_search_tool_result"},` +
		`{"citations":[{"cited_text":"c","document_index":0,"document_title":"","end_char_index":4,"start_char_index":0,"type":"char_location"}],"text":"x","type":"text"}]`
	if string(got) != want {
		t.Fatalf("content =\n%s\nwant\n%s", got, want)
	}
}

func TestCorruptProviderBlocksFailCleanly(t *testing.T) {
	cases := map[string]llm.ContentPart{
		"invalid json":     {Type: llm.ContentServerToolUse, ProviderBlock: json.RawMessage(`{"type":`)},
		"array not object": {Type: llm.ContentServerToolResult, ProviderBlock: json.RawMessage(`[1,2]`)},
		"whitespace":       {Type: llm.ContentProviderBlock, ProviderBlock: json.RawMessage("  \n")},
		"bare string":      {Type: llm.ContentProviderBlock, ProviderBlock: json.RawMessage(`"x"`)},
		"result no type":   {Type: llm.ContentServerToolResult, ToolResultID: "x"},
		"bad citation":     {Type: llm.ContentText, Text: "x", Citations: []llm.Citation{{Type: "t", Raw: json.RawMessage(`{`)}}},
	}
	for name, part := range cases {
		t.Run(name, func(t *testing.T) {
			msg := llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentPart{part}}
			_, err := newTestClient().buildRequest(llm.Request{Messages: []llm.Message{msg}}, false)
			if err == nil {
				t.Fatal("want an error, not a request the API rejects")
			}
		})
	}
}

func runStream(t *testing.T, sse string) ([]llm.StreamEvent, []error) {
	t.Helper()
	var events []llm.StreamEvent
	var errs []error
	c := newTestClient()
	c.parseStream(strings.NewReader(sse), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			errs = append(errs, err)
			return true
		}
		events = append(events, ev)
		return true
	})
	return events, errs
}

func sse(lines ...string) string {
	var b strings.Builder
	for _, l := range lines {
		b.WriteString("data: " + l + "\n\n")
	}
	return b.String()
}

func TestStreamServerToolBreaks(t *testing.T) {
	t.Run("server input over limit", func(t *testing.T) {
		chunk := `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"` + strings.Repeat("a", 1<<16) + `"}}`
		lines := []string{`{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"s","name":"web_search","input":{}}}`}
		for n := 0; n <= llm.DefaultMaxToolInputBytes>>16; n++ {
			lines = append(lines, chunk)
		}
		_, errs := runStream(t, sse(lines...))
		if len(errs) == 0 || !strings.Contains(errs[0].Error(), "exceeded") {
			t.Fatalf("errs = %v", errs)
		}
	})
	t.Run("missing stop before message_delta", func(t *testing.T) {
		events, errs := runStream(t, sse(
			`{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"s","name":"web_search","input":{}}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"q\"}"}}`,
			`{"type":"message_delta","delta":{"stop_reason":"pause_turn"},"usage":{"output_tokens":3}}`,
		))
		if len(errs) != 0 {
			t.Fatal(errs)
		}
		done := events[len(events)-1]
		if len(done.Parts) != 1 || done.Parts[0].ToolInput.(map[string]any)["query"] != "q" {
			t.Fatalf("parts = %+v", done.Parts)
		}
		if !strings.Contains(string(done.Parts[0].ProviderBlock), `"input":{"query":"q"}`) {
			t.Fatalf("provider block = %s", done.Parts[0].ProviderBlock)
		}
	})
	t.Run("broken input JSON keeps the started block", func(t *testing.T) {
		events, _ := runStream(t, sse(
			`{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"s","name":"web_search","input":{}}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":"}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}`,
		))
		done := events[len(events)-1]
		if !json.Valid(done.Parts[0].ProviderBlock) {
			t.Fatalf("provider block not valid JSON: %s", done.Parts[0].ProviderBlock)
		}
	})
	t.Run("deltas without block start", func(t *testing.T) {
		events, errs := runStream(t, sse(
			`{"type":"content_block_delta","index":7,"delta":{"type":"text_delta","text":"orphan"}}`,
			`{"type":"content_block_delta","index":7,"delta":{"type":"citations_delta","citation":{"type":"char_location","cited_text":"c"}}}`,
			`{"type":"content_block_delta","index":8,"delta":{"type":"citations_delta","citation":"not an object"}}`,
			`{"type":"content_block_delta","index":9,"delta":{"type":"input_json_delta","partial_json":"{}"}}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}`,
		))
		if len(errs) != 0 {
			t.Fatal(errs)
		}
		done := events[len(events)-1]
		if len(done.Parts) == 0 || done.Parts[0].Text != "orphan" || len(done.Parts[0].Citations) != 1 {
			t.Fatalf("parts = %+v", done.Parts)
		}
	})
	t.Run("malformed content block", func(t *testing.T) {
		_, errs := runStream(t, sse(
			`{"type":"content_block_start","index":0,"content_block":"text"}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}`,
		))
		if len(errs) == 0 {
			t.Fatal("want an error for a non-object content block")
		}
	})
	t.Run("truncated after server call", func(t *testing.T) {
		_, errs := runStream(t, sse(
			`{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"s","name":"web_search","input":{}}}`,
		))
		if len(errs) == 0 {
			t.Fatal("want an unexpected-EOF error")
		}
	})
	t.Run("usage from message_start when delta has none", func(t *testing.T) {
		events, _ := runStream(t, sse(
			`{"type":"message_start","message":{"usage":{"input_tokens":5,"server_tool_use":{"web_search_requests":2},"iterations":[{"type":"message","input_tokens":5,"output_tokens":0}]},"container":{"id":"cx"}}}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}`,
		))
		done := events[len(events)-1]
		if done.ServerToolUse.WebSearchRequests != 2 || len(done.Iterations) != 1 || done.ContainerID != "cx" {
			t.Fatalf("done = %+v", done)
		}
	})
}
