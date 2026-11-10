package anthropic

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// originalContent returns the fixture's content array in compact form, the
// bytes a verbatim replay must reproduce.
func originalContent(t *testing.T, name string) []byte {
	t.Helper()
	var doc struct {
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(loadFixture(t, name), &doc); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := json.Compact(&buf, doc.Content); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// TestPausedTurnWithEmptySignedThinkingReplaysByteIdentical: Claude 4.7+
// return thinking blocks with empty text and a signature by default. On a
// same-provider continuation they must go back unchanged, signature included
// — the docs: "a thinking block with empty text is still a block", and
// dropping one fails the preserved-thinking chain check.
func TestPausedTurnWithEmptySignedThinkingReplaysByteIdentical(t *testing.T) {
	want := originalContent(t, "web_search_paused_thinking.json")
	cases := map[string]llm.ThinkingLevel{
		"always-on model, ThinkingLevel unset": llm.ThinkingNone,
		"ThinkingLevel set":                    llm.ThinkingHigh,
	}
	for name, level := range cases {
		t.Run(name, func(t *testing.T) {
			c := newTestClient()
			resp, _ := parseFixture(t, c, "web_search_paused_thinking.json")
			if resp.Parts[0].Thinking != "" || resp.Parts[0].ThinkingSignature == "" {
				t.Fatalf("fixture part 0 = %+v", resp.Parts[0])
			}
			conv := llm.NewConversation("")
			conv.Append(llm.NewTextMessage(llm.RoleUser, "Look up the example gadget."))
			conv.AddResponse("anthropic", resp)
			data, _ := json.Marshal(conv)
			restored, err := llm.LoadConversation(data)
			if err != nil {
				t.Fatal(err)
			}
			for _, msgs := range [][]llm.Message{conv.Messages, restored.Messages} {
				req := llm.Request{ThinkingLevel: level, Messages: llm.NormalizeForModel(msgs, "anthropic", "claude-opus-5-5")}
				next, err := llm.ContinuePausedTurn(llm.Request{ThinkingLevel: level, Messages: req.Messages[:1]}, resp)
				if err != nil {
					t.Fatal(err)
				}
				for _, r := range []llm.Request{req, next} {
					apiReq, err := c.buildRequest(r, false)
					if err != nil {
						t.Fatal(err)
					}
					got, _ := json.Marshal(apiReq.Messages[1].Content)
					if !bytes.Equal(got, want) {
						t.Fatalf("paused content not byte-identical\n got %s\nwant %s", got, want)
					}
				}
			}
		})
	}
}

func TestEmptySignedThinkingInToolLoop(t *testing.T) {
	c := newTestClient()
	raw := `{"id":"m","model":"claude-opus-5-5","stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1},"content":[` +
		`{"type":"thinking","thinking":"","signature":"sig-loop"},` +
		`{"type":"tool_use","id":"toolu_1","name":"lookup","input":{"id":"a"}}]}`
	var apiResp anthropicResponse
	if err := json.Unmarshal([]byte(raw), &apiResp); err != nil {
		t.Fatal(err)
	}
	resp := c.parseResponse(&apiResp)
	am := llm.NewAssistantMessage(resp)
	am.Provider = "anthropic"
	msgs := llm.NormalizeForModel([]llm.Message{
		llm.NewTextMessage(llm.RoleUser, "q"), am, llm.NewToolResultMessage("toolu_1", "ok", false),
	}, "anthropic", "claude-opus-5-5")
	apiReq, err := c.buildRequest(llm.Request{Messages: msgs}, false)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := json.Marshal(apiReq.Messages[1].Content[0])
	if string(got) != `{"type":"thinking","thinking":"","signature":"sig-loop"}` {
		t.Fatalf("thinking block = %s", got)
	}

	// A model whose thinking is off for this request still drops it: the API
	// rejects thinking blocks then.
	off := &Client{config: llm.Config{Model: "claude-opus-4-8", MaxTokens: 1024}, providerName: "anthropic"}
	apiReq, err = off.buildRequest(llm.Request{Messages: msgs}, false)
	if err != nil {
		t.Fatal(err)
	}
	if blocks := apiReq.Messages[1].Content; len(blocks) != 1 {
		t.Fatalf("thinking off: blocks = %v", blocks)
	}

	// Another provider gets nothing of an empty block.
	other := llm.NormalizeForProvider([]llm.Message{llm.NewTextMessage(llm.RoleUser, "q"), am}, "gemini")
	for _, p := range other[1].Content {
		if p.Type == llm.ContentThinking {
			t.Fatalf("empty thinking crossed providers: %+v", p)
		}
	}
}

func TestStreamedEmptySignedThinkingKeepsSignature(t *testing.T) {
	events, errs := runStream(t, sse(
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":"","x_future":1}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"EuYB+/sig=="}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"server_tool_use","id":"s","name":"web_search","input":{}}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"q\"}"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"message_delta","delta":{"stop_reason":"pause_turn"},"usage":{"output_tokens":3}}`,
	))
	if len(errs) != 0 {
		t.Fatal(errs)
	}
	done := events[len(events)-1]
	msg := llm.NewAssistantMessage(&llm.Response{Parts: done.Parts, StopReason: done.StopReason})
	apiReq, err := newTestClient().buildRequest(llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "q"), msg}}, false)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := json.Marshal(apiReq.Messages[1].Content[0])
	if string(got) != `{"signature":"EuYB+/sig==","thinking":"","type":"thinking","x_future":1}` {
		t.Fatalf("streamed thinking block = %s", got)
	}
}

func TestStopDetails(t *testing.T) {
	resp, _ := parseFixture(t, newTestClient(), "refusal.json")
	d := resp.StopDetails
	if resp.StopReason != llm.StopRefusal || d == nil {
		t.Fatalf("stop = %q, details = %+v", resp.StopReason, d)
	}
	if d.Type != "refusal" || d.Category != "cyber" || d.RecommendedModel != "claude-opus-4-8" || d.Explanation == "" {
		t.Errorf("details = %+v", d)
	}
	if !bytes.Contains(d.Raw, []byte(`"x_future_field":{"k":1}`)) {
		t.Errorf("Raw lost an unknown field: %s", d.Raw)
	}

	paused, _ := parseFixture(t, newTestClient(), "web_search_paused.json")
	if paused.StopDetails != nil {
		t.Errorf("null stop_details decoded as %+v", paused.StopDetails)
	}

	events, _ := runStream(t, sse(
		`{"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"type":"refusal","category":null,"fallback_credit_token":"tok","fallback_has_prefill_claim":true}},"usage":{"output_tokens":0}}`,
	))
	sd := events[len(events)-1].StopDetails
	if sd == nil || sd.Category != "" || sd.FallbackCreditToken != "tok" || !sd.FallbackHasPrefillClaim {
		t.Fatalf("streamed details = %+v", sd)
	}

	for raw, wantNil := range map[string]bool{`null`: true, `"refusal"`: true, `[1]`: true, `{`: true, `{"category":7,"type":"refusal"}`: false} {
		got := llm.StopDetailsFromRaw(json.RawMessage(raw))
		if (got == nil) != wantNil {
			t.Errorf("StopDetailsFromRaw(%s) = %+v", raw, got)
		}
		if got != nil && (got.Category != "" || got.Type != "refusal") {
			t.Errorf("mistyped field leaked: %+v", got)
		}
	}
}

func TestBetaHeaderSentWithoutThinking(t *testing.T) {
	var headers []string
	var bodies []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		headers = append(headers, r.Header.Get("anthropic-beta"))
		b, _ := io.ReadAll(r.Body)
		bodies = append(bodies, string(b))
		if strings.Contains(r.Header.Get("Accept"), "event-stream") {
			_, _ = w.Write([]byte(sse(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`)))
			return
		}
		_, _ = w.Write(loadFixture(t, "web_search_final.json"))
	}))
	defer srv.Close()
	c, err := New(llm.Config{Model: "claude-opus-5-5", APIKey: "k", BaseURL: srv.URL, MaxTokens: 64,
		BetaFeatures: []string{"server-side-fallback-2026-07-01", "fallback-credit-2026-07-01"}})
	if err != nil {
		t.Fatal(err)
	}
	req := llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "q")}, SamplingParams: map[string]any{"fallbacks": "default"}}
	if _, err := c.Complete(t.Context(), req); err != nil {
		t.Fatal(err)
	}
	for _, err := range c.Stream(t.Context(), req) {
		if err != nil {
			t.Fatal(err)
		}
	}
	for i, h := range headers {
		if h != "server-side-fallback-2026-07-01,fallback-credit-2026-07-01" {
			t.Errorf("request %d anthropic-beta = %q", i, h)
		}
	}
	if len(headers) != 2 {
		t.Fatalf("requests = %d", len(headers))
	}
	for i, b := range bodies {
		if !strings.Contains(b, `"fallbacks":"default"`) {
			t.Errorf("request %d body lacks fallbacks: %s", i, b)
		}
	}
}

func TestCacheTTLWire(t *testing.T) {
	c := newTestClient()
	req := llm.Request{
		System: "sys", SystemCacheControl: true, SystemCacheTTL: llm.CacheTTL1h,
		Tools:       []llm.Tool{{Name: "t", Description: "d", CacheControl: true, CacheTTL: llm.CacheTTL1h}},
		ServerTools: []llm.ServerTool{{Type: llm.WebSearch20250305, Name: "web_search", CacheControl: true, CacheTTL: llm.CacheTTL1h}},
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentPart{
			{Type: llm.ContentText, Text: "long", CacheControl: true, CacheTTL: llm.CacheTTL1h},
			{Type: llm.ContentText, Text: "tail", CacheControl: true},
		}}},
	}
	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := json.Marshal(apiReq)
	for _, want := range []string{
		`"system":[{"cache_control":{"type":"ephemeral","ttl":"1h"},"text":"sys","type":"text"}]`,
		`{"name":"t","description":"d","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral","ttl":"1h"}}`,
		`{"cache_control":{"type":"ephemeral","ttl":"1h"},"name":"web_search","type":"web_search_20250305"}`,
		`{"cache_control":{"type":"ephemeral","ttl":"1h"},"text":"long","type":"text"}`,
		`{"cache_control":{"type":"ephemeral"},"text":"tail","type":"text"}`,
	} {
		if !strings.Contains(string(body), want) {
			t.Errorf("body lacks %s\nbody: %s", want, body)
		}
	}

	for name, r := range map[string]llm.Request{
		"system":              {System: "s", SystemCacheControl: true, SystemCacheTTL: "2h"},
		"tool":                {Tools: []llm.Tool{{Name: "t", CacheControl: true, CacheTTL: "1H"}}},
		"server tool":         {ServerTools: []llm.ServerTool{{Type: llm.WebSearch20250305, Name: "web_search", CacheControl: true, CacheTTL: "60m"}}},
		"part":                {Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentPart{{Type: llm.ContentText, Text: "x", CacheControl: true, CacheTTL: " 1h"}}}}},
		"ttl without marker":  {Tools: []llm.Tool{{Name: "t", CacheTTL: llm.CacheTTL1h}}},
		"1h after 5m":         {Tools: []llm.Tool{{Name: "t", CacheControl: true}}, System: "s", SystemCacheControl: true, SystemCacheTTL: llm.CacheTTL1h},
		"1h message after 5m": {Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentPart{{Type: llm.ContentText, Text: "a", CacheControl: true}, {Type: llm.ContentText, Text: "b", CacheControl: true, CacheTTL: llm.CacheTTL1h}}}}},
	} {
		if _, err := c.buildRequest(r, false); err == nil {
			t.Errorf("%s: bad TTL accepted", name)
		}
	}
}

func TestOneHourCacheWritesParsedAndPriced(t *testing.T) {
	resp, _ := parseFixture(t, newTestClient(), "web_search_paused_thinking.json")
	if resp.CacheCreationTokens != 3000 || resp.CacheCreation1hTokens != 2000 {
		t.Fatalf("cache writes = %d, 1h = %d", resp.CacheCreationTokens, resp.CacheCreation1hTokens)
	}
	var u llm.Usage
	u.AddResponse(resp)
	// Opus 5.5: input $4, output $20, 5m write $5, 1h write $8, search $10/1k.
	want := (900*4.0+120*20.0+1000*5.0+2000*8.0)/1e6 + 0.01
	if diff := u.Cost - want; diff > 1e-12 || diff < -1e-12 {
		t.Fatalf("cost = %.10f, want %.10f", u.Cost, want)
	}
	if u.CacheCreation1hTokens != 2000 {
		t.Errorf("Usage.CacheCreation1hTokens = %d", u.CacheCreation1hTokens)
	}
}

func TestHandBuiltEmptySignedThinkingIsSent(t *testing.T) {
	msg := llm.Message{Role: llm.RoleAssistant, Provider: "anthropic", Content: []llm.ContentPart{
		{Type: llm.ContentThinking, ThinkingSignature: "sig-hand"},
		{Type: llm.ContentText, Text: "a"},
	}}
	apiReq, err := newTestClient().buildRequest(llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "q"), msg}}, false)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := json.Marshal(apiReq.Messages[1].Content[0])
	if string(got) != `{"signature":"sig-hand","thinking":"","type":"thinking"}` {
		t.Fatalf("block = %s", got)
	}
}

// TestThinkingReplayPerModelWithoutThinkingLevel: without a ThinkingLevel a
// model gets its own "off" form. Always-on models (thinking omitted) and
// between_tools (Sonnet 5.5) still write thinking blocks, which must go back
// unchanged; a model sent {"type":"disabled"} gets them dropped (the API
// rejects thinking blocks then).
func TestThinkingReplayPerModelWithoutThinkingLevel(t *testing.T) {
	want := `{"type":"thinking","thinking":"","signature":"sig-m"}`
	cases := []struct {
		model, thinking string
		replayed        bool
	}{
		{"claude-opus-5-5", "", true},
		{"claude-fable-5-2", "", true},
		{"claude-fable-5-1", "", true},
		{"claude-sonnet-5-5", "between_tools", true},
		{"claude-sonnet-5", "disabled", false},
		{"claude-opus-4-8", "disabled", false},
		{"claude-haiku-4-5", "", false},
	}
	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			c := &Client{config: llm.Config{Model: tc.model, MaxTokens: 1024}, providerName: "anthropic"}
			raw := `{"id":"m","model":"` + tc.model + `","stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1},"content":[` +
				want + `,{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}]}`
			var apiResp anthropicResponse
			if err := json.Unmarshal([]byte(raw), &apiResp); err != nil {
				t.Fatal(err)
			}
			am := llm.NewAssistantMessage(c.parseResponse(&apiResp))
			am.Provider = "anthropic"
			msgs := llm.NormalizeForModel([]llm.Message{
				llm.NewTextMessage(llm.RoleUser, "q"), am, llm.NewToolResultMessage("toolu_1", "ok", false),
			}, "anthropic", tc.model)
			apiReq, err := c.buildRequest(llm.Request{Messages: msgs}, false)
			if err != nil {
				t.Fatal(err)
			}
			gotThinking := ""
			if apiReq.Thinking != nil {
				gotThinking = apiReq.Thinking.Type
			}
			if gotThinking != tc.thinking {
				t.Fatalf("thinking = %q, want %q", gotThinking, tc.thinking)
			}
			first, _ := json.Marshal(apiReq.Messages[1].Content[0])
			if replayed := string(first) == want; replayed != tc.replayed {
				t.Fatalf("replayed = %v (first block %s), want %v", replayed, first, tc.replayed)
			}
		})
	}
}

func TestInterleavedBetaOnlyWithBudgetThinking(t *testing.T) {
	var headers []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		headers = append(headers, r.Header.Get("anthropic-beta"))
		_, _ = w.Write(loadFixture(t, "web_search_final.json"))
	}))
	defer srv.Close()
	run := func(model string, level llm.ThinkingLevel, betas []string) string {
		t.Helper()
		headers = nil
		c, err := New(llm.Config{Model: model, APIKey: "k", BaseURL: srv.URL, MaxTokens: 8192, BetaFeatures: betas})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := c.Complete(t.Context(), llm.Request{ThinkingLevel: level, Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "q")}}); err != nil {
			t.Fatal(err)
		}
		return headers[0]
	}
	defaults := llm.DefaultConfig().BetaFeatures
	if h := run("claude-haiku-4-5", llm.ThinkingLow, defaults); h != interleavedThinkingBeta {
		t.Errorf("budget thinking: anthropic-beta = %q", h)
	}
	if h := run("claude-haiku-4-5", llm.ThinkingNone, defaults); h != "" {
		t.Errorf("no thinking: anthropic-beta = %q, want none", h)
	}
	if h := run("claude-opus-5-5", llm.ThinkingHigh, defaults); h != "" {
		t.Errorf("adaptive thinking: anthropic-beta = %q, want none", h)
	}
	if h := run("claude-opus-5-5", llm.ThinkingNone, append(defaults, "server-side-fallback-2026-07-01")); h != "server-side-fallback-2026-07-01" {
		t.Errorf("user beta: anthropic-beta = %q", h)
	}
}
