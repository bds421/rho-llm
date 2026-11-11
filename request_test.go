package llm_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider"
	"github.com/bds421/rho-llm/provider/anthropic"
	"github.com/bds421/rho-llm/provider/gemini"
	"github.com/bds421/rho-llm/provider/openaicompat"
	"github.com/bds421/rho-llm/provider/openairesponses"
)

func TestToolInputNumberDriftIsWireSafe(t *testing.T) {
	// Known Go-JSON behavior: integers in ToolInput (typed any) deserialize as
	// float64. This is a documented limitation for callers that type-assert the
	// in-memory value — but the WIRE bytes are identical, so a resend (the thing
	// that actually matters) is unaffected.
	conv := llm.NewConversation("")
	conv.Append(llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentPart{
		{Type: llm.ContentToolUse, ToolUseID: "c", ToolName: "f", ToolInput: map[string]any{"page": 3}},
	}})

	blob, err := json.Marshal(conv)
	if err != nil {
		t.Fatal(err)
	}
	got, err := llm.LoadConversation(blob)
	if err != nil {
		t.Fatal(err)
	}

	// In-memory type drifts int -> float64 (documented)...
	in := got.Messages[0].Content[0].ToolInput.(map[string]any)
	if _, isFloat := in["page"].(float64); !isFloat {
		t.Errorf("expected float64 after reload (Go JSON), got %T", in["page"])
	}
	// ...but the re-serialized bytes are byte-identical, so resend is safe.
	reblob, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if string(blob) != string(reblob) {
		t.Errorf("wire form changed across reload:\n in: %s\nout: %s", blob, reblob)
	}
}

func captureBody(t *testing.T, jsonResp string) (*httptest.Server, *[]byte) {
	t.Helper()
	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, jsonResp)
	}))
	return srv, &body
}

// A message whose content all filtered out must not be emitted with a null/empty
// content field that the provider rejects.
func TestOpenAIFamilyBuildRequestDropsEmptyContentMessage(t *testing.T) {
	emptyAsst := llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentPart{{Type: llm.ContentText, Text: ""}}}
	msgs := []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi"), emptyAsst}

	t.Run("openaicompat", func(t *testing.T) {
		srv, body := captureBody(t, `{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{}}`)
		defer srv.Close()
		c, err := openaicompat.New(llm.Config{Provider: "openai", Model: "gpt-4.1", APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if _, err := c.Complete(context.Background(), llm.Request{MaxTokens: 16, Messages: msgs}); err != nil {
			t.Fatalf("Complete: %v", err)
		}
		if bytes.Contains(*body, []byte(`"content":null`)) {
			t.Fatalf("openaicompat emitted a null-content message: %s", *body)
		}
	})

	t.Run("openairesponses", func(t *testing.T) {
		srv, body := captureBody(t, `{"id":"r","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{}}`)
		defer srv.Close()
		c, err := openairesponses.New(llm.Config{Provider: "openai_responses", Model: "gpt-5.2", APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if _, err := c.Complete(context.Background(), llm.Request{MaxTokens: 16, Messages: msgs}); err != nil {
			t.Fatalf("Complete: %v", err)
		}
		if bytes.Contains(*body, []byte(`"content":null`)) {
			t.Fatalf("openairesponses emitted a null-content message: %s", *body)
		}
	})
}

// A message whose content all gets filtered out (here: an assistant turn with
// only an empty-text part) must not be emitted as an empty-content message —
// Anthropic rejects "content": [] / null. The adapter should drop it (matching
// NormalizeForProvider), not send a malformed message.
func TestAnthropicBuildRequestDropsEmptyContentMessage(t *testing.T) {
	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"m","model":"claude-sonnet-4-6","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer srv.Close()

	client, err := anthropic.New(llm.Config{Provider: "anthropic", Model: "claude-sonnet-4-6", APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = client.Complete(context.Background(), llm.Request{
		MaxTokens: 16,
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "hi"),
			{Role: llm.RoleAssistant, Content: []llm.ContentPart{{Type: llm.ContentText, Text: ""}}}, // all-empty
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if bytes.Contains(body, []byte(`"content":[]`)) || bytes.Contains(body, []byte(`"content":null`)) {
		t.Fatalf("emitted an empty-content message (Anthropic rejects it): %s", body)
	}
}

// A per-request Model override that is an ALIAS (or mixed-case variant) must be
// resolved to its canonical registry ID before it hits the provider wire — the
// provider only recognizes the exact ID (MiniMax wants "MiniMax-M3", not
// "minimax"/"minimax-m3"). factory.go resolves cfg.Model at construction, but a
// Request{Model: alias} override bypassed that and shipped the alias verbatim.
func TestPerRequestAliasResolvesToCanonicalWireModel(t *testing.T) {
	cases := []struct{ provider, reqModel, wantWire string }{
		{"zai", "glm", "glm-5.3"},
		{"minimax", "minimax-m3", "MiniMax-M3"}, // lowercase variant of the mixed-case key
		{"moonshot", "kimi", "kimi-k3.1-pro"},
	}
	for _, tc := range cases {
		t.Run(tc.provider+"/"+tc.reqModel, func(t *testing.T) {
			var gotModel string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				var req struct {
					Model string `json:"model"`
				}
				_ = json.Unmarshal(body, &req)
				gotModel = req.Model
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprint(w, `{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
			}))
			defer srv.Close()

			client, err := llm.NewClient(llm.Config{Provider: tc.provider, Model: tc.wantWire, APIKey: "sk-test", BaseURL: srv.URL})
			if err != nil {
				t.Fatalf("NewClient(%s): %v", tc.provider, err)
			}
			defer client.Close()

			_, _ = client.Complete(context.Background(), llm.Request{
				Model:    tc.reqModel, // per-request alias/variant override
				Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
			})
			if gotModel != tc.wantWire {
				t.Fatalf("wire model = %q, want canonical %q — per-request alias not resolved before the wire", gotModel, tc.wantWire)
			}
		})
	}
}

func TestSamplingParamsBatchBodiesMatchLiveRequests(t *testing.T) {
	for i, p := range tcProtocols {
		t.Run(p.name, func(t *testing.T) {
			c, err := p.new(llm.Config{Provider: p.name, Model: p.model, APIKey: "k", BaseURL: "http://localhost"})
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			var build func(llm.Request) (json.RawMessage, error)
			switch c := c.(type) {
			case *openaicompat.Client:
				build = c.BuildChatBatchLineBody
			case *openairesponses.Client:
				build = c.BuildResponsesBatchLineBody
			case *anthropic.Client:
				build = c.BuildMessageBatchParams
			case *gemini.Client:
				build = c.BuildGenerateContentBody
			default:
				t.Fatalf("unsupported batch codec %T", c)
			}
			for _, tc := range []struct {
				name    string
				params  map[string]any
				wantErr bool
			}{
				{"passthrough", map[string]any{"seed": 42, "provider": map[string]any{"order": []string{"a", "b"}}}, false},
				{"reserved_model", map[string]any{"model": "hijacked"}, true},
				{"reserved_stream", map[string]any{"stream": true}, true},
				{"reserved_absent_tools", map[string]any{"tools": []any{}}, true},
				{"empty_key", map[string]any{"": 1}, true},
				{"unmarshalable_value", map[string]any{"bad": make(chan int)}, true},
			} {
				t.Run(tc.name, func(t *testing.T) {
					req := llm.Request{MaxTokens: 16, Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")}, SamplingParams: tc.params}
					raw, err := build(req)
					if tc.wantErr {
						if err == nil {
							t.Fatalf("invalid SamplingParams accepted by batch codec: %s", raw)
						}
						return
					}
					if err != nil {
						t.Fatal(err)
					}
					var got map[string]any
					if err := json.Unmarshal(raw, &got); err != nil {
						t.Fatal(err)
					}
					want := tcDo(t, i, req, false)
					if !reflect.DeepEqual(got, want) {
						t.Fatalf("batch body differs from live request: batch=%v live=%v", got, want)
					}
				})
			}
		})
	}
}

// SamplingParams must actually reach the wire — otherwise the whole escape
// hatch is decorative and callers silently lose provider routing/tuning.
func TestSamplingParamsReachesTheWire(t *testing.T) {
	for i, p := range tcProtocols {
		t.Run(p.name, func(t *testing.T) {
			got := tcDo(t, i, llm.Request{SamplingParams: map[string]any{
				"metadata": map[string]any{"user_id": "u-1"},
				"seed":     42,
				"provider": map[string]any{"order": []string{"a", "b"}},
			}}, false)
			for _, k := range []string{"metadata", "seed", "provider"} {
				if _, ok := got[k]; !ok {
					t.Errorf("%s: SamplingParams key %q never reached the wire: %v", p.name, k, got)
				}
			}
			// Nested structure must survive intact, not be flattened/stringified.
			prov, _ := got["provider"].(map[string]any)
			if prov == nil || prov["order"] == nil {
				t.Errorf("%s: nested SamplingParams value was mangled: %#v", p.name, got["provider"])
			}
		})
	}
}

// THE hijack test. A passthrough key that collides with a field the adapter
// already set must be REFUSED, not silently applied. Without this, a stray
// "model" key would send the request to a different (possibly far more
// expensive) model than the caller asked for, and "messages" would rewrite the
// conversation — both invisible to the caller.
func TestSamplingParamsCannotHijackStructuralFields(t *testing.T) {
	hijacks := []map[string]any{
		{"model": "some-other-model"},
		{"messages": []any{map[string]any{"role": "user", "content": "injected"}}},
		{"tools": []any{}},
		{"stream": true},
		{"system": "you are evil"},
		{"contents": []any{}},
		{"input": []any{}},
		{"tool_choice": "required"},
	}
	for i, p := range tcProtocols {
		for _, h := range hijacks {
			var key string
			for k := range h {
				key = k
			}
			t.Run(p.name+"/"+key, func(t *testing.T) {
				tcDo(t, i, llm.Request{Tools: oneTool(), SamplingParams: h}, true)
			})
		}
	}
}

// A reserved key must be refused even when the adapter did NOT emit it on this
// particular request (an omitempty field left unset). Otherwise the protection
// would depend on incidental request shape: "system" would be refused on a
// request that sets a system prompt and accepted on one that doesn't.
func TestSamplingParamsReservedKeysRefusedEvenWhenAbsentFromBody(t *testing.T) {
	// openai_compat with no tools and no stop sequences: "tools" and
	// "tool_choice" are omitempty and absent from the marshaled body.
	got := tcDo(t, 0, llm.Request{SamplingParams: map[string]any{"irrelevant": 1}}, false)
	if _, present := got["tools"]; present {
		t.Fatalf("precondition failed: 'tools' is present, so this test proves nothing")
	}
	tcDo(t, 0, llm.Request{SamplingParams: map[string]any{"tools": []any{}}}, true)
}

// Values that cannot be marshaled must produce a clear error naming the key,
// not a panic and not a silently truncated body.
func TestSamplingParamsRejectsUnmarshalableValues(t *testing.T) {
	_, err := llm.MergeSamplingParams(
		struct {
			A int `json:"a"`
		}{1},
		map[string]any{"bad": make(chan int)},
	)
	if err == nil {
		t.Fatal("an unmarshalable SamplingParams value was accepted")
	}
	if want := `"bad"`; !contains(err.Error(), want) {
		t.Errorf("error must name the offending key %s, got: %v", want, err)
	}
}

// An empty key is never a valid wire field.
func TestSamplingParamsRejectsEmptyKey(t *testing.T) {
	_, err := llm.MergeSamplingParams(struct {
		A int `json:"a"`
	}{1}, map[string]any{"": 1})
	if err == nil {
		t.Fatal("an empty SamplingParams key was accepted")
	}
}

// A nil/empty map must be a no-op that leaves the body byte-identical — the
// non-passthrough path is the overwhelmingly common one.
func TestSamplingParamsNilIsNoOp(t *testing.T) {
	type req struct {
		A int      `json:"a"`
		B []string `json:"b,omitempty"`
	}
	plain, err := json.Marshal(req{A: 1})
	if err != nil {
		t.Fatal(err)
	}
	for _, params := range []map[string]any{nil, {}} {
		got, err := llm.MergeSamplingParams(req{A: 1}, params)
		if err != nil {
			t.Fatalf("MergeSamplingParams(%v): %v", params, err)
		}
		if string(got) != string(plain) {
			t.Errorf("SamplingParams(%v) altered the body: %s vs %s", params, got, plain)
		}
	}
}

// Merging into a non-object body must error rather than produce garbage.
func TestSamplingParamsRejectsNonObjectBody(t *testing.T) {
	for _, tc := range []struct {
		name string
		body any
	}{
		{"nil", nil},
		{"nil_pointer", (*struct{})(nil)},
		{"nil_map", map[string]any(nil)},
		{"nil_slice", []int(nil)},
		{"raw_null", json.RawMessage(`null`)},
		{"array", []int{1, 2, 3}},
		{"string", "request"},
		{"number", 42},
		{"boolean", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body, err := llm.MergeSamplingParams(tc.body, map[string]any{"x": 1})
			if err == nil || !contains(err.Error(), "non-object request body") {
				t.Fatalf("expected non-object error, got body=%s err=%v", body, err)
			}
			if body != nil {
				t.Fatalf("rejected request returned a body: %s", body)
			}
		})
	}
}

// The error must be deterministic when several keys offend — an error that
// changes between identical runs is far harder to debug.
func TestSamplingParamsErrorIsDeterministic(t *testing.T) {
	body := struct {
		Model    string `json:"model"`
		Messages []any  `json:"messages"`
	}{"m", nil}
	var first string
	for i := 0; i < 20; i++ {
		_, err := llm.MergeSamplingParams(body, map[string]any{
			"model": "x", "messages": []any{}, "system": "y", "tools": []any{},
		})
		if err == nil {
			t.Fatal("hijack was accepted")
		}
		if i == 0 {
			first = err.Error()
		} else if err.Error() != first {
			t.Fatalf("non-deterministic error: %q then %q", first, err.Error())
		}
	}
}

// RawStopReason must preserve the provider's own vocabulary. The bug this
// guards: normalization maps "length"→"max_tokens" and the original is lost, so
// a caller can never tell a real max-tokens stop from a provider-specific one
// that happened to normalize the same way.
func TestRawStopReasonSurvivesNormalization(t *testing.T) {
	cases := []struct {
		raw        string
		wantNormal string
	}{
		{"stop", llm.StopEndTurn},
		{"length", llm.StopMaxTokens},
		{"tool_calls", llm.StopToolUse},
		{"content_filter", llm.StopRefusal},    // RawStopReason keeps "content_filter"
		{"some_new_reason", "some_new_reason"}, // unmapped: passes through verbatim
	}
	for _, c := range cases {
		t.Run(c.raw, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprintf(w, `{"id":"x","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":%q}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`, c.raw)
			}))
			defer srv.Close()
			cl, err := openaicompat.New(llm.Config{Provider: "openai", Model: "gpt-4.1", APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
			if err != nil {
				t.Fatal(err)
			}
			defer cl.Close()
			resp, err := cl.Complete(context.Background(), llm.Request{
				MaxTokens: 16,
				Messages:  []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
			})
			if err != nil {
				t.Fatalf("Complete: %v", err)
			}
			if resp.StopReason != c.wantNormal {
				t.Errorf("StopReason = %q, want %q", resp.StopReason, c.wantNormal)
			}
			if resp.RawStopReason != c.raw {
				t.Errorf("RawStopReason = %q, want the provider's own %q — normalization erased it",
					resp.RawStopReason, c.raw)
			}
		})
	}
}

// A synthesized stop reason has no provider raw value, so RawStopReason must
// stay EMPTY rather than echoing a value the provider never sent. The bug this
// guards: fabricating provenance that looks authoritative but isn't.
func TestSynthesizedStopReasonHasNoRawValue(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// A [DONE] with no finish_reason — the known local-server spec violation.
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()
	cl, err := openaicompat.New(llm.Config{Provider: "openai", Model: "gpt-4.1", APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
	if err != nil {
		t.Fatal(err)
	}
	defer cl.Close()

	var sawDone bool
	for ev, err := range cl.Stream(context.Background(), llm.Request{
		MaxTokens: 16,
		Messages:  []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
	}) {
		if err != nil {
			t.Fatalf("Stream: %v", err)
		}
		if ev.Type == llm.EventDone {
			sawDone = true
			if ev.StopReason == "" {
				t.Error("synthesized StopReason should be set")
			}
			if ev.RawStopReason != "" {
				t.Errorf("RawStopReason = %q, want empty — the provider never sent a finish_reason, so reporting one is fabricated provenance", ev.RawStopReason)
			}
		}
	}
	if !sawDone {
		t.Fatal("no EventDone — assertion would be vacuous")
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (len(sub) == 0 || indexOf(s, sub) >= 0)
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}

// A ToolChoice/SamplingParams set on a Session's base Request must survive into
// each turn — including across a SwitchProvider handoff, where the whole point
// of a provider-neutral ToolChoice is that it re-translates to the new
// provider's shape instead of being dropped.
func TestBaseRequestCarriesToolChoiceAndSamplingParams(t *testing.T) {
	conv := llm.NewConversation("", oneTool()...)
	conv.Messages = []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")}
	base := llm.Request{
		Model:          "m",
		ToolChoice:     llm.ForceTool("get_weather"),
		SamplingParams: map[string]any{"top_p": 0.25},
		Tools:          oneTool(),
	}
	got := conv.ToRequest(base)
	if got.ToolChoice == nil || got.ToolChoice.Name != "get_weather" {
		t.Errorf("ToolChoice lost in ToRequest: %#v", got.ToolChoice)
	}
	if got.SamplingParams["top_p"] != 0.25 {
		t.Errorf("SamplingParams lost in ToRequest: %#v", got.SamplingParams)
	}
}

// A request struct that cannot be marshaled at all must surface that error
// rather than being reported as a SamplingParams problem.
func TestSamplingParamsPropagatesMarshalFailure(t *testing.T) {
	bad := struct {
		Ch chan int `json:"ch"`
	}{make(chan int)}
	if _, err := llm.MergeSamplingParams(bad, map[string]any{"x": 1}); err == nil {
		t.Fatal("an unmarshalable request struct was accepted")
	}
	// Same failure with no params at all — the early-return path.
	if _, err := llm.MergeSamplingParams(bad, nil); err == nil {
		t.Fatal("an unmarshalable request struct was accepted with nil params")
	}
}

// A json_schema response format with no Schema is meaningless and serializes to
// "schema": null, which the provider rejects. The adapter should degrade to plain
// JSON mode (json_object) rather than emit a null schema.
func TestOpenAICompatJSONSchemaWithoutSchemaDegrades(t *testing.T) {
	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"{}"},"finish_reason":"stop"}],"usage":{}}`)
	}))
	defer srv.Close()

	c, err := openaicompat.New(llm.Config{Provider: "openai", Model: "gpt-4.1", APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = c.Complete(context.Background(), llm.Request{
		MaxTokens:      16,
		Messages:       []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
		ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSONSchema, Name: "x"}, // no Schema
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if bytes.Contains(body, []byte(`"schema":null`)) {
		t.Fatalf("emitted a null json_schema (provider rejects it): %s", body)
	}
}

// tcProtocols pairs each wire protocol with a constructor, a model that routes
// to it, and the minimal valid response body that protocol returns.
var tcProtocols = []struct {
	name  string
	model string
	resp  string
	new   func(llm.Config) (llm.Client, error)
}{
	{
		name: "openai_compat", model: "gpt-4.1",
		resp: `{"id":"x","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`,
		new:  func(c llm.Config) (llm.Client, error) { return openaicompat.New(c) },
	},
	{
		name: "anthropic", model: "claude-opus-5",
		resp: `{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"model":"m","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`,
		new:  func(c llm.Config) (llm.Client, error) { return anthropic.New(c) },
	},
	{
		name: "gemini", model: "gemini-2.5-flash",
		resp: `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}`,
		new:  func(c llm.Config) (llm.Client, error) { return gemini.New(c) },
	},
	{
		name: "openai_responses", model: "gpt-5.6-sol",
		resp: `{"id":"resp_1","object":"response","model":"m","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1}}`,
		new:  func(c llm.Config) (llm.Client, error) { return openairesponses.New(c) },
	},
}

// tcDo issues one Complete against a capturing server and returns the decoded
// request body. It fails the test if the request errored (unless wantErr).
func tcDo(t *testing.T, proto int, req llm.Request, wantErr bool) map[string]any {
	t.Helper()
	p := tcProtocols[proto]
	srv, raw := captureBody(t, p.resp)
	defer srv.Close()
	c, err := p.new(llm.Config{Provider: p.name, Model: p.model, APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
	if err != nil {
		t.Fatalf("%s New: %v", p.name, err)
	}
	defer c.Close()
	req.MaxTokens = 16
	req.Messages = []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")}
	_, err = c.Complete(context.Background(), req)
	if wantErr {
		if err == nil {
			t.Fatalf("%s: expected an error, got none", p.name)
		}
		return nil
	}
	if err != nil {
		t.Fatalf("%s Complete: %v", p.name, err)
	}
	if len(*raw) == 0 {
		t.Fatalf("%s: server never received a body — assertion would be vacuous", p.name)
	}
	var m map[string]any
	if err := json.Unmarshal(*raw, &m); err != nil {
		t.Fatalf("%s: body is not a JSON object: %v (%s)", p.name, err, *raw)
	}
	return m
}

// tcField is the wire field each protocol carries tool choice in.
func tcField(proto int) string {
	if tcProtocols[proto].name == "gemini" {
		return "toolConfig"
	}
	return "tool_choice"
}

func oneTool() []llm.Tool {
	return []llm.Tool{{
		Name:        "get_weather",
		Description: "weather",
		InputSchema: map[string]any{"type": "object"},
	}}
}

// A ToolChoice that constrains tool use MUST reach the wire. The bug this
// guards: a caller says "you must call a tool", the adapter drops it, and the
// model answers with prose instead — a silent, provider-side behavior change
// that no error ever surfaces.
func TestToolChoiceReachesTheWire(t *testing.T) {
	for i, p := range tcProtocols {
		for _, mode := range []llm.ToolChoiceMode{llm.ToolChoiceNone, llm.ToolChoiceRequired, llm.ToolChoiceTool} {
			t.Run(p.name+"/"+string(mode), func(t *testing.T) {
				tc := &llm.ToolChoice{Mode: mode}
				if mode == llm.ToolChoiceTool {
					tc.Name = "get_weather"
				}
				got := tcDo(t, i, llm.Request{Tools: oneTool(), ToolChoice: tc}, false)
				v, ok := got[tcField(i)]
				if !ok {
					t.Fatalf("%s: ToolChoice{%s} never reached the wire — no %q field in %v",
						p.name, mode, tcField(i), got)
				}
				if mode == llm.ToolChoiceTool {
					if enc, _ := json.Marshal(v); !strings.Contains(string(enc), "get_weather") {
						t.Errorf("%s: forced tool name lost on the wire: %s", p.name, enc)
					}
				}
			})
		}
	}
}

// Auto is every provider's default, so it must NOT be emitted. The bug this
// guards: shipping an explicit "auto" to a provider that rejects the field, or
// that treats an explicit value differently from its absence.
func TestToolChoiceAutoIsOmitted(t *testing.T) {
	for i, p := range tcProtocols {
		t.Run(p.name, func(t *testing.T) {
			got := tcDo(t, i, llm.Request{Tools: oneTool(), ToolChoice: llm.NewToolChoice(llm.ToolChoiceAuto)}, false)
			for _, field := range []string{"tool_choice", "toolConfig"} {
				if _, present := got[field]; present {
					t.Errorf("%s: ToolChoiceAuto emitted %q — auto is the default and must be omitted", p.name, field)
				}
			}
		})
	}
}

// A tool constraint with NO tools is a 400 on real providers, so the adapter
// must not ship one. The bug this guards: a caller sets ToolChoice but forgets
// Tools, and every request fails at the provider with an opaque error.
func TestToolChoiceWithoutToolsIsNotSent(t *testing.T) {
	for i, p := range tcProtocols {
		t.Run(p.name, func(t *testing.T) {
			got := tcDo(t, i, llm.Request{ToolChoice: llm.NewToolChoice(llm.ToolChoiceRequired)}, false)
			for _, field := range []string{"tool_choice", "toolConfig"} {
				if _, present := got[field]; present {
					t.Errorf("%s: sent %q with an empty tools array — providers reject this", p.name, field)
				}
			}
		})
	}
}

// A malformed ToolChoice must fail LOUDLY at the library boundary, not sail
// through as an opaque provider 400 — or, worse, as a silently ignored field.
func TestMalformedToolChoiceIsRejected(t *testing.T) {
	bad := []struct {
		name string
		tc   *llm.ToolChoice
	}{
		{"mode tool with no name", &llm.ToolChoice{Mode: llm.ToolChoiceTool}},
		{"empty mode", &llm.ToolChoice{}},
		{"unknown mode", &llm.ToolChoice{Mode: "definitely-not-a-mode"}},
		{"name without tool mode", &llm.ToolChoice{Mode: llm.ToolChoiceRequired, Name: "get_weather"}},
		{"name on auto", &llm.ToolChoice{Mode: llm.ToolChoiceAuto, Name: "get_weather"}},
	}
	for i, p := range tcProtocols {
		for _, b := range bad {
			t.Run(p.name+"/"+b.name, func(t *testing.T) {
				tcDo(t, i, llm.Request{Tools: oneTool(), ToolChoice: b.tc}, true)
			})
		}
	}
}

// A nil ToolChoice is the documented "provider default" and must never panic or
// error — it is the overwhelmingly common path.
func TestNilToolChoiceIsInert(t *testing.T) {
	if err := (*llm.ToolChoice)(nil).Validate(); err != nil {
		t.Fatalf("nil ToolChoice must validate clean, got %v", err)
	}
	for i, p := range tcProtocols {
		t.Run(p.name, func(t *testing.T) {
			got := tcDo(t, i, llm.Request{Tools: oneTool()}, false)
			for _, field := range []string{"tool_choice", "toolConfig"} {
				if _, present := got[field]; present {
					t.Errorf("%s: nil ToolChoice still emitted %q", p.name, field)
				}
			}
		})
	}
}

// TestReasoningModelMaxTokensWireFormat verifies that reasoning models get the
// correct max tokens parameter based on their provider: OpenAI/xAI use
// max_completion_tokens, while Mistral/Groq/Ollama use standard max_tokens.
func TestReasoningModelMaxTokensWireFormat(t *testing.T) {
	tests := []struct {
		provider               string
		model                  string
		wantMaxTokens          bool // expect max_tokens in wire request
		wantMaxCompletionToken bool // expect max_completion_tokens in wire request
	}{
		// OpenAI reasoning models → max_completion_tokens (gpt-5 family uses Responses API, tested separately)
		{"openai", "o3", false, true},
		// xAI reasoning models → max_completion_tokens
		{"xai", "grok-4.3", false, true},
		// Mistral reasoning models → max_tokens
		{"mistral", "mistral-small-2603", true, false},
		// Other openai_compat reasoning hosts → max_tokens
		{"dashscope", "qwen3.6-plus", true, false},
		// Ollama reasoning models → max_tokens
		{"ollama", "qwen3:8b", true, false},
		// Non-reasoning models → max_tokens
		{"openai", "gpt-4.1", true, false},
		{"mistral", "mistral-large-2512", true, false},
	}

	for _, tt := range tests {
		t.Run(tt.provider+"/"+tt.model, func(t *testing.T) {
			var captured []byte
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Errorf("read body: %v", err)
				}
				captured = body
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{
					"id": "chatcmpl-test",
					"object": "chat.completion",
					"model": "` + tt.model + `",
					"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
					"usage": {"prompt_tokens": 10, "completion_tokens": 2}
				}`))
			}))
			defer srv.Close()

			cfg := llm.Config{
				Provider:  tt.provider,
				Model:     tt.model,
				APIKey:    "test-key",
				BaseURL:   srv.URL,
				MaxTokens: 1024,
				Timeout:   5 * time.Second,
			}
			client, err := llm.NewClient(cfg)
			if err != nil {
				t.Fatalf("NewClient: %v", err)
			}
			defer client.Close()

			_, err = client.Complete(context.Background(), llm.Request{
				Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
			})
			if err != nil {
				t.Fatalf("Complete: %v", err)
			}

			wireStr := string(captured)

			hasMaxTokens := strings.Contains(wireStr, `"max_tokens"`)
			hasMaxCompletionTokens := strings.Contains(wireStr, `"max_completion_tokens"`)

			if hasMaxTokens != tt.wantMaxTokens {
				t.Errorf("max_tokens present=%v, want %v\nwire: %s", hasMaxTokens, tt.wantMaxTokens, wireStr)
			}
			if hasMaxCompletionTokens != tt.wantMaxCompletionToken {
				t.Errorf("max_completion_tokens present=%v, want %v\nwire: %s", hasMaxCompletionTokens, tt.wantMaxCompletionToken, wireStr)
			}

			// When temperature is nil (default), it should be omitted from wire.
			// When explicitly set, non-reasoning models on non-OpenAI/xAI providers should include it.
		})
	}
}

// TestTemperatureNilOmitted verifies that nil temperature is not sent on the wire.
func TestTemperatureNilOmitted(t *testing.T) {
	adapters := []struct {
		provider string
		model    string
		checkKey string
	}{
		{"openai", "gpt-4.1", `"temperature"`},
		{"anthropic", "claude-sonnet-4-6", `"temperature"`},
		{"gemini", "gemini-3.1-flash-lite", `"temperature"`},
	}

	for _, tt := range adapters {
		t.Run(tt.provider, func(t *testing.T) {
			var captured []byte
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				captured = body
				w.Header().Set("Content-Type", "application/json")
				switch tt.provider {
				case "anthropic":
					_, _ = w.Write([]byte(`{"id":"msg-1","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":2}}`))
				case "gemini":
					_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2}}`))
				default:
					_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"` + tt.model + `","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2}}`))
				}
			}))
			defer srv.Close()

			cfg := llm.Config{
				Provider:  tt.provider,
				Model:     tt.model,
				APIKey:    "test-key",
				BaseURL:   srv.URL,
				MaxTokens: 100,
				Timeout:   5 * time.Second,
				// Temperature is nil (default) — should be omitted
			}
			client, err := llm.NewClient(cfg)
			if err != nil {
				t.Fatalf("NewClient: %v", err)
			}
			defer client.Close()

			_, err = client.Complete(context.Background(), llm.Request{
				Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
			})
			if err != nil {
				t.Fatalf("Complete: %v", err)
			}

			if strings.Contains(string(captured), tt.checkKey) {
				t.Errorf("nil temperature should be omitted from wire for %s\nwire: %s", tt.provider, string(captured))
			}
		})
	}
}

// TestTemperatureExplicitZero verifies that *float64(0.0) IS sent on the wire.
func TestTemperatureExplicitZero(t *testing.T) {
	var captured []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured = body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4.1","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2}}`))
	}))
	defer srv.Close()

	zero := 0.0
	cfg := llm.Config{
		Provider:    "openai",
		Model:       "gpt-4.1",
		APIKey:      "test-key",
		BaseURL:     srv.URL,
		MaxTokens:   100,
		Temperature: &zero,
		Timeout:     5 * time.Second,
	}
	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	_, err = client.Complete(context.Background(), llm.Request{
		Messages:    []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
		Temperature: &zero,
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	if !strings.Contains(string(captured), `"temperature"`) {
		t.Errorf("explicit zero temperature should be present on wire\nwire: %s", string(captured))
	}
}
