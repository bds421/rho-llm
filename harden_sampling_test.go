package llm_test

// Hardening pass 16 (cont.) — SamplingParams passthrough and RawStopReason.
// The threat model for SamplingParams is hijacking: it writes arbitrary keys
// into the request body, so the tests below try to use it to redirect the
// request to another model, rewrite its history, swap its tools, and smuggle
// unserializable values. For RawStopReason the threat is erasure: normalization
// must not destroy what the provider actually said.

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/provider/openaicompat"
)

// SamplingParams must actually reach the wire — otherwise the whole escape
// hatch is decorative and callers silently lose provider routing/tuning.
func TestSamplingParamsReachesTheWire(t *testing.T) {
	for i, p := range tcProtocols {
		t.Run(p.name, func(t *testing.T) {
			got := tcDo(t, i, llm.Request{SamplingParams: map[string]any{
				"top_p":    0.5,
				"seed":     42,
				"provider": map[string]any{"order": []string{"a", "b"}},
			}}, false)
			for _, k := range []string{"top_p", "seed", "provider"} {
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
	if _, err := llm.MergeSamplingParams([]int{1, 2, 3}, map[string]any{"x": 1}); err == nil {
		t.Fatal("merging into a JSON array was accepted")
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
		{"content_filter", "content_filter"}, // unmapped: passes through verbatim
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

// DefaultConfig's model must equal the registry's Anthropic default. The bug
// this guards: config.go hardcodes a model ID that silently drifts from
// defaultModels every time the flagship moves — which is exactly what had
// happened (DefaultConfig said claude-sonnet-4-6 long after the registry
// default moved on).
func TestDefaultConfigTracksRegistryDefault(t *testing.T) {
	got := llm.DefaultConfig().Model
	want := llm.GetDefaultModel("anthropic")
	if got != want {
		t.Errorf("DefaultConfig().Model = %q, registry default = %q — the two have drifted apart", got, want)
	}
	if _, ok := llm.GetModelInfo(got); !ok {
		t.Errorf("DefaultConfig().Model = %q is not in the registry at all", got)
	}
}
