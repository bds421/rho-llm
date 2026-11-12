package anthropic

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// signedHistory is a tool turn whose assistant message carries a signed
// (empty, display "omitted") thinking block, as newer Claude models return.
func signedHistory() []llm.Message {
	return []llm.Message{
		llm.NewTextMessage(llm.RoleUser, "q"),
		{Role: llm.RoleAssistant, Provider: "anthropic", Content: []llm.ContentPart{
			{Type: llm.ContentThinking, ThinkingSignature: "sig-1"},
			{Type: llm.ContentToolUse, ToolUseID: "toolu_1", ToolName: "f", ToolInput: map[string]any{}},
		}},
		llm.NewToolResultMessage("toolu_1", "ok", false),
	}
}

type capture struct {
	mu     sync.Mutex
	beta   []string
	bodies []map[string]any
}

func (c *capture) last(t *testing.T) (string, map[string]any) {
	t.Helper()
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.bodies) == 0 {
		t.Fatal("no request reached the server")
	}
	return c.beta[len(c.beta)-1], c.bodies[len(c.bodies)-1]
}

func bindingServer(t *testing.T, reply string) (*httptest.Server, *capture) {
	t.Helper()
	cp := &capture{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		cp.mu.Lock()
		cp.beta = append(cp.beta, r.Header.Get("anthropic-beta"))
		cp.bodies = append(cp.bodies, body)
		cp.mu.Unlock()
		if strings.HasPrefix(reply, "event:") {
			w.Header().Set("Content-Type", "text/event-stream")
		}
		_, _ = io.WriteString(w, reply)
	}))
	t.Cleanup(srv.Close)
	return srv, cp
}

const okReply = `{"id":"m","model":"claude-opus-5-5","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1},"content":[{"type":"text","text":"hi"}]}`

func binding(body map[string]any) (thinkingType string, behavior any) {
	th, _ := body["thinking"].(map[string]any)
	if th == nil {
		return "", nil
	}
	thinkingType, _ = th["type"].(string)
	if bb, ok := th["block_binding"].(map[string]any); ok {
		return thinkingType, bb["prefix_mismatch_behavior"]
	}
	return thinkingType, nil
}

// Newer Claude models answer a 400 when replayed thinking belongs to a changed
// conversation; this adapter itself changes it (it hoists RoleSystem
// messages). The request now asks for drop_block whenever it replays signed
// thinking under adaptive thinking (P1).
func TestThinkingBindingSentOnlyWhenReplaying(t *testing.T) {
	srv, cp := bindingServer(t, okReply)
	cases := []struct {
		name      string
		cfg       llm.Config
		req       llm.Request
		wantType  string
		wantBind  any
		wantBetaH bool
	}{
		{"always-on model, no level", llm.Config{Model: "claude-opus-5-5"}, llm.Request{Messages: signedHistory()}, "adaptive", "drop_block", true},
		{"adaptive level", llm.Config{Model: "claude-opus-5-5"}, llm.Request{ThinkingLevel: llm.ThinkingHigh, Messages: signedHistory()}, "adaptive", "drop_block", true},
		{"error mode", llm.Config{Model: "claude-fable-5-2", ThinkingBinding: llm.ThinkingBindingError}, llm.Request{Messages: signedHistory()}, "adaptive", "error", true},
		{"off", llm.Config{Model: "claude-opus-5-5", ThinkingBinding: llm.ThinkingBindingOff}, llm.Request{Messages: signedHistory()}, "", nil, false},
		{"nothing replayed", llm.Config{Model: "claude-opus-5-5"}, llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "q")}}, "", nil, false},
		// between_tools and budget thinking reject block_binding.
		{"between_tools", llm.Config{Model: "claude-sonnet-5-5"}, llm.Request{Messages: signedHistory()}, "between_tools", nil, false},
		{"budget thinking", llm.Config{Model: "claude-sonnet-4-6", MaxTokens: 16000}, llm.Request{ThinkingLevel: llm.ThinkingLow, Messages: signedHistory()}, "enabled", nil, false},
		// Unsigned thinking is not replayed, so nothing can be bound.
		{"unsigned thinking", llm.Config{Model: "claude-opus-5-5"}, llm.Request{Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "q"),
			{Role: llm.RoleAssistant, Content: []llm.ContentPart{{Type: llm.ContentThinking, Thinking: "x"}, {Type: llm.ContentText, Text: "a"}}},
			llm.NewTextMessage(llm.RoleUser, "q2"),
		}}, "", nil, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.cfg.APIKey, tc.cfg.BaseURL = "k", srv.URL
			c, err := New(tc.cfg)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := c.Complete(t.Context(), tc.req); err != nil {
				t.Fatal(err)
			}
			beta, body := cp.last(t)
			gotType, gotBind := binding(body)
			if gotType != tc.wantType || gotBind != tc.wantBind {
				t.Errorf("thinking = %q binding = %v, want %q / %v", gotType, gotBind, tc.wantType, tc.wantBind)
			}
			if has := strings.Contains(beta, ThinkingBindingBeta); has != tc.wantBetaH {
				t.Errorf("anthropic-beta = %q, binding beta wanted: %v", beta, tc.wantBetaH)
			}
		})
	}
}

// The field without the beta is a 400, so the beta must go out even when the
// caller cleared BetaFeatures, and only once when the caller lists it too.
func TestThinkingBindingBetaNeverMissingOrDoubled(t *testing.T) {
	srv, cp := bindingServer(t, okReply)
	for _, betas := range [][]string{nil, {ThinkingBindingBeta}, {"server-side-fallback-2026-07-01", ThinkingBindingBeta}} {
		c, err := New(llm.Config{Model: "claude-opus-5-5", APIKey: "k", BaseURL: srv.URL, BetaFeatures: betas})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := c.Complete(t.Context(), llm.Request{Messages: signedHistory()}); err != nil {
			t.Fatal(err)
		}
		beta, _ := cp.last(t)
		if strings.Count(beta, ThinkingBindingBeta) != 1 {
			t.Errorf("BetaFeatures %v: anthropic-beta = %q", betas, beta)
		}
	}
}

func TestThinkingBindingRefusesUnknownMode(t *testing.T) {
	c := &Client{config: llm.Config{Model: "claude-opus-5-5", ThinkingBinding: "drop"}}
	if _, err := c.buildRequest(llm.Request{Messages: signedHistory()}, false); err == nil || !strings.Contains(err.Error(), "ThinkingBinding") {
		t.Fatalf("err = %v", err)
	}
}

// The drop report reaches the caller on Complete, on Stream's EventDone and on
// the folded stream; a report in an unexpected shape is ignored, not fatal.
func TestInputTransformationsReported(t *testing.T) {
	report := `[{"type":"thinking_dropped","path":"messages.1.content.0","reason":"prefix_binding_mismatch"},{"path":"no type"}]`
	want := []llm.InputTransformation{{Type: "thinking_dropped", Path: "messages.1.content.0", Reason: "prefix_binding_mismatch"}}
	check := func(name string, got []llm.InputTransformation) {
		t.Helper()
		if len(got) != 1 || got[0] != want[0] {
			t.Errorf("%s: transformations = %+v", name, got)
		}
	}

	srv, _ := bindingServer(t, strings.Replace(okReply, `"content":`, `"input_transformations":`+report+`,"content":`, 1))
	c, _ := New(llm.Config{Model: "claude-opus-5-5", APIKey: "k", BaseURL: srv.URL})
	resp, err := c.Complete(t.Context(), llm.Request{Messages: signedHistory()})
	if err != nil {
		t.Fatal(err)
	}
	check("Complete", resp.InputTransformations)

	sse := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-opus-5-5\",\"usage\":{\"input_tokens\":3},\"input_transformations\":" + report + "}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	ssrv, _ := bindingServer(t, sse)
	sc, _ := New(llm.Config{Model: "claude-opus-5-5", APIKey: "k", BaseURL: ssrv.URL})
	var acc llm.StreamAccumulator
	var done *llm.StreamEvent
	for ev, err := range sc.Stream(t.Context(), llm.Request{Messages: signedHistory()}) {
		if err != nil {
			t.Fatal(err)
		}
		acc.Add(ev)
		if ev.Type == llm.EventDone {
			done = &ev
		}
	}
	if done == nil {
		t.Fatal("no EventDone")
	}
	check("EventDone", done.InputTransformations)
	check("folded stream", acc.Response().InputTransformations)

	for _, bad := range []string{`"input_transformations":"x",`, `"input_transformations":{"a":1},`, `"input_transformations":null,`, `"input_transformations":[],`} {
		bsrv, _ := bindingServer(t, strings.Replace(okReply, `"content":`, bad+`"content":`, 1))
		bc, _ := New(llm.Config{Model: "claude-opus-5-5", APIKey: "k", BaseURL: bsrv.URL})
		resp, err := bc.Complete(t.Context(), llm.Request{Messages: signedHistory()})
		if err != nil || resp.InputTransformations != nil || resp.Content != "hi" {
			t.Errorf("%s: resp=%+v err=%v", bad, resp, err)
		}
	}
}
