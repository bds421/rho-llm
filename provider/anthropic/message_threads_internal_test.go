package anthropic

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// Message threads (beta MessageThreadsBeta): create sends everything and a
// thread marker; continue sends only new messages, leaves system and tools
// off the wire (a 400 even as null) and keeps every fixed field identical.

func threadClient(t *testing.T, cfg llm.Config, reply string) (*Client, *capture) {
	t.Helper()
	srv, cp := bindingServer(t, reply)
	cfg.APIKey, cfg.BaseURL = "k", srv.URL
	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return c, cp
}

var threadTool = llm.Tool{Name: "f", Description: "d", InputSchema: map[string]any{"type": "object"}}

func threadReq(th *llm.Thread, msgs ...llm.Message) llm.Request {
	return llm.Request{
		System: "sys", Tools: []llm.Tool{threadTool}, ToolChoice: &llm.ToolChoice{Mode: llm.ToolChoiceNone},
		Messages: msgs, Thread: th,
	}
}

func TestThreadCreateWire(t *testing.T) {
	c, cp := threadClient(t, llm.Config{Model: "claude-opus-5-5"}, okReply)
	if _, err := c.Complete(t.Context(), threadReq(&llm.Thread{}, llm.NewTextMessage(llm.RoleUser, "q"))); err != nil {
		t.Fatal(err)
	}
	beta, body := cp.last(t)
	if !reflect.DeepEqual(body["thread"], map[string]any{"type": "create"}) {
		t.Errorf("thread = %v", body["thread"])
	}
	if body["system"] == nil || body["tools"] == nil {
		t.Errorf("create must carry system and tools: %v", body)
	}
	if !strings.Contains(beta, MessageThreadsBeta) {
		t.Errorf("anthropic-beta = %q", beta)
	}
}

func TestThreadContinueWire(t *testing.T) {
	c, cp := threadClient(t, llm.Config{Model: "claude-opus-5-5"}, okReply)
	req := threadReq(&llm.Thread{PreviousMessageID: "msg_01"}, llm.NewTextMessage(llm.RoleUser, "next"))
	req.ContainerID = "container_1"
	if _, err := c.Complete(t.Context(), req); err != nil {
		t.Fatal(err)
	}
	beta, body := cp.last(t)
	if !reflect.DeepEqual(body["thread"], map[string]any{"type": "continue", "previous_message_id": "msg_01"}) {
		t.Errorf("thread = %v", body["thread"])
	}
	for _, key := range []string{"system", "tools"} {
		if _, present := body[key]; present {
			t.Errorf("a continuation must not send %q, even as null", key)
		}
	}
	if msgs, _ := body["messages"].([]any); len(msgs) != 1 {
		t.Errorf("messages = %v", body["messages"])
	}
	// Fixed fields ride every request: tool_choice is built from the tools the
	// wire body leaves out, and the container is never stored by the API.
	if tc, _ := body["tool_choice"].(map[string]any); tc["type"] != "none" {
		t.Errorf("tool_choice = %v", body["tool_choice"])
	}
	if body["container"] != "container_1" {
		t.Errorf("container = %v", body["container"])
	}
	if !strings.Contains(beta, MessageThreadsBeta) {
		t.Errorf("anthropic-beta = %q", beta)
	}
}

// A paused turn resumes with "messages": [] (not null).
func TestThreadContinueWithNoMessagesSendsAnEmptyList(t *testing.T) {
	c, cp := threadClient(t, llm.Config{Model: "claude-opus-5-5"}, okReply)
	if _, err := c.Complete(t.Context(), threadReq(&llm.Thread{PreviousMessageID: "msg_01"})); err != nil {
		t.Fatal(err)
	}
	_, body := cp.last(t)
	msgs, ok := body["messages"].([]any)
	if !ok || len(msgs) != 0 {
		t.Errorf("messages = %#v, want []", body["messages"])
	}
}

// New system text cannot ride a continuation: refuse it rather than drop the
// instruction or rewrite the thread's prompt. Nothing is sent.
func TestThreadContinueRefusesSystemMessages(t *testing.T) {
	c, cp := threadClient(t, llm.Config{Model: "claude-opus-5-5"}, okReply)
	_, err := c.Complete(t.Context(), threadReq(&llm.Thread{PreviousMessageID: "msg_01"},
		llm.NewSystemMessage("new rules"), llm.NewTextMessage(llm.RoleUser, "q")))
	if err == nil || !strings.Contains(err.Error(), "system") {
		t.Fatalf("err = %v", err)
	}
	if len(cp.bodies) != 0 {
		t.Error("a request was sent")
	}
	// On create the same message is hoisted into system as always.
	if _, err := c.Complete(t.Context(), threadReq(&llm.Thread{}, llm.NewSystemMessage("rules"), llm.NewTextMessage(llm.RoleUser, "q"))); err != nil {
		t.Fatalf("create with a system message: %v", err)
	}
}

// Without a Thread nothing about the request changes: no marker, no beta.
func TestNoThreadLeavesRequestsUnchanged(t *testing.T) {
	c, cp := threadClient(t, llm.Config{Model: "claude-opus-5-5", BetaFeatures: []string{"other-beta-1"}}, okReply)
	if _, err := c.Complete(t.Context(), threadReq(nil, llm.NewTextMessage(llm.RoleUser, "q"))); err != nil {
		t.Fatal(err)
	}
	beta, body := cp.last(t)
	if _, present := body["thread"]; present {
		t.Errorf("thread key present: %v", body["thread"])
	}
	if beta != "other-beta-1" {
		t.Errorf("anthropic-beta = %q", beta)
	}
}

func TestThreadBetaJoinsConfiguredBetasOnce(t *testing.T) {
	for _, configured := range [][]string{nil, {"other-beta-1"}, {MessageThreadsBeta}, {"a", MessageThreadsBeta, "b"}} {
		c, cp := threadClient(t, llm.Config{Model: "claude-opus-5-5", BetaFeatures: configured}, okReply)
		if _, err := c.Complete(t.Context(), threadReq(&llm.Thread{}, llm.NewTextMessage(llm.RoleUser, "q"))); err != nil {
			t.Fatal(err)
		}
		beta, _ := cp.last(t)
		if strings.Count(beta, MessageThreadsBeta) != 1 {
			t.Errorf("configured %v: anthropic-beta = %q", configured, beta)
		}
		for _, want := range configured {
			if !strings.Contains(beta, want) {
				t.Errorf("configured %v: %q dropped from %q", configured, want, beta)
			}
		}
	}
}

// thinking and the beta set are fixed fields: a continuation that differs from
// its create is a thread_fingerprint_mismatch 400. The create here replays no
// thinking and the continuation sends none, yet both must carry block_binding.
func TestThreadFixedFieldsMatchBetweenCreateAndContinue(t *testing.T) {
	for _, model := range []string{"claude-opus-5-5", "claude-fable-5-2"} {
		for _, level := range []llm.ThinkingLevel{llm.ThinkingNone, llm.ThinkingHigh} {
			t.Run(model+"/"+string(level), func(t *testing.T) {
				c, cp := threadClient(t, llm.Config{Model: model}, okReply)
				mk := func(th *llm.Thread) (string, map[string]any) {
					req := threadReq(th, llm.NewTextMessage(llm.RoleUser, "q"))
					req.ThinkingLevel = level
					req.Effort = ""
					if _, err := c.Complete(t.Context(), req); err != nil {
						t.Fatal(err)
					}
					return cp.last(t)
				}
				cb, cbody := mk(&llm.Thread{})
				kb, kbody := mk(&llm.Thread{PreviousMessageID: "msg_01"})
				if cb != kb {
					t.Errorf("anthropic-beta differs: create %q, continue %q", cb, kb)
				}
				for _, field := range []string{"thinking", "output_config", "tool_choice"} {
					if !reflect.DeepEqual(cbody[field], kbody[field]) {
						t.Errorf("%s differs: create %v, continue %v", field, cbody[field], kbody[field])
					}
				}
				if _, bound := binding(cbody); bound == nil {
					t.Errorf("thinking carries no block_binding on a thread: %v", cbody["thinking"])
				}
			})
		}
	}
}

func TestThreadWithBindingOffAddsNoBinding(t *testing.T) {
	c, cp := threadClient(t, llm.Config{Model: "claude-opus-5-5", ThinkingBinding: llm.ThinkingBindingOff}, okReply)
	if _, err := c.Complete(t.Context(), threadReq(&llm.Thread{}, llm.NewTextMessage(llm.RoleUser, "q"))); err != nil {
		t.Fatal(err)
	}
	beta, body := cp.last(t)
	if _, bound := binding(body); bound != nil || strings.Contains(beta, ThinkingBindingBeta) {
		t.Errorf("binding off: thinking %v beta %q", body["thinking"], beta)
	}
}

// Budget and between_tools thinking reject block_binding; a thread must not
// force it on them.
func TestThreadDoesNotAddBindingWhereItIsRejected(t *testing.T) {
	cases := map[string]llm.Config{
		"between_tools": {Model: "claude-sonnet-5-5"},
		"budget":        {Model: "claude-sonnet-4-6", MaxTokens: 16000},
	}
	for name, cfg := range cases {
		c, cp := threadClient(t, cfg, okReply)
		req := threadReq(&llm.Thread{}, llm.NewTextMessage(llm.RoleUser, "q"))
		if name == "budget" {
			req.ThinkingLevel = llm.ThinkingLow
		}
		if _, err := c.Complete(t.Context(), req); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		beta, body := cp.last(t)
		if _, bound := binding(body); bound != nil || strings.Contains(beta, ThinkingBindingBeta) {
			t.Errorf("%s: thinking %v beta %q", name, body["thinking"], beta)
		}
	}
}

func TestThreadIDsAreEscapedOnTheWire(t *testing.T) {
	c, cp := threadClient(t, llm.Config{Model: "claude-opus-5-5"}, okReply)
	hostile := "msg\"}],\"system\":\"x\n </script>"
	if _, err := c.Complete(t.Context(), threadReq(&llm.Thread{PreviousMessageID: hostile}, llm.NewTextMessage(llm.RoleUser, "q"))); err != nil {
		t.Fatal(err)
	}
	_, body := cp.last(t)
	th, _ := body["thread"].(map[string]any)
	if th["previous_message_id"] != hostile {
		t.Errorf("id did not round-trip: %v", th)
	}
	if _, present := body["system"]; present {
		t.Error("the id injected a system field")
	}
}

func TestThreadResponseIDOnCompleteAndStream(t *testing.T) {
	c, _ := threadClient(t, llm.Config{Model: "claude-opus-5-5"}, strings.Replace(okReply, `"id":"m"`, `"id":"msg_complete"`, 1))
	resp, err := c.Complete(t.Context(), threadReq(&llm.Thread{}, llm.NewTextMessage(llm.RoleUser, "q")))
	if err != nil || resp.ID != "msg_complete" {
		t.Fatalf("complete: %+v %v", resp, err)
	}

	sse := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_streamed\",\"model\":\"claude-opus-5-5\",\"usage\":{\"input_tokens\":3}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	sc, _ := threadClient(t, llm.Config{Model: "claude-opus-5-5"}, sse)
	got, err := llm.CollectStream(sc.Stream(t.Context(), threadReq(&llm.Thread{}, llm.NewTextMessage(llm.RoleUser, "q"))))
	if err != nil || got.ID != "msg_streamed" || got.Content != "hi" {
		t.Fatalf("stream: %+v %v", got, err)
	}
}

// A stream that breaks after message_start still reports the ID, so the
// caller can tell which turn the API may hold.
func TestThreadStreamErrorCarriesTheID(t *testing.T) {
	sse := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_broken\",\"model\":\"m\",\"usage\":{\"input_tokens\":3}}}\n\n" +
		"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"x\"}}\n\n"
	c, _ := threadClient(t, llm.Config{Model: "claude-opus-5-5"}, sse)
	var sawErr llm.StreamEvent
	for ev, err := range c.Stream(t.Context(), threadReq(&llm.Thread{}, llm.NewTextMessage(llm.RoleUser, "q"))) {
		if err != nil {
			sawErr = ev
		}
	}
	if sawErr.ID != "msg_broken" {
		t.Errorf("error event = %+v", sawErr)
	}
}

// The API's thread errors come back as classifiable APIErrors on both paths.
func TestThreadErrorsAreClassified(t *testing.T) {
	body := `{"type":"error","error":{"type":"not_found_error","message":"gone","details":{"error_code":"thread_not_found"}}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	c, err := New(llm.Config{Model: "claude-opus-5-5", APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	req := threadReq(&llm.Thread{PreviousMessageID: "msg_01"}, llm.NewTextMessage(llm.RoleUser, "q"))
	_, err = c.Complete(t.Context(), req)
	if got := llm.ThreadErrorCode(err); got != llm.ThreadNotFound {
		t.Errorf("Complete: code %q from %v", got, err)
	}
	for _, err := range c.Stream(t.Context(), req) {
		if err != nil {
			if got := llm.ThreadErrorCode(err); got != llm.ThreadNotFound {
				t.Errorf("Stream: code %q from %v", got, err)
			}
			return
		}
	}
	t.Error("stream produced no error")
}

func TestThreadsAreRefusedInBatches(t *testing.T) {
	c, err := New(llm.Config{Model: "claude-opus-5-5", APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.BuildMessageBatchParams(threadReq(&llm.Thread{}, llm.NewTextMessage(llm.RoleUser, "q"))); err == nil {
		t.Error("a batch item with a thread is rejected by the API; refuse it before upload")
	}
	raw, err := c.BuildMessageBatchParams(threadReq(nil, llm.NewTextMessage(llm.RoleUser, "q")))
	if err != nil || strings.Contains(string(raw), "thread") {
		t.Errorf("plain batch params: %s %v", raw, err)
	}
	var probe map[string]any
	if json.Unmarshal(raw, &probe) != nil {
		t.Error("params are not JSON")
	}
}
