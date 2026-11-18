package gemini

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// hopServer answers each request with the next scripted body (the last one
// repeats) and records the decoded request bodies.
type hopServer struct {
	t      *testing.T
	bodies []string
	status []int // optional per-hop status; 0 = 200
	mu     sync.Mutex
	reqs   []map[string]any
}

func (h *hopServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	raw, _ := io.ReadAll(r.Body)
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		h.t.Errorf("request body is not JSON: %v", err)
	}
	h.mu.Lock()
	h.reqs = append(h.reqs, got)
	n := len(h.reqs) - 1
	h.mu.Unlock()
	i := min(n, len(h.bodies)-1)
	if n < len(h.status) && h.status[n] != 0 {
		w.WriteHeader(h.status[n])
	}
	if strings.Contains(r.URL.RawQuery, "alt=sse") {
		w.Header().Set("Content-Type", "text/event-stream")
	}
	_, _ = io.WriteString(w, h.bodies[i])
}

func (h *hopServer) requests() []map[string]any {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]map[string]any(nil), h.reqs...)
}

func newHopClient(t *testing.T, h *hopServer, model string) *Client {
	t.Helper()
	h.t = t
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	c, err := New(llm.Config{Provider: "gemini", Model: model, APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func userReq() llm.Request {
	return llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "prove it")}, MaxTokens: 262144}
}

// A two-hop turn: the second request is the first plus the token (no partial
// output in contents), parts are joined and usage is summed.
func TestLongDecodeCompleteJoinsHops(t *testing.T) {
	h := &hopServer{bodies: []string{
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"think ","thought":true},{"text":"Part one. "}]},"finishReason":"CONTINUATION","continuationToken":"T1"}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":5,"thoughtsTokenCount":30,"totalTokenCount":135}}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"Part two."}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":95,"candidatesTokenCount":7,"thoughtsTokenCount":10,"totalTokenCount":117}}`,
	}}
	c := newHopClient(t, h, "gemini-4-carbon")
	resp, err := c.Complete(t.Context(), userReq())
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "Part one. Part two." || resp.Thinking != "think " {
		t.Errorf("content %q thinking %q", resp.Content, resp.Thinking)
	}
	if resp.StopReason != llm.StopEndTurn || resp.RawStopReason != "STOP" {
		t.Errorf("stop %q raw %q, want end_turn/STOP", resp.StopReason, resp.RawStopReason)
	}
	// 200 prompt tokens over two hops, 95 of them cache reads.
	if resp.InputTokens != 105 || resp.CacheReadTokens != 95 || resp.OutputTokens != 12 || resp.ThinkingTokens != 40 {
		t.Errorf("usage in=%d cache=%d out=%d think=%d, want 105/95/12/40",
			resp.InputTokens, resp.CacheReadTokens, resp.OutputTokens, resp.ThinkingTokens)
	}
	reqs := h.requests()
	if len(reqs) != 2 {
		t.Fatalf("%d requests, want 2", len(reqs))
	}
	if _, ok := reqs[0]["continuationToken"]; ok {
		t.Error("first hop sent a continuationToken")
	}
	if reqs[1]["continuationToken"] != "T1" {
		t.Errorf("second hop continuationToken = %v, want T1 at the top level", reqs[1]["continuationToken"])
	}
	first, _ := json.Marshal(reqs[0]["contents"])
	second, _ := json.Marshal(reqs[1]["contents"])
	if string(first) != string(second) {
		t.Errorf("contents changed between hops (partial output must not be added):\n%s\n%s", first, second)
	}
	gc0, _ := json.Marshal(reqs[0]["generationConfig"])
	gc1, _ := json.Marshal(reqs[1]["generationConfig"])
	if string(gc0) != string(gc1) {
		t.Errorf("generationConfig changed between hops:\n%s\n%s", gc0, gc1)
	}
}

// A server that never stops answering CONTINUATION is bounded, and the turn
// is reported as truncated rather than failed or successful.
func TestLongDecodeCompleteHopBound(t *testing.T) {
	h := &hopServer{bodies: []string{
		`{"candidates":[{"content":{"parts":[{"text":"x"}]},"finishReason":"CONTINUATION","continuationToken":"again"}]}`,
	}}
	c := newHopClient(t, h, "gemini-4-hydrogen")
	resp, err := c.Complete(t.Context(), userReq())
	if err != nil {
		t.Fatal(err)
	}
	if n := len(h.requests()); n != maxContinuationHops {
		t.Errorf("%d requests, want the bound %d", n, maxContinuationHops)
	}
	if resp.StopReason != llm.StopMaxTokens || resp.RawStopReason != finishContinuation {
		t.Errorf("stop %q raw %q, want max_tokens/CONTINUATION", resp.StopReason, resp.RawStopReason)
	}
	if resp.Content != strings.Repeat("x", maxContinuationHops) {
		t.Errorf("content length %d, want %d", len(resp.Content), maxContinuationHops)
	}
	if resp.InputTokens != llm.TokensNotReported {
		t.Errorf("no hop reported usage, InputTokens = %d", resp.InputTokens)
	}
}

// CONTINUATION without a token cannot be continued: one request, truncated.
func TestLongDecodeCompleteNoToken(t *testing.T) {
	h := &hopServer{bodies: []string{
		`{"candidates":[{"content":{"parts":[{"text":"x"}]},"finishReason":"CONTINUATION"}]}`,
	}}
	c := newHopClient(t, h, "gemini-4-carbon")
	resp, err := c.Complete(t.Context(), userReq())
	if err != nil {
		t.Fatal(err)
	}
	if n := len(h.requests()); n != 1 {
		t.Errorf("%d requests, want 1", n)
	}
	if resp.StopReason != llm.StopMaxTokens {
		t.Errorf("stop %q, want max_tokens", resp.StopReason)
	}
}

// A failing later hop fails the turn: no half answer comes back as success.
func TestLongDecodeCompleteLaterHopFails(t *testing.T) {
	h := &hopServer{
		bodies: []string{
			`{"candidates":[{"content":{"parts":[{"text":"x"}]},"finishReason":"CONTINUATION","continuationToken":"T1"}]}`,
			`{"error":{"code":400,"message":"bad token","status":"INVALID_ARGUMENT"}}`,
		},
		status: []int{0, http.StatusBadRequest},
	}
	c := newHopClient(t, h, "gemini-4-carbon")
	resp, err := c.Complete(t.Context(), userReq())
	if err == nil {
		t.Fatalf("got %+v, want the second hop's error", resp)
	}
	var apiErr *llm.APIError
	if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusBadRequest {
		t.Errorf("err = %v, want an APIError 400", err)
	}
}

// A later hop with no candidate (blocked) ends the turn as a refusal, not as
// the earlier hop's text with a stale CONTINUATION.
func TestLongDecodeCompleteLaterHopBlocked(t *testing.T) {
	h := &hopServer{bodies: []string{
		`{"candidates":[{"content":{"parts":[{"text":"x"}]},"finishReason":"CONTINUATION","continuationToken":"T1"}]}`,
		`{"promptFeedback":{"blockReason":"SAFETY"}}`,
	}}
	c := newHopClient(t, h, "gemini-4-carbon")
	resp, err := c.Complete(t.Context(), userReq())
	if err != nil {
		t.Fatal(err)
	}
	if resp.StopReason != llm.StopRefusal {
		t.Errorf("stop %q, want refusal", resp.StopReason)
	}
}

// Tool calls in different hops keep distinct IDs, and a tool call still
// turns the final STOP into tool_use.
func TestLongDecodeCompleteToolCallsAcrossHops(t *testing.T) {
	h := &hopServer{bodies: []string{
		`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{"a":1}},"thoughtSignature":"sig"}]},"finishReason":"CONTINUATION","continuationToken":"T1"}]}`,
		`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{"a":2}}}]},"finishReason":"STOP"}]}`,
	}}
	c := newHopClient(t, h, "gemini-4-carbon")
	resp, err := c.Complete(t.Context(), userReq())
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.ToolCalls) != 2 || resp.ToolCalls[0].ID == resp.ToolCalls[1].ID {
		t.Fatalf("tool calls %+v, want two with distinct IDs", resp.ToolCalls)
	}
	if resp.ToolCalls[0].ThoughtSignature != "sig" || resp.ToolCalls[1].ThoughtSignature != "" {
		t.Errorf("signatures %q %q: only the first call carries one", resp.ToolCalls[0].ThoughtSignature, resp.ToolCalls[1].ThoughtSignature)
	}
	if resp.StopReason != llm.StopToolUse {
		t.Errorf("stop %q, want tool_use", resp.StopReason)
	}
}

func sse(chunks ...string) string {
	var b strings.Builder
	for _, c := range chunks {
		fmt.Fprintf(&b, "data: %s\n\n", c)
	}
	return b.String()
}

func collect(t *testing.T, c *Client, req llm.Request) ([]llm.StreamEvent, error) {
	t.Helper()
	var events []llm.StreamEvent
	for ev, err := range c.Stream(t.Context(), req) {
		if err != nil {
			return events, err
		}
		events = append(events, ev)
	}
	return events, nil
}

// A streamed two-hop turn: events from both hops, one EventDone with the
// summed usage, a mid-stream checkpoint superseded by the final token.
func TestLongDecodeStreamJoinsHops(t *testing.T) {
	h := &hopServer{bodies: []string{
		sse(`{"candidates":[{"content":{"parts":[{"text":"a"}]}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":1}}`,
			`{"candidates":[{"content":{"parts":[{"text":"b"}]},"continuationToken":"CHECKPOINT"}]}`,
			`{"candidates":[{"content":{"parts":[{"text":""}]},"finishReason":"CONTINUATION","continuationToken":"T1"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2}}`),
		sse(`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{}}}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"c"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":3}}`),
	}}
	c := newHopClient(t, h, "gemini-4-carbon")
	events, err := collect(t, c, userReq())
	if err != nil {
		t.Fatal(err)
	}
	var text string
	var dones []llm.StreamEvent
	for _, ev := range events {
		switch ev.Type {
		case llm.EventContent:
			text += ev.Text
		case llm.EventDone:
			dones = append(dones, ev)
		}
	}
	if text != "abc" {
		t.Errorf("text %q, want abc", text)
	}
	if len(dones) != 1 {
		t.Fatalf("%d EventDone, want exactly 1: %+v", len(dones), dones)
	}
	if d := dones[0]; d.StopReason != llm.StopToolUse || d.InputTokens != 20 || d.OutputTokens != 5 {
		t.Errorf("done %+v, want tool_use with input 20, output 5", d)
	}
	reqs := h.requests()
	if len(reqs) != 2 || reqs[1]["continuationToken"] != "T1" {
		t.Fatalf("requests %v, want 2 with the final token T1", reqs)
	}
}

// A consumer that stops during the first hop starts no second request.
func TestLongDecodeStreamConsumerBreak(t *testing.T) {
	h := &hopServer{bodies: []string{
		sse(`{"candidates":[{"content":{"parts":[{"text":"a"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"b"}]},"finishReason":"CONTINUATION","continuationToken":"T1"}]}`),
	}}
	c := newHopClient(t, h, "gemini-4-carbon")
	for range c.Stream(t.Context(), userReq()) {
		break
	}
	if n := len(h.requests()); n != 1 {
		t.Errorf("%d requests after the consumer stopped, want 1", n)
	}
}

// A stream that drops during a later hop is an error, not a quiet end.
func TestLongDecodeStreamLaterHopDrops(t *testing.T) {
	h := &hopServer{bodies: []string{
		sse(`{"candidates":[{"content":{"parts":[{"text":"a"}]},"finishReason":"CONTINUATION","continuationToken":"T1"}]}`),
		sse(`{"candidates":[{"content":{"parts":[{"text":"b"}]}}]}`),
	}}
	c := newHopClient(t, h, "gemini-4-carbon")
	_, err := collect(t, c, userReq())
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Errorf("err = %v, want io.ErrUnexpectedEOF", err)
	}
}

// Usage reported by only one hop is that hop's, and a turn whose hops never
// report usage stays TokensNotReported (not 0, not negative).
func TestLongDecodeStreamPartialUsage(t *testing.T) {
	cases := map[string]struct {
		second  string
		wantIn  int
		wantOut int
	}{
		"only last hop": {`{"candidates":[{"content":{"parts":[{"text":"b"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":3}}`, 7, 3},
		"no hop":        {`{"candidates":[{"content":{"parts":[{"text":"b"}]},"finishReason":"STOP"}]}`, llm.TokensNotReported, llm.TokensNotReported},
		"negative":      {`{"candidates":[{"content":{"parts":[{"text":"b"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":-5,"candidatesTokenCount":-1}}`, 0, 0},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			h := &hopServer{bodies: []string{
				sse(`{"candidates":[{"content":{"parts":[{"text":"a"}]},"finishReason":"CONTINUATION","continuationToken":"T1"}]}`),
				sse(tc.second),
			}}
			c := newHopClient(t, h, "gemini-4-carbon")
			events, err := collect(t, c, userReq())
			if err != nil {
				t.Fatal(err)
			}
			done := events[len(events)-1]
			if done.Type != llm.EventDone || done.InputTokens != tc.wantIn || done.OutputTokens != tc.wantOut {
				t.Errorf("done %+v, want input %d output %d", done, tc.wantIn, tc.wantOut)
			}
		})
	}
}

// The single-hop parser (and a hop that may not continue) reports
// CONTINUATION as a truncated turn.
func TestParseStreamContinuationIsMaxTokens(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gemini-4-carbon"}, providerName: "gemini"}
	var done llm.StreamEvent
	c.parseStream(strings.NewReader(sse(`{"candidates":[{"content":{"parts":[{"text":"a"}]},"finishReason":"CONTINUATION","continuationToken":"T1"}]}`)),
		func(ev llm.StreamEvent, err error) bool {
			if err != nil {
				t.Fatal(err)
			}
			if ev.Type == llm.EventDone {
				done = ev
			}
			return true
		})
	if done.StopReason != llm.StopMaxTokens || done.RawStopReason != finishContinuation {
		t.Errorf("done %+v, want max_tokens/CONTINUATION", done)
	}
}

// Concurrent long-decode turns on one client keep their hops apart (-race).
func TestLongDecodeConcurrent(t *testing.T) {
	var calls atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var req struct {
			Contents          []geminiContent `json:"contents"`
			ContinuationToken string          `json:"continuationToken"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		who := req.Contents[0].Parts[0].Text
		if req.ContinuationToken == "" {
			fmt.Fprintf(w, `{"candidates":[{"content":{"parts":[{"text":"%s-1 "}]},"finishReason":"CONTINUATION","continuationToken":"tok-%s"}]}`, who, who)
			return
		}
		if req.ContinuationToken != "tok-"+who {
			t.Errorf("request %q carried token %q", who, req.ContinuationToken)
		}
		fmt.Fprintf(w, `{"candidates":[{"content":{"parts":[{"text":"%s-2"}]},"finishReason":"STOP"}]}`, who)
	}))
	defer srv.Close()
	c, err := New(llm.Config{Provider: "gemini", Model: "gemini-4-carbon", APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := range 16 {
		wg.Go(func() {
			who := fmt.Sprintf("r%d", i)
			resp, err := c.Complete(t.Context(), llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, who)}})
			if err != nil {
				t.Error(err)
				return
			}
			if resp.Content != who+"-1 "+who+"-2" {
				t.Errorf("%s got %q", who, resp.Content)
			}
		})
	}
	wg.Wait()
	if calls.Load() != 32 {
		t.Errorf("%d calls, want 32", calls.Load())
	}
}
