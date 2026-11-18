package geminiinteractions

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

func sse(events ...string) string {
	var b strings.Builder
	for _, e := range events {
		b.WriteString("event: x\ndata: " + e + "\n\n")
	}
	return b.String()
}

func collect(t *testing.T, c *Client, req llm.Request) (evs []llm.StreamEvent, err error) {
	t.Helper()
	for ev, e := range c.Stream(t.Context(), req) {
		if e != nil {
			return evs, e
		}
		evs = append(evs, ev)
	}
	return evs, nil
}

// Live-captured shapes: thought summary + signature, text deltas, usage on
// interaction.completed.
func TestStreamTextThoughtAndUsage(t *testing.T) {
	c, cap := serve(t, 200, sse(
		`{"interaction":{"object":"interaction","model":"gemini-3.8-flash"},"event_type":"interaction.created"}`,
		`{"status":"in_progress","event_type":"interaction.status_update"}`,
		`{"index":0,"step":{"type":"thought"},"event_type":"step.start"}`,
		`{"index":0,"delta":{"content":{"text":"hmm ","type":"text"},"type":"thought_summary"},"event_type":"step.delta"}`,
		`{"index":0,"delta":{"signature":"SIGS","type":"thought_signature"},"event_type":"step.delta"}`,
		`{"index":0,"event_type":"step.stop"}`,
		`{"index":1,"step":{"type":"model_output"},"event_type":"step.start"}`,
		`{"index":1,"delta":{"text":"Hi","type":"text"},"event_type":"step.delta"}`,
		`{"index":1,"delta":{"text":"!","type":"text"},"event_type":"step.delta"}`,
		`{"index":1,"event_type":"step.stop"}`,
		`{"interaction":{"status":"completed","usage":{"total_input_tokens":3,"total_cached_tokens":1,"total_output_tokens":9,"total_thought_tokens":4},"model":"gemini-3.8-flash"},"event_type":"interaction.completed"}`,
		`[DONE]`,
	), "text/event-stream")
	evs, err := collect(t, c, user("hi"))
	if err != nil {
		t.Fatal(err)
	}
	if cap.last()["stream"] != true {
		t.Error("stream flag not sent")
	}
	acc := &llm.StreamAccumulator{}
	for _, ev := range evs {
		acc.Add(ev)
	}
	r := acc.Response()
	if r.Content != "Hi!" || r.Thinking != "hmm " || r.ThinkingSignature != "SIGS" {
		t.Errorf("%+v", r)
	}
	last := evs[len(evs)-1]
	if last.Type != llm.EventDone || last.StopReason != llm.StopEndTurn || last.InputTokens != 2 || last.CacheReadTokens != 1 || last.OutputTokens != 9 || last.ThinkingTokens != 4 {
		t.Errorf("done %+v", last)
	}
}

func TestStreamParallelToolCallsWithSplitArguments(t *testing.T) {
	c, _ := serve(t, 200, sse(
		`{"index":0,"step":{"type":"thought"},"event_type":"step.start"}`,
		`{"index":0,"delta":{"signature":"S","type":"thought_signature"},"event_type":"step.delta"}`,
		`{"index":0,"event_type":"step.stop"}`,
		`{"index":1,"step":{"id":"call_9","type":"function_call","name":"get_weather","arguments":{}},"event_type":"step.start"}`,
		`{"index":1,"delta":{"arguments":"{\"ci","type":"arguments_delta"},"event_type":"step.delta"}`,
		`{"index":1,"delta":{"arguments":"ty\":\"Paris\"}","type":"arguments_delta"},"event_type":"step.delta"}`,
		`{"index":1,"event_type":"step.stop"}`,
		`{"index":2,"step":{"id":"call_8","type":"function_call","name":"get_time","arguments":{}},"event_type":"step.start"}`,
		`{"index":2,"event_type":"step.stop"}`,
		`{"interaction":{"status":"requires_action","usage":{"total_input_tokens":50,"total_output_tokens":32}},"event_type":"interaction.completed"}`,
		`[DONE]`,
	), "text/event-stream")
	evs, err := collect(t, c, user("x"))
	if err != nil {
		t.Fatal(err)
	}
	var calls []*llm.ToolCall
	for _, ev := range evs {
		if ev.Type == llm.EventToolUse {
			calls = append(calls, ev.ToolCall)
		}
	}
	if len(calls) != 2 || calls[0].ID != "call_9" || calls[0].Input.(map[string]any)["city"] != "Paris" {
		t.Fatalf("calls %+v", calls)
	}
	if m, ok := calls[1].Input.(map[string]any); !ok || len(m) != 0 {
		t.Errorf("no-arg call input %#v", calls[1].Input)
	}
	if last := evs[len(evs)-1]; last.StopReason != llm.StopToolUse || last.ThinkingSignature != "S" {
		t.Errorf("done %+v", last)
	}
}

func TestStreamIncompleteMapsToMaxTokens(t *testing.T) {
	c, _ := serve(t, 200, sse(
		`{"index":1,"delta":{"text":"Cats","type":"text"},"event_type":"step.delta"}`,
		`{"interaction":{"status":"incomplete","usage":{"total_input_tokens":9,"total_output_tokens":16}},"event_type":"interaction.completed"}`,
	), "text/event-stream")
	evs, err := collect(t, c, user("x"))
	if err != nil {
		t.Fatal(err)
	}
	if last := evs[len(evs)-1]; last.StopReason != llm.StopMaxTokens || last.RawStopReason != "incomplete" {
		t.Errorf("%+v", last)
	}
}

// A stream that stops before interaction.completed is a retryable truncation,
// not a silent end; a stream without usage reports it as not reported.
func TestStreamTruncationAndMissingUsage(t *testing.T) {
	for name, body := range map[string]string{
		"eof":        sse(`{"index":1,"delta":{"text":"par","type":"text"},"event_type":"step.delta"}`),
		"done-only":  sse(`{"index":1,"delta":{"text":"par","type":"text"},"event_type":"step.delta"}`, `[DONE]`),
		"empty":      "",
		"mid-line":   `data: {"event_type":"step.del`,
		"keepalives": ": ping\n\ndata: \n\n",
	} {
		c, _ := serve(t, 200, body, "text/event-stream")
		_, err := collect(t, c, user("x"))
		if !errors.Is(err, io.ErrUnexpectedEOF) && !strings.Contains(fmt.Sprint(err), "malformed") {
			t.Errorf("%s: err %v, want ErrUnexpectedEOF", name, err)
		}
	}
	c, _ := serve(t, 200, sse(`{"interaction":{"status":"completed"},"event_type":"interaction.completed"}`), "text/event-stream")
	evs, err := collect(t, c, user("x"))
	if err != nil || evs[0].InputTokens != llm.TokensNotReported || evs[0].OutputTokens != llm.TokensNotReported {
		t.Errorf("%+v %v", evs, err)
	}
}

func TestStreamHostileEvents(t *testing.T) {
	// A malformed event is reported and the stream goes on; unknown events, a
	// delta for an unknown index, a step.stop without a start and a null
	// delta or interaction are ignored.
	c, _ := serve(t, 200, sse(
		`{not json`,
		`{"event_type":"mystery.event","x":[1,2]}`,
		`{"index":7,"delta":{"arguments":"{}","type":"arguments_delta"},"event_type":"step.delta"}`,
		`{"index":7,"event_type":"step.stop"}`,
		`{"index":1,"delta":null,"event_type":"step.delta"}`,
		`{"index":1,"delta":{"text":"ok","type":"text"},"event_type":"step.delta"}`,
		`{"interaction":null,"event_type":"interaction.completed"}`,
	), "text/event-stream")
	var errs int
	var text string
	var done bool
	for ev, err := range c.Stream(t.Context(), user("x")) {
		if err != nil {
			errs++
			continue
		}
		text += ev.Text
		done = done || ev.Type == llm.EventDone
	}
	if errs != 1 || text != "ok" || !done {
		t.Errorf("errs=%d text=%q done=%v", errs, text, done)
	}
}

func TestStreamHTTPAndInBandErrors(t *testing.T) {
	c, _ := serve(t, 404, `{"error":{"message":"Model 'nope' not found.","code":"not_found"}}`, "application/json")
	_, err := collect(t, c, user("x"))
	var apiErr *llm.APIError
	if !errors.As(err, &apiErr) || apiErr.StatusCode != 404 {
		t.Errorf("404: %v", err)
	}
	c, _ = serve(t, 200, sse(
		`{"index":1,"delta":{"text":"part","type":"text"},"event_type":"step.delta"}`,
		`{"event_type":"error","error":{"message":"boom sekrit-key","code":"unavailable"}}`,
		`{"index":1,"delta":{"text":"never","type":"text"},"event_type":"step.delta"}`,
	), "text/event-stream")
	evs, err := collect(t, c, user("x"))
	if !errors.As(err, &apiErr) || apiErr.StatusCode != 503 || strings.Contains(apiErr.Error(), "sekrit-key") {
		t.Errorf("in-band: %v", err)
	}
	if len(evs) != 1 {
		t.Errorf("events after error: %+v", evs)
	}
	// failed arrives as a terminal status update: StopError, not a hang.
	c, _ = serve(t, 200, sse(`{"status":"failed","event_type":"interaction.status_update"}`), "text/event-stream")
	evs, err = collect(t, c, user("x"))
	if err != nil || evs[len(evs)-1].StopReason != llm.StopError {
		t.Errorf("failed: %+v %v", evs, err)
	}
}

func TestStreamToolInputLimit(t *testing.T) {
	big := strings.Repeat("a", 4096)
	c, _ := serve(t, 200, sse(
		`{"index":0,"step":{"id":"c","type":"function_call","name":"f","arguments":{}},"event_type":"step.start"}`,
		fmt.Sprintf(`{"index":0,"delta":{"arguments":"{\"x\":\"%s","type":"arguments_delta"},"event_type":"step.delta"}`, big),
	), "text/event-stream")
	c.config.MaxToolInputBytes = 1024
	_, err := collect(t, c, user("x"))
	if err == nil || !strings.Contains(err.Error(), "tool input exceeded") {
		t.Errorf("err %v", err)
	}
}

// Consumer stops early: the stream must stop yielding and not panic.
func TestStreamEarlyBreak(t *testing.T) {
	c, _ := serve(t, 200, sse(
		`{"index":1,"delta":{"text":"a","type":"text"},"event_type":"step.delta"}`,
		`{"index":1,"delta":{"text":"b","type":"text"},"event_type":"step.delta"}`,
		`{"interaction":{"status":"completed"},"event_type":"interaction.completed"}`,
	), "text/event-stream")
	n := 0
	for _, err := range c.Stream(t.Context(), user("x")) {
		if err != nil {
			t.Fatal(err)
		}
		n++
		break
	}
	if n != 1 {
		t.Errorf("n=%d", n)
	}
}

func TestConcurrentCalls(t *testing.T) {
	var hits atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		var m struct {
			Input []struct {
				Content []struct {
					Text string `json:"text"`
				} `json:"content"`
			} `json:"input"`
		}
		_ = json.NewDecoder(r.Body).Decode(&m)
		who := m.Input[0].Content[0].Text
		fmt.Fprintf(w, `{"status":"completed","steps":[{"type":"model_output","content":[{"type":"text","text":%q}]}]}`, "re:"+who)
	}))
	defer srv.Close()
	c, err := New(llm.Config{Provider: "gemini-interactions", Model: "m", APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			who := fmt.Sprintf("w%d", i)
			r, err := c.Complete(t.Context(), user(who))
			if err != nil || r.Content != "re:"+who {
				t.Errorf("%s: %v %+v", who, err, r)
			}
		}()
	}
	wg.Wait()
	if hits.Load() != 20 {
		t.Errorf("hits %d", hits.Load())
	}
}

func TestNewValidatesAndNames(t *testing.T) {
	if _, err := New(llm.Config{Model: "m"}); err == nil {
		t.Error("missing key accepted")
	}
	c, err := New(llm.Config{Model: "m", APIKey: "k"})
	if err != nil || c.Provider() != "gemini-interactions" || c.Model() != "m" || !strings.HasSuffix(c.endpoint, "/v1beta/interactions") {
		t.Errorf("%+v %v", c, err)
	}
	c, _ = New(llm.Config{Model: "m", APIKey: "k", ProviderName: "mine", BaseURL: "http://127.0.0.1:1/root/"})
	if c.Provider() != "mine" || c.endpoint != "http://127.0.0.1:1/root/interactions" {
		t.Errorf("%q %q", c.Provider(), c.endpoint)
	}
	if err := c.Close(); err != nil {
		t.Error(err)
	}
}
