package openairesponses

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// Wire shapes below follow docs.perplexity.ai/api-reference/agent-post (the
// OpenAPI schema, 2026-11-18); nothing was recorded from a live server.

const perplexityAgentResponse = `{
  "id": "resp_abc",
  "object": "response",
  "created_at": 1760000000,
  "status": "completed",
  "model": "perplexity/sonar",
  "effort": "fast@2026-11-11.0",
  "output": [
    {"type": "search_results", "id": "sr_1", "results": [{"id": 1, "title": "T", "url": "https://example.com"}]},
    {"type": "message", "id": "msg_1", "role": "assistant", "content": [
      {"type": "output_text", "text": "Paris [1].", "annotations": [{"type": "url_citation", "url": "https://example.com"}]}
    ]}
  ],
  "usage": {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150,
    "cost": {"currency": "USD", "input_cost": 0.00003, "output_cost": 0.000075, "total_cost": 0.000105}}
}`

func perplexityClient(t *testing.T, baseURL string) *Client {
	t.Helper()
	c, err := New(llm.Config{Provider: "perplexity-agent", Model: "perplexity/sonar", APIKey: "pplx-test", BaseURL: baseURL})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func userReq(system string) llm.Request {
	return llm.Request{System: system, Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "Capital of France?")}}
}

func TestPerplexityCompleteWire(t *testing.T) {
	var gotPath, gotAuth string
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath, gotAuth = r.URL.Path, r.Header.Get("Authorization")
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(perplexityAgentResponse))
	}))
	defer srv.Close()

	c := perplexityClient(t, srv.URL) // bare host: the dialect adds /v1
	req := userReq("Be brief.")
	req.ThinkingLevel = llm.ThinkingMax
	resp, err := c.Complete(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if gotPath != "/v1/responses" || gotAuth != "Bearer pplx-test" {
		t.Fatalf("path/auth = %q/%q", gotPath, gotAuth)
	}
	if gotBody["instructions"] != "Be brief." || gotBody["model"] != "perplexity/sonar" || gotBody["store"] != false {
		t.Fatalf("body = %v", gotBody)
	}
	for _, m := range gotBody["input"].([]any) {
		if m.(map[string]any)["role"] == "system" {
			t.Fatalf("system prompt also sent as an input item: %v", gotBody["input"])
		}
	}
	if _, has := gotBody["max_output_tokens"]; has {
		t.Fatalf("max_output_tokens sent while unset (optional on the Agent API): %v", gotBody)
	}
	if eff := gotBody["reasoning"].(map[string]any)["effort"]; eff != "max" {
		t.Fatalf("reasoning.effort = %v, want max (the Agent API accepts it)", eff)
	}
	if resp.Content != "Paris [1]." || resp.Model != "perplexity/sonar" || resp.StopReason != llm.StopEndTurn ||
		resp.InputTokens != 120 || resp.OutputTokens != 30 {
		t.Fatalf("resp = %+v", resp)
	}
	if c.Provider() != "perplexity-agent" {
		t.Fatalf("Provider() = %q", c.Provider())
	}
}

// A BaseURL that already names the path, or another host, is left alone, and
// the OpenAI dialect is unchanged.
func TestDialectBaseURLAndOpenAIUnchanged(t *testing.T) {
	d := perplexityDialect
	for in, want := range map[string]string{
		"https://api.perplexity.ai":     "https://api.perplexity.ai/v1",
		"https://api.perplexity.ai/":    "https://api.perplexity.ai/v1",
		"https://api.perplexity.ai/v1":  "https://api.perplexity.ai/v1",
		"https://gw.example.com/pplx":   "https://gw.example.com/pplx",
		"::not a url::":                 "::not a url::",
		"":                              "",
		"https://api.perplexity.ai/v1/": "https://api.perplexity.ai/v1/",
	} {
		if got := d.fixBaseURL(in); got != want {
			t.Errorf("fixBaseURL(%q) = %q, want %q", in, got, want)
		}
	}
	if got := (dialect{}).fixBaseURL("https://api.openai.com"); got != "https://api.openai.com" {
		t.Errorf("openai dialect rewrote the base URL: %q", got)
	}

	// OpenAI keeps the system input item and clamps max to xhigh.
	c := &Client{providerName: "openai", config: llm.Config{Model: "gpt-6.1-sol"}}
	req := userReq("sys")
	req.ThinkingLevel = llm.ThinkingMax
	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatal(err)
	}
	if apiReq.Instructions != "" || apiReq.Reasoning.Effort != "xhigh" {
		t.Fatalf("openai dialect changed: instructions=%q effort=%q", apiReq.Instructions, apiReq.Reasoning.Effort)
	}
	if first, ok := apiReq.Input[0].(responsesInputMsg); !ok || first.Role != "system" {
		t.Fatalf("openai lost its system input item: %+v", apiReq.Input)
	}
	// The Sonar preset name carries no dialect: only perplexity-agent opts in.
	if got := dialectFor(llm.Config{Provider: "perplexity"}); got != (dialect{}) {
		t.Errorf("provider perplexity got the Agent API dialect: %+v", got)
	}
	// A custom provider name on the Perplexity host still gets the dialect.
	if dialectFor(llm.Config{Provider: "my-gateway", BaseURL: "https://API.Perplexity.ai/v1"}).name != "perplexity-agent" {
		t.Error("host match missed")
	}
	if dialectFor(llm.Config{Provider: "openai", BaseURL: "https://api.perplexity.ai.evil.example/v1"}).name != "" {
		t.Error("lookalike host matched the Perplexity dialect")
	}
}

// The documented event set: lifecycle, reasoning.* progress and search events
// (unknown to the OpenAI adapter) must be skipped, text streams, the terminal
// response carries usage.
func TestPerplexityStreamDocumentedEvents(t *testing.T) {
	c := &Client{providerName: "perplexity-agent", config: llm.Config{}}
	events, err := runStream(t, c,
		`{"type":"response.created","sequence_number":1,"response":{"id":"r","status":"in_progress"}}`,
		`{"type":"response.in_progress","sequence_number":2}`,
		`{"type":"response.reasoning.started","sequence_number":3}`,
		`{"type":"response.reasoning.search_queries","sequence_number":4,"queries":["capital of france"]}`,
		`{"type":"response.reasoning.search_results","sequence_number":5,"results":[{"id":1,"url":"https://example.com"}]}`,
		`{"type":"response.reasoning.stopped","sequence_number":6}`,
		`{"type":"response.output_item.added","sequence_number":7,"output_index":1,"item":{"type":"message","id":"m"}}`,
		`{"type":"response.output_text.delta","sequence_number":8,"item_id":"m","output_index":1,"content_index":0,"delta":"Par"}`,
		`{"type":"response.output_text.delta","sequence_number":9,"item_id":"m","output_index":1,"content_index":0,"delta":"is [1]."}`,
		`{"type":"response.output_text.done","sequence_number":10,"item_id":"m","output_index":1,"content_index":0,"text":"Paris [1]."}`,
		`{"type":"response.output_item.done","sequence_number":11,"output_index":1,"item":{"type":"message","id":"m"}}`,
		`{"type":"response.completed","sequence_number":12,"response":{"id":"r","status":"completed","usage":{"input_tokens":120,"output_tokens":30,"total_tokens":150}}}`,
		`[DONE]`)
	if err != nil {
		t.Fatal(err)
	}
	var text strings.Builder
	for _, ev := range events {
		if ev.Type == llm.EventContent {
			text.WriteString(ev.Text)
		}
	}
	done := events[len(events)-1]
	if text.String() != "Paris [1]." || done.Type != llm.EventDone || done.StopReason != llm.StopEndTurn || done.InputTokens != 120 || done.OutputTokens != 30 {
		t.Fatalf("text=%q done=%+v", text.String(), done)
	}
}

// response.failed carries the error at the top level and no response. It used
// to end as a clean, empty completion.
func TestStreamFailedEventWithTopLevelError(t *testing.T) {
	for name, line := range map[string]string{
		"top-level error":  `{"type":"response.failed","sequence_number":3,"error":{"code":"rate_limit_exceeded","message":"slow down"}}`,
		"nested, no state": `{"type":"response.failed","response":{"error":{"code":"server_error","message":"boom"}}}`,
		"no detail at all": `{"type":"response.failed"}`,
	} {
		t.Run(name, func(t *testing.T) {
			c := &Client{providerName: "perplexity-agent", config: llm.Config{}}
			events, err := runStream(t, c, `{"type":"response.output_text.delta","delta":"par"}`, line)
			var apiErr *llm.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("err = %v (events %+v), want an *llm.APIError, not a completion", err, events)
			}
			for _, ev := range events {
				if ev.Type == llm.EventDone {
					t.Fatalf("a failed stream emitted Done: %+v", events)
				}
			}
		})
	}
	c := &Client{providerName: "perplexity-agent", config: llm.Config{}}
	_, err := runStream(t, c, `{"type":"response.failed","error":{"code":"rate_limit_exceeded","message":"slow down"}}`)
	if !llm.IsRateLimited(err) {
		t.Fatalf("err = %v, want rate-limit classification", err)
	}
}

// A function call delivered only whole (output_item.done, or only in the
// terminal response) reaches the caller exactly once, with or without the
// OpenAI arguments events.
func TestStreamFunctionCallDeliveryPaths(t *testing.T) {
	call := `{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"x\"}"}`
	completed := func(output string) string {
		return `{"type":"response.completed","response":{"status":"completed","output":[` + output + `],"usage":{"input_tokens":1,"output_tokens":1}}}`
	}
	for name, lines := range map[string][]string{
		"item.done only":        {`{"type":"response.output_item.done","item":` + call + `}`, completed(call)},
		"terminal output only":  {completed(call)},
		"arguments.done first":  {`{"type":"response.function_call_arguments.done","item_id":"fc_1","name":"lookup","call_id":"call_1","arguments":"{\"q\":\"x\"}"}`, `{"type":"response.output_item.done","item":` + call + `}`, completed(call)},
		"deltas then item.done": {`{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\"q\":"}`, `{"type":"response.output_item.done","item":` + call + `}`, completed(call)},
	} {
		t.Run(name, func(t *testing.T) {
			c := &Client{providerName: "perplexity-agent", config: llm.Config{}}
			events, err := runStream(t, c, lines...)
			if err != nil {
				t.Fatal(err)
			}
			var calls []*llm.ToolCall
			for _, ev := range events {
				if ev.Type == llm.EventToolUse {
					calls = append(calls, ev.ToolCall)
				}
			}
			done := events[len(events)-1]
			if len(calls) != 1 || calls[0].Name != "lookup" || calls[0].ID != "call_1" || fmt.Sprint(calls[0].Input) != "map[q:x]" {
				t.Fatalf("calls = %+v, want exactly one lookup(q=x)", calls)
			}
			if done.StopReason != llm.StopToolUse {
				t.Fatalf("stop = %q, want tool_use", done.StopReason)
			}
		})
	}
}

// Two distinct calls are both kept, and hostile function-call payloads are
// bounded or survive.
func TestStreamFunctionCallHostile(t *testing.T) {
	c := &Client{providerName: "perplexity-agent", config: llm.Config{}}
	events, err := runStream(t, c,
		`{"type":"response.output_item.done","item":{"type":"function_call","id":"a","call_id":"c1","name":"f","arguments":"{}"}}`,
		`{"type":"response.output_item.done","item":{"type":"function_call","id":"b","call_id":"c2","name":"g","arguments":"not json"}}`,
		`{"type":"response.output_item.done","item":"garbage"}`,
		`{"type":"response.completed","response":{"status":"completed"}}`)
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, ev := range events {
		if ev.Type == llm.EventToolUse {
			n++
		}
	}
	if n != 2 {
		t.Fatalf("tool calls = %d, want 2", n)
	}

	c = &Client{providerName: "perplexity-agent", config: llm.Config{MaxToolInputBytes: 16}}
	big := `{"type":"response.output_item.done","item":{"type":"function_call","id":"a","call_id":"c","name":"f","arguments":"` + strings.Repeat("x", 64) + `"}}`
	if _, err := runStream(t, c, big); err == nil || !strings.Contains(err.Error(), "exceeded") {
		t.Fatalf("oversized item.done arguments: err = %v, want a bound error", err)
	}
	c = &Client{providerName: "perplexity-agent", config: llm.Config{MaxToolInputBytes: 16}}
	doneBig := `{"type":"response.completed","response":{"status":"completed","output":[{"type":"function_call","id":"a","call_id":"c","name":"f","arguments":"` + strings.Repeat("x", 64) + `"}]}}`
	if _, err := runStream(t, c, doneBig); err == nil || !strings.Contains(err.Error(), "exceeded") {
		t.Fatalf("oversized terminal arguments: err = %v, want a bound error", err)
	}
}

func TestPerplexityErrorEnvelopeAndConcurrency(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer pplx-test" {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":{"code":"unauthorized","message":"bad key","type":"invalid_request_error"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(perplexityAgentResponse))
	}))
	defer srv.Close()

	c := perplexityClient(t, srv.URL)
	var wg sync.WaitGroup
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if resp, err := c.Complete(context.Background(), userReq("s")); err != nil || resp.Content != "Paris [1]." {
				t.Errorf("resp=%v err=%v", resp, err)
			}
		}()
	}
	wg.Wait()

	bad, err := New(llm.Config{Provider: "perplexity-agent", Model: "perplexity/sonar", APIKey: "wrong", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	_, err = bad.Complete(context.Background(), userReq(""))
	var apiErr *llm.APIError
	if !errors.As(err, &apiErr) || apiErr.StatusCode != 401 || strings.Contains(err.Error(), "wrong") {
		t.Fatalf("err = %v, want a 401 APIError without the key", err)
	}
}

// OpenAI's own event order (arguments.delta, arguments.done, output_item.done,
// completed with the same output) still emits each call exactly once, in order,
// with the arguments the done event carried.
func TestStreamOpenAIShapedCallsEmitOnce(t *testing.T) {
	c := &Client{providerName: "openai", config: llm.Config{Model: "gpt-6.1-sol"}}
	a := `{"type":"function_call","id":"fc_a","call_id":"call_a","name":"f","arguments":"{\"n\":1}"}`
	b := `{"type":"function_call","id":"fc_b","call_id":"call_b","name":"g","arguments":"{\"n\":2}"}`
	events, err := runStream(t, c,
		`{"type":"response.created","response":{"id":"r"}}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_a","call_id":"call_a","name":"f"}}`,
		`{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_b","call_id":"call_b","name":"g"}}`,
		`{"type":"response.function_call_arguments.delta","item_id":"fc_a","output_index":0,"delta":"{\"n\":"}`,
		`{"type":"response.function_call_arguments.delta","item_id":"fc_b","output_index":1,"delta":"{\"n\":"}`,
		`{"type":"response.function_call_arguments.delta","item_id":"fc_a","output_index":0,"delta":"1}"}`,
		`{"type":"response.function_call_arguments.delta","item_id":"fc_b","output_index":1,"delta":"2}"}`,
		`{"type":"response.function_call_arguments.done","item_id":"fc_a","output_index":0,"arguments":"{\"n\":1}"}`,
		`{"type":"response.output_item.done","output_index":0,"item":`+a+`}`,
		`{"type":"response.function_call_arguments.done","item_id":"fc_b","output_index":1,"arguments":"{\"n\":2}"}`,
		`{"type":"response.output_item.done","output_index":1,"item":`+b+`}`,
		`{"type":"response.completed","response":{"status":"completed","output":[`+a+`,`+b+`],"usage":{"input_tokens":5,"output_tokens":3}}}`)
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, ev := range events {
		if ev.Type == llm.EventToolUse {
			got = append(got, ev.ToolCall.ID+":"+ev.ToolCall.Name+":"+fmt.Sprint(ev.ToolCall.Input))
		}
	}
	want := []string{"call_a:f:map[n:1]", "call_b:g:map[n:2]"}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("calls = %v, want %v", got, want)
	}
	if done := events[len(events)-1]; done.Type != llm.EventDone || done.StopReason != llm.StopToolUse {
		t.Fatalf("done = %+v", done)
	}
}
