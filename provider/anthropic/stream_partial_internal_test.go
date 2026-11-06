package anthropic

import (
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"
	"testing/iotest"

	llm "github.com/bds421/rho-llm"
)

const streamStart = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","model":"claude-opus-5-5","usage":{"input_tokens":120,"output_tokens":2,"cache_creation_input_tokens":30,"cache_read_input_tokens":4000,"cache_creation":{"ephemeral_1h_input_tokens":30}}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Partial ans"}}

`

func streamFrom(c *Client, r io.Reader) func(func(llm.StreamEvent, error) bool) {
	return func(yield func(llm.StreamEvent, error) bool) { c.parseStream(r, yield) }
}

func TestBrokenStreamKeepsUsage(t *testing.T) {
	c := &Client{config: llm.Config{Model: "claude-opus-5-5"}, providerName: "anthropic"}
	overloaded := streamStart + "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n"
	hostile := map[string]io.Reader{
		"EOF after deltas":   strings.NewReader(streamStart),
		"read error":         io.MultiReader(strings.NewReader(streamStart), iotest.ErrReader(errors.New("connection reset"))),
		"overloaded mid-way": strings.NewReader(overloaded),
	}
	for name, r := range hostile {
		resp, err := llm.CollectStream(streamFrom(c, r))
		if err == nil {
			t.Errorf("%s: no error", name)
			continue
		}
		if resp.InputTokens != 120 || resp.CacheReadTokens != 4000 || resp.CacheCreationTokens != 30 ||
			resp.CacheCreation1hTokens != 30 || resp.OutputTokens != 2 || resp.Model != "claude-opus-5-5" {
			t.Errorf("%s: partial usage lost: %+v", name, resp)
		}
		if resp.Content != "Partial ans" || len(resp.Parts) != 1 || resp.Parts[0].Text != "Partial ans" {
			t.Errorf("%s: partial content: %q parts %+v", name, resp.Content, resp.Parts)
		}
	}

	// The overloaded event keeps its classification: retryable, overloaded.
	_, err := llm.CollectStream(streamFrom(c, strings.NewReader(overloaded)))
	if !llm.IsRetryable(err) || !llm.IsOverloaded(err) {
		t.Errorf("overloaded_error: %v (retryable %v)", err, llm.IsRetryable(err))
	}
	var apiErr *llm.APIError
	if !errors.As(err, &apiErr) || apiErr.StatusCode != 529 {
		t.Errorf("overloaded_error status: %v", err)
	}

	// Before message_start there is no usage to report: an empty event.
	resp, err := llm.CollectStream(streamFrom(c, strings.NewReader("event: ping\ndata: {\"type\":\"ping\"}\n\n")))
	if err == nil || resp.InputTokens != 0 || resp.Model != "" {
		t.Errorf("before message_start: %+v, %v", resp, err)
	}
}

func TestStreamDoneCarriesTheAnsweringModel(t *testing.T) {
	c := &Client{config: llm.Config{Model: "claude-opus-5-5"}, providerName: "anthropic"}
	done := streamStart + `event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":9}}

`
	// With a refusal fallback, message_start names the model that answered.
	resp, err := llm.CollectStream(streamFrom(c, strings.NewReader(strings.Replace(done, `"model":"claude-opus-5-5"`, `"model":"claude-opus-5"`, 1))))
	if err != nil || resp.Model != "claude-opus-5" || resp.OutputTokens != 9 || resp.StopReason != llm.StopEndTurn {
		t.Errorf("done: %+v, %v", resp, err)
	}
}

func TestStreamErrorStatuses(t *testing.T) {
	for errType, retryable := range map[string]bool{
		"overloaded_error": true, "rate_limit_error": true, "api_error": true, "timeout_error": true,
		"invalid_request_error": false, "authentication_error": false, "unknown_kind": true,
	} {
		if got := llm.IsRetryable(streamError(errType, "m", llm.Config{})); got != retryable {
			t.Errorf("%s: retryable %v, want %v", errType, got, retryable)
		}
	}
}

func TestIterationOneHourShareParsed(t *testing.T) {
	var apiResp anthropicResponse
	raw := `{"id":"m","model":"claude-haiku-5-5","content":[],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1,
		"iterations":[{"type":"message","model":"claude-haiku-5-5","input_tokens":1,"output_tokens":1,"cache_creation_input_tokens":50,
		"cache_creation":{"ephemeral_5m_input_tokens":20,"ephemeral_1h_input_tokens":30}}]}}`
	if err := json.Unmarshal([]byte(raw), &apiResp); err != nil {
		t.Fatal(err)
	}
	resp := (&Client{}).parseResponse(&apiResp)
	if len(resp.Iterations) != 1 || resp.Iterations[0].CacheCreationTokens != 50 || resp.Iterations[0].CacheCreation1hTokens != 30 {
		t.Errorf("iterations %+v", resp.Iterations)
	}
}

// message_delta usage is cumulative: with server tools it has the input and
// cache counts of every step, which replace message_start's first-step ones.
func TestStreamDoneTakesCumulativeUsage(t *testing.T) {
	c := &Client{config: llm.Config{Model: "claude-haiku-5-5"}, providerName: "anthropic"}
	sse := streamStart + `event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":11700,"output_tokens":190,"cache_read_input_tokens":4200,"cache_creation_input_tokens":60,"cache_creation":{"ephemeral_1h_input_tokens":60},"server_tool_use":{"web_search_requests":1}}}

`
	resp, err := llm.CollectStream(streamFrom(c, strings.NewReader(sse)))
	if err != nil || resp.InputTokens != 11700 || resp.OutputTokens != 190 || resp.CacheReadTokens != 4200 ||
		resp.CacheCreationTokens != 60 || resp.CacheCreation1hTokens != 60 || resp.ServerToolUse.WebSearchRequests != 1 {
		t.Errorf("%+v, %v", resp, err)
	}
	// Without counts in message_delta, message_start's stay.
	resp, _ = llm.CollectStream(streamFrom(c, strings.NewReader(streamStart+"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":9}}\n\n")))
	if resp.InputTokens != 120 || resp.CacheReadTokens != 4000 {
		t.Errorf("start usage lost: %+v", resp)
	}
}

// A mid-stream error that echoes the request key never carries it on.
func TestStreamErrorRedactsTheKey(t *testing.T) {
	const key = "test-secret-key-0123456789abcdef"
	c := &Client{config: llm.Config{Model: "claude-haiku-5-5", APIKey: key}, providerName: "anthropic"}
	sse := streamStart + "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"authentication_error\",\"message\":\"invalid x-api-key " + key + "\"}}\n\n"
	_, err := llm.CollectStream(streamFrom(c, strings.NewReader(sse)))
	if err == nil || strings.Contains(err.Error(), key) {
		t.Errorf("key leaked or no error: %v", err)
	}
	var apiErr *llm.APIError
	if !errors.As(err, &apiErr) || strings.Contains(apiErr.Message, key) || apiErr.StatusCode != 401 {
		t.Errorf("APIError %+v", apiErr)
	}
}
