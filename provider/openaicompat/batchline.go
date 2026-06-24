package openaicompat

import (
	"encoding/json"
	"fmt"

	"github.com/bds421/rho-llm"
)

// Batch line codec — exposes this adapter's request translation and response parsing
// for the OpenAI Batch API (provider/openaibatch), so a batched /v1/chat/completions
// line is built and parsed by the SAME code path as a live Complete call. Keeping the
// wire format in one place is the whole point: a parity test asserts the batch line
// body is byte-identical to what Complete POSTs for the same Request.

// BatchTranslator constructs a throwaway adapter used purely as a codec for batch
// line bodies. New performs no I/O (it only resolves the base URL and auth), so the
// returned *Client is safe to use without ever issuing a request. The batch driver
// builds one translator per endpoint kind and reuses it across all lines.
func BatchTranslator(cfg llm.Config) (*Client, error) {
	return New(cfg)
}

// BuildChatBatchLineBody builds the "body" object for one OpenAI batch line targeting
// /v1/chat/completions, reusing the adapter's full request translation (system prompt,
// tool-result splitting into role:"tool" messages, multimodal content arrays,
// reasoning-model max_completion_tokens handling). No network call; stream is false.
func (c *Client) BuildChatBatchLineBody(req llm.Request) (json.RawMessage, error) {
	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		return nil, err
	}
	return json.Marshal(apiReq)
}

// ParseChatBatchResultBody parses one batch output line's response.body (a
// /v1/chat/completions response) into the neutral llm.Response, reusing the adapter's
// normalization (stop-reason mapping, tool-call/reasoning extraction, token usage).
func (c *Client) ParseChatBatchResultBody(raw json.RawMessage) (*llm.Response, error) {
	var apiResp openaiResponse
	if err := json.Unmarshal(raw, &apiResp); err != nil {
		return nil, fmt.Errorf("openaicompat: decode batch result body: %w", err)
	}
	return c.parseResponse(&apiResp), nil
}
