package openairesponses

import (
	"encoding/json"
	"fmt"

	"github.com/bds421/rho-llm"
)

// Batch line codec — exposes this adapter's request translation and response parsing
// for the OpenAI Batch API (provider/openaibatch), so a batched /v1/responses line is
// built and parsed by the SAME code path as a live Complete call. buildRequest already
// sets store:false, which is exactly what batch processing needs.

// BatchTranslator constructs a throwaway adapter used purely as a codec for batch
// line bodies. New performs no I/O, so the returned *Client is safe to use without
// ever issuing a request.
func BatchTranslator(cfg llm.Config) (*Client, error) {
	return New(cfg)
}

// BuildResponsesBatchLineBody builds the "body" object for one OpenAI batch line
// targeting /v1/responses, reusing the adapter's request translation (reasoning effort,
// max_output_tokens, tool conversion). No network call; stream is false.
func (c *Client) BuildResponsesBatchLineBody(req llm.Request) (json.RawMessage, error) {
	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		return nil, err
	}
	return json.Marshal(apiReq)
}

// ParseResponsesBatchResultBody parses one batch output line's response.body (a
// /v1/responses response) into the neutral llm.Response, reusing the adapter's
// normalization (output-item walking, reasoning, token usage).
func (c *Client) ParseResponsesBatchResultBody(raw json.RawMessage) (*llm.Response, error) {
	var apiResp responsesResponse
	if err := json.Unmarshal(raw, &apiResp); err != nil {
		return nil, fmt.Errorf("openairesponses: decode batch result body: %w", err)
	}
	return c.parseResponse(&apiResp), nil
}
