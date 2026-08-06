package anthropic

import (
	"encoding/json"
	"fmt"

	"github.com/bds421/rho-llm"
)

// BatchTranslator constructs a throwaway adapter used purely as a codec for
// Message Batches params/results. New performs no I/O beyond SafeHTTPClient
// construction; the returned client is never used for live Complete calls.
func BatchTranslator(cfg llm.Config) (*Client, error) {
	return New(cfg)
}

// BuildMessageBatchParams builds the Messages API params object for one
// Anthropic Message Batch request entry, reusing the live Complete encoder
// (stream forced false).
func (c *Client) BuildMessageBatchParams(req llm.Request) (json.RawMessage, error) {
	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		return nil, err
	}
	// Batch params must not request streaming.
	apiReq.Stream = false
	return json.Marshal(apiReq)
}

// ParseMessageBatchResult parses one succeeded Message object from a batch
// results JSONL line into the neutral Response.
func (c *Client) ParseMessageBatchResult(raw json.RawMessage) (*llm.Response, error) {
	var apiResp anthropicResponse
	if err := json.Unmarshal(raw, &apiResp); err != nil {
		return nil, fmt.Errorf("anthropic: decode batch result message: %w", err)
	}
	return c.parseResponse(&apiResp), nil
}
