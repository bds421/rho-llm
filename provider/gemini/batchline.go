package gemini

import (
	"encoding/json"
	"fmt"

	"github.com/bds421/rho-llm"
)

// BatchTranslator constructs a throwaway codec client for Gemini Batch.
func BatchTranslator(cfg llm.Config) (*Client, error) {
	return New(cfg)
}

// BuildGenerateContentBody builds one generateContent request body for batch.
func (c *Client) BuildGenerateContentBody(req llm.Request) (json.RawMessage, error) {
	apiReq, err := c.buildRequest(req)
	if err != nil {
		return nil, err
	}
	return json.Marshal(apiReq)
}

// ParseGenerateContentBody parses a generateContent response into llm.Response.
func (c *Client) ParseGenerateContentBody(raw json.RawMessage, model string) (*llm.Response, error) {
	var apiResp geminiResponse
	if err := json.Unmarshal(raw, &apiResp); err != nil {
		return nil, fmt.Errorf("gemini: decode batch result: %w", err)
	}
	return c.parseResponse(&apiResp, model), nil
}

// BaseURL returns the resolved models base URL (for batch path construction).
func (c *Client) BaseURL() string { return c.baseURL }
