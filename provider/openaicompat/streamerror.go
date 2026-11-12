package openaicompat

import (
	"encoding/json"
	"strconv"
	"strings"

	llm "github.com/bds421/rho-llm"
)

// providerError is the error object OpenAI-compatible servers put inside a
// 200 response or a stream chunk instead of an HTTP status: OpenAI and
// Together send {"message","type","code":"…"}, OpenRouter and vLLM a numeric
// code ({"code":502,"message":…}). Code is kept raw because it is either.
type providerError struct {
	Message string          `json:"message"`
	Type    string          `json:"type"`
	Code    json.RawMessage `json:"code"`
}

// errorTypeStatus maps the string type/code of an in-band error onto the HTTP
// status the same error carries as a response status, so retry and failover
// classify it alike.
var errorTypeStatus = map[string]int{
	"invalid_request_error":   400,
	"context_length_exceeded": 400,
	"bad_request":             400,
	"authentication_error":    401,
	"invalid_api_key":         401,
	"permission_error":        403,
	"not_found_error":         404,
	"model_not_found":         404,
	"rate_limit_exceeded":     429,
	"rate_limit_error":        429,
	"insufficient_quota":      429,
	"server_error":            500,
	"internal_error":          500,
	"timeout":                 504,
	"overloaded_error":        529,
}

// status picks the HTTP status for an in-band error: a numeric code in the
// 4xx/5xx range wins, then a known type or string code; anything else is a
// server error (500, retryable before any output).
func (e *providerError) status() int {
	code := strings.Trim(string(e.Code), `"`)
	if n, err := strconv.Atoi(code); err == nil && n >= 400 && n <= 599 {
		return n
	}
	if s, ok := errorTypeStatus[code]; ok {
		return s
	}
	if s, ok := errorTypeStatus[e.Type]; ok {
		return s
	}
	return 500
}

// inBandError turns an in-band error object into the same *llm.APIError an
// HTTP error status would produce. The message gets the same hygiene as an
// error body: the server may echo the key, so the deployment's secrets are
// scrubbed first.
func (c *Client) inBandError(e *providerError) error {
	message := e.Message
	if e.Type != "" {
		message = e.Type + ": " + message
	}
	if message == "" {
		message = "provider reported an error"
	}
	message = llm.RedactSecrets(message, c.config)
	return llm.NewAPIErrorFromStatusWithLimit(c.providerName, e.status(), message, c.config.EffectiveMaxErrorMessageLen())
}
