package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"strings"
)

// This file centralizes the HTTP plumbing shared by every provider adapter so the
// security-critical pieces — bounded reads and error construction — live in one
// place. Wire-format translation stays per-adapter; only the transport is shared.

// NewJSONRequest builds a POST request carrying a JSON body, with Content-Type set.
// Adapters add their provider-specific auth/version headers afterwards.
func NewJSONRequest(ctx context.Context, url string, body []byte) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return req, nil
}

// ErrorFromResponse reads a non-2xx response body (bounded by cfg's effective error
// limits) and returns the matching *APIError. Centralizes the bounded error read +
// classification that every adapter needs.
func ErrorFromResponse(provider string, resp *http.Response, cfg Config) error {
	body, readErr := io.ReadAll(io.LimitReader(resp.Body, cfg.EffectiveMaxErrorBodyBytes()))
	if readErr != nil {
		slog.Warn("failed to read error response body", "provider", provider, "error", readErr)
	}
	return NewAPIErrorFromStatusWithLimit(provider, resp.StatusCode, string(body), cfg.EffectiveMaxErrorMessageLen())
}

// DecodeJSONResponse decodes a 2xx JSON response body into out, bounded by cfg's
// effective response-body limit.
func DecodeJSONResponse(resp *http.Response, cfg Config, out any) error {
	return json.NewDecoder(io.LimitReader(resp.Body, cfg.EffectiveMaxResponseBodyBytes())).Decode(out)
}

// SSEData extracts the payload of a server-sent-events "data: " line.
// ok is false for every other line (event names, comments, blank keep-alives),
// which SSE consumers skip. Shared by all streaming adapters.
func SSEData(line string) (data string, ok bool) {
	if !strings.HasPrefix(line, "data: ") {
		return "", false
	}
	data = strings.TrimPrefix(line, "data: ")
	if data == "" {
		// "data: " with an empty value carries no event (keep-alive/padding some
		// servers emit). Report not-ok so callers skip it instead of trying — and
		// failing — to JSON-parse an empty string, which would surface as a
		// spurious mid-stream error that aborts an otherwise-complete turn.
		return "", false
	}
	return data, true
}
