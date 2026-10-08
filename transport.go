package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

// This file centralizes the HTTP plumbing shared by every provider adapter so the
// security-critical pieces — bounded reads and error construction — live in one
// place. Wire-format translation stays per-adapter; only the transport is shared.

// HTTPRequestFactory rebuilds one request for a retry attempt. Bodies must be
// fresh because net/http consumes them.
type HTTPRequestFactory func(context.Context) (*http.Request, error)

// HTTPCallOptions tunes one DoHTTPWithOptions call. The zero value is
// exactly DoHTTP's behaviour.
type HTTPCallOptions struct {
	// NonIdempotent marks a request that creates a durable or billable
	// resource — a batch job, generated images, synthesized speech. A resend
	// after the provider already acted on it would duplicate the job or the
	// charge, so such a request is retried only when the provider certainly
	// did not act: HTTP 429 or 503 (rejected before processing) or a
	// connection that was never established (DNS/dial/proxy-connect failure).
	// 408/500/502/504 and transport errors after the request may have been
	// sent are returned after the first attempt.
	NonIdempotent bool
}

// DoHTTP executes a provider HTTP request with the common retry policy,
// retry hooks, proxy-aware client, and caller cancellation semantics. The final
// HTTP response is returned even for non-2xx status so callers can decode its
// provider error body. Auth rotation and circuit breaking remain properties of
// NewClientWithKeys; a single Config contains only one credential.
//
// Attempts: Config.MaxRetries when set (minimum 3), otherwise
// DefaultHTTPMaxAttempts. Backoff follows Config.RetryPolicy, stretched to a
// provider Retry-After hint (capped at MaxRetryAfter). Config.RetryBudget,
// when positive, stops the sequence before a backoff that would overrun it.
// DoHTTP treats the request as idempotent; use DoHTTPWithOptions with
// NonIdempotent for creates.
func DoHTTP(ctx context.Context, cfg Config, client *http.Client, build HTTPRequestFactory) (*http.Response, error) {
	return DoHTTPWithOptions(ctx, cfg, client, build, HTTPCallOptions{})
}

// DoHTTPWithOptions is DoHTTP with per-call options (see HTTPCallOptions).
func DoHTTPWithOptions(
	ctx context.Context, cfg Config, client *http.Client, build HTTPRequestFactory, opts HTTPCallOptions,
) (*http.Response, error) {
	if client == nil {
		var err error
		client, err = NewSafeHTTPClient(cfg)
		if err != nil {
			return nil, err
		}
	}
	attempts := 1
	if !cfg.DisableRetries {
		attempts = cfg.MaxRetries
		if attempts <= 0 {
			attempts = DefaultHTTPMaxAttempts
		}
		if attempts < 3 {
			attempts = 3
		}
	}
	policy := DefaultRetryPolicy
	if cfg.RetryPolicy != nil {
		policy = *cfg.RetryPolicy
	}
	provider := cfg.ProviderName
	if provider == "" {
		provider = cfg.Provider
	}
	budget := newRetryBudget(cfg.RetryBudget)
	exhausted := func(attempt int, err error) {
		if cfg.RetryHook != nil {
			cfg.RetryHook(RetryEvent{Type: RetryExhausted, Attempt: attempt, Err: err, Provider: provider})
		}
	}
	var lastErr error
	for attempt := 0; attempt < attempts; attempt++ {
		req, err := build(ctx)
		if err != nil {
			return nil, err
		}
		resp, err := client.Do(req)
		if err == nil && !retryableHTTPStatus(resp.StatusCode) {
			return resp, nil
		}
		var delay time.Duration
		if err == nil {
			apiErr := NewAPIErrorFromStatus(provider, resp.StatusCode, resp.Status)
			if wait, ok := parseRetryAfter(resp.Header, time.Now()); ok {
				apiErr.RetryAfter = wait
			}
			lastErr = apiErr
			if opts.NonIdempotent && !idempotentSafeStatus(resp.StatusCode) {
				// The provider may have committed the create: hand the
				// response back without a resend.
				return resp, nil
			}
			delay = backoffFor(policy, attempt, apiErr.RetryAfter)
			if attempt+1 == attempts || !budget.allows(delay) {
				exhausted(attempt, lastErr)
				return resp, nil
			}
			_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
			_ = resp.Body.Close()
		} else {
			lastErr = err
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			retryable := IsRetryable(err)
			if retryable && opts.NonIdempotent && !isDialError(err) {
				// The request may have reached the provider before the
				// transport failed; a resend could duplicate the create.
				return nil, err
			}
			if !retryable {
				return nil, err
			}
			delay = backoffFor(policy, attempt, 0)
			if attempt+1 == attempts || !budget.allows(delay) {
				exhausted(attempt, err)
				return nil, err
			}
		}
		if cfg.RetryHook != nil {
			cfg.RetryHook(RetryEvent{Type: RetryAttemptFailed, Attempt: attempt, Err: lastErr, Provider: provider})
		}
		if cfg.RetryHook != nil {
			cfg.RetryHook(RetryEvent{Type: RetryBackingOff, Attempt: attempt, Err: lastErr, Backoff: delay, Provider: provider})
		}
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-timer.C:
		}
	}
	return nil, fmt.Errorf("llm: retry loop exhausted: %w", lastErr)
}

// idempotentSafeStatus reports a status at which the provider rejected the
// request before acting on it, so even a create may be resent.
func idempotentSafeStatus(status int) bool {
	return status == http.StatusTooManyRequests || status == http.StatusServiceUnavailable
}

func retryableHTTPStatus(status int) bool {
	return status == http.StatusRequestTimeout || status == http.StatusTooManyRequests ||
		status == http.StatusInternalServerError || status == http.StatusBadGateway ||
		status == http.StatusServiceUnavailable || status == http.StatusGatewayTimeout
}

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
		slog.Debug("failed to read error response body", "provider", provider, "error", readErr)
	}
	// A provider may echo the request credential in its error body; scrub the
	// deployment's own key and BaseURL secrets before the text becomes an error
	// that callers log or show.
	message := redactProfileSecrets(string(body), cfg.APIKey, cfg.BaseURL)
	apiErr := NewAPIErrorFromStatusWithLimit(provider, resp.StatusCode, message, cfg.EffectiveMaxErrorMessageLen())
	if wait, ok := parseRetryAfter(resp.Header, time.Now()); ok {
		apiErr.RetryAfter = wait
	}
	return apiErr
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
