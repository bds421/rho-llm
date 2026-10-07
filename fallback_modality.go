package llm

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/url"
	"strings"
)

// NewFallbackModalityClient returns a ModalityClient that tries primary first
// and each fallback in order. A later deployment is used only when the
// previous one failed at the provider — an APIError (any HTTP status) or a
// transport failure. Request validation errors and caller cancellation are
// returned as-is: they would fail every deployment the same way, so they never
// cost another call.
//
// Failover replaces retries: every deployment except the last has
// DisableRetries forced on, so a 5xx on the primary hands over immediately
// instead of backing off first (the default retries up to ten times over
// minutes). The last deployment keeps its own retry settings.
//
// Each request's Model is rewritten to the deployment being tried. Provider()
// and Model() report the primary. Close closes every deployment.
//
// Usage reporting: each deployment reports its own attempt to its own
// Config.UsageHook, with UsageEvent.Attempt set to its position in the chain
// and Fallback true for every deployment but the primary, so a failover shows
// up as two events (the failed primary with Err set, then the fallback). A
// fallback whose UsageHook is nil inherits the primary's, so one hook on the
// primary sees the whole chain; give a fallback its own hook to route it
// elsewhere.
//
// Motivation: dedicated preview models can break without notice — on
// 2026-10-07 gemini-3.5-transcribe began rejecting every request with HTTP 400
// while gemini-3.5-flash-lite kept working.
func NewFallbackModalityClient(primary Config, fallbacks ...Config) (ModalityClient, error) {
	configs := append([]Config{primary}, fallbacks...)
	chain := &fallbackModalityClient{}
	for i, cfg := range configs {
		if i < len(configs)-1 {
			cfg.DisableRetries = true
		}
		if cfg.UsageHook == nil {
			cfg.UsageHook = primary.UsageHook
		}
		client, err := NewModalityClient(cfg)
		if err != nil {
			_ = chain.Close()
			return nil, fmt.Errorf("llm: fallback deployment %d (%s/%s): %w", i, cfg.Provider, cfg.Model, err)
		}
		chain.clients = append(chain.clients, client)
	}
	return chain, nil
}

type fallbackModalityClient struct {
	clients []ModalityClient
}

// isProviderFailure reports an error the provider or the network caused, for
// which another deployment may succeed.
func isProviderFailure(ctx context.Context, err error) bool {
	if ctx.Err() != nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return true
	}
	var netErr net.Error
	var urlErr *url.Error
	return errors.As(err, &netErr) || errors.As(err, &urlErr)
}

// tryEach runs call against each deployment until one succeeds or a failure is
// not the provider's. The returned error wraps the LAST attempt's error, so
// IsRateLimited/IsAuthError/errors.As classify the final outcome (what the
// caller should act on); earlier attempts are kept as text.
func tryEach[T any](ctx context.Context, clients []ModalityClient, call func(context.Context, ModalityClient) (T, error)) (T, error) {
	var zero T
	var errs []error
	for i, client := range clients {
		result, err := call(withModalityAttempt(ctx, i), client)
		if err == nil {
			return result, nil
		}
		errs = append(errs, fmt.Errorf("%s/%s: %w", client.Provider(), client.Model(), err))
		if i == len(clients)-1 || !isProviderFailure(ctx, err) {
			break
		}
	}
	last := errs[len(errs)-1]
	if len(errs) == 1 {
		return zero, last
	}
	earlier := make([]string, 0, len(errs)-1)
	for _, err := range errs[:len(errs)-1] {
		earlier = append(earlier, err.Error())
	}
	return zero, fmt.Errorf("llm: all fallback deployments failed; last: %w (earlier: %s)", last, strings.Join(earlier, "; "))
}

func (c *fallbackModalityClient) GenerateEmbeddings(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error) {
	return tryEach(ctx, c.clients, func(ctx context.Context, client ModalityClient) (*EmbeddingResponse, error) {
		req.Model = client.Model()
		return client.GenerateEmbeddings(ctx, req)
	})
}

func (c *fallbackModalityClient) GenerateImages(ctx context.Context, req ImageRequest) (*ImageResponse, error) {
	return tryEach(ctx, c.clients, func(ctx context.Context, client ModalityClient) (*ImageResponse, error) {
		req.Model = client.Model()
		return client.GenerateImages(ctx, req)
	})
}

func (c *fallbackModalityClient) SynthesizeSpeech(ctx context.Context, req SpeechRequest) (*SpeechResponse, error) {
	return tryEach(ctx, c.clients, func(ctx context.Context, client ModalityClient) (*SpeechResponse, error) {
		req.Model = client.Model()
		return client.SynthesizeSpeech(ctx, req)
	})
}

func (c *fallbackModalityClient) TranscribeAudio(ctx context.Context, req TranscriptionRequest) (string, error) {
	return tryEach(ctx, c.clients, func(ctx context.Context, client ModalityClient) (string, error) {
		req.Model = client.Model()
		return client.TranscribeAudio(ctx, req)
	})
}

func (c *fallbackModalityClient) Provider() string { return c.clients[0].Provider() }
func (c *fallbackModalityClient) Model() string    { return c.clients[0].Model() }

// Close closes every deployment; it is safe to call more than once.
func (c *fallbackModalityClient) Close() error {
	var errs []error
	for _, client := range c.clients {
		if err := client.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}
