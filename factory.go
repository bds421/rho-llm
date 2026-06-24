package llm

import (
	"fmt"
	"log/slog"
)

// NewClient creates an LLM client based on the provider configuration.
// All clients get automatic retry with exponential backoff on transient errors
// (429, 503, 502). Use NewClientWithKeys for multi-key rotation.
func NewClient(cfg Config) (Client, error) {
	return NewClientWithKeys(cfg, []string{cfg.APIKey})
}

// NewClientWithKeys creates an LLM client with optional multiple API keys for rotation.
// Every client goes through PooledClient to get retry/backoff on transient errors —
// a nil or empty keys slice falls back to cfg.APIKey, exactly like NewClient.
// Keys may use the format "apikey|baseurl" to override the base URL per key.
func NewClientWithKeys(cfg Config, keys []string) (Client, error) {
	if len(keys) == 0 {
		keys = []string{cfg.APIKey}
	}
	return newPooledClient(cfg, keys)
}

// NewBatchClient creates an asynchronous batch client for bulk request processing.
// Batch is a separate execution mode from Complete/Stream — submit → poll → fetch —
// so it returns a BatchClient, not a Client. Only providers whose preset advertises
// SupportsBatch (currently first-party OpenAI) are accepted; anything else returns a
// clear error rather than falling through to a nil-factory panic.
//
// cfg.Model is optional here: the model is taken per-item from each BatchItem's
// Request/Embedding, so a single batch client can submit mixed models (subject to the
// driver's single-endpoint homogeneity rule).
func NewBatchClient(cfg Config) (BatchClient, error) {
	cfg.Model = ResolveModelAlias(cfg.Model)

	if cfg.Timeout <= 0 {
		cfg.Timeout = DefaultTimeout
	}

	preset, known := PresetFor(cfg.Provider)
	if !known {
		return nil, fmt.Errorf("unknown provider %q: the batch API is not supported (set a known provider)", cfg.Provider)
	}
	if !preset.SupportsBatch {
		return nil, fmt.Errorf("provider %q does not support the batch API", cfg.Provider)
	}

	// Opt-in SSRF hardening, consistent with newSingleClient.
	if err := CheckBaseURL(cfg); err != nil {
		return nil, err
	}

	protocol := ResolveProtocol(cfg)
	factory := getBatchProviderFactory(protocol)
	if factory == nil {
		return nil, fmt.Errorf("no registered batch driver for protocol %q (provider %q)", protocol, cfg.Provider)
	}
	return factory(cfg)
}

// newSingleClient creates a single (non-pooled) client based on protocol routing.
func newSingleClient(cfg Config) (Client, error) {
	// Resolve model alias to its full identifier
	cfg.Model = ResolveModelAlias(cfg.Model)

	if cfg.Model == "" {
		return nil, fmt.Errorf("model is required")
	}
	if cfg.MaxTokens < 0 {
		return nil, fmt.Errorf("MaxTokens must be >= 0, got %d", cfg.MaxTokens)
	}
	if cfg.Temperature != nil && *cfg.Temperature < 0 {
		return nil, fmt.Errorf("Temperature must be >= 0, got %f", *cfg.Temperature)
	}

	// Apply timeout floor — prevents unbounded HTTP clients when callers
	// construct Config manually without calling DefaultConfig().
	if cfg.Timeout <= 0 {
		cfg.Timeout = DefaultTimeout
	}

	// Unknown providers must specify a BaseURL for custom endpoints.
	if _, known := PresetFor(cfg.Provider); !known && cfg.BaseURL == "" {
		return nil, fmt.Errorf("unknown provider %q: set BaseURL for custom providers", cfg.Provider)
	}

	// Opt-in SSRF hardening — also covers the per-key "apikey|baseurl" override,
	// since pooled clients construct each profile through newSingleClient.
	if err := CheckBaseURL(cfg); err != nil {
		return nil, err
	}

	protocol := ResolveProtocol(cfg)

	// Look up the registered provider factory for this protocol
	factory := getProviderFactory(protocol)
	if factory == nil {
		return nil, fmt.Errorf("unsupported protocol %q for provider %q (no registered driver)", protocol, cfg.Provider)
	}

	client, err := factory(cfg)

	if err != nil {
		return nil, err
	}

	if cfg.LogRequests {
		client = WithLogging(client)
	}

	return client, nil
}

// newPooledClient creates a pooled client with auth rotation.
func newPooledClient(cfg Config, keys []string) (Client, error) {
	slog.Info("creating pooled client", "profiles", len(keys), "provider", cfg.Provider)

	// Logging is applied once at the pool level, not per-inner-client.
	// Without this, each rotated inner client gets its own LoggingClient,
	// and the pool-level wrapper doubles the output.
	wantLog := cfg.LogRequests

	clientFunc := func(profile AuthProfile) (Client, error) {
		cfgCopy := cfg
		cfgCopy.APIKey = profile.APIKey
		cfgCopy.LogRequests = false // prevent inner LoggingClient wrapping
		if profile.BaseURL != "" {
			cfgCopy.BaseURL = profile.BaseURL
		}
		return newSingleClient(cfgCopy)
	}

	pc, err := NewPooledClient(cfg, keys, clientFunc)
	if err != nil {
		return nil, err
	}

	if wantLog {
		return WithLogging(pc), nil
	}
	return pc, nil
}
