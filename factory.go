package llm

import (
	"fmt"
	"log/slog"
)

// NewClient creates an LLM client based on the provider configuration.
// If cfg.APIKey is set, the client gets automatic retry with exponential backoff
// on transient errors (429, 503, 502). Use NewClientWithKeys for multi-key rotation.
func NewClient(cfg Config) (Client, error) {
	if cfg.APIKey != "" {
		return NewClientWithKeys(cfg, []string{cfg.APIKey})
	}
	return newSingleClient(cfg)
}

// NewClientWithKeys creates an LLM client with optional multiple API keys for rotation.
// Even single-key clients go through PooledClient to get retry/backoff on transient errors.
func NewClientWithKeys(cfg Config, keys []string) (Client, error) {
	if len(keys) >= 1 {
		return newPooledClient(cfg, keys)
	}

	// No keys provided - use config's APIKey directly
	return newSingleClient(cfg)
}

// newSingleClient creates a single (non-pooled) client based on protocol routing.
func newSingleClient(cfg Config) (Client, error) {
	// Resolve model alias to its full identifier
	cfg.Model = ResolveModelAlias(cfg.Model)

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

	clientFunc := func(profile AuthProfile) (Client, error) {
		cfgCopy := cfg
		cfgCopy.APIKey = profile.APIKey
		if profile.BaseURL != "" {
			cfgCopy.BaseURL = profile.BaseURL
		}
		return newSingleClient(cfgCopy)
	}

	return NewPooledClient(cfg, keys, clientFunc)
}
