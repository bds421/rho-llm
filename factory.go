package llm

import (
	"context"
	"fmt"
	"iter"
	"log/slog"
)

// NewClient creates an LLM client based on the provider configuration.
// All clients get automatic retry with exponential backoff on transient errors
// (429, 503, 502). Use NewClientWithKeys for multi-key rotation.
func NewClient(cfg Config) (Client, error) {
	if cfg.DisableRetries {
		return newSingleClient(cfg)
	}
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
	if cfg.DisableRetries {
		if len(keys) != 1 {
			return nil, fmt.Errorf("llm: retries-disabled client requires exactly one credential profile")
		}
		cfg.APIKey = keys[0]
		return newSingleClient(cfg)
	}
	return newPooledClient(cfg, keys)
}

// NewModalityClient creates a durable non-chat client through the modality
// adapter registered for the selected wire protocol. The returned client owns a
// persistent SafeHTTPClient and applies the same retry, cancellation, bounded
// response, error classification, and request capability checks as chat clients.
// Callers should reuse the client and Close it when the worker shuts down.
func NewModalityClient(cfg Config) (ModalityClient, error) {
	cfg.Model = configuredModel(cfg)
	if cfg.Model == "" {
		return nil, fmt.Errorf("model is required")
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = DefaultTimeout
	}
	if _, known := PresetFor(cfg.Provider); !known && cfg.BaseURL == "" {
		return nil, fmt.Errorf("unknown provider %q: set BaseURL for custom providers", cfg.Provider)
	}
	if err := CheckBaseURL(cfg); err != nil {
		return nil, err
	}
	driver, err := modalityDriverFor(cfg)
	if err != nil {
		return nil, err
	}
	client, err := driver.New(cfg)
	if err != nil {
		return nil, err
	}
	return &capabilityValidatedModalityClient{ModalityClient: client, cfg: cfg}, nil
}

type capabilityValidatedModalityClient struct {
	ModalityClient
	cfg Config
}

func (client *capabilityValidatedModalityClient) GenerateEmbeddings(
	ctx context.Context, req EmbeddingRequest,
) (*EmbeddingResponse, error) {
	if err := ValidateEmbeddingRequest(client.cfg, req); err != nil {
		return nil, err
	}
	return client.ModalityClient.GenerateEmbeddings(ctx, req)
}

func (client *capabilityValidatedModalityClient) GenerateImages(
	ctx context.Context, req ImageRequest,
) (*ImageResponse, error) {
	if err := ValidateImageRequest(client.cfg, req); err != nil {
		return nil, err
	}
	return client.ModalityClient.GenerateImages(ctx, req)
}

func (client *capabilityValidatedModalityClient) SynthesizeSpeech(
	ctx context.Context, req SpeechRequest,
) (*SpeechResponse, error) {
	if err := ValidateSpeechRequest(client.cfg, req); err != nil {
		return nil, err
	}
	return client.ModalityClient.SynthesizeSpeech(ctx, req)
}

func (client *capabilityValidatedModalityClient) TranscribeAudio(
	ctx context.Context, req TranscriptionRequest,
) (string, error) {
	if err := ValidateTranscriptionRequest(client.cfg, req); err != nil {
		return "", err
	}
	return client.ModalityClient.TranscribeAudio(ctx, req)
}

// NewBatchClient creates an asynchronous batch client for bulk request processing.
// Batch is a separate execution mode from Complete/Stream — submit → poll → fetch —
// so it returns a BatchClient, not a Client. Only providers whose preset advertises
// SupportsBatch (OpenAI, Anthropic Message Batches, Gemini Batch) are accepted;
// anything else returns a clear error rather than falling through to a nil-factory panic.
//
// cfg.Model is optional here: the model is taken per-item from each BatchItem's
// Request/Embedding, so a single batch client can submit mixed models (subject to the
// driver's single-endpoint homogeneity rule).
func NewBatchClient(cfg Config) (BatchClient, error) {
	cfg.Model = configuredModel(cfg)

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
	cfg.Model = configuredModel(cfg)

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
	client = &capabilityValidatedClient{Client: client, cfg: cfg}

	if cfg.LogRequests {
		client = WithLogging(client)
	}

	return client, nil
}

// capabilityValidatedClient binds every chat dispatch to the exact reviewed
// capabilities carried by the client config. Provider adapters remain concerned
// only with wire translation; this wrapper is applied once by the factory.
type capabilityValidatedClient struct {
	Client
	cfg Config
}

func (client *capabilityValidatedClient) Complete(ctx context.Context, req Request) (*Response, error) {
	if err := ValidateRequestCapabilities(client.cfg, req, false); err != nil {
		return nil, err
	}
	return client.Client.Complete(ctx, req)
}

func (client *capabilityValidatedClient) Stream(ctx context.Context, req Request) iter.Seq2[StreamEvent, error] {
	if err := ValidateRequestCapabilities(client.cfg, req, true); err != nil {
		return func(yield func(StreamEvent, error) bool) {
			yield(StreamEvent{}, err)
		}
	}
	return client.Client.Stream(ctx, req)
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
