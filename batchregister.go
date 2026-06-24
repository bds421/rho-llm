package llm

// Batch driver registry — the asynchronous counterpart to register.go. Batch
// drivers self-register per protocol in their package init() and are wired through
// provider/all.go (the database/sql driver pattern). Kept separate from the Client
// registry so a protocol can support synchronous completion without batch, or vice
// versa.

// BatchProviderFactory is a constructor that creates a BatchClient from a Config.
type BatchProviderFactory func(cfg Config) (BatchClient, error)

// batchProviders is populated by init() functions in provider sub-packages.
// Immutable after init — no mutex needed.
var batchProviders = make(map[string]BatchProviderFactory)

// RegisterBatchProvider registers a batch driver for the given protocol name.
// Called from init() in provider sub-packages (init runs sequentially before main).
func RegisterBatchProvider(protocol string, factory BatchProviderFactory) {
	batchProviders[protocol] = factory
}

// getBatchProviderFactory returns the registered batch factory for a protocol, or nil.
func getBatchProviderFactory(protocol string) BatchProviderFactory {
	return batchProviders[protocol]
}
