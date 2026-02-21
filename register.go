package llm

// ProviderFactory is a constructor that creates a Client from a Config.
type ProviderFactory func(cfg Config) (Client, error)

// providers is populated by init() functions in provider sub-packages.
// Immutable after init — no mutex needed.
var providers = make(map[string]ProviderFactory)

// RegisterProvider registers a provider factory for the given protocol name.
// Called from init() in provider sub-packages (init runs sequentially before main).
func RegisterProvider(protocol string, factory ProviderFactory) {
	providers[protocol] = factory
}

// getProviderFactory returns the registered factory for a protocol, or nil.
func getProviderFactory(protocol string) ProviderFactory {
	return providers[protocol]
}
