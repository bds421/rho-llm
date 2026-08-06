package llm

// modalityDrivers is populated by provider package init functions. Go runs init
// sequentially before main, so the registry is immutable during normal use.
var modalityDrivers = make(map[string]ModalityDriver)

// RegisterModalityDriver registers one non-chat adapter for a wire protocol.
// Provider packages normally call this from init alongside RegisterProvider.
func RegisterModalityDriver(protocol string, driver ModalityDriver) {
	if protocol == "" {
		panic("llm: cannot register modality driver with empty protocol")
	}
	if driver == nil {
		panic("llm: cannot register nil modality driver for " + protocol)
	}
	if _, exists := modalityDrivers[protocol]; exists {
		panic("llm: modality driver already registered for " + protocol)
	}
	modalityDrivers[protocol] = driver
}

func getModalityDriver(protocol string) ModalityDriver {
	return modalityDrivers[protocol]
}
