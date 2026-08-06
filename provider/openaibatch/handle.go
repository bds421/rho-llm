package openaibatch

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"unicode/utf8"

	"github.com/bds421/rho-llm"
)

const (
	adapterStateVersion = 1
	maxProviderIDBytes  = 256
)

// adapterState is the OpenAI Files + Batches resume contract. It is sealed into
// llm.BatchHandle.AdapterState but remains invisible to the provider-neutral API.
type adapterState struct {
	Endpoint     string `json:"endpoint"`
	InputFileID  string `json:"input_file_id"`
	OutputFileID string `json:"output_file_id,omitempty"`
	ErrorFileID  string `json:"error_file_id,omitempty"`
}

func operationForEndpoint(endpoint string) (llm.BatchOperationKind, error) {
	switch endpoint {
	case endpointChat, endpointResponses:
		return llm.BatchOperationCompletion, nil
	case endpointEmbeddings:
		return llm.BatchOperationEmbedding, nil
	default:
		return "", fmt.Errorf("openaibatch: invalid adapter endpoint")
	}
}

func encodeAdapterState(object *batchObject) (json.RawMessage, llm.BatchOperationKind, error) {
	if object == nil {
		return nil, "", fmt.Errorf("openaibatch: missing provider batch")
	}
	state := adapterState{
		Endpoint: object.Endpoint, InputFileID: object.InputFileID,
		OutputFileID: object.OutputFileID, ErrorFileID: object.ErrorFileID,
	}
	operation, err := validateAdapterState(state)
	if err != nil {
		return nil, "", err
	}
	raw, err := json.Marshal(state)
	if err != nil {
		return nil, "", fmt.Errorf("openaibatch: encode adapter state: %w", err)
	}
	return raw, operation, nil
}

func decodeAdapterState(handle llm.BatchHandle, provider string) (adapterState, error) {
	if err := handle.Validate(); err != nil {
		return adapterState{}, fmt.Errorf("openaibatch: invalid batch handle: %w", err)
	}
	if handle.Provider != provider {
		return adapterState{}, fmt.Errorf("openaibatch: batch handle provider mismatch")
	}
	if handle.AdapterStateVersion != adapterStateVersion {
		return adapterState{}, fmt.Errorf("openaibatch: unsupported adapter state version %d", handle.AdapterStateVersion)
	}
	var state adapterState
	decoder := json.NewDecoder(bytes.NewReader(handle.AdapterState))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&state); err != nil {
		return adapterState{}, fmt.Errorf("openaibatch: decode adapter state: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return adapterState{}, fmt.Errorf("openaibatch: decode adapter state: trailing data")
	}
	operation, err := validateAdapterState(state)
	if err != nil {
		return adapterState{}, err
	}
	if operation != handle.Operation {
		return adapterState{}, fmt.Errorf("openaibatch: adapter state operation mismatch")
	}
	return state, nil
}

func validateAdapterState(state adapterState) (llm.BatchOperationKind, error) {
	if err := validateProviderID("input file id", state.InputFileID, true); err != nil {
		return "", err
	}
	for label, value := range map[string]string{
		"output file id": state.OutputFileID, "error file id": state.ErrorFileID,
	} {
		if err := validateProviderID(label, value, false); err != nil {
			return "", err
		}
	}
	return operationForEndpoint(state.Endpoint)
}

func validateProviderID(label, value string, required bool) error {
	if value == "" && !required {
		return nil
	}
	if value == "" || value != strings.TrimSpace(value) || len(value) > maxProviderIDBytes ||
		!utf8.ValidString(value) {
		return fmt.Errorf("openaibatch: invalid %s", label)
	}
	for _, character := range value {
		if character < 0x21 || character > 0x7e {
			return fmt.Errorf("openaibatch: invalid %s", label)
		}
	}
	return nil
}
