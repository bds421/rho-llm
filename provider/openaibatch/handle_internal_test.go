package openaibatch

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/bds421/rho-llm"
)

func validAdapterHandle() llm.BatchHandle {
	return llm.BatchHandle{
		SchemaVersion:       llm.BatchSchemaVersion,
		Provider:            "openai",
		ID:                  "batch-1",
		Operation:           llm.BatchOperationCompletion,
		Status:              llm.BatchRunning,
		AdapterStateVersion: adapterStateVersion,
		AdapterState: json.RawMessage(
			`{"endpoint":"/v1/chat/completions","input_file_id":"file-in"}`,
		),
	}
}

func TestAdapterStateIsStrictAndOperationBound(t *testing.T) {
	tests := map[string]func(*llm.BatchHandle){
		"version": func(handle *llm.BatchHandle) { handle.AdapterStateVersion++ },
		"unknown field": func(handle *llm.BatchHandle) {
			handle.AdapterState = json.RawMessage(
				`{"endpoint":"/v1/chat/completions","input_file_id":"file-in","extra":true}`,
			)
		},
		"unknown endpoint": func(handle *llm.BatchHandle) {
			handle.AdapterState = json.RawMessage(
				`{"endpoint":"/v1/unknown","input_file_id":"file-in"}`,
			)
		},
		"missing input file": func(handle *llm.BatchHandle) {
			handle.AdapterState = json.RawMessage(
				`{"endpoint":"/v1/chat/completions","input_file_id":""}`,
			)
		},
		"operation mismatch": func(handle *llm.BatchHandle) {
			handle.Operation = llm.BatchOperationEmbedding
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			handle := validAdapterHandle()
			mutate(&handle)
			if _, err := decodeAdapterState(handle, "openai"); err == nil {
				t.Fatal("corrupt adapter state accepted")
			}
		})
	}
}

func TestProviderStatusNormalizationIsClosed(t *testing.T) {
	for provider, expected := range map[string]llm.BatchStatus{
		"validating":  llm.BatchQueued,
		"in_progress": llm.BatchRunning,
		"finalizing":  llm.BatchRunning,
		"completed":   llm.BatchCompleted,
		"failed":      llm.BatchFailed,
		"expired":     llm.BatchExpired,
		"cancelling":  llm.BatchCancelling,
		"cancelled":   llm.BatchCancelled,
	} {
		actual, err := normalizeStatus(provider)
		if err != nil || actual != expected {
			t.Fatalf("normalize %q = %q, %v", provider, actual, err)
		}
	}
	if _, err := normalizeStatus("future"); err == nil || !strings.Contains(err.Error(), "unknown") {
		t.Fatalf("unknown status was accepted: %v", err)
	}
}
