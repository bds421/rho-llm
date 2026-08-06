package openaibatch

// White-box tests for the unexported batch internals: endpoint/kind inference and
// per-line result parsing. These reach functions an external test cannot.

import (
	"testing"

	llm "github.com/bds421/rho-llm"
)

func testCfg() llm.Config {
	cfg := llm.DefaultConfig()
	cfg.Provider = "openai"
	cfg.APIKey = "k"
	cfg.Model = "gpt-5.3-chat-latest"
	return cfg
}

func TestInferKindChatHomogeneous(t *testing.T) {
	items := []llm.BatchItem{
		{ItemID: "a", Request: &llm.Request{Model: "gpt-5.3-chat-latest"}},
		{ItemID: "b", Request: &llm.Request{Model: "gpt-5.3-chat-latest"}},
	}
	k, err := inferKind(testCfg(), items)
	if err != nil {
		t.Fatalf("inferKind: %v", err)
	}
	if k != kindChat {
		t.Fatalf("expected kindChat, got %v", k)
	}
}

func TestInferKindResponsesModelRoutes(t *testing.T) {
	// A ResponsesAPI model must route to the /v1/responses endpoint, exactly like a
	// live Complete call would.
	items := []llm.BatchItem{{ItemID: "a", Request: &llm.Request{Model: "gpt-5.5"}}}
	k, err := inferKind(testCfg(), items)
	if err != nil {
		t.Fatalf("inferKind: %v", err)
	}
	if k != kindResponses {
		t.Fatalf("expected kindResponses for gpt-5.5, got %v", k)
	}
}

func TestInferKindMixedRejected(t *testing.T) {
	items := []llm.BatchItem{
		{ItemID: "a", Request: &llm.Request{Model: "gpt-5.3-chat-latest"}},
		{ItemID: "b", Embedding: &llm.EmbeddingRequest{Model: "text-embedding-3-small", Input: []string{"x"}}},
	}
	if _, err := inferKind(testCfg(), items); err == nil {
		t.Fatal("expected mixed-kind (chat+embedding) error")
	}
}

func TestInferKindDuplicateRejected(t *testing.T) {
	items := []llm.BatchItem{
		{ItemID: "dup", Request: &llm.Request{Model: "gpt-5.3-chat-latest"}},
		{ItemID: "dup", Request: &llm.Request{Model: "gpt-5.3-chat-latest"}},
	}
	if _, err := inferKind(testCfg(), items); err == nil {
		t.Fatal("expected duplicate custom_id error")
	}
}

func TestInferKindEmptyRejected(t *testing.T) {
	if _, err := inferKind(testCfg(), nil); err == nil {
		t.Fatal("expected error for empty items")
	}
}

func TestOpenAIBatchRequestCountBoundary(t *testing.T) {
	if err := validateBatchRequestCount(maxBatchRequests); err != nil {
		t.Fatalf("request count cap rejected: %v", err)
	}
	if err := validateBatchRequestCount(maxBatchRequests + 1); err == nil {
		t.Fatal("request count cap+1 accepted")
	}
}

func TestOpenAIBatchInputSizeBoundary(t *testing.T) {
	if maxBatchInputBytes != 200_000_000 {
		t.Fatalf("OpenAI batch input limit = %d, want 200 MB", maxBatchInputBytes)
	}
	if err := validateBatchInputBytes(maxBatchInputBytes); err != nil {
		t.Fatalf("input size cap rejected: %v", err)
	}
	if err := validateBatchInputBytes(maxBatchInputBytes + 1); err == nil {
		t.Fatal("input size cap+1 accepted")
	}
}

func TestBuildBatchJSONLInputSizeBoundary(t *testing.T) {
	items := []llm.BatchItem{{
		ItemID: "a",
		Embedding: &llm.EmbeddingRequest{
			Model: "text-embedding-3-small",
			Input: []string{"small fixture"},
		},
	}}
	encoded, err := buildBatchJSONL(kindEmbeddings, items, endpointEmbeddings, nil, nil, 4_096)
	if err != nil {
		t.Fatalf("build fixture: %v", err)
	}
	if _, err := buildBatchJSONL(kindEmbeddings, items, endpointEmbeddings, nil, nil, len(encoded)); err != nil {
		t.Fatalf("exact encoded cap rejected: %v", err)
	}
	if _, err := buildBatchJSONL(kindEmbeddings, items, endpointEmbeddings, nil, nil, len(encoded)-1); err == nil {
		t.Fatal("encoded cap+1 accepted")
	}
}

func TestOpenAIBatchEmbeddingInputBoundaryAcrossLines(t *testing.T) {
	if maxBatchEmbeddingInputs != 50_000 {
		t.Fatalf("OpenAI embedding input limit = %d, want 50,000", maxBatchEmbeddingInputs)
	}
	total, err := addBatchEmbeddingInputs(0, 25_000)
	if err != nil {
		t.Fatalf("first line: %v", err)
	}
	total, err = addBatchEmbeddingInputs(total, 25_000)
	if err != nil || total != maxBatchEmbeddingInputs {
		t.Fatalf("cross-line exact cap: total=%d err=%v", total, err)
	}
	if _, err := addBatchEmbeddingInputs(total, 1); err == nil {
		t.Fatal("cross-line embedding input cap+1 accepted")
	}
}

func newTestClient(t *testing.T) *Client {
	t.Helper()
	c, err := New(testCfg())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

func TestParseOutputLineBlankSkipped(t *testing.T) {
	c := newTestClient(t)
	if _, ok := c.parseOutputLine(endpointChat, []byte("   \n"), nil, nil); ok {
		t.Fatal("expected ok=false for a blank line")
	}
}

func TestParseOutputLineMalformedIsError(t *testing.T) {
	c := newTestClient(t)
	res, ok := c.parseOutputLine(endpointChat, []byte("{ not json"), nil, nil)
	if !ok {
		t.Fatal("expected ok=true for a non-empty line")
	}
	if res.Error == nil {
		t.Fatal("expected an error result for a malformed line")
	}
}

func TestParseOutputLineExplicitError(t *testing.T) {
	c := newTestClient(t)
	raw := []byte(`{"custom_id":"x","error":{"message":"boom"}}`)
	res, ok := c.parseOutputLine(endpointChat, raw, nil, nil)
	if !ok || res.ItemID != "x" || res.Error == nil {
		t.Fatalf("expected correlated error result, got ok=%v res=%+v", ok, res)
	}
}
