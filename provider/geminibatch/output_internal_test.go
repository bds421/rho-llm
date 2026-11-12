package geminibatch

import (
	"encoding/json"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

const genItem = `{"metadata":{"key":"item-a"},"response":{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}}`
const embItem = `{"metadata":{"key":"item-e"},"response":{"embedding":{"values":[0.5,0.25]}}}`

func newTestClient(t *testing.T) *Client {
	t.Helper()
	c, err := New(llm.Config{Provider: "gemini", Model: "gemini-3.6-flash", APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func stateOf(t *testing.T, c *Client, body string, op llm.BatchOperationKind) adapterState {
	t.Helper()
	var wire batchResource
	if err := json.Unmarshal([]byte(body), &wire); err != nil {
		t.Fatalf("decode %s: %v", body, err)
	}
	h, err := c.toHandle(&wire, op, "gemini-3.6-flash")
	if err != nil {
		t.Fatal(err)
	}
	st, err := decodeState(*h, c.providerName)
	if err != nil {
		t.Fatal(err)
	}
	return st
}

// The API documents inline results as an object wrapping the array, under
// response, response.output or metadata.output. Only a bare array under dest
// or response was read, so documented results failed to decode (B28).
func TestDocumentedOutputShapesDecode(t *testing.T) {
	c := newTestClient(t)
	wrapped := `{"inlinedResponses":[` + genItem + `]}`
	for name, body := range map[string]string{
		"response wrapped":        `{"name":"batches/j","metadata":{"state":"BATCH_STATE_SUCCEEDED"},"done":true,"response":{"inlinedResponses":` + wrapped + `}}`,
		"response.output wrapped": `{"name":"batches/j","metadata":{"state":"BATCH_STATE_SUCCEEDED"},"done":true,"response":{"output":{"inlinedResponses":` + wrapped + `}}}`,
		"metadata.output wrapped": `{"name":"batches/j","metadata":{"state":"BATCH_STATE_SUCCEEDED","output":{"inlinedResponses":` + wrapped + `}},"done":true}`,
		"dest bare array":         `{"name":"batches/j","state":"BATCH_STATE_SUCCEEDED","dest":{"inlinedResponses":[` + genItem + `]}}`,
	} {
		st := stateOf(t, c, body, llm.BatchOperationCompletion)
		res, err := c.parseCompletionResults(st.InlinedResponses, st.Model)
		if err != nil || len(res) != 1 || res[0].ItemID != "item-a" || res[0].Response == nil || res[0].Response.Content != "ok" {
			t.Errorf("%s: results=%+v err=%v", name, res, err)
		}
	}
	// Embedding batches: EmbedContentBatchOutput.inlinedResponses (wrapped).
	st := stateOf(t, c, `{"name":"batches/e","metadata":{"state":"BATCH_STATE_SUCCEEDED"},"done":true,"response":{"output":{"inlinedResponses":{"inlinedResponses":[`+embItem+`]}}}}`, llm.BatchOperationEmbedding)
	res, err := c.parseEmbeddingResults(st.InlinedEmbedResponses)
	if err != nil || len(res) != 1 || res[0].Embedding == nil || len(res[0].Embedding.Embeddings[0].Vector) != 2 {
		t.Fatalf("embedding results=%+v err=%v", res, err)
	}
}

// Results written to a file instead of inline were returned as an empty,
// successful result set.
func TestResponsesFileIsReported(t *testing.T) {
	c := newTestClient(t)
	st := stateOf(t, c, `{"name":"batches/j","metadata":{"state":"BATCH_STATE_SUCCEEDED"},"done":true,"response":{"responsesFile":"files/out-1"}}`, llm.BatchOperationCompletion)
	if st.ResponsesFile != "files/out-1" {
		t.Fatalf("responses file not kept: %+v", st)
	}
	var wire batchResource
	if err := json.Unmarshal([]byte(`{"name":"batches/j","response":{"inlinedResponses":"nonsense"}}`), &wire); err == nil || !strings.Contains(err.Error(), "cannot unmarshal") {
		t.Fatalf("malformed inline results: err = %v", err)
	}
}
