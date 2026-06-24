package openaibatch

// Hardening round 2 — white-box adversarial coverage for the unexported defensive
// branches the PR suite never reached: parseOutputLine's translator-missing /
// neither-nor / unknown-endpoint / embeddings paths, and the early-return error
// guards in New, getObject, endpointFor, and buildLineBody. These must yield bounded
// error results — never a panic or a nil-deref.

import (
	"context"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/provider/openaicompat"
	"github.com/bds421/rho-llm/provider/openairesponses"
)

// A 2xx responses line with no translator wired must degrade to an error result, not
// dereference a nil *openairesponses.Client. (Results always builds the translator, so
// this branch is only reachable white-box — exactly why it was at 0%.)
func TestParseOutputLineResponsesNoTranslatorIsError(t *testing.T) {
	c := newTestClient(t)
	raw := []byte(`{"custom_id":"x","response":{"status_code":200,"body":{"id":"r","status":"completed","output":[]}}}`)
	res, ok := c.parseOutputLine(endpointResponses, raw, nil, nil) // respT == nil
	if !ok {
		t.Fatal("expected ok=true for a non-empty line")
	}
	if res.Error == nil || !strings.Contains(res.Error.Message, "no responses translator") {
		t.Fatalf("expected a 'no responses translator' error result, got %+v", res)
	}
}

func TestParseOutputLineChatNoTranslatorIsError(t *testing.T) {
	c := newTestClient(t)
	raw := []byte(`{"custom_id":"x","response":{"status_code":200,"body":{"id":"r","choices":[]}}}`)
	res, ok := c.parseOutputLine(endpointChat, raw, nil, nil) // chatT == nil
	if !ok || res.Error == nil || !strings.Contains(res.Error.Message, "no chat translator") {
		t.Fatalf("expected a 'no chat translator' error result, got ok=%v res=%+v", ok, res)
	}
}

// A line with neither a response nor an error object is malformed-but-correlatable:
// it must become an error result keyed by its custom_id, not silently vanish.
func TestParseOutputLineNeitherResponseNorError(t *testing.T) {
	c := newTestClient(t)
	for _, raw := range [][]byte{
		[]byte(`{"custom_id":"x"}`),
		[]byte(`{"custom_id":"x","error":null,"response":null}`),
	} {
		res, ok := c.parseOutputLine(endpointChat, raw, nil, nil)
		if !ok || res.CustomID != "x" || res.Error == nil {
			t.Fatalf("expected a correlated error result for %q, got ok=%v res=%+v", raw, ok, res)
		}
		if !strings.Contains(res.Error.Message, "neither response nor error") {
			t.Fatalf("expected 'neither response nor error' message, got %q", res.Error.Message)
		}
	}
}

// The embeddings result branch needs no translator; a valid body parses, a malformed
// one degrades to an error result.
func TestParseOutputLineEmbeddings(t *testing.T) {
	c := newTestClient(t)
	good := []byte(`{"custom_id":"e","response":{"status_code":200,"body":{"model":"m","data":[{"index":0,"embedding":[0.1,0.2]}],"usage":{"prompt_tokens":3}}}}`)
	res, ok := c.parseOutputLine(endpointEmbeddings, good, nil, nil)
	if !ok || res.Error != nil || res.Embedding == nil || len(res.Embedding.Embeddings) != 1 {
		t.Fatalf("expected a parsed embedding, got ok=%v res=%+v", ok, res)
	}

	bad := []byte(`{"custom_id":"e","response":{"status_code":200,"body":"not-an-object"}}`)
	res2, _ := c.parseOutputLine(endpointEmbeddings, bad, nil, nil)
	if res2.Error == nil || !strings.Contains(res2.Error.Message, "decode embeddings result body") {
		t.Fatalf("expected an embeddings-decode error, got %+v", res2)
	}
}

// An unrecognized endpoint must produce a bounded error result, not panic.
func TestParseOutputLineUnknownEndpoint(t *testing.T) {
	c := newTestClient(t)
	raw := []byte(`{"custom_id":"x","response":{"status_code":200,"body":{}}}`)
	res, ok := c.parseOutputLine("/v1/something-new", raw, nil, nil)
	if !ok || res.Error == nil || !strings.Contains(res.Error.Message, "unknown batch endpoint") {
		t.Fatalf("expected 'unknown batch endpoint' error, got ok=%v res=%+v", ok, res)
	}
}

func TestNewRequiresAPIKey(t *testing.T) {
	cfg := testCfg()
	cfg.APIKey = "" // openai is not a no-auth provider
	if _, err := New(cfg); err == nil || !strings.Contains(err.Error(), "API key is required") {
		t.Fatalf("expected missing-API-key error, got: %v", err)
	}
}

func TestGetObjectEmptyIDRejected(t *testing.T) {
	c := newTestClient(t)
	// Empty id must fail before any network effect (no server is running here).
	if _, err := c.getObject(context.Background(), ""); err == nil || !strings.Contains(err.Error(), "empty batch id") {
		t.Fatalf("expected empty-batch-id error, got: %v", err)
	}
}

func TestEndpointForUnknownKind(t *testing.T) {
	if got := endpointFor(kindUnknown); got != "" {
		t.Fatalf("endpointFor(kindUnknown) must be empty, got %q", got)
	}
}

func TestBuildLineBodyUnknownKind(t *testing.T) {
	if _, err := buildLineBody(kindUnknown, llm.BatchItem{CustomID: "x"}, nil, nil); err == nil {
		t.Fatal("expected an error for an unknown batch kind")
	}
}

// A hostile/corrupt result file may carry a well-formed envelope (custom_id + 2xx
// response) but a body that is not a valid chat/responses object. The translator's
// decode must fail into a bounded error result, not a panic — and the custom_id
// correlation must survive so the caller knows which item failed.
func TestParseOutputLineChatBodyDecodeError(t *testing.T) {
	c := newTestClient(t)
	tr, err := openaicompat.BatchTranslator(testCfg())
	if err != nil {
		t.Fatalf("BatchTranslator: %v", err)
	}
	raw := []byte(`{"custom_id":"x","response":{"status_code":200,"body":"not-a-chat-object"}}`)
	res, ok := c.parseOutputLine(endpointChat, raw, tr, nil)
	if !ok || res.CustomID != "x" || res.Error == nil || !strings.Contains(res.Error.Message, "decode chat result body") {
		t.Fatalf("expected a correlated 'decode chat result body' error, got ok=%v res=%+v", ok, res)
	}
}

func TestParseOutputLineResponsesBodyDecodeError(t *testing.T) {
	c := newTestClient(t)
	tr, err := openairesponses.BatchTranslator(testCfg())
	if err != nil {
		t.Fatalf("BatchTranslator: %v", err)
	}
	raw := []byte(`{"custom_id":"y","response":{"status_code":200,"body":42}}`)
	res, ok := c.parseOutputLine(endpointResponses, raw, nil, tr)
	if !ok || res.CustomID != "y" || res.Error == nil || !strings.Contains(res.Error.Message, "decode responses result body") {
		t.Fatalf("expected a correlated 'decode responses result body' error, got ok=%v res=%+v", ok, res)
	}
}
