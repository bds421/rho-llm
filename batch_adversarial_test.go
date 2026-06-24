package llm_test

// Adversarial tests for the Batch API. Per CLAUDE.md: these must genuinely try to
// break the code — bad item sets fail before any network effect, hostile/mixed/
// malformed result lines are handled per-line, oversize downloads are rejected, and
// the serialized handle round-trips so Results works after a restart with no items.
//
// (Named batch_adversarial_test.go, NOT batch_test.go — that file already exists and
// tests unrelated Mock/discovery wrappers.)

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	llm "github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/provider/openaicompat"

	// Register the OpenAI batch driver (and, transitively, the chat/responses adapters).
	_ "github.com/bds421/rho-llm/provider/openaibatch"
)

// ---- helpers ---------------------------------------------------------------

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func newBatchClient(t *testing.T, baseURL string) llm.BatchClient {
	t.Helper()
	cfg := llm.DefaultConfig()
	cfg.Provider = "openai"
	cfg.APIKey = "test-key"
	cfg.BaseURL = baseURL
	cfg.Model = "gpt-5.3-chat-latest"
	bc, err := llm.NewBatchClient(cfg)
	if err != nil {
		t.Fatalf("NewBatchClient: %v", err)
	}
	t.Cleanup(func() { _ = bc.Close() })
	return bc
}

func chatItem(id string) llm.BatchItem {
	return llm.BatchItem{CustomID: id, Request: &llm.Request{
		Model:     "gpt-5.3-chat-latest",
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentPart{{Type: llm.ContentText, Text: "hi " + id}}}},
		MaxTokens: 64,
	}}
}

func embItem(id string) llm.BatchItem {
	return llm.BatchItem{CustomID: id, Embedding: &llm.EmbeddingRequest{Model: "text-embedding-3-small", Input: []string{"hi " + id}}}
}

func chatBody(content string, in, out int) map[string]any {
	return map[string]any{
		"id": "resp", "model": "gpt-5.3-chat-latest",
		"choices": []any{map[string]any{"index": 0, "message": map[string]any{"role": "assistant", "content": content}, "finish_reason": "stop"}},
		"usage":   map[string]any{"prompt_tokens": in, "completion_tokens": out, "total_tokens": in + out},
	}
}

func successLine(id string, body map[string]any) string {
	b, _ := json.Marshal(map[string]any{"custom_id": id, "response": map[string]any{"status_code": 200, "request_id": "r", "body": body}})
	return string(b) + "\n"
}

// batchEnv serves the OpenAI Files+Batches REST surface from in-memory canned data.
type batchEnv struct {
	endpoint     string // batch endpoint, e.g. "/v1/chat/completions"
	status       string // status returned by GET /batches/{id}
	outputFileID string
	errorFileID  string
	files        map[string]string // file_id -> raw content for /files/{id}/content
	uploaded     [][]byte          // captured JSONL uploads
	mu           sync.Mutex
}

func (e *batchEnv) batchObject() map[string]any {
	return map[string]any{
		"id": "batch-1", "status": e.status, "endpoint": e.endpoint,
		"input_file_id": "file-in", "output_file_id": e.outputFileID, "error_file_id": e.errorFileID,
		"created_at": 1700000000, "request_counts": map[string]int{"total": 1, "completed": 1, "failed": 0},
	}
}

func (e *batchEnv) handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/files":
			body, _ := io.ReadAll(r.Body)
			e.mu.Lock()
			e.uploaded = append(e.uploaded, body)
			e.mu.Unlock()
			writeJSON(w, map[string]any{"id": "file-in"})
		case r.Method == http.MethodPost && r.URL.Path == "/batches":
			var req map[string]any
			_ = json.NewDecoder(r.Body).Decode(&req)
			obj := e.batchObject()
			obj["status"] = "validating"
			if ep, ok := req["endpoint"]; ok {
				obj["endpoint"] = ep
			}
			writeJSON(w, obj)
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/cancel"):
			obj := e.batchObject()
			obj["status"] = "cancelled"
			writeJSON(w, obj)
		case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/batches/"):
			writeJSON(w, e.batchObject())
		case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/files/") && strings.HasSuffix(r.URL.Path, "/content"):
			id := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/files/"), "/content")
			content, ok := e.files[id]
			if !ok {
				http.Error(w, "no such file", http.StatusNotFound)
				return
			}
			_, _ = io.WriteString(w, content)
		default:
			http.Error(w, "unexpected "+r.Method+" "+r.URL.Path, http.StatusInternalServerError)
		}
	}
}

// ---- Submit validation: must fail BEFORE any network effect -----------------

func countingServer(t *testing.T) (*httptest.Server, *int32) {
	t.Helper()
	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&n, 1)
		writeJSON(w, map[string]any{"id": "file-in"})
	}))
	t.Cleanup(srv.Close)
	return srv, &n
}

func TestBatchSubmitMixedKindsRejectedNoNetwork(t *testing.T) {
	srv, n := countingServer(t)
	bc := newBatchClient(t, srv.URL)
	_, err := bc.Submit(context.Background(), []llm.BatchItem{chatItem("a"), embItem("b")}, llm.BatchOptions{})
	if err == nil {
		t.Fatal("expected homogeneity error for mixed chat+embedding batch")
	}
	if !strings.Contains(err.Error(), "homogeneous") {
		t.Fatalf("expected homogeneity error, got: %v", err)
	}
	if got := atomic.LoadInt32(n); got != 0 {
		t.Fatalf("expected 0 network requests on invalid submit, got %d", got)
	}
}

func TestBatchSubmitChatResponsesMixedRejected(t *testing.T) {
	srv, n := countingServer(t)
	bc := newBatchClient(t, srv.URL)
	resp := chatItem("b")
	resp.Request.Model = "gpt-5.5" // ResponsesAPI model → /v1/responses
	_, err := bc.Submit(context.Background(), []llm.BatchItem{chatItem("a"), resp}, llm.BatchOptions{})
	if err == nil || !strings.Contains(err.Error(), "homogeneous") {
		t.Fatalf("expected homogeneity error mixing chat+responses, got: %v", err)
	}
	if got := atomic.LoadInt32(n); got != 0 {
		t.Fatalf("expected 0 network requests, got %d", got)
	}
}

func TestBatchSubmitTaggedUnionRejected(t *testing.T) {
	srv, n := countingServer(t)
	bc := newBatchClient(t, srv.URL)
	both := llm.BatchItem{CustomID: "x", Request: chatItem("x").Request, Embedding: embItem("x").Embedding}
	neither := llm.BatchItem{CustomID: "y"}
	for name, item := range map[string]llm.BatchItem{"both": both, "neither": neither} {
		if _, err := bc.Submit(context.Background(), []llm.BatchItem{item}, llm.BatchOptions{}); err == nil {
			t.Fatalf("%s: expected tagged-union error", name)
		}
	}
	if got := atomic.LoadInt32(n); got != 0 {
		t.Fatalf("expected 0 network requests, got %d", got)
	}
}

func TestBatchSubmitDuplicateCustomIDRejected(t *testing.T) {
	srv, n := countingServer(t)
	bc := newBatchClient(t, srv.URL)
	_, err := bc.Submit(context.Background(), []llm.BatchItem{chatItem("dup"), chatItem("dup")}, llm.BatchOptions{})
	if err == nil || !strings.Contains(err.Error(), "duplicate custom_id") {
		t.Fatalf("expected duplicate custom_id error, got: %v", err)
	}
	if got := atomic.LoadInt32(n); got != 0 {
		t.Fatalf("expected 0 network requests, got %d", got)
	}
}

func TestBatchSubmitEmptyAndBlankIDRejected(t *testing.T) {
	srv, n := countingServer(t)
	bc := newBatchClient(t, srv.URL)
	if _, err := bc.Submit(context.Background(), nil, llm.BatchOptions{}); err == nil {
		t.Fatal("expected error for empty items")
	}
	if _, err := bc.Submit(context.Background(), []llm.BatchItem{chatItem("")}, llm.BatchOptions{}); err == nil {
		t.Fatal("expected error for empty custom_id")
	}
	if got := atomic.LoadInt32(n); got != 0 {
		t.Fatalf("expected 0 network requests, got %d", got)
	}
}

// ---- Submit happy path ------------------------------------------------------

func TestBatchSubmitUploadsJSONL(t *testing.T) {
	env := &batchEnv{endpoint: "/v1/chat/completions", status: "validating"}
	srv := httptest.NewServer(env.handler())
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)

	h, err := bc.Submit(context.Background(), []llm.BatchItem{chatItem("a"), chatItem("b")}, llm.BatchOptions{})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if h.ID != "batch-1" || h.Endpoint != "/v1/chat/completions" {
		t.Fatalf("unexpected handle: %+v", h)
	}
	env.mu.Lock()
	defer env.mu.Unlock()
	if len(env.uploaded) != 1 {
		t.Fatalf("expected 1 multipart upload, got %d", len(env.uploaded))
	}
	// The multipart body must carry two JSONL lines, each with its custom_id and the
	// absolute /v1 url.
	body := string(env.uploaded[0])
	for _, want := range []string{`"custom_id":"a"`, `"custom_id":"b"`, `"url":"/v1/chat/completions"`} {
		if !strings.Contains(body, want) {
			t.Fatalf("uploaded JSONL missing %q; got:\n%s", want, body)
		}
	}
}

// ---- Results robustness -----------------------------------------------------

func TestBatchResultsHappyCorrelation(t *testing.T) {
	env := &batchEnv{
		endpoint: "/v1/chat/completions", status: "completed", outputFileID: "file-out",
		files: map[string]string{"file-out": successLine("a", chatBody("hello-a", 10, 5)) + successLine("b", chatBody("hello-b", 7, 3))},
	}
	srv := httptest.NewServer(env.handler())
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)

	results, err := bc.Results(context.Background(), "batch-1")
	if err != nil {
		t.Fatalf("Results: %v", err)
	}
	got := map[string]string{}
	for _, r := range results {
		if r.Error != nil {
			t.Fatalf("unexpected error for %q: %v", r.CustomID, r.Error)
		}
		if r.Response == nil {
			t.Fatalf("nil response for %q", r.CustomID)
		}
		got[r.CustomID] = r.Response.Content
	}
	if got["a"] != "hello-a" || got["b"] != "hello-b" {
		t.Fatalf("bad correlation: %+v", got)
	}
}

func TestBatchResultsMalformedLineBecomesError(t *testing.T) {
	env := &batchEnv{
		endpoint: "/v1/chat/completions", status: "completed", outputFileID: "file-out",
		files: map[string]string{"file-out": "{ this is not json }\n" + successLine("ok", chatBody("good", 1, 1))},
	}
	srv := httptest.NewServer(env.handler())
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)

	results, err := bc.Results(context.Background(), "batch-1")
	if err != nil {
		t.Fatalf("Results should not abort on one bad line: %v", err)
	}
	var sawError, sawGood bool
	for _, r := range results {
		if r.Error != nil {
			sawError = true
		}
		if r.CustomID == "ok" && r.Response != nil && r.Response.Content == "good" {
			sawGood = true
		}
	}
	if !sawError || !sawGood {
		t.Fatalf("expected one error result AND the good line to still parse; got %+v", results)
	}
}

func TestBatchResultsNon2xxLineIsError(t *testing.T) {
	rateLine, _ := json.Marshal(map[string]any{"custom_id": "rl", "response": map[string]any{"status_code": 429, "body": map[string]any{"error": map[string]any{"message": "slow down"}}}})
	env := &batchEnv{
		endpoint: "/v1/chat/completions", status: "completed", outputFileID: "file-out",
		files: map[string]string{"file-out": string(rateLine) + "\n"},
	}
	srv := httptest.NewServer(env.handler())
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)

	results, err := bc.Results(context.Background(), "batch-1")
	if err != nil {
		t.Fatalf("Results: %v", err)
	}
	if len(results) != 1 || results[0].Error == nil {
		t.Fatalf("expected one error result, got %+v", results)
	}
	if !llm.IsRateLimited(results[0].Error) {
		t.Fatalf("expected 429 to classify as rate-limited, got %v", results[0].Error)
	}
}

func TestBatchResultsErrorFileMerged(t *testing.T) {
	errLine, _ := json.Marshal(map[string]any{"custom_id": "bad", "error": map[string]any{"message": "validation failed"}})
	env := &batchEnv{
		endpoint: "/v1/chat/completions", status: "completed", outputFileID: "file-out", errorFileID: "file-err",
		files: map[string]string{
			"file-out": successLine("good", chatBody("ok", 1, 1)),
			"file-err": string(errLine) + "\n",
		},
	}
	srv := httptest.NewServer(env.handler())
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)

	results, err := bc.Results(context.Background(), "batch-1")
	if err != nil {
		t.Fatalf("Results: %v", err)
	}
	byID := map[string]llm.BatchResult{}
	for _, r := range results {
		byID[r.CustomID] = r
	}
	if byID["good"].Response == nil {
		t.Fatal("expected success from output file")
	}
	if byID["bad"].Error == nil || !strings.Contains(byID["bad"].Error.Message, "validation failed") {
		t.Fatalf("expected merged error from error file, got %+v", byID["bad"])
	}
}

func TestBatchResultsOversizeDownloadRejected(t *testing.T) {
	big := strings.Repeat("x", 4096)
	env := &batchEnv{
		endpoint: "/v1/chat/completions", status: "completed", outputFileID: "file-out",
		files: map[string]string{"file-out": big},
	}
	srv := httptest.NewServer(env.handler())
	defer srv.Close()

	cfg := llm.DefaultConfig()
	cfg.Provider = "openai"
	cfg.APIKey = "k"
	cfg.BaseURL = srv.URL
	cfg.MaxBatchDownloadBytes = 1024 // far below the 4096-byte file
	bc, err := llm.NewBatchClient(cfg)
	if err != nil {
		t.Fatalf("NewBatchClient: %v", err)
	}
	defer bc.Close()

	if _, err := bc.Results(context.Background(), "batch-1"); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("expected oversize download to be rejected, got: %v", err)
	}
}

func TestBatchResultsFailedNoFilesNoPanic(t *testing.T) {
	env := &batchEnv{endpoint: "/v1/chat/completions", status: "failed"} // no output/error files
	srv := httptest.NewServer(env.handler())
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)

	results, err := bc.Results(context.Background(), "batch-1")
	if err != nil {
		t.Fatalf("Results on failed batch should not error: %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("expected no results for a failed batch with no files, got %d", len(results))
	}
}

func TestBatchResultsEmbeddings(t *testing.T) {
	embBody := map[string]any{"model": "text-embedding-3-small", "data": []any{map[string]any{"index": 0, "embedding": []float64{0.1, 0.2, 0.3}}}, "usage": map[string]any{"prompt_tokens": 4}}
	env := &batchEnv{
		endpoint: "/v1/embeddings", status: "completed", outputFileID: "file-out",
		files: map[string]string{"file-out": successLine("e", embBody)},
	}
	srv := httptest.NewServer(env.handler())
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)

	results, err := bc.Results(context.Background(), "batch-1")
	if err != nil {
		t.Fatalf("Results: %v", err)
	}
	if len(results) != 1 || results[0].Embedding == nil || len(results[0].Embedding.Embeddings) != 1 {
		t.Fatalf("expected one embedding result, got %+v", results)
	}
	if v := results[0].Embedding.Embeddings[0].Vector; len(v) != 3 {
		t.Fatalf("expected 3-dim vector, got %v", v)
	}
}

// ---- Resume / serialization -------------------------------------------------

func TestBatchHandleRoundTripAndResultsFromEndpoint(t *testing.T) {
	env := &batchEnv{
		endpoint: "/v1/chat/completions", status: "completed", outputFileID: "file-out",
		files: map[string]string{"file-out": successLine("a", chatBody("resumed", 1, 1))},
	}
	srv := httptest.NewServer(env.handler())
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)

	h, err := bc.Get(context.Background(), "batch-1")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	raw, err := json.Marshal(h)
	if err != nil {
		t.Fatalf("marshal handle: %v", err)
	}
	loaded, err := llm.LoadBatchHandle(raw)
	if err != nil {
		t.Fatalf("LoadBatchHandle: %v", err)
	}
	if loaded.Endpoint != "/v1/chat/completions" || loaded.ID != "batch-1" {
		t.Fatalf("round-trip lost fields: %+v", loaded)
	}
	// Results works from the (deserialized) batch id alone — the endpoint on the
	// server-side object drives the parser.
	results, err := bc.Results(context.Background(), loaded.ID)
	if err != nil || len(results) != 1 || results[0].Response == nil || results[0].Response.Content != "resumed" {
		t.Fatalf("resume Results failed: %v / %+v", err, results)
	}
}

func TestLoadBatchHandleRejectsSchemaVersions(t *testing.T) {
	if _, err := llm.LoadBatchHandle([]byte(`{"schema_version":2,"id":"x"}`)); err == nil {
		t.Fatal("expected rejection of newer schema_version 2")
	}
	// Absent/zero version is treated as current (forward-compat from pre-versioned writes).
	if _, err := llm.LoadBatchHandle([]byte(`{"id":"x"}`)); err != nil {
		t.Fatalf("absent schema_version should be accepted, got: %v", err)
	}
}

// ---- Routing ----------------------------------------------------------------

func TestNewBatchClientUnsupportedProvider(t *testing.T) {
	cfg := llm.DefaultConfig()
	cfg.Provider = "anthropic"
	cfg.APIKey = "k"
	if _, err := llm.NewBatchClient(cfg); err == nil || !strings.Contains(err.Error(), "does not support the batch API") {
		t.Fatalf("expected unsupported-provider error, got: %v", err)
	}
}

func TestNewBatchClientUnknownProvider(t *testing.T) {
	cfg := llm.DefaultConfig()
	cfg.Provider = "totally-unknown"
	cfg.APIKey = "k"
	if _, err := llm.NewBatchClient(cfg); err == nil {
		t.Fatal("expected error for unknown provider")
	}
}

// ---- Parity: batch line body == live POST body ------------------------------

func TestBatchChatLineBodyMatchesLivePost(t *testing.T) {
	var posted []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/chat/completions" {
			posted, _ = io.ReadAll(r.Body)
			writeJSON(w, chatBody("ok", 1, 1))
			return
		}
		http.Error(w, "unexpected", http.StatusInternalServerError)
	}))
	defer srv.Close()

	cfg := llm.DefaultConfig()
	cfg.Provider = "openai"
	cfg.APIKey = "k"
	cfg.BaseURL = srv.URL
	cfg.Model = "gpt-5.3-chat-latest"

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	req := llm.Request{
		Model:     "gpt-5.3-chat-latest",
		System:    "be terse",
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentPart{{Type: llm.ContentText, Text: "hi <there> & you"}}}},
		MaxTokens: 100,
	}
	if _, err := client.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	tr, err := openaicompat.BatchTranslator(cfg)
	if err != nil {
		t.Fatalf("BatchTranslator: %v", err)
	}
	batchBody, err := tr.BuildChatBatchLineBody(req)
	if err != nil {
		t.Fatalf("BuildChatBatchLineBody: %v", err)
	}
	if !bytes.Equal(bytes.TrimSpace(posted), bytes.TrimSpace(batchBody)) {
		t.Fatalf("batch line body must equal live POST body:\n live  = %s\n batch = %s", posted, batchBody)
	}
}

// ---- Cost: batch is exactly half ------------------------------------------

func TestEstimateCostBatchIsHalf(t *testing.T) {
	in := llm.CostInput{Model: "gpt-5.3-chat-latest", InputTokens: 1_000_000, OutputTokens: 1_000_000}
	full := llm.EstimateCost(in)
	in.Batch = true
	half := llm.EstimateCost(in)
	if full <= 0 {
		t.Fatal("expected nonzero sync cost (model must have pricing)")
	}
	if math.Abs(half-full*0.5) > 1e-9 {
		t.Fatalf("batch cost %v should be half of sync cost %v", half, full)
	}
}

// A misconfigured (negative) process-wide download cap must never make the bounded
// read return zero bytes and silently truncate results — the effective cap clamps to
// the positive default instead. Not parallel: it briefly mutates a process-wide var.
func TestEffectiveMaxBatchDownloadBytesNeverNonPositive(t *testing.T) {
	saved := llm.MaxBatchDownloadBytes
	t.Cleanup(func() { llm.MaxBatchDownloadBytes = saved })
	llm.MaxBatchDownloadBytes = -5 // hostile/misconfigured global cap

	cfg := llm.DefaultConfig() // no per-config override → falls through to the global
	if got := cfg.EffectiveMaxBatchDownloadBytes(); got <= 0 {
		t.Fatalf("download cap must never be non-positive (would silently truncate results); got %d", got)
	}
}

// ---- Concurrency: race-free under -race ------------------------------------

func TestBatchConcurrentRaceFree(t *testing.T) {
	env := &batchEnv{
		endpoint: "/v1/chat/completions", status: "completed", outputFileID: "file-out",
		files: map[string]string{"file-out": successLine("a", chatBody("ok", 1, 1))},
	}
	srv := httptest.NewServer(env.handler())
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)

	var wg sync.WaitGroup
	for i := 0; i < 24; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ctx := context.Background()
			switch i % 4 {
			case 0:
				_, _ = bc.Submit(ctx, []llm.BatchItem{chatItem("c")}, llm.BatchOptions{})
			case 1:
				_, _ = bc.Get(ctx, "batch-1")
			case 2:
				_, _ = bc.Results(ctx, "batch-1")
			case 3:
				_, _ = bc.Cancel(ctx, "batch-1")
			}
		}(i)
	}
	wg.Wait()
}
