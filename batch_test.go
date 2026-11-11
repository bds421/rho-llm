package llm_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider/openaibatch"
	"github.com/bds421/rho-llm/provider/openaicompat"
)

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
	cfg.Model = "gpt-4.1-mini"
	cfg.DisableRetries = true
	bc, err := llm.NewBatchClient(cfg)
	if err != nil {
		t.Fatalf("NewBatchClient: %v", err)
	}
	t.Cleanup(func() { _ = bc.Close() })
	return bc
}

func chatItem(id string) llm.BatchItem {
	return llm.BatchItem{ItemID: id, Request: &llm.Request{
		Model:     "gpt-4.1-mini",
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentPart{{Type: llm.ContentText, Text: "hi " + id}}}},
		MaxTokens: 64,
	}}
}

func embItem(id string) llm.BatchItem {
	return llm.BatchItem{ItemID: id, Embedding: &llm.EmbeddingRequest{Model: "text-embedding-3-small", Input: []string{"hi " + id}}}
}

func chatBody(content string, in, out int) map[string]any {
	return map[string]any{
		"id": "resp", "model": "gpt-4.1-mini",
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
	_, err := bc.Submit(context.Background(), []llm.BatchItem{chatItem("a"), embItem("b")}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour})
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
	_, err := bc.Submit(context.Background(), []llm.BatchItem{chatItem("a"), resp}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour})
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
	both := llm.BatchItem{ItemID: "x", Request: chatItem("x").Request, Embedding: embItem("x").Embedding}
	neither := llm.BatchItem{ItemID: "y"}
	for name, item := range map[string]llm.BatchItem{"both": both, "neither": neither} {
		if _, err := bc.Submit(context.Background(), []llm.BatchItem{item}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour}); err == nil {
			t.Fatalf("%s: expected tagged-union error", name)
		}
	}
	if got := atomic.LoadInt32(n); got != 0 {
		t.Fatalf("expected 0 network requests, got %d", got)
	}
}

func TestBatchSubmitDuplicateItemIDRejected(t *testing.T) {
	srv, n := countingServer(t)
	bc := newBatchClient(t, srv.URL)
	_, err := bc.Submit(context.Background(), []llm.BatchItem{chatItem("dup"), chatItem("dup")}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour})
	if err == nil || !strings.Contains(err.Error(), "duplicate item_id") {
		t.Fatalf("expected duplicate item_id error, got: %v", err)
	}
	if got := atomic.LoadInt32(n); got != 0 {
		t.Fatalf("expected 0 network requests, got %d", got)
	}
}

func TestBatchSubmitEmptyAndBlankIDRejected(t *testing.T) {
	srv, n := countingServer(t)
	bc := newBatchClient(t, srv.URL)
	if _, err := bc.Submit(context.Background(), nil, llm.BatchOptions{MaxTurnaround: 24 * time.Hour}); err == nil {
		t.Fatal("expected error for empty items")
	}
	if _, err := bc.Submit(context.Background(), []llm.BatchItem{chatItem("")}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour}); err == nil {
		t.Fatal("expected error for empty custom_id")
	}
	if got := atomic.LoadInt32(n); got != 0 {
		t.Fatalf("expected 0 network requests, got %d", got)
	}
}

func TestBatchSubmitUploadsJSONL(t *testing.T) {
	env := &batchEnv{endpoint: "/v1/chat/completions", status: "validating"}
	srv := httptest.NewServer(env.handler())
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)

	h, err := bc.Submit(context.Background(), []llm.BatchItem{chatItem("a"), chatItem("b")}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if h.ID != "batch-1" || h.Operation != llm.BatchOperationCompletion {
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

func TestBatchResultsHappyCorrelation(t *testing.T) {
	env := &batchEnv{
		endpoint: "/v1/chat/completions", status: "completed", outputFileID: "file-out",
		files: map[string]string{"file-out": successLine("a", chatBody("hello-a", 10, 5)) + successLine("b", chatBody("hello-b", 7, 3))},
	}
	srv := httptest.NewServer(env.handler())
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)

	results, err := bc.Results(context.Background(), validBatchHandle())
	if err != nil {
		t.Fatalf("Results: %v", err)
	}
	got := map[string]string{}
	for _, r := range results {
		if r.Error != nil {
			t.Fatalf("unexpected error for %q: %v", r.ItemID, r.Error)
		}
		if r.Response == nil {
			t.Fatalf("nil response for %q", r.ItemID)
		}
		got[r.ItemID] = r.Response.Content
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

	results, err := bc.Results(context.Background(), validBatchHandle())
	if err != nil {
		t.Fatalf("Results should not abort on one bad line: %v", err)
	}
	var sawError, sawGood bool
	for _, r := range results {
		if r.Error != nil {
			sawError = true
		}
		if r.ItemID == "ok" && r.Response != nil && r.Response.Content == "good" {
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

	results, err := bc.Results(context.Background(), validBatchHandle())
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

	results, err := bc.Results(context.Background(), validBatchHandle())
	if err != nil {
		t.Fatalf("Results: %v", err)
	}
	byID := map[string]llm.BatchResult{}
	for _, r := range results {
		byID[r.ItemID] = r
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

	if _, err := bc.Results(context.Background(), validBatchHandle()); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("expected oversize download to be rejected, got: %v", err)
	}
}

func TestBatchResultsFailedNoFilesNoPanic(t *testing.T) {
	env := &batchEnv{endpoint: "/v1/chat/completions", status: "failed"} // no output/error files
	srv := httptest.NewServer(env.handler())
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)

	results, err := bc.Results(context.Background(), validBatchHandle())
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

	results, err := bc.Results(context.Background(), validEmbeddingBatchHandle())
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

func TestBatchHandleRoundTripAndResultsFromEndpoint(t *testing.T) {
	env := &batchEnv{
		endpoint: "/v1/chat/completions", status: "completed", outputFileID: "file-out",
		files: map[string]string{"file-out": successLine("a", chatBody("resumed", 1, 1))},
	}
	srv := httptest.NewServer(env.handler())
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)

	h, err := bc.Get(context.Background(), validBatchHandle())
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
	if loaded.Operation != llm.BatchOperationCompletion || loaded.ID != "batch-1" ||
		!bytes.Equal(loaded.AdapterState, h.AdapterState) {
		t.Fatalf("round-trip lost fields: %+v", loaded)
	}
	// Results works from the complete deserialized handle; adapter state selects
	// the exact response parser without exposing endpoint vocabulary to callers.
	results, err := bc.Results(context.Background(), *loaded)
	if err != nil || len(results) != 1 || results[0].Response == nil || results[0].Response.Content != "resumed" {
		t.Fatalf("resume Results failed: %v / %+v", err, results)
	}
}

func TestLoadBatchHandleRejectsSchemaVersions(t *testing.T) {
	if _, err := llm.LoadBatchHandle([]byte(`{"schema_version":1,"id":"x"}`)); err == nil {
		t.Fatal("expected rejection of older schema_version 1")
	}
	if _, err := llm.LoadBatchHandle([]byte(`{"provider":"openai","id":"x"}`)); err == nil {
		t.Fatal("expected rejection of missing schema_version")
	}
}

func TestNewBatchClientUnsupportedProvider(t *testing.T) {
	cfg := llm.DefaultConfig()
	// Groq remains chat-only openai_compat without SupportsBatch.
	cfg.Provider = "groq"
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
	cfg.Model = "gpt-4.1-mini"

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	req := llm.Request{
		Model:     "gpt-4.1-mini",
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
				_, _ = bc.Submit(ctx, []llm.BatchItem{chatItem("c")}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour})
			case 1:
				_, _ = bc.Get(ctx, validBatchHandle())
			case 2:
				_, _ = bc.Results(ctx, validBatchHandle())
			case 3:
				_, _ = bc.Cancel(ctx, validBatchHandle())
			}
		}(i)
	}
	wg.Wait()
}

// failingHandler serves the Files+Batches surface but fails one named step.
func failOnStep(t *testing.T, failPath string, code int) llm.BatchClient {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, failPath) {
			http.Error(w, `{"error":{"message":"step failed"}}`, code)
			return
		}
		switch {
		case r.URL.Path == "/files":
			writeJSON(w, map[string]any{"id": "file-in"})
		case r.URL.Path == "/batches":
			writeJSON(w, map[string]any{"id": "batch-1", "status": "validating", "endpoint": "/v1/chat/completions", "input_file_id": "file-in"})
		default:
			writeJSON(w, map[string]any{"id": "batch-1", "status": "completed", "endpoint": "/v1/chat/completions", "input_file_id": "file-in"})
		}
	}))
	t.Cleanup(srv.Close)
	return newBatchClient(t, srv.URL)
}

func TestBatchSubmitUploadFailure(t *testing.T) {
	bc := failOnStep(t, "/files", http.StatusInternalServerError)
	if _, err := bc.Submit(context.Background(), []llm.BatchItem{chatItem("a")}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour}); err == nil {
		t.Fatal("expected Submit to fail when the file upload returns non-2xx")
	}
}

func TestBatchSubmitCreateFailure(t *testing.T) {
	bc := failOnStep(t, "/batches", http.StatusBadRequest)
	if _, err := bc.Submit(context.Background(), []llm.BatchItem{chatItem("a")}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour}); err == nil {
		t.Fatal("expected Submit to fail when batch creation returns non-2xx")
	}
}

func TestBatchSubmitUploadNoIDRejected(t *testing.T) {
	// A 2xx upload that returns no file id must not proceed to create a batch that
	// references an empty input file.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/files" {
			writeJSON(w, map[string]any{}) // 200 but no "id"
			return
		}
		http.Error(w, "should not reach create", http.StatusInternalServerError)
	}))
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)
	if _, err := bc.Submit(context.Background(), []llm.BatchItem{chatItem("a")}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour}); err == nil || !strings.Contains(err.Error(), "no id") {
		t.Fatalf("expected 'no id' error, got: %v", err)
	}
}

func TestBatchCancelTransportFailureSurfaced(t *testing.T) {
	// A non-2xx on the cancel endpoint must surface as an error, not a nil handle.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/cancel") {
			http.Error(w, `{"error":{"message":"cannot cancel"}}`, http.StatusConflict)
			return
		}
		http.Error(w, "unexpected", http.StatusInternalServerError)
	}))
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)
	if _, err := bc.Cancel(context.Background(), validBatchHandle()); err == nil {
		t.Fatal("expected Cancel to surface a non-2xx from the cancel endpoint")
	}
}

func TestBatchResultsDownloadFailureSurfaced(t *testing.T) {
	// The batch is completed with an output file id, but the file content endpoint
	// fails: Results must surface the error, not return partial/empty results.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/content"):
			http.Error(w, "gone", http.StatusInternalServerError)
		case strings.HasPrefix(r.URL.Path, "/batches/"):
			writeJSON(w, map[string]any{"id": "batch-1", "status": "completed", "endpoint": "/v1/chat/completions", "input_file_id": "file-in", "output_file_id": "file-out"})
		default:
			http.Error(w, "unexpected", http.StatusInternalServerError)
		}
	}))
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)
	if _, err := bc.Results(context.Background(), validBatchHandle()); err == nil {
		t.Fatal("expected Results to surface a download failure")
	}
}

func TestBatchResultsBatchLevelErrorsSurfaced(t *testing.T) {
	// A batch that failed validation carries top-level errors.data[] and NO result
	// files. Those messages must surface as error results.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]any{
			"id": "batch-1", "status": "failed", "endpoint": "/v1/chat/completions", "input_file_id": "file-in",
			"errors": map[string]any{"data": []any{
				map[string]any{"code": "invalid_request", "message": "line 1: bad model", "line": 1},
				map[string]any{"code": "invalid_request", "message": "line 2: missing url"},
			}},
		})
	}))
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)

	results, err := bc.Results(context.Background(), validBatchHandle())
	if err != nil {
		t.Fatalf("Results: %v", err)
	}
	if len(results) != 2 || results[0].Error == nil || results[1].Error == nil {
		t.Fatalf("expected 2 batch-level error results, got %+v", results)
	}
	if !strings.Contains(results[0].Error.Message, "bad model") {
		t.Fatalf("batch-level error message not surfaced: %+v", results[0].Error)
	}
}

func resultsWithFileOfSize(t *testing.T, capBytes, fileBytes int) error {
	t.Helper()
	// A single valid JSONL line padded (via a long custom_id) to exactly fileBytes,
	// so the body size is deterministic.
	line := map[string]any{"custom_id": "x", "response": map[string]any{"status_code": 200, "body": chatBody("ok", 1, 1)}}
	base, _ := json.Marshal(line)
	pad := fileBytes - (len(base) + 1) // +1 for the trailing newline
	if pad < 0 {
		t.Fatalf("fileBytes %d too small (min %d)", fileBytes, len(base)+1)
	}
	line["custom_id"] = "x" + strings.Repeat("p", pad)
	content, _ := json.Marshal(line)
	body := string(content) + "\n"
	if len(body) != fileBytes {
		t.Fatalf("padding math off: got %d want %d", len(body), fileBytes)
	}

	env := &batchEnv{
		endpoint: "/v1/chat/completions", status: "completed", outputFileID: "file-out",
		files: map[string]string{"file-out": body},
	}
	srv := httptest.NewServer(env.handler())
	t.Cleanup(srv.Close)

	cfg := llm.DefaultConfig()
	cfg.Provider = "openai"
	cfg.APIKey = "k"
	cfg.BaseURL = srv.URL
	cfg.MaxBatchDownloadBytes = capBytes
	bc, err := llm.NewBatchClient(cfg)
	if err != nil {
		t.Fatalf("NewBatchClient: %v", err)
	}
	t.Cleanup(func() { _ = bc.Close() })
	_, err = bc.Results(context.Background(), validBatchHandle())
	return err
}

func TestBatchDownloadBoundaryExactCapSucceeds(t *testing.T) {
	const limit = 2048
	// A file of exactly the cap must succeed (the read peeks one byte past the cap).
	if err := resultsWithFileOfSize(t, limit, limit); err != nil {
		t.Fatalf("a file of exactly the cap (%d) must download, got: %v", limit, err)
	}
}

func TestBatchDownloadBoundaryOneOverCapFails(t *testing.T) {
	const limit = 2048
	// One byte over the cap must be rejected, not silently truncated.
	err := resultsWithFileOfSize(t, limit, limit+1)
	if err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("a file of cap+1 must be rejected as oversize, got: %v", err)
	}
}

func TestBatchResultsLastLineNoTrailingNewline(t *testing.T) {
	// The final output line may lack a trailing '\n'; it must still be parsed.
	body := successLine("a", chatBody("hello-a", 1, 1)) + strings.TrimRight(successLine("b", chatBody("hello-b", 1, 1)), "\n")
	env := &batchEnv{
		endpoint: "/v1/chat/completions", status: "completed", outputFileID: "file-out",
		files: map[string]string{"file-out": body},
	}
	srv := httptest.NewServer(env.handler())
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)

	results, err := bc.Results(context.Background(), validBatchHandle())
	if err != nil {
		t.Fatalf("Results: %v", err)
	}
	got := map[string]bool{}
	for _, r := range results {
		got[r.ItemID] = true
	}
	if !got["a"] || !got["b"] {
		t.Fatalf("a newline-less final line was dropped; got %+v", got)
	}
}

func TestBatchResultsHugeSingleLineNotCapped(t *testing.T) {
	// A single result line far larger than bufio.Scanner's 64KB default must parse —
	// the impl uses bufio.Reader.ReadBytes, not Scanner.
	huge := strings.Repeat("z", 200*1024) // 200 KB of content in one line
	env := &batchEnv{
		endpoint: "/v1/chat/completions", status: "completed", outputFileID: "file-out",
		files: map[string]string{"file-out": successLine("big", chatBody(huge, 1, 1))},
	}
	srv := httptest.NewServer(env.handler())
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)

	results, err := bc.Results(context.Background(), validBatchHandle())
	if err != nil {
		t.Fatalf("Results: %v", err)
	}
	if len(results) != 1 || results[0].Response == nil || len(results[0].Response.Content) != len(huge) {
		t.Fatalf("expected the 200KB line parsed intact, got %d results / content len %d", len(results), func() int {
			if len(results) > 0 && results[0].Response != nil {
				return len(results[0].Response.Content)
			}
			return -1
		}())
	}
}

func TestBatchResultsDedupAcrossOutputAndErrorFiles(t *testing.T) {
	// The same custom_id appearing in BOTH the output and error files must collapse to
	// a single result (first occurrence — the output/success — wins).
	errLine, _ := json.Marshal(map[string]any{"custom_id": "dup", "error": map[string]any{"message": "also failed"}})
	env := &batchEnv{
		endpoint: "/v1/chat/completions", status: "completed", outputFileID: "file-out", errorFileID: "file-err",
		files: map[string]string{
			"file-out": successLine("dup", chatBody("won", 1, 1)),
			"file-err": string(errLine) + "\n",
		},
	}
	srv := httptest.NewServer(env.handler())
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)

	results, err := bc.Results(context.Background(), validBatchHandle())
	if err != nil {
		t.Fatalf("Results: %v", err)
	}
	n := 0
	for _, r := range results {
		if r.ItemID == "dup" {
			n++
			if r.Response == nil || r.Response.Content != "won" {
				t.Fatalf("expected the output (success) line to win, got %+v", r)
			}
		}
	}
	if n != 1 {
		t.Fatalf("expected exactly one result for the duplicated custom_id, got %d", n)
	}
}

func TestBatchStatusTerminal(t *testing.T) {
	// A bug that flips any of these would either strand a poll forever (terminal
	// misread as non-terminal) or end it prematurely (vice-versa).
	terminal := []llm.BatchStatus{llm.BatchCompleted, llm.BatchFailed, llm.BatchExpired, llm.BatchCancelled}
	nonTerminal := []llm.BatchStatus{
		llm.BatchQueued, llm.BatchRunning, llm.BatchCancelling,
		llm.BatchStatus("some_future_status_we_dont_know"), llm.BatchStatus(""),
	}
	for _, s := range terminal {
		if !s.Terminal() {
			t.Errorf("status %q must be terminal", s)
		}
	}
	for _, s := range nonTerminal {
		if s.Terminal() {
			t.Errorf("status %q must NOT be terminal", s)
		}
	}
}

// transitionServer returns a server whose GET /batches/{id} walks through the given
// status sequence (repeating the last), counting GETs.
func transitionServer(t *testing.T, statuses ...string) (*httptest.Server, *int32) {
	t.Helper()
	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/batches/") {
			i := int(atomic.AddInt32(&n, 1)) - 1
			if i >= len(statuses) {
				i = len(statuses) - 1
			}
			writeJSON(w, map[string]any{"id": "batch-1", "status": statuses[i], "endpoint": "/v1/chat/completions", "input_file_id": "file-in"})
			return
		}
		http.Error(w, "unexpected "+r.URL.Path, http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)
	return srv, &n
}

func TestWaitForBatchPollsToTerminal(t *testing.T) {
	srv, n := transitionServer(t, "validating", "in_progress", "completed")
	bc := newBatchClient(t, srv.URL)

	h, err := llm.WaitForBatch(context.Background(), bc, validBatchHandle(), time.Millisecond)
	if err != nil {
		t.Fatalf("WaitForBatch: %v", err)
	}
	if h == nil || h.Status != llm.BatchCompleted {
		t.Fatalf("expected to land on completed, got %+v", h)
	}
	if got := atomic.LoadInt32(n); got < 3 {
		t.Fatalf("expected at least 3 polls to walk to terminal, got %d", got)
	}
}

func TestWaitForBatchTerminalFirstPollNoSleep(t *testing.T) {
	// Terminal on the first Get must return immediately even with a non-positive
	// pollInterval (which would otherwise be substituted with the 30s default) —
	// the terminal check returns before the select ever sleeps.
	srv, _ := transitionServer(t, "completed")
	bc := newBatchClient(t, srv.URL)

	done := make(chan struct{})
	go func() {
		h, err := llm.WaitForBatch(context.Background(), bc, validBatchHandle(), 0) // 0 -> default 30s, but terminal first
		if err != nil || h == nil || h.Status != llm.BatchCompleted {
			t.Errorf("expected immediate completed, got %+v err=%v", h, err)
		}
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("WaitForBatch slept on a terminal-first batch (pollInterval<=0 default leaked into the wait)")
	}
}

// Cancel only after the real client's Get has decoded a successful response.
// Signaling from the HTTP handler would still race with response decoding.
type cancelAfterBatchGet struct {
	llm.BatchClient
	cancel context.CancelFunc
}

func (bc cancelAfterBatchGet) Get(ctx context.Context, handle llm.BatchHandle) (*llm.BatchHandle, error) {
	h, err := bc.BatchClient.Get(ctx, handle)
	if err == nil {
		bc.cancel()
	}
	return h, err
}

func TestWaitForBatchHonorsContextCancel(t *testing.T) {
	// A batch that never terminates must not block forever: cancelling ctx returns
	// the last observed handle together with ctx.Err(). Exercises the select's
	// <-ctx.Done() branch specifically.
	srv, _ := transitionServer(t, "in_progress") // never terminal
	bc := newBatchClient(t, srv.URL)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	bc = cancelAfterBatchGet{BatchClient: bc, cancel: cancel}
	var last *llm.BatchHandle
	var werr error
	done := make(chan struct{})
	go func() {
		last, werr = llm.WaitForBatch(ctx, bc, validBatchHandle(), time.Hour)
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("WaitForBatch did not return after ctx cancel")
	}
	if !errors.Is(werr, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", werr)
	}
	if last == nil || last.Status != llm.BatchRunning {
		t.Fatalf("expected the last observed (in_progress) handle on cancel, got %+v", last)
	}
}

func TestWaitForBatchGetErrorSurfaced(t *testing.T) {
	// A hard transport error from Get must be surfaced, not swallowed; the first-poll
	// failure returns a nil last handle.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)

	last, err := llm.WaitForBatch(context.Background(), bc, validBatchHandle(), time.Millisecond)
	if err == nil {
		t.Fatal("expected WaitForBatch to surface the Get error")
	}
	if last != nil {
		t.Fatalf("expected nil last handle when the first Get fails, got %+v", last)
	}
}

func TestBatchUnknownProviderStatusFailsClosed(t *testing.T) {
	srv, _ := transitionServer(t, "future_provider_state")
	bc := newBatchClient(t, srv.URL)
	if _, err := bc.Get(context.Background(), validBatchHandle()); err == nil ||
		!strings.Contains(err.Error(), "unknown provider batch status") {
		t.Fatalf("expected unknown status rejection, got %v", err)
	}
}

func responsesBody(text string, in, out int) map[string]any {
	return map[string]any{
		"id": "resp_1", "model": "gpt-5.5", "status": "completed",
		"output": []any{map[string]any{
			"type": "message", "role": "assistant",
			"content": []any{map[string]any{"type": "output_text", "text": text}},
		}},
		"usage": map[string]any{"input_tokens": in, "output_tokens": out},
	}
}

func responsesItem(id string) llm.BatchItem {
	return llm.BatchItem{ItemID: id, Request: &llm.Request{
		Model:     "gpt-5.5", // a ResponsesAPI model → routes to /v1/responses
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentPart{{Type: llm.ContentText, Text: "q " + id}}}},
		MaxTokens: 64,
	}}
}

func TestBatchSubmitAndResultsResponsesRoundTrip(t *testing.T) {
	env := &batchEnv{
		endpoint: "/v1/responses", status: "completed", outputFileID: "file-out",
		files: map[string]string{"file-out": successLine("q1", responsesBody("ALPHA", 10, 5))},
	}
	srv := httptest.NewServer(env.handler())
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)

	// Submit must build /v1/responses lines (covers openairesponses.BuildResponsesBatchLineBody).
	h, err := bc.Submit(context.Background(), []llm.BatchItem{responsesItem("q1")}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour})
	if err != nil {
		t.Fatalf("Submit responses: %v", err)
	}
	if h.Operation != llm.BatchOperationCompletion {
		t.Fatalf("expected completion operation, got %q", h.Operation)
	}
	env.mu.Lock()
	body := string(env.uploaded[len(env.uploaded)-1])
	env.mu.Unlock()
	if !strings.Contains(body, `"url":"/v1/responses"`) {
		t.Fatalf("uploaded JSONL should target /v1/responses; got:\n%s", body)
	}

	// Results must parse a /v1/responses body (covers ParseResponsesBatchResultBody).
	results, err := bc.Results(context.Background(), *h)
	if err != nil {
		t.Fatalf("Results: %v", err)
	}
	if len(results) != 1 || results[0].Response == nil || results[0].Response.Content != "ALPHA" {
		t.Fatalf("expected parsed responses content ALPHA, got %+v", results)
	}
	if results[0].ItemID != "q1" {
		t.Fatalf("expected item_id correlation q1, got %q", results[0].ItemID)
	}
}

func TestBatchSubmitEmbeddingsUploadsJSONL(t *testing.T) {
	env := &batchEnv{endpoint: "/v1/embeddings", status: "validating"}
	srv := httptest.NewServer(env.handler())
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)

	h, err := bc.Submit(context.Background(), []llm.BatchItem{embItem("e1"), embItem("e2")}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour})
	if err != nil {
		t.Fatalf("Submit embeddings: %v", err)
	}
	if h.Operation != llm.BatchOperationEmbedding {
		t.Fatalf("expected embedding operation, got %q", h.Operation)
	}
	env.mu.Lock()
	body := string(env.uploaded[0])
	env.mu.Unlock()
	for _, want := range []string{`"custom_id":"e1"`, `"url":"/v1/embeddings"`, `"model":"text-embedding-3-small"`} {
		if !strings.Contains(body, want) {
			t.Fatalf("uploaded embeddings JSONL missing %q; got:\n%s", want, body)
		}
	}
}

func TestLoadBatchHandleRejectsMalformedJSON(t *testing.T) {
	if _, err := llm.LoadBatchHandle([]byte("{ not json")); err == nil {
		t.Fatal("expected a decode error for malformed handle JSON")
	}
}

func TestBuildEmbeddingsBatchLineBodyRejectsEmptyInput(t *testing.T) {
	if _, err := openaicompat.BuildEmbeddingsBatchLineBody(llm.EmbeddingRequest{Model: "m"}); err == nil {
		t.Fatal("expected an error for an embeddings request with no input")
	}
}

func TestBatchCancelEmptyIDRejected(t *testing.T) {
	// The empty-id guard must fail before any network effect.
	srv, n := countingServer(t)
	bc := newBatchClient(t, srv.URL)
	bad := validBatchHandle()
	bad.ID = ""
	if _, err := bc.Cancel(context.Background(), bad); err == nil {
		t.Fatal("expected Cancel to reject an empty batch id")
	}
	if got := atomic.LoadInt32(n); got != 0 {
		t.Fatalf("expected 0 network requests on empty-id Cancel, got %d", got)
	}
}
