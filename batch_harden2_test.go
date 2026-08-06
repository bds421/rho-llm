package llm_test

// Hardening round 2 (transport layer) — adversarial coverage for the network failure
// paths the PR suite left untested: a non-2xx on upload / create / download, an upload
// with no file id, batch-level validation errors with no result files, the exact
// oversize-download boundary (cap vs cap+1), a final line with no trailing newline, a
// single line larger than bufio's default buffer, and cross-file custom_id dedup.

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
)

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

// ---- oversize download boundary: cap succeeds, cap+1 fails ------------------

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

// ---- forEachLine robustness -------------------------------------------------

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
