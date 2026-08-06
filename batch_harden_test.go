package llm_test

// Hardening round 1 for the Batch API — closes the structural coverage holes the
// PR's adversarial suite left at 0%: WaitForBatch (polling + cancellation + error
// propagation), BatchStatus.Terminal, the entire /v1/responses batch path (build +
// parse), the embeddings Submit path, and Usage.AddBatchResponse (half-cost + the
// TokensNotReported clamp). These reuse the batchEnv / helpers from
// batch_adversarial_test.go (same package).

import (
	"context"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/provider/openaicompat"
)

// ---- BatchStatus.Terminal ---------------------------------------------------

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

// ---- WaitForBatch -----------------------------------------------------------

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

func TestWaitForBatchHonorsContextCancel(t *testing.T) {
	// A batch that never terminates must not block forever: cancelling ctx returns
	// the last observed handle together with ctx.Err(). Exercises the select's
	// <-ctx.Done() branch specifically.
	srv, _ := transitionServer(t, "in_progress") // never terminal
	bc := newBatchClient(t, srv.URL)

	ctx, cancel := context.WithCancel(context.Background())
	var last *llm.BatchHandle
	var werr error
	done := make(chan struct{})
	go func() {
		last, werr = llm.WaitForBatch(ctx, bc, validBatchHandle(), 30*time.Millisecond)
		close(done)
	}()
	time.Sleep(15 * time.Millisecond) // let the first Get succeed and the loop enter its sleep
	cancel()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("WaitForBatch did not return after ctx cancel")
	}
	if werr == nil {
		t.Fatal("expected a context error after cancel")
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

// ---- /v1/responses batch path (build + parse), the PR suite left it at 0% ----

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

// ---- embeddings Submit path (BuildEmbeddingsBatchLineBody was 0%) -----------

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

// ---- Usage.AddBatchResponse (half cost, same tokens, sentinel clamp) --------

func TestUsageAddBatchResponseHalfCostSameTokens(t *testing.T) {
	resp := &llm.Response{Model: "gpt-5.3-chat-latest", InputTokens: 1_000_000, OutputTokens: 1_000_000}

	var syncU, batchU llm.Usage
	syncU.AddResponse(resp)
	batchU.AddBatchResponse(resp)

	if syncU.InputTokens != batchU.InputTokens || syncU.OutputTokens != batchU.OutputTokens {
		t.Fatalf("token accumulation must be identical: sync=%+v batch=%+v", syncU, batchU)
	}
	if syncU.Cost <= 0 {
		t.Fatal("expected a nonzero sync cost (model must have pricing)")
	}
	if math.Abs(batchU.Cost-syncU.Cost*0.5) > 1e-9 {
		t.Fatalf("batch cost %v must be half of sync cost %v", batchU.Cost, syncU.Cost)
	}
}

func TestUsageAddBatchResponseClampsTokensNotReported(t *testing.T) {
	// A streamed turn that never reported usage carries the -1 sentinel; folding it at
	// batch pricing must not push tokens or cost negative.
	resp := &llm.Response{Model: "gpt-5.3-chat-latest", InputTokens: llm.TokensNotReported, OutputTokens: llm.TokensNotReported}
	var u llm.Usage
	u.AddBatchResponse(resp)
	if u.InputTokens != 0 || u.OutputTokens != 0 {
		t.Fatalf("sentinel (-1) must clamp to 0, got %+v", u)
	}
	if u.Cost < 0 {
		t.Fatalf("cost must never be negative, got %v", u.Cost)
	}
}

// Sanity: AddBatchResponse/AddResponse are nil-safe (shared add()).
func TestAddBatchResponseNilSafe(t *testing.T) {
	var u llm.Usage
	u.AddBatchResponse(nil)
	u.AddResponse(nil)
	if (u != llm.Usage{}) {
		t.Fatalf("nil response must be a no-op, got %+v", u)
	}
}

// ---- remaining hostile-input edges -----------------------------------------

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
