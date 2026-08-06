package llm_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
)

const testRequestDigest = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

func validBatchHandle() llm.BatchHandle {
	return llm.BatchHandle{
		SchemaVersion:       llm.BatchSchemaVersion,
		Provider:            "openai",
		ID:                  "batch-1",
		Operation:           llm.BatchOperationCompletion,
		Status:              llm.BatchRunning,
		RequestCounts:       llm.BatchRequestCounts{Total: 2, Completed: 1},
		CreatedAt:           time.Unix(1_700_000_000, 0).UTC(),
		ExpiresAt:           time.Unix(1_700_086_400, 0).UTC(),
		Metadata:            map[string]string{"purpose": "test"},
		RecoveryKey:         "run:stable-1",
		RequestDigest:       testRequestDigest,
		AdapterStateVersion: 1,
		AdapterState:        json.RawMessage(`{"endpoint":"/v1/chat/completions","input_file_id":"file-in"}`),
	}
}

func validEmbeddingBatchHandle() llm.BatchHandle {
	handle := validBatchHandle()
	handle.Operation = llm.BatchOperationEmbedding
	handle.AdapterState = json.RawMessage(`{"endpoint":"/v1/embeddings","input_file_id":"file-in"}`)
	return handle
}

func TestBatchOptionsRequireBoundedTurnaround(t *testing.T) {
	for _, turnaround := range []time.Duration{0, -time.Second, llm.MaxBatchTurnaround + time.Second} {
		if err := (llm.BatchOptions{MaxTurnaround: turnaround}).Validate(); err == nil {
			t.Fatalf("turnaround %s was accepted", turnaround)
		}
	}
	if err := (llm.BatchOptions{MaxTurnaround: time.Hour}).Validate(); err != nil {
		t.Fatalf("neutral bounded turnaround rejected: %v", err)
	}
}

func TestBatchHandleValidateRejectsCorruptResumeState(t *testing.T) {
	valid := validBatchHandle()
	if err := valid.Validate(); err != nil {
		t.Fatalf("valid handle rejected: %v", err)
	}
	tests := map[string]func(*llm.BatchHandle){
		"schema":        func(h *llm.BatchHandle) { h.SchemaVersion++ },
		"provider":      func(h *llm.BatchHandle) { h.Provider = "" },
		"id whitespace": func(h *llm.BatchHandle) { h.ID = " batch" },
		"operation":     func(h *llm.BatchHandle) { h.Operation = "" },
		"status":        func(h *llm.BatchHandle) { h.Status = "" },
		"state version": func(h *llm.BatchHandle) { h.AdapterStateVersion = 0 },
		"state":         func(h *llm.BatchHandle) { h.AdapterState = nil },
		"counts":        func(h *llm.BatchHandle) { h.RequestCounts.Failed = 2 },
		"expiry":        func(h *llm.BatchHandle) { h.ExpiresAt = h.CreatedAt },
		"recovery":      func(h *llm.BatchHandle) { h.RecoveryKey = "bad key" },
		"digest":        func(h *llm.BatchHandle) { h.RequestDigest = "sha256:no" },
		"metadata":      func(h *llm.BatchHandle) { h.Metadata[" bad"] = "x" },
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			h := validBatchHandle()
			h.Metadata = map[string]string{"purpose": "test"}
			mutate(&h)
			if err := h.Validate(); err == nil {
				t.Fatalf("corrupt handle accepted: %+v", h)
			}
		})
	}
}

func TestLoadBatchHandleStrictJSON(t *testing.T) {
	h := validBatchHandle()
	raw, _ := json.Marshal(h)
	if _, err := llm.LoadBatchHandle(raw); err != nil {
		t.Fatalf("round trip: %v", err)
	}
	for _, hostile := range [][]byte{
		append(append([]byte(nil), raw...), []byte(` {}`)...),
		[]byte(`{"schema_version":1,"provider":"openai","id":"b","operation":"completion","status":"queued","request_counts":{},"adapter_state_version":1,"adapter_state":{}}`),
		[]byte(`{"provider":"openai","id":"b","operation":"completion","status":"queued","request_counts":{},"adapter_state_version":1,"adapter_state":{}}`),
		[]byte(`{"schema_version":2,"provider":"openai","id":"b","operation":"completion","status":"queued","request_counts":{},"adapter_state_version":1,"adapter_state":{},"unknown":true}`),
	} {
		if _, err := llm.LoadBatchHandle(hostile); err == nil {
			t.Fatalf("hostile handle accepted: %s", hostile)
		}
	}
}

func TestBatchRecoveryKeyRejectedBeforeNetwork(t *testing.T) {
	srv, requests := countingServer(t)
	bc := newBatchClient(t, srv.URL)
	_, err := bc.Submit(context.Background(), []llm.BatchItem{chatItem("a")}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour, RecoveryKey: "bad key"})
	if err == nil {
		t.Fatal("expected invalid recovery key")
	}
	if got := atomic.LoadInt32(requests); got != 0 {
		t.Fatalf("invalid recovery identity made %d network requests", got)
	}
}

func TestOpenAIBatchTurnaroundRejectedBeforeNetwork(t *testing.T) {
	srv, requests := countingServer(t)
	bc := newBatchClient(t, srv.URL)
	_, err := bc.Submit(context.Background(), []llm.BatchItem{chatItem("a")}, llm.BatchOptions{
		MaxTurnaround: 23 * time.Hour,
	})
	if err == nil || !strings.Contains(err.Error(), "exactly 24h") {
		t.Fatalf("expected exact turnaround rejection, got %v", err)
	}
	if got := atomic.LoadInt32(requests); got != 0 {
		t.Fatalf("invalid turnaround made %d network requests", got)
	}
}

func recoveryObject(id, recoveryKey, digest string) map[string]any {
	return map[string]any{
		"id": id, "status": "in_progress", "endpoint": "/v1/chat/completions",
		"input_file_id": "file-in", "created_at": 1_700_000_000,
		"request_counts": map[string]int{"total": 1, "completed": 0, "failed": 0},
		"metadata":       map[string]string{"rho_recovery_key": recoveryKey, "rho_request_digest": digest},
	}
}

func TestBatchRecoverMissingAndConflict(t *testing.T) {
	for name, objects := range map[string][]map[string]any{
		"missing":  {recoveryObject("batch-other", "other-key", testRequestDigest)},
		"conflict": {recoveryObject("batch-1", "stable-key", testRequestDigest), recoveryObject("batch-2", "stable-key", testRequestDigest)},
	} {
		t.Run(name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				writeJSON(w, map[string]any{"data": objects, "has_more": false})
			}))
			defer srv.Close()
			bc := newBatchClient(t, srv.URL)
			_, err := bc.Recover(context.Background(), "stable-key")
			if name == "missing" && !errors.Is(err, llm.ErrBatchNotFound) {
				t.Fatalf("expected not found, got %v", err)
			}
			if name == "conflict" && !errors.Is(err, llm.ErrBatchRecoveryConflict) {
				t.Fatalf("expected conflict, got %v", err)
			}
		})
	}
}

type durableBatchServer struct {
	mu              sync.Mutex
	metadata        map[string]string
	created         bool
	uploads         int
	creates         int
	cancels         int
	status          string
	ambiguousCreate bool
	ambiguousCancel bool
}

func (state *durableBatchServer) object() map[string]any {
	metadata := make(map[string]string, len(state.metadata))
	for key, value := range state.metadata {
		metadata[key] = value
	}
	return map[string]any{
		"id": "batch-1", "status": state.status, "endpoint": "/v1/chat/completions",
		"input_file_id": "file-in", "created_at": 1_700_000_000,
		"request_counts": map[string]int{"total": 1, "completed": 0, "failed": 0},
		"metadata":       metadata,
	}
}

func (state *durableBatchServer) handler(w http.ResponseWriter, r *http.Request) {
	state.mu.Lock()
	defer state.mu.Unlock()
	switch {
	case r.Method == http.MethodGet && r.URL.Path == "/batches":
		data := []any{}
		if state.created {
			data = append(data, state.object())
		}
		writeJSON(w, map[string]any{"data": data, "has_more": false})
	case r.Method == http.MethodPost && r.URL.Path == "/files":
		state.uploads++
		writeJSON(w, map[string]string{"id": "file-in"})
	case r.Method == http.MethodPost && r.URL.Path == "/batches":
		state.creates++
		var body struct {
			Metadata map[string]string `json:"metadata"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		state.metadata = body.Metadata
		state.created = true
		state.status = "validating"
		if state.ambiguousCreate {
			http.Error(w, "accepted then disconnected", http.StatusInternalServerError)
			return
		}
		writeJSON(w, state.object())
	case r.Method == http.MethodGet && r.URL.Path == "/batches/batch-1":
		writeJSON(w, state.object())
	case r.Method == http.MethodPost && r.URL.Path == "/batches/batch-1/cancel":
		state.cancels++
		state.status = "cancelling"
		if state.ambiguousCancel {
			http.Error(w, "accepted then disconnected", http.StatusInternalServerError)
			return
		}
		writeJSON(w, state.object())
	default:
		http.Error(w, "unexpected "+r.Method+" "+r.URL.Path, http.StatusInternalServerError)
	}
}

func TestBatchSubmitRecoversAmbiguousCreateAndDoesNotResubmit(t *testing.T) {
	state := &durableBatchServer{status: "in_progress", ambiguousCreate: true}
	srv := httptest.NewServer(http.HandlerFunc(state.handler))
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)
	opts := llm.BatchOptions{MaxTurnaround: 24 * time.Hour, RecoveryKey: "stable-run-1", Metadata: map[string]string{"purpose": "test"}}
	handle, err := bc.Submit(context.Background(), []llm.BatchItem{chatItem("a")}, opts)
	if err != nil {
		t.Fatalf("recover ambiguous create: %v", err)
	}
	if handle.RecoveryKey != opts.RecoveryKey || handle.RequestDigest == "" {
		t.Fatalf("recovery binding missing: %+v", handle)
	}
	state.mu.Lock()
	state.ambiguousCreate = false
	state.mu.Unlock()
	second, err := bc.Submit(context.Background(), []llm.BatchItem{chatItem("a")}, opts)
	if err != nil || second.ID != handle.ID {
		t.Fatalf("replay did not recover existing batch: %+v / %v", second, err)
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	if state.uploads != 1 || state.creates != 1 {
		t.Fatalf("replay duplicated provider effects: uploads=%d creates=%d", state.uploads, state.creates)
	}
}

func TestBatchSubmitRecoveryKeyCannotNameDifferentManifest(t *testing.T) {
	state := &durableBatchServer{status: "validating"}
	srv := httptest.NewServer(http.HandlerFunc(state.handler))
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)
	opts := llm.BatchOptions{MaxTurnaround: 24 * time.Hour, RecoveryKey: "stable-run-2"}
	if _, err := bc.Submit(context.Background(), []llm.BatchItem{chatItem("a")}, opts); err != nil {
		t.Fatalf("first submit: %v", err)
	}
	_, err := bc.Submit(context.Background(), []llm.BatchItem{chatItem("different")}, opts)
	if !errors.Is(err, llm.ErrBatchRecoveryConflict) {
		t.Fatalf("expected manifest conflict, got %v", err)
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	if state.uploads != 1 || state.creates != 1 {
		t.Fatalf("conflict made new provider effects: uploads=%d creates=%d", state.uploads, state.creates)
	}
}

func TestBatchSubmitPOSTsAreAlwaysSingleDispatch(t *testing.T) {
	var uploads atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/files" {
			uploads.Add(1)
			http.Error(w, "retryable", http.StatusInternalServerError)
			return
		}
		http.Error(w, "unexpected", http.StatusInternalServerError)
	}))
	defer srv.Close()
	cfg := llm.DefaultConfig()
	cfg.Provider = "openai"
	cfg.APIKey = "test-key"
	cfg.BaseURL = srv.URL
	cfg.Model = "gpt-5.3-chat-latest"
	bc, err := llm.NewBatchClient(cfg) // default retries enabled: Submit must still dispatch once.
	if err != nil {
		t.Fatalf("NewBatchClient: %v", err)
	}
	defer bc.Close()
	if _, err = bc.Submit(context.Background(), []llm.BatchItem{chatItem("a")}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour}); err == nil {
		t.Fatal("expected upload failure")
	}
	if got := uploads.Load(); got != 1 {
		t.Fatalf("non-idempotent upload dispatched %d times", got)
	}
}

func TestBatchCancelReconcilesAmbiguousResponse(t *testing.T) {
	state := &durableBatchServer{status: "in_progress", ambiguousCancel: true, created: true}
	srv := httptest.NewServer(http.HandlerFunc(state.handler))
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)
	handle, err := bc.Cancel(context.Background(), validBatchHandle())
	if err != nil || handle.Status != llm.BatchCancelling {
		t.Fatalf("cancel did not reconcile: %+v / %v", handle, err)
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	if state.cancels != 1 {
		t.Fatalf("cancel POST dispatched %d times", state.cancels)
	}
}

func TestBatchCancelTerminalDoesNotPOST(t *testing.T) {
	state := &durableBatchServer{status: "completed", created: true}
	srv := httptest.NewServer(http.HandlerFunc(state.handler))
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)
	handle, err := bc.Cancel(context.Background(), validBatchHandle())
	if err != nil || handle.Status != llm.BatchCompleted {
		t.Fatalf("terminal cancel reconciliation: %+v / %v", handle, err)
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	if state.cancels != 0 {
		t.Fatalf("terminal batch received %d cancel POSTs", state.cancels)
	}
}

func TestBatchRecoverRejectsInvalidKeyWithoutNetwork(t *testing.T) {
	var requests atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
	}))
	defer srv.Close()
	bc := newBatchClient(t, srv.URL)
	_, err := bc.Recover(context.Background(), strings.Repeat("x", llm.MaxBatchRecoveryKeyBytes+1))
	if err == nil || requests.Load() != 0 {
		t.Fatalf("invalid recovery key result: err=%v requests=%d", err, requests.Load())
	}
}
