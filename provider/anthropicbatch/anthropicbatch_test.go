package anthropicbatch_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider/anthropicbatch"
)

func TestAnthropicBatchSubmitGetResultsCancel(t *testing.T) {
	var created bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/messages/batches") && !strings.Contains(r.URL.Path, "cancel"):
			body, _ := io.ReadAll(r.Body)
			var payload struct {
				Requests []struct {
					CustomID string          `json:"custom_id"`
					Params   json.RawMessage `json:"params"`
				} `json:"requests"`
			}
			if err := json.Unmarshal(body, &payload); err != nil {
				t.Errorf("decode submit: %v", err)
			}
			if len(payload.Requests) != 1 || payload.Requests[0].CustomID != "item-1" {
				t.Errorf("unexpected requests: %+v", payload.Requests)
			}
			var params map[string]any
			if err := json.Unmarshal(payload.Requests[0].Params, &params); err != nil {
				t.Errorf("params: %v", err)
			}
			if params["model"] != "claude-sonnet-5" {
				t.Errorf("model = %v", params["model"])
			}
			if params["stream"] == true {
				t.Error("batch params must not stream")
			}
			created = true
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"id":"msgbatch_01",
				"type":"message_batch",
				"processing_status":"in_progress",
				"request_counts":{"processing":1,"succeeded":0,"errored":0,"canceled":0,"expired":0},
				"created_at":"2026-08-06T10:00:00Z",
				"expires_at":"2026-08-07T10:00:00Z",
				"results_url":"https://api.anthropic.com/v1/messages/batches/msgbatch_01/results"
			}`))
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/messages/batches/msgbatch_01"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"id":"msgbatch_01",
				"type":"message_batch",
				"processing_status":"ended",
				"request_counts":{"processing":0,"succeeded":1,"errored":0,"canceled":0,"expired":0},
				"created_at":"2026-08-06T10:00:00Z",
				"expires_at":"2026-08-07T10:00:00Z",
				"ended_at":"2026-08-06T10:05:00Z",
				"results_url":"https://api.anthropic.com/v1/messages/batches/msgbatch_01/results"
			}`))
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/results"):
			w.Header().Set("Content-Type", "application/x-jsonl")
			_, _ = w.Write([]byte(`{"custom_id":"item-1","result":{"type":"succeeded","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-5","content":[{"type":"text","text":"hello batch"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":2}}}}` + "\n"))
		case r.Method == http.MethodPost && strings.Contains(r.URL.Path, "/cancel"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"id":"msgbatch_01",
				"type":"message_batch",
				"processing_status":"canceling",
				"request_counts":{"processing":1,"succeeded":0,"errored":0,"canceled":0,"expired":0},
				"created_at":"2026-08-06T10:00:00Z",
				"expires_at":"2026-08-07T10:00:00Z"
			}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	cfg := llm.Config{
		Provider: "anthropic", Model: "claude-sonnet-5", APIKey: "test-key",
		BaseURL: srv.URL, DisableProxy: true, DisableRetries: true, Timeout: 5 * time.Second,
	}
	bc, err := llm.NewBatchClient(cfg)
	if err != nil {
		t.Fatalf("NewBatchClient: %v", err)
	}
	defer bc.Close()

	handle, err := bc.Submit(context.Background(), []llm.BatchItem{{
		ItemID: "item-1",
		Request: &llm.Request{
			Model: "claude-sonnet-5", MaxTokens: 64,
			Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
		},
	}}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if !created || handle.ID != "msgbatch_01" || handle.Status != llm.BatchRunning {
		t.Fatalf("handle after submit = %+v created=%v", handle, created)
	}
	raw, err := json.Marshal(handle)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := llm.LoadBatchHandle(raw)
	if err != nil {
		t.Fatalf("LoadBatchHandle: %v", err)
	}

	got, err := bc.Get(context.Background(), *loaded)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got.Status != llm.BatchCompleted || got.RequestCounts.Completed != 1 {
		t.Fatalf("Get handle = %+v", got)
	}

	results, err := bc.Results(context.Background(), *got)
	if err != nil {
		t.Fatalf("Results: %v", err)
	}
	if len(results) != 1 || results[0].ItemID != "item-1" || results[0].Response == nil || results[0].Response.Content != "hello batch" {
		t.Fatalf("results = %+v", results)
	}

	// Cancel path against a running-shaped handle.
	running := *handle
	canceled, err := bc.Cancel(context.Background(), running)
	if err != nil {
		t.Fatalf("Cancel: %v", err)
	}
	if canceled.Status != llm.BatchCancelling {
		t.Fatalf("cancel status = %s", canceled.Status)
	}
}

func TestAnthropicBatchRejectsEmbeddingsAndNonBatchProvider(t *testing.T) {
	cfg := llm.Config{
		Provider: "anthropic", APIKey: "k", BaseURL: "http://127.0.0.1:9",
		DisableProxy: true, DisableRetries: true,
	}
	bc, err := llm.NewBatchClient(cfg)
	if err != nil {
		t.Fatalf("NewBatchClient: %v", err)
	}
	defer bc.Close()
	_, err = bc.Submit(context.Background(), []llm.BatchItem{{
		ItemID:    "e1",
		Embedding: &llm.EmbeddingRequest{Model: "x", Input: []string{"a"}},
	}}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour})
	if err == nil || !strings.Contains(err.Error(), "embeddings") {
		t.Fatalf("expected embeddings rejection, got %v", err)
	}

	if _, err := llm.NewBatchClient(llm.Config{Provider: "groq", APIKey: "k"}); err == nil {
		t.Fatal("groq must not advertise batch")
	}
}
