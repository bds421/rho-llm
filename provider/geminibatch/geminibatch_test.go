package geminibatch_test

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
	_ "github.com/bds421/rho-llm/provider/geminibatch"
)

func TestGeminiBatchCompletionSubmitGetResults(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && strings.Contains(r.URL.Path, "batchGenerateContent"):
			body, _ := io.ReadAll(r.Body)
			if !strings.Contains(string(body), "item-a") {
				t.Errorf("missing item key in body: %s", body)
			}
			w.Header().Set("Content-Type", "application/json")
			// Live API uses BATCH_STATE_*; offline suite covers both naming styles.
			_, _ = w.Write([]byte(`{"name":"batches/job1","metadata":{"state":"BATCH_STATE_RUNNING"}}`))
		case r.Method == http.MethodGet && strings.Contains(r.URL.Path, "batches/job1"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"name":"batches/job1",
				"state":"BATCH_STATE_SUCCEEDED",
				"dest":{"inlinedResponses":[
					{"metadata":{"key":"item-a"},"response":{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}}
				]}
			}`))
		case r.Method == http.MethodPost && strings.Contains(r.URL.Path, ":cancel"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	cfg := llm.Config{
		Provider: "gemini", Model: "gemini-3.6-flash", APIKey: "k",
		BaseURL: srv.URL + "/v1beta/models", DisableProxy: true, DisableRetries: true,
		Timeout: 5 * time.Second,
	}
	bc, err := llm.NewBatchClient(cfg)
	if err != nil {
		t.Fatalf("NewBatchClient: %v", err)
	}
	defer bc.Close()

	handle, err := bc.Submit(context.Background(), []llm.BatchItem{{
		ItemID: "item-a",
		Request: &llm.Request{
			Model:    "gemini-3.6-flash",
			Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
		},
	}}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if handle.ID != "job1" || handle.Status != llm.BatchRunning {
		t.Fatalf("handle=%+v", handle)
	}
	raw, _ := json.Marshal(handle)
	if _, err := llm.LoadBatchHandle(raw); err != nil {
		t.Fatalf("LoadBatchHandle: %v", err)
	}
	got, err := bc.Get(context.Background(), *handle)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got.Status != llm.BatchCompleted {
		t.Fatalf("status=%s", got.Status)
	}
	results, err := bc.Results(context.Background(), *got)
	if err != nil {
		t.Fatalf("Results: %v", err)
	}
	if len(results) != 1 || results[0].Response == nil || results[0].Response.Content != "ok" {
		t.Fatalf("results=%+v", results)
	}
}

func TestGeminiBatchEmbeddingSubmit(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && strings.Contains(r.URL.Path, "asyncBatchEmbedContent") {
			body, _ := io.ReadAll(r.Body)
			if !strings.Contains(string(body), "emb-1") {
				t.Errorf("body missing emb-1: %s", body)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"name":"batches/embjob","metadata":{"state":"JOB_STATE_PENDING"}}`))
			return
		}
		if r.Method == http.MethodGet {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"name":"batches/embjob","state":"JOB_STATE_SUCCEEDED",
				"dest":{"inlinedEmbedContentResponses":[
					{"metadata":{"key":"emb-1"},"response":{"embedding":{"values":[0.1,0.2]}}}
				]}
			}`))
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	cfg := llm.Config{
		Provider: "gemini", APIKey: "k", BaseURL: srv.URL + "/v1beta/models",
		DisableProxy: true, DisableRetries: true, Timeout: 5 * time.Second,
	}
	bc, err := llm.NewBatchClient(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer bc.Close()
	handle, err := bc.Submit(context.Background(), []llm.BatchItem{{
		ItemID:    "emb-1",
		Embedding: &llm.EmbeddingRequest{Model: "gemini-embedding-001", Input: []string{"hello"}},
	}}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	got, err := bc.Get(context.Background(), *handle)
	if err != nil {
		t.Fatal(err)
	}
	results, err := bc.Results(context.Background(), *got)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Embedding == nil || len(results[0].Embedding.Embeddings) != 1 {
		t.Fatalf("results=%+v", results)
	}
}
