package anthropicbatch_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
)

// H5: batch creation is not idempotent — a 500 the provider returned after
// committing the batch must not be resent (duplicate batch, double charge).
// 503/429 were rejected before processing and still retry.
func TestAnthropicBatchSubmitNotRetriedOn500(t *testing.T) {
	for _, tc := range []struct {
		status int
		want   int32
	}{{http.StatusInternalServerError, 1}, {http.StatusBadGateway, 1}, {http.StatusServiceUnavailable, 3}} {
		var hits atomic.Int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			hits.Add(1)
			w.WriteHeader(tc.status)
		}))
		bc, err := llm.NewBatchClient(llm.Config{
			Provider: "anthropic", Model: "claude-sonnet-5", APIKey: "test-key",
			BaseURL: srv.URL, DisableProxy: true, Timeout: 5 * time.Second,
			RetryPolicy: &llm.RetryPolicy{BaseDelay: time.Millisecond, MaxDelay: time.Millisecond, Factor: 1},
		})
		if err != nil {
			t.Fatal(err)
		}
		_, err = bc.Submit(context.Background(), []llm.BatchItem{{
			ItemID:  "item-1",
			Request: &llm.Request{Model: "claude-sonnet-5", MaxTokens: 16, Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")}},
		}}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour})
		_ = bc.Close()
		srv.Close()
		if err == nil {
			t.Errorf("status %d: want an error", tc.status)
		}
		if got := hits.Load(); got != tc.want {
			t.Errorf("status %d: submit hits = %d, want %d", tc.status, got, tc.want)
		}
	}
}
