package geminibatch_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
)

// H5: batchGenerateContent creates a billable job — never resent after a 500.
func TestGeminiBatchSubmitNotRetriedOn500(t *testing.T) {
	for _, tc := range []struct {
		status int
		want   int32
	}{{http.StatusInternalServerError, 1}, {http.StatusGatewayTimeout, 1}, {http.StatusTooManyRequests, 3}} {
		var hits atomic.Int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			hits.Add(1)
			w.WriteHeader(tc.status)
		}))
		bc, err := llm.NewBatchClient(llm.Config{
			Provider: "gemini", Model: "gemini-3.6-flash", APIKey: "k",
			BaseURL: srv.URL + "/v1beta/models", DisableProxy: true, Timeout: 5 * time.Second,
			RetryPolicy: &llm.RetryPolicy{BaseDelay: time.Millisecond, MaxDelay: time.Millisecond, Factor: 1},
		})
		if err != nil {
			t.Fatal(err)
		}
		_, err = bc.Submit(context.Background(), []llm.BatchItem{{
			ItemID:  "item-a",
			Request: &llm.Request{Model: "gemini-3.6-flash", Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")}},
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
