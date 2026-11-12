package anthropicbatch

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

func newTestClient(t *testing.T, cfg llm.Config) *Client {
	t.Helper()
	cfg.Provider = "anthropic"
	cfg.APIKey = "k"
	if cfg.Model == "" {
		cfg.Model = "claude-sonnet-5"
	}
	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// One result line holds a whole message, so a long answer is a long line. The
// 256 KB SSE line cap failed the whole download (B26).
func TestResultsWithALongAnswer(t *testing.T) {
	text := strings.Repeat("word ", 80_000) // 400 KB
	line, _ := json.Marshal(map[string]any{"custom_id": "item-1", "result": map[string]any{"type": "succeeded", "message": map[string]any{
		"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5",
		"content": []any{map[string]any{"type": "text", "text": text}}, "stop_reason": "end_turn",
		"usage": map[string]any{"input_tokens": 1, "output_tokens": 80000},
	}}})
	results, err := newTestClient(t, llm.Config{}).parseResults(strings.NewReader(string(line) + "\n"))
	if err != nil {
		t.Fatalf("parseResults: %v", err)
	}
	if len(results) != 1 || results[0].Response == nil || len(results[0].Response.Content) != len(text) {
		t.Fatalf("results = %+v", results)
	}
}

// The download is held to MaxBatchDownloadBytes; the driver ignored it.
func TestResultsHonourTheDownloadCap(t *testing.T) {
	body := strings.Repeat(`{"custom_id":"x","result":{"type":"expired"}}`+"\n", 100)
	c := newTestClient(t, llm.Config{MaxBatchDownloadBytes: 1000})
	if _, err := c.parseResults(strings.NewReader(body)); err == nil || !strings.Contains(err.Error(), "MaxBatchDownloadBytes") {
		t.Fatalf("err = %v, want the download cap error", err)
	}
	// Exactly at the cap is fine.
	exact := body[:len(body)/10]
	c = newTestClient(t, llm.Config{MaxBatchDownloadBytes: len(exact)})
	if res, err := c.parseResults(strings.NewReader(exact)); err != nil || len(res) != 10 {
		t.Fatalf("a file of exactly the cap: %d results, err %v", len(res), err)
	}
}

// An errored item nests its error object: {"type":"error","error":{…}}. Only
// the flat shape was read, so the provider's message was lost (B27).
func TestErroredItemKeepsTheProviderMessage(t *testing.T) {
	body := `{"custom_id":"a","result":{"type":"errored","error":{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens: 99999999 > 64000"}}}}` + "\n" +
		`{"custom_id":"b","result":{"type":"errored","error":{"type":"invalid_request_error","message":"flat shape"}}}` + "\n" +
		`{"custom_id":"c","result":{"type":"errored","error":{"type":"error","error":{}}}}` + "\n" +
		`{"custom_id":"d","result":{"type":"expired"}}` + "\n"
	results, err := newTestClient(t, llm.Config{}).parseResults(strings.NewReader(body))
	if err != nil || len(results) != 4 {
		t.Fatalf("results=%v err=%v", results, err)
	}
	for i, want := range []string{
		"errored: invalid_request_error: max_tokens: 99999999 > 64000",
		"errored: invalid_request_error: flat shape",
		"errored: batch item failed",
		"expired: batch item failed",
	} {
		if results[i].Error == nil || results[i].Error.Message != want {
			t.Errorf("item %s: error = %+v, want %q", results[i].ItemID, results[i].Error, want)
		}
	}
}

// A batch handle is often loaded back from storage. Its ID went into the URL
// path as is, so "../" or "?" reached another endpoint with the key (B29).
func TestBatchIDIsCheckedBeforeItBecomesAPath(t *testing.T) {
	for _, id := range []string{"msgbatch_013Zva2CMHLNnXjNJJKqJ2EF", "msgbatch-1"} {
		if err := checkBatchID(id); err != nil {
			t.Errorf("valid id %q refused: %v", id, err)
		}
	}
	for _, id := range []string{"", "..", "../../v1/files", "msgbatch_1/../../x", "msgbatch_1?beta=x", "msgbatch_1#x", "a b", "msgbatch_1%2F..", strings.Repeat("a", 129)} {
		if err := checkBatchID(id); err == nil {
			t.Errorf("hostile id %q accepted", id)
		}
	}
}

// The same through the client: a tampered handle never reaches the network.
func TestTamperedHandleSendsNoRequest(t *testing.T) {
	var paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.RawPath+r.URL.Path+"?"+r.URL.RawQuery)
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()
	c := newTestClient(t, llm.Config{BaseURL: srv.URL, DisableRetries: true})
	handle, err := c.toHandle(&batchObject{ID: "msgbatch_1", ProcessingStatus: "in_progress"})
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"../../v1/files", "msgbatch_1?x=1", ".."} {
		h := *handle
		h.ID = id
		_, getErr := c.Get(context.Background(), h)
		_, resErr := c.Results(context.Background(), h)
		_, cancelErr := c.Cancel(context.Background(), h)
		if getErr == nil || resErr == nil || cancelErr == nil {
			t.Fatalf("id %q: Get/Results/Cancel errors = %v / %v / %v", id, getErr, resErr, cancelErr)
		}
	}
	if len(paths) != 0 {
		t.Fatalf("tampered handles reached the server: %v", paths)
	}
}
