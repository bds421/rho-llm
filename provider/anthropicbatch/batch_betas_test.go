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
)

// Batch creation sends Config.BetaFeatures (interleaved thinking only with
// budget thinking) and the items' output_config.format; a capability the
// model lacks is refused before anything is sent.
func TestBatchSendsBetasAndFormat(t *testing.T) {
	var beta string
	var params map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		beta = r.Header.Get("anthropic-beta")
		body, _ := io.ReadAll(r.Body)
		var payload struct {
			Requests []struct {
				Params map[string]any `json:"params"`
			} `json:"requests"`
		}
		_ = json.Unmarshal(body, &payload)
		if len(payload.Requests) > 0 {
			params = payload.Requests[0].Params
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msgbatch_01","type":"message_batch","processing_status":"in_progress",
			"request_counts":{"processing":1,"succeeded":0,"errored":0,"canceled":0,"expired":0},
			"created_at":"2026-08-06T10:00:00Z","expires_at":"2026-08-07T10:00:00Z"}`))
	}))
	defer srv.Close()

	submit := func(cfg llm.Config, req llm.Request) error {
		cfg.APIKey, cfg.BaseURL, cfg.DisableProxy, cfg.DisableRetries, cfg.Timeout = "k", srv.URL, true, true, 5*time.Second
		bc, err := llm.NewBatchClient(cfg)
		if err != nil {
			return err
		}
		defer bc.Close()
		req.Messages = []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")}
		_, err = bc.Submit(context.Background(), []llm.BatchItem{{ItemID: "item-1", Request: &req}}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour})
		return err
	}
	schema := map[string]any{"type": "object", "properties": map[string]any{}, "additionalProperties": false}
	betas := []string{"server-side-fallback-2026-07-01", "interleaved-thinking-2025-05-14"}

	err := submit(llm.Config{Provider: "anthropic", Model: "claude-haiku-5-5", BetaFeatures: betas},
		llm.Request{Model: "claude-haiku-5-5", MaxTokens: 64, ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSONSchema, Schema: schema}})
	if err != nil {
		t.Fatal(err)
	}
	if beta != "server-side-fallback-2026-07-01" {
		t.Errorf("anthropic-beta = %q", beta)
	}
	if oc, _ := params["output_config"].(map[string]any); oc["format"] == nil {
		t.Errorf("format dropped: %v", params)
	}

	if err := submit(llm.Config{Provider: "anthropic", Model: "claude-haiku-4-5", BetaFeatures: betas},
		llm.Request{Model: "claude-haiku-4-5", MaxTokens: 4096, ThinkingLevel: llm.ThinkingLow}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(beta, "interleaved-thinking-2025-05-14") {
		t.Errorf("budget thinking: anthropic-beta = %q", beta)
	}

	beta, params = "", nil
	custom := llm.Config{Provider: "anthropic", Model: "claude-example-1",
		ModelCapabilities: llm.Capabilities(llm.CapabilityChat, llm.CapabilityBatch)}
	if err := submit(custom, llm.Request{Model: "claude-example-1", MaxTokens: 64,
		ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSONSchema, Schema: schema}}); err == nil || params != nil {
		t.Errorf("unsupported ResponseFormat: err %v, sent %v", err, params)
	}
}
