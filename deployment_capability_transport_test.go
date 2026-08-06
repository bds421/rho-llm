package llm_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider/anthropic"
	_ "github.com/bds421/rho-llm/provider/gemini"
	_ "github.com/bds421/rho-llm/provider/openaicompat"
)

func TestClientFactoryEnforcesExactDeploymentCapabilitiesBeforeTransport(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"response-a","model":"tenant/exact-model","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	cfg := llm.Config{
		Provider: "vllm", Model: "tenant/exact-model", BaseURL: server.URL,
		ModelCapabilities: llm.Capabilities(llm.CapabilityChat),
		DisableProxy:      true, DisableRetries: true,
	}
	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	_, err = client.Complete(context.Background(), llm.Request{
		Model: cfg.Model, Tools: []llm.Tool{{Name: "forbidden_tool"}},
	})
	if err == nil || !strings.Contains(err.Error(), "does not support tools") {
		t.Fatalf("undeclared tool capability error = %v", err)
	}
	if calls.Load() != 0 {
		t.Fatalf("unsupported request reached provider %d time(s)", calls.Load())
	}

	response, err := client.Complete(context.Background(), llm.Request{Model: cfg.Model})
	if err != nil {
		t.Fatal(err)
	}
	if response.Content != "ok" || calls.Load() != 1 {
		t.Fatalf("response=%+v calls=%d", response, calls.Load())
	}

	_, err = client.Complete(context.Background(), llm.Request{Model: "tenant/other-model"})
	if err == nil || !strings.Contains(err.Error(), "cannot authorize") {
		t.Fatalf("different-model error = %v", err)
	}
	if calls.Load() != 1 {
		t.Fatalf("different model reached provider %d time(s)", calls.Load())
	}
}

func TestUnsupportedTemperatureFailsBeforeTransportWhileNilUsesProviderDefault(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}`)
	}))
	defer server.Close()

	client, err := llm.NewClient(llm.Config{
		Provider: "gemini", Model: "gemini-3.6-flash", APIKey: "test",
		ModelCapabilities: llm.Capabilities(llm.CapabilityChat, llm.CapabilityTemperature),
		BaseURL:           server.URL, DisableProxy: true, DisableRetries: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	temperature := 0.2
	_, err = client.Complete(context.Background(), llm.Request{Temperature: &temperature})
	if err == nil || !strings.Contains(err.Error(), "sampling_temperature") {
		t.Fatalf("explicit temperature error = %v", err)
	}
	if calls.Load() != 0 {
		t.Fatalf("unsupported temperature reached provider %d time(s)", calls.Load())
	}

	response, err := client.Complete(context.Background(), llm.Request{})
	if err != nil {
		t.Fatal(err)
	}
	if response.Content != "ok" || calls.Load() != 1 {
		t.Fatalf("response=%+v calls=%d", response, calls.Load())
	}
}

func TestAnthropicReasoningTemperatureConflictFailsBeforeTransport(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"message-a","type":"message","role":"assistant","model":"tenant/claude","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	client, err := llm.NewClient(llm.Config{
		Provider: "anthropic", Model: "tenant/claude", APIKey: "test",
		ModelCapabilities: llm.Capabilities(
			llm.CapabilityChat, llm.CapabilityReasoning, llm.CapabilityTemperature,
		),
		BaseURL: server.URL, DisableProxy: true, DisableRetries: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	incompatible := 0.2
	_, err = client.Complete(context.Background(), llm.Request{
		ThinkingLevel: llm.ThinkingLow,
		Temperature:   &incompatible,
	})
	if err == nil || !strings.Contains(err.Error(), "requires sampling temperature 1") {
		t.Fatalf("incompatible reasoning temperature error = %v", err)
	}
	if calls.Load() != 0 {
		t.Fatalf("incompatible request reached provider %d time(s)", calls.Load())
	}

	exact := 1.0
	response, err := client.Complete(context.Background(), llm.Request{
		ThinkingLevel: llm.ThinkingLow,
		Temperature:   &exact,
	})
	if err != nil {
		t.Fatal(err)
	}
	if response.Content != "ok" || calls.Load() != 1 {
		t.Fatalf("response=%+v calls=%d", response, calls.Load())
	}
}
