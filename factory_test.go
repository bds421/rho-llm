package llm_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider"
	_ "github.com/bds421/rho-llm/provider/anthropic"
	_ "github.com/bds421/rho-llm/provider/gemini"
	_ "github.com/bds421/rho-llm/provider/openaicompat"
)

// Attack: a negative MaxTokens must be REJECTED, never silently floored into a
// valid-looking request. Flooring a negative would hide a caller bug.
func TestBreakNegativeMaxTokensRejectedNotFloored(t *testing.T) {
	_, err := llm.NewClient(llm.Config{
		Provider: "anthropic", Model: "claude-sonnet-4-6", APIKey: "k", MaxTokens: -1,
	})
	if err == nil {
		t.Fatal("MaxTokens=-1 accepted: a caller bug was silently converted into a valid request")
	}
	if !strings.Contains(err.Error(), "MaxTokens") {
		t.Errorf("error = %q, want it to name MaxTokens", err.Error())
	}
}

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
		AnthropicVersion: llm.AnthropicVersion20230601,
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

// TestFactoryDefaultModels verifies default models are set correctly.
func TestFactoryDefaultModels(t *testing.T) {
	tests := []struct {
		provider string
		expected string
	}{
		{"anthropic", "claude-sonnet-5-5"},
		{"claude", "claude-sonnet-5-5"},
		{"xai", "grok-4.7"},
		{"grok", "grok-4.7"},
		{"openai", "gpt-6.1-sol"},
		{"meta", "muse-spark-1.3"},
		{"gemini", "gemini-3.5-flash-lite"},
		{"google", "gemini-3.5-flash-lite"},
	}

	for _, tc := range tests {
		t.Run(tc.provider, func(t *testing.T) {
			model := llm.GetDefaultModel(tc.provider)
			if model != tc.expected {
				t.Errorf("Expected %s for %s, got %s", tc.expected, tc.provider, model)
			}
		})
	}
}

// TestFactoryUnknownProvider verifies error handling for unknown providers with no BaseURL.
func TestFactoryUnknownProvider(t *testing.T) {
	cfg := llm.Config{
		Provider: "unknown",
		APIKey:   "test",
	}

	_, err := llm.NewClient(cfg)
	if err == nil {
		t.Error("Expected error for unknown provider with no base URL")
	}
}

// TestOpenAICompatCustomBaseURL verifies custom BaseURL is used.
func TestOpenAICompatCustomBaseURL(t *testing.T) {
	cfg := llm.Config{
		Provider:   "custom",
		Model:      "my-model",
		APIKey:     "test-key",
		BaseURL:    "http://localhost:9999/v1",
		AuthHeader: "Bearer",
		MaxTokens:  100,
		Timeout:    10 * time.Second,
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create custom client: %v", err)
	}
	defer client.Close()

	if client.Provider() != "custom" {
		t.Errorf("Expected provider 'custom', got %q", client.Provider())
	}
	if client.Model() != "my-model" {
		t.Errorf("Expected model 'my-model', got %q", client.Model())
	}
}

// TestProviderNameOverride verifies ProviderName overrides Client.Provider().
func TestProviderNameOverride(t *testing.T) {
	cfg := llm.Config{
		Provider:     "openai",
		Model:        "gpt-4",
		APIKey:       "test-key",
		ProviderName: "my-proxy",
		MaxTokens:    100,
		Timeout:      10 * time.Second,
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	if client.Provider() != "my-proxy" {
		t.Errorf("Expected provider 'my-proxy', got %q", client.Provider())
	}
}

// TestNewClientWithLogRequests verifies LogRequests config wraps client.
func TestNewClientWithLogRequests(t *testing.T) {
	cfg := llm.Config{
		Provider:    "ollama",
		Model:       "llama3",
		LogRequests: true,
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}
	defer client.Close()

	// Client should be wrapped with logging (we can't easily verify, but it shouldn't panic)
	if client.Provider() != "ollama" {
		t.Errorf("Provider() = %q, want ollama", client.Provider())
	}
}

// TestNewClientEmptyModel verifies that creating a client without a model
// returns a clear error.
func TestNewClientEmptyModel(t *testing.T) {
	cfg := llm.Config{
		Provider: "anthropic",
		APIKey:   "test-key",
	}
	_, err := llm.NewClient(cfg)
	if err == nil {
		t.Fatal("expected error for empty model, got nil")
	}
	if !strings.Contains(err.Error(), "model is required") {
		t.Errorf("unexpected error: %v", err)
	}
}

// TestNegativeMaxTokensRejected verifies MaxTokens < 0 returns error.
func TestNegativeMaxTokensRejected(t *testing.T) {
	cfg := llm.Config{
		Provider:  "ollama",
		Model:     "llama3",
		MaxTokens: -1,
	}

	_, err := llm.NewClient(cfg)
	if err == nil {
		t.Fatal("NewClient with negative MaxTokens should fail")
	}
	if !strings.Contains(err.Error(), "MaxTokens") {
		t.Errorf("error = %q, want mention of MaxTokens", err.Error())
	}
}

// TestNegativeTemperatureRejected verifies Temperature < 0 returns error.
func TestNegativeTemperatureRejected(t *testing.T) {
	cfg := llm.Config{
		Provider:    "ollama",
		Model:       "llama3",
		MaxTokens:   100,
		Temperature: floatPtr(-0.5),
	}

	_, err := llm.NewClient(cfg)
	if err == nil {
		t.Fatal("NewClient with negative Temperature should fail")
	}
	if !strings.Contains(err.Error(), "Temperature") {
		t.Errorf("error = %q, want mention of Temperature", err.Error())
	}
}

// TestThinkingLevelRejectedForOpenAICompat verifies that ThinkingLevel on an
// OpenAI-compatible host without a documented reasoning control returns an
// error instead of being dropped silently. Hosts with one (openai, groq,
// deepseek, …) send it in their own dialect since P2.
func TestThinkingLevelRejectedForOpenAICompat(t *testing.T) {
	providers := []string{"xai", "mistral", "cohere", "minimax"}

	for _, provider := range providers {
		t.Run(provider, func(t *testing.T) {
			cfg := llm.Config{
				Provider: provider,
				Model:    "test-model",
				ModelCapabilities: llm.Capabilities(
					llm.CapabilityChat, llm.CapabilityReasoning,
				),
				APIKey:        "test-key",
				MaxTokens:     100,
				ThinkingLevel: llm.ThinkingHigh,
				Timeout:       5 * time.Second,
			}

			client, err := llm.NewClient(cfg)
			if err != nil {
				t.Fatalf("NewClient failed: %v", err)
			}
			defer client.Close()

			_, err = client.Complete(context.Background(), llm.Request{
				Messages:      []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
				ThinkingLevel: llm.ThinkingHigh,
			})
			if err == nil {
				t.Fatal("Complete with ThinkingLevel on openai_compat should fail")
			}
			if !strings.Contains(err.Error(), "ThinkingLevel is not supported") {
				t.Errorf("error = %q, want pre-transport reasoning rejection", err.Error())
			}
		})
	}
}

// TestThinkingLevelFromConfigRoutesToResponsesAPI verifies that ThinkingLevel
// set on Config for a ResponsesAPI model auto-routes to the Responses API provider.
func TestThinkingLevelFromConfigRoutesToResponsesAPI(t *testing.T) {
	cfg := llm.Config{
		Provider:      "openai",
		Model:         "gpt-5",
		APIKey:        "test-key",
		MaxTokens:     100,
		ThinkingLevel: llm.ThinkingMedium,
		Timeout:       5 * time.Second,
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient failed: %v", err)
	}
	defer client.Close()

	// Should route to Responses API and get an auth error (not a ThinkingLevel rejection).
	_, err = client.Complete(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
	})
	if err == nil {
		t.Fatal("expected API error with test-key")
	}
	if strings.Contains(err.Error(), "ThinkingLevel") {
		t.Errorf("ResponsesAPI model should not reject ThinkingLevel, got: %v", err)
	}
}

// TestThinkingBudgetTokens verifies the token budget mapping for all levels.
func TestThinkingBudgetTokens(t *testing.T) {
	tests := []struct {
		level        llm.ThinkingLevel
		customBudget int
		want         int
	}{
		{llm.ThinkingNone, 0, 0},
		{llm.ThinkingMinimal, 0, 1024},
		{llm.ThinkingLow, 0, 4096},
		{llm.ThinkingMedium, 0, 16384},
		{llm.ThinkingHigh, 0, 65536},
		{llm.ThinkingXHigh, 0, 128000},
		{llm.ThinkingMax, 0, 128000},
		// Custom budget overrides level default
		{llm.ThinkingLow, 8000, 8000},
		{llm.ThinkingHigh, 1000, 1000},
		// Custom budget with ThinkingNone
		{llm.ThinkingNone, 5000, 5000},
	}

	for _, tc := range tests {
		got := llm.ThinkingBudgetTokens(tc.level, tc.customBudget)
		if got != tc.want {
			t.Errorf("ThinkingBudgetTokens(%q, %d) = %d, want %d", tc.level, tc.customBudget, got, tc.want)
		}
	}
}

// TestClampThinkingBudget verifies centralized budget clamping.
func TestClampThinkingBudget(t *testing.T) {
	// No clamping needed
	got := llm.ClampThinkingBudget("test", "model", 1000, 65536)
	if got != 1000 {
		t.Errorf("ClampThinkingBudget(1000, 65536) = %d, want 1000", got)
	}

	// Clamping needed
	got = llm.ClampThinkingBudget("test", "model", 200000, 65536)
	if got != 65536 {
		t.Errorf("ClampThinkingBudget(200000, 65536) = %d, want 65536", got)
	}

	// Zero maxTokens = no clamping
	got = llm.ClampThinkingBudget("test", "model", 200000, 0)
	if got != 200000 {
		t.Errorf("ClampThinkingBudget(200000, 0) = %d, want 200000", got)
	}
}

// TestUnknownProviderRequiresBaseURL verifies that unknown providers without BaseURL fail.
func TestUnknownProviderRequiresBaseURL(t *testing.T) {
	// Unknown provider without BaseURL → error
	cfg := llm.Config{
		Provider:  "typo-provider",
		Model:     "some-model",
		APIKey:    "test-key",
		MaxTokens: 100,
	}
	_, err := llm.NewClient(cfg)
	if err == nil {
		t.Fatal("expected error for unknown provider without BaseURL")
	}
	if !strings.Contains(err.Error(), "unknown provider") {
		t.Errorf("error = %q, want mention of 'unknown provider'", err.Error())
	}

	// Unknown provider WITH BaseURL → succeeds
	cfg.BaseURL = "http://localhost:9999/v1"
	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("unknown provider with BaseURL should succeed: %v", err)
	}
	client.Close()
}

// TestNewClientAppliesMaxTokensFloor verifies that a Config built as a struct
// literal — the form every example and README snippet uses — gets a usable
// MaxTokens instead of 0. Anthropic rejects a streaming request with
// "max_tokens cannot be 0" (HTTP 400), so leaving the zero value in place
// made the documented streaming recipe fail against the live API.
//
// This mirrors the existing Timeout floor in NewClient: DefaultConfig() is not
// on the path when callers construct Config directly.
func TestNewClientAppliesMaxTokensFloor(t *testing.T) {
	c, err := llm.NewClient(llm.Config{
		Provider: "anthropic",
		Model:    "claude-sonnet-4-6",
		APIKey:   "test-key-not-used",
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer c.Close()

	// The floor must be observable on the wire, not just defined as a
	// constant: assert the adapter would send a non-zero max_tokens.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		mt, _ := body["max_tokens"].(float64)
		if mt <= 0 {
			t.Errorf("wire max_tokens = %v, want > 0", body["max_tokens"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"m","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer srv.Close()

	c2, err := llm.NewClient(llm.Config{
		Provider: "anthropic",
		Model:    "claude-sonnet-4-6",
		APIKey:   "test-key-not-used",
		BaseURL:  srv.URL,
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer c2.Close()

	if _, err := c2.Complete(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
	}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
}

// TestBatchAndModalityClientsApplyMaxTokensFloor verifies that the
// DefaultMaxTokens floor reaches every client constructor, not just NewClient.
//
// NewBatchClient and NewModalityClient apply the Timeout floor but originally
// skipped MaxTokens. The Anthropic batch encoder reuses the live buildRequest,
// which falls back to Config.MaxTokens, so a Config built as a struct literal
// produced batch entries carrying "max_tokens": 0 — the same HTTP 400 the
// single-request path was fixed for, still reachable through the batch API.
// The wire-level assertion lives in provider/anthropic (batchline_internal);
// here we pin that both constructors succeed and normalize the config.
func TestBatchAndModalityClientsApplyMaxTokensFloor(t *testing.T) {
	bc, err := llm.NewBatchClient(llm.Config{
		Provider: "anthropic", Model: "claude-sonnet-4-6", APIKey: "test-key",
	})
	if err != nil {
		t.Fatalf("NewBatchClient: %v", err)
	}
	defer bc.Close()

	// Modality clients exist only for protocols with a registered driver;
	// openai has one, anthropic does not.
	mc, err := llm.NewModalityClient(llm.Config{
		Provider: "openai", Model: "gpt-4.1", APIKey: "test-key",
	})
	if err != nil {
		t.Fatalf("NewModalityClient: %v", err)
	}
	defer mc.Close()
}
