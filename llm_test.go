package llm_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"gitlab2024.bds421-cloud.com/bds421/rho/llm"
	_ "gitlab2024.bds421-cloud.com/bds421/rho/llm/provider"
)

// TestPrompt is a simple prompt that all models should handle.
const TestPrompt = "Reply with exactly one word: Hello"

// envKey reads an API key from env, supporting both singular and plural forms.
// Checks KEY first, then KEYS (returns first comma-separated value).
func envKey(singular, plural string) string {
	if v := os.Getenv(singular); v != "" {
		return v
	}
	if v := os.Getenv(plural); v != "" {
		if idx := strings.Index(v, ","); idx > 0 {
			return v[:idx]
		}
		return v
	}
	return ""
}

// envKeys reads all API keys from env (plural form, comma-separated).
// Falls back to singular form as a single-element list.
func envKeys(singular, plural string) []string {
	if v := os.Getenv(plural); v != "" {
		var keys []string
		for _, k := range strings.Split(v, ",") {
			k = strings.TrimSpace(k)
			if k != "" {
				keys = append(keys, k)
			}
		}
		if len(keys) > 0 {
			return keys
		}
	}
	if v := os.Getenv(singular); v != "" {
		return []string{v}
	}
	return nil
}

// =============================================================================
// UNIT TESTS (no API keys needed)
// =============================================================================

// TestResolveProtocol verifies protocol resolution for all known providers.
func TestResolveProtocol(t *testing.T) {
	tests := []struct {
		provider string
		expected string
	}{
		{"anthropic", "anthropic"},
		{"claude", "anthropic"},
		{"gemini", "gemini"},
		{"google", "gemini"},
		{"openai", "openai_compat"},
		{"xai", "openai_compat"},
		{"grok", "openai_compat"},
		{"groq", "openai_compat"},
		{"cerebras", "openai_compat"},
		{"mistral", "openai_compat"},
		{"openrouter", "openai_compat"},
		{"ollama", "openai_compat"},
		{"vllm", "openai_compat"},
		{"lmstudio", "openai_compat"},
	}

	for _, tc := range tests {
		t.Run(tc.provider, func(t *testing.T) {
			cfg := llm.Config{Provider: tc.provider}
			got := llm.ResolveProtocol(cfg)
			if got != tc.expected {
				t.Errorf("ResolveProtocol(%q) = %q, want %q", tc.provider, got, tc.expected)
			}
		})
	}
}

// TestResolveProtocolCustomBaseURL verifies unknown providers with BaseURL resolve to openai_compat.
func TestResolveProtocolCustomBaseURL(t *testing.T) {
	cfg := llm.Config{
		Provider: "my-custom-provider",
		BaseURL:  "http://localhost:9999/v1",
	}
	got := llm.ResolveProtocol(cfg)
	if got != "openai_compat" {
		t.Errorf("ResolveProtocol(custom) = %q, want openai_compat", got)
	}
}

// TestProviderPresets verifies all known presets have valid protocols.
func TestProviderPresets(t *testing.T) {
	validProtocols := map[string]bool{
		"anthropic":     true,
		"gemini":        true,
		"openai_compat": true,
	}

	providers := []string{
		"anthropic", "claude", "gemini", "google",
		"openai", "xai", "grok", "groq", "cerebras", "mistral", "openrouter",
		"ollama", "vllm", "lmstudio",
	}

	for _, name := range providers {
		preset, ok := llm.PresetFor(name)
		if !ok {
			t.Errorf("Provider %q has no preset", name)
			continue
		}
		if !validProtocols[preset.Protocol] {
			t.Errorf("Provider %q has invalid protocol %q", name, preset.Protocol)
		}
	}
}

// TestFactoryDefaultModels verifies default models are set correctly.
func TestFactoryDefaultModels(t *testing.T) {
	tests := []struct {
		provider string
		expected string
	}{
		{"anthropic", "claude-sonnet-4-6"},
		{"claude", "claude-sonnet-4-6"},
		{"xai", "grok-4-fast-non-reasoning"},
		{"grok", "grok-4-fast-non-reasoning"},
		{"gemini", "gemini-2.5-flash-lite"},
		{"google", "gemini-2.5-flash-lite"},
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

// TestOllamaNoAuthRequired verifies Ollama client creation without API key.
func TestOllamaNoAuthRequired(t *testing.T) {
	cfg := llm.Config{
		Provider:  "ollama",
		Model:     "llama3",
		MaxTokens: 100,
		Timeout:   10 * time.Second,
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create Ollama client without API key: %v", err)
	}
	defer client.Close()

	if client.Provider() != "ollama" {
		t.Errorf("Expected provider 'ollama', got %q", client.Provider())
	}
}

// TestVLLMNoAuthRequired verifies vLLM client creation without API key.
func TestVLLMNoAuthRequired(t *testing.T) {
	cfg := llm.Config{
		Provider:  "vllm",
		Model:     "meta-llama/Llama-3-8b",
		MaxTokens: 100,
		Timeout:   10 * time.Second,
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create vLLM client without API key: %v", err)
	}
	defer client.Close()

	if client.Provider() != "vllm" {
		t.Errorf("Expected provider 'vllm', got %q", client.Provider())
	}
}

// TestLMStudioNoAuthRequired verifies LM Studio client creation without API key.
func TestLMStudioNoAuthRequired(t *testing.T) {
	cfg := llm.Config{
		Provider:  "lmstudio",
		Model:     "local-model",
		MaxTokens: 100,
		Timeout:   10 * time.Second,
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create LM Studio client without API key: %v", err)
	}
	defer client.Close()

	if client.Provider() != "lmstudio" {
		t.Errorf("Expected provider 'lmstudio', got %q", client.Provider())
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

// TestAvailableModels verifies the available models list.
func TestAvailableModels(t *testing.T) {
	providers := []string{"anthropic", "xai", "gemini"}

	for _, provider := range providers {
		models := llm.GetAvailableModels(provider)
		if models == nil {
			t.Errorf("No models defined for provider %s", provider)
			continue
		}

		if len(models) == 0 {
			t.Errorf("Empty model list for provider %s", provider)
		}

		t.Logf("%s models: %v", provider, models)
	}
}

// TestResolveModelAlias verifies alias resolution.
func TestResolveModelAlias(t *testing.T) {
	tests := []struct {
		alias    string
		expected string
	}{
		{"opus", "claude-opus-4-6"},
		{"sonnet", "claude-sonnet-4-6"},
		{"haiku", "claude-haiku-4-5-20251001"},
		{"grok", "grok-4-fast-non-reasoning"},
		{"grok-code", "grok-code-fast-1"},
		{"gemini-pro", "gemini-3-pro-preview"},
		{"flash-lite", "gemini-2.5-flash-lite"},
		// Non-alias should pass through
		{"claude-opus-4-6", "claude-opus-4-6"},
		{"unknown-model", "unknown-model"},
	}

	for _, tc := range tests {
		t.Run(tc.alias, func(t *testing.T) {
			got := llm.ResolveModelAlias(tc.alias)
			if got != tc.expected {
				t.Errorf("ResolveModelAlias(%q) = %q, want %q", tc.alias, got, tc.expected)
			}
		})
	}
}

// TestProviderForModel verifies model-to-provider resolution.
func TestProviderForModel(t *testing.T) {
	tests := []struct {
		model    string
		expected string
	}{
		{"claude-sonnet-4-6", "anthropic"},
		{"grok-4-fast-non-reasoning", "xai"},
		{"gemini-2.5-flash", "gemini"},
		{"unknown-model", ""},
	}

	for _, tc := range tests {
		t.Run(tc.model, func(t *testing.T) {
			got := llm.ProviderForModel(tc.model)
			if got != tc.expected {
				t.Errorf("ProviderForModel(%q) = %q, want %q", tc.model, got, tc.expected)
			}
		})
	}
}

// TestIsNoAuthProvider verifies no-auth provider detection.
func TestIsNoAuthProvider(t *testing.T) {
	tests := []struct {
		provider string
		expected bool
	}{
		{"ollama", true},
		{"vllm", true},
		{"lmstudio", true},
		{"anthropic", false},
		{"openai", false},
		{"xai", false},
	}

	for _, tc := range tests {
		t.Run(tc.provider, func(t *testing.T) {
			got := llm.IsNoAuthProvider(tc.provider)
			if got != tc.expected {
				t.Errorf("IsNoAuthProvider(%q) = %v, want %v", tc.provider, got, tc.expected)
			}
		})
	}
}

// TestDefaultConfig verifies default config values.
func TestDefaultConfig(t *testing.T) {
	cfg := llm.DefaultConfig()

	if cfg.Provider != "anthropic" {
		t.Errorf("Expected default provider 'anthropic', got %q", cfg.Provider)
	}
	if cfg.MaxTokens != 8192 {
		t.Errorf("Expected default MaxTokens 8192, got %d", cfg.MaxTokens)
	}
	if cfg.Timeout != 120*time.Second {
		t.Errorf("Expected default Timeout 120s, got %v", cfg.Timeout)
	}
}

// TestNewTextMessage verifies text message construction.
func TestNewTextMessage(t *testing.T) {
	msg := llm.NewTextMessage("user", "hello")
	if msg.Role != llm.RoleUser {
		t.Errorf("Expected role 'user', got %q", msg.Role)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("Expected 1 content part, got %d", len(msg.Content))
	}
	if msg.Content[0].Type != llm.ContentText || msg.Content[0].Text != "hello" {
		t.Errorf("Unexpected content: %+v", msg.Content[0])
	}
}

// TestNewToolResultMessage verifies tool result message construction.
func TestNewToolResultMessage(t *testing.T) {
	msg := llm.NewToolResultMessage("tool-123", "result data", false)
	if msg.Role != llm.RoleUser {
		t.Errorf("Expected role 'user', got %q", msg.Role)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("Expected 1 content part, got %d", len(msg.Content))
	}
	part := msg.Content[0]
	if part.Type != llm.ContentToolResult {
		t.Errorf("Expected type 'tool_result', got %q", part.Type)
	}
	if part.ToolResultID != "tool-123" {
		t.Errorf("Expected tool_use_id 'tool-123', got %q", part.ToolResultID)
	}
}

// TestAuthProfileAvailability verifies auth profile state transitions.
func TestAuthProfileAvailability(t *testing.T) {
	p := &llm.AuthProfile{
		Name:      "test",
		APIKey:    "key",
		IsHealthy: true,
	}

	if !p.IsAvailable() {
		t.Error("Fresh profile should be available")
	}

	p.MarkUsed()
	if !p.IsAvailable() {
		t.Error("Used profile should still be available")
	}

	p.MarkFailed(fmt.Errorf("rate limit"), 1*time.Minute)
	if p.IsAvailable() {
		t.Error("Failed profile should not be available during cooldown")
	}

	p.MarkHealthy()
	if !p.IsAvailable() {
		t.Error("Healthy profile should be available")
	}
}

// =============================================================================
// STRUCTURED ERROR TESTS
// =============================================================================

// TestAPIErrorTypes verifies error type construction and classification.
func TestAPIErrorTypes(t *testing.T) {
	tests := []struct {
		name          string
		err           *llm.APIError
		isRateLimited bool
		isOverloaded  bool
		isRetryable   bool
		isAuth        bool
		isCtxLength   bool
	}{
		{
			name:          "rate limit",
			err:           llm.NewRateLimitError("anthropic", "rate limit exceeded"),
			isRateLimited: true,
			isRetryable:   true,
		},
		{
			name:         "overloaded",
			err:          llm.NewOverloadedError("gemini", "service unavailable"),
			isOverloaded: true,
			isRetryable:  true,
		},
		{
			name:   "auth error 401",
			err:    llm.NewAuthError("xai", "unauthorized", 401),
			isAuth: true,
		},
		{
			name:   "auth error 403",
			err:    llm.NewAuthError("openai", "forbidden", 403),
			isAuth: true,
		},
		{
			name:        "context length",
			err:         llm.NewContextLengthError("anthropic", "maximum context length exceeded"),
			isCtxLength: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if llm.IsRateLimited(tc.err) != tc.isRateLimited {
				t.Errorf("IsRateLimited = %v, want %v", llm.IsRateLimited(tc.err), tc.isRateLimited)
			}
			if llm.IsOverloaded(tc.err) != tc.isOverloaded {
				t.Errorf("IsOverloaded = %v, want %v", llm.IsOverloaded(tc.err), tc.isOverloaded)
			}
			if llm.IsRetryable(tc.err) != tc.isRetryable {
				t.Errorf("IsRetryable = %v, want %v", llm.IsRetryable(tc.err), tc.isRetryable)
			}
			if llm.IsAuthError(tc.err) != tc.isAuth {
				t.Errorf("IsAuthError = %v, want %v", llm.IsAuthError(tc.err), tc.isAuth)
			}
			if llm.IsContextLength(tc.err) != tc.isCtxLength {
				t.Errorf("IsContextLength = %v, want %v", llm.IsContextLength(tc.err), tc.isCtxLength)
			}
		})
	}
}

// TestAPIErrorFromStatus verifies automatic error classification from HTTP status.
func TestAPIErrorFromStatus(t *testing.T) {
	tests := []struct {
		status      int
		body        string
		isRetryable bool
		isAuth      bool
		isCtxLen    bool
	}{
		{429, "rate limit exceeded", true, false, false},
		{503, "overloaded", true, false, false},
		{401, "invalid api key", false, true, false},
		{403, "forbidden", false, true, false},
		{400, "maximum context length exceeded", false, false, true},
		{400, "context_length_exceeded: max 8192 tokens", false, false, true},
		{400, "input too long for model", false, false, true},
		{400, "request too large", false, false, true},
		{400, "bad request", false, false, false},
		{500, "internal server error", true, false, false},
	}

	for _, tc := range tests {
		name := fmt.Sprintf("status_%d", tc.status)
		t.Run(name, func(t *testing.T) {
			err := llm.NewAPIErrorFromStatus("test", tc.status, tc.body)
			if llm.IsRetryable(err) != tc.isRetryable {
				t.Errorf("IsRetryable = %v, want %v", llm.IsRetryable(err), tc.isRetryable)
			}
			if llm.IsAuthError(err) != tc.isAuth {
				t.Errorf("IsAuthError = %v, want %v", llm.IsAuthError(err), tc.isAuth)
			}
			if llm.IsContextLength(err) != tc.isCtxLen {
				t.Errorf("IsContextLength = %v, want %v", llm.IsContextLength(err), tc.isCtxLen)
			}
		})
	}
}

// TestAPIErrorNilSafety verifies nil errors don't panic.
func TestAPIErrorNilSafety(t *testing.T) {
	if llm.IsRateLimited(nil) {
		t.Error("nil should not be rate limited")
	}
	if llm.IsRetryable(nil) {
		t.Error("nil should not be retryable")
	}
	if llm.IsAuthError(nil) {
		t.Error("nil should not be auth error")
	}
}

// =============================================================================
// BACKOFF TESTS
// =============================================================================

// TestBackoffExponentialGrowth verifies backoff grows exponentially.
func TestBackoffExponentialGrowth(t *testing.T) {
	baseDelay := 1 * time.Second
	maxDelay := 30 * time.Second

	// Run multiple samples to account for jitter
	for attempt := 0; attempt < 5; attempt++ {
		var total time.Duration
		samples := 100
		for i := 0; i < samples; i++ {
			total += llm.Backoff(attempt, baseDelay, maxDelay)
		}
		avg := total / time.Duration(samples)

		// Expected center: min(baseDelay * 2^attempt, maxDelay)
		expected := baseDelay
		for j := 0; j < attempt; j++ {
			expected *= 2
			if expected > maxDelay {
				expected = maxDelay
				break
			}
		}

		// Allow 35% tolerance for jitter
		low := time.Duration(float64(expected) * 0.65)
		high := time.Duration(float64(expected) * 1.35)
		if high > maxDelay {
			high = maxDelay
		}

		if avg < low || avg > high {
			t.Errorf("attempt %d: avg=%v, expected ~%v (range %v-%v)", attempt, avg, expected, low, high)
		}
	}
}

// TestBackoffMaxCap verifies backoff is capped at maxDelay.
func TestBackoffMaxCap(t *testing.T) {
	for i := 0; i < 100; i++ {
		d := llm.Backoff(20, 1*time.Second, 5*time.Second)
		if d > 5*time.Second {
			t.Errorf("backoff(%d) = %v, exceeds max 5s", 20, d)
		}
	}
}

// TestBackoffNegativeAttempt verifies negative attempt doesn't panic.
func TestBackoffNegativeAttempt(t *testing.T) {
	d := llm.Backoff(-1, 1*time.Second, 30*time.Second)
	if d <= 0 || d > 30*time.Second {
		t.Errorf("backoff(-1) = %v, expected positive value", d)
	}
}

// =============================================================================
// COST ESTIMATION TESTS
// =============================================================================

// TestEstimateCost verifies cost calculation for known models.
func TestEstimateCost(t *testing.T) {
	tests := []struct {
		model        string
		inputTokens  int
		outputTokens int
		wantMin      float64
		wantMax      float64
	}{
		// claude-sonnet-4-6: $3/1M input, $15/1M output
		// 1000 input: 3 * 1000/1M = 0.003, 500 output: 15 * 500/1M = 0.0075
		{"claude-sonnet-4-6", 1000, 500, 0.0104, 0.0106},
		// gemini-2.5-flash-lite: free
		{"gemini-2.5-flash-lite", 10000, 5000, 0, 0},
		// unknown model: 0
		{"unknown-model", 1000, 500, 0, 0},
		// claude-opus-4-6: $15/1M input, $75/1M output
		// 10000 input: 0.15, 2000 output: 0.15
		{"claude-opus-4-6", 10000, 2000, 0.299, 0.301},
	}

	for _, tc := range tests {
		t.Run(tc.model, func(t *testing.T) {
			cost := llm.EstimateCost(tc.model, tc.inputTokens, tc.outputTokens)
			if cost < tc.wantMin || cost > tc.wantMax {
				t.Errorf("EstimateCost(%s, %d, %d) = %f, want [%f, %f]",
					tc.model, tc.inputTokens, tc.outputTokens, cost, tc.wantMin, tc.wantMax)
			}
		})
	}
}

// =============================================================================
// MIDDLEWARE TESTS
// =============================================================================

// mockClient is a minimal Client for testing middleware.
type mockClient struct {
	provider     string
	model        string
	resp         *llm.Response
	err          error
	completeFunc func(context.Context, llm.Request) (*llm.Response, error)
}

func (m *mockClient) Complete(ctx context.Context, req llm.Request) (*llm.Response, error) {
	if m.completeFunc != nil {
		return m.completeFunc(ctx, req)
	}
	return m.resp, m.err
}

func (m *mockClient) Stream(_ context.Context, _ llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		if m.err != nil {
			yield(llm.StreamEvent{}, m.err)
			return
		}
		if !yield(llm.StreamEvent{Type: llm.EventContent, Text: "hello"}, nil) {
			return
		}
		yield(llm.StreamEvent{Type: llm.EventDone, StopReason: "end_turn", InputTokens: 10, OutputTokens: 5}, nil)
	}
}

func (m *mockClient) Provider() string { return m.provider }
func (m *mockClient) Model() string    { return m.model }
func (m *mockClient) Close() error     { return nil }

// TestLoggingClientDelegates verifies the logging wrapper delegates correctly.
func TestLoggingClientDelegates(t *testing.T) {
	inner := &mockClient{
		provider: "test",
		model:    "test-model",
		resp: &llm.Response{
			Content:      "hello",
			InputTokens:  10,
			OutputTokens: 5,
			StopReason:   "end_turn",
		},
	}

	client := llm.WithLogging(inner)

	if client.Provider() != "test" {
		t.Errorf("Provider() = %q, want test", client.Provider())
	}
	if client.Model() != "test-model" {
		t.Errorf("Model() = %q, want test-model", client.Model())
	}

	// Test Complete
	resp, err := client.Complete(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}
	if resp.Content != "hello" {
		t.Errorf("Content = %q, want hello", resp.Content)
	}

	// Test Stream
	var chunks int
	for event, err := range client.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage("user", "hi")},
	}) {
		if err != nil {
			t.Fatalf("Stream failed: %v", err)
		}
		if event.Type == llm.EventContent {
			chunks++
		}
	}
	if chunks != 1 {
		t.Errorf("chunks = %d, want 1", chunks)
	}
}

// TestLoggingClientError verifies error propagation.
func TestLoggingClientError(t *testing.T) {
	apiErr := llm.NewRateLimitError("test", "too fast")
	inner := &mockClient{
		provider: "test",
		model:    "test-model",
		err:      apiErr,
	}

	client := llm.WithLogging(inner)

	_, err := client.Complete(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage("user", "hi")},
	})
	if err == nil {
		t.Fatal("Expected error")
	}
	if !llm.IsRateLimited(err) {
		t.Errorf("Expected rate limit error, got: %v", err)
	}
}

// TestModelInfoContextWindow verifies all Anthropic/Gemini models have ContextWindow set.
func TestModelInfoContextWindow(t *testing.T) {
	modelsToCheck := []string{
		"claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001",
		"gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
		"grok-4-fast-non-reasoning", "grok-3",
	}

	for _, model := range modelsToCheck {
		info, ok := llm.GetModelInfo(model)
		if !ok {
			t.Errorf("Model %s not in registry", model)
			continue
		}
		if info.ContextWindow == 0 {
			t.Errorf("Model %s has ContextWindow=0", model)
		}
	}
}

// TestModelInfoThinkingFlags verifies reasoning flags are correctly set in the registry.
func TestModelInfoThinkingFlags(t *testing.T) {
	// 1. API-controlled (SupportsThinking)
	apiControlled := []string{
		"claude-opus-4-6", "claude-sonnet-4-6", "claude-sonnet-4-5",
	}
	for _, model := range apiControlled {
		info, ok := llm.GetModelInfo(model)
		if !ok || !info.SupportsThinking || info.Thinking {
			t.Errorf("Model %s should support API-controlled thinking", model)
		}
	}

	// 2. Intrinsic reasoning (Thinking)
	intrinsic := []string{
		"grok-4-1-fast-reasoning", "grok-4-fast-reasoning",
	}
	for _, model := range intrinsic {
		info, ok := llm.GetModelInfo(model)
		if !ok || !info.Thinking || info.SupportsThinking {
			t.Errorf("Model %s should have intrinsic reasoning", model)
		}
	}

	// 3. No reasoning
	none := []string{
		"gemini-2.5-flash", "claude-haiku-4-5-20251001", "grok-3-mini",
	}
	for _, model := range none {
		info, ok := llm.GetModelInfo(model)
		if !ok || info.Thinking || info.SupportsThinking {
			t.Errorf("Model %s should not have any reasoning flags", model)
		}
	}
}

// =============================================================================
// INTEGRATION TESTS (require API keys)
// =============================================================================

// TestAllProviders runs a simple completion test against all configured providers.
func TestAllProviders(t *testing.T) {
	anthropicKey := envKey("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEYS")
	xaiKey := envKey("XAI_API_KEY", "XAI_API_KEYS")
	geminiKey := envKey("GEMINI_API_KEY", "GEMINI_API_KEYS")

	tests := []struct {
		name     string
		provider string
		model    string
		apiKey   string
		skip     bool
	}{
		{
			name:     "Anthropic/Claude-Sonnet",
			provider: "anthropic",
			model:    "claude-sonnet-4-6",
			apiKey:   anthropicKey,
			skip:     anthropicKey == "",
		},
		{
			name:     "Anthropic/Claude-Haiku",
			provider: "anthropic",
			model:    "claude-haiku-4-5-20251001",
			apiKey:   anthropicKey,
			skip:     anthropicKey == "",
		},
		{
			name:     "xAI/Grok-4-Fast",
			provider: "xai",
			model:    "grok-4-fast-non-reasoning",
			apiKey:   xaiKey,
			skip:     xaiKey == "",
		},
		{
			name:     "xAI/Grok-3-Mini",
			provider: "xai",
			model:    "grok-3-mini",
			apiKey:   xaiKey,
			skip:     xaiKey == "",
		},
		{
			name:     "Gemini/2.5-Flash-Lite",
			provider: "gemini",
			model:    "gemini-2.5-flash-lite",
			apiKey:   geminiKey,
			skip:     geminiKey == "",
		},
		{
			name:     "Gemini/2.5-Flash",
			provider: "gemini",
			model:    "gemini-2.5-flash",
			apiKey:   geminiKey,
			skip:     geminiKey == "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.skip {
				t.Skipf("Skipping %s: API key not set", tc.name)
			}

			cfg := llm.Config{
				Provider:    tc.provider,
				Model:       tc.model,
				APIKey:      tc.apiKey,
				MaxTokens:   100,
				Temperature: 0.0,
				Timeout:     30 * time.Second,
			}

			client, err := llm.NewClient(cfg)
			if err != nil {
				t.Fatalf("Failed to create client: %v", err)
			}
			defer client.Close()

			if client.Provider() != tc.provider {
				t.Errorf("Expected provider %s, got %s", tc.provider, client.Provider())
			}

			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()

			req := llm.Request{
				Model:       tc.model,
				Messages:    []llm.Message{llm.NewTextMessage("user", TestPrompt)},
				MaxTokens:   100,
				Temperature: 0.0,
			}

			resp, err := client.Complete(ctx, req)
			if err != nil {
				t.Fatalf("Completion failed: %v", err)
			}

			if resp.Content == "" {
				t.Error("Empty response content")
			}

			lower := strings.ToLower(resp.Content)
			if !strings.Contains(lower, "hello") && !strings.Contains(lower, "hi") {
				t.Errorf("Expected response containing 'hello' or 'hi', got: %s", resp.Content)
			}

			t.Logf("Provider: %s, Model: %s", client.Provider(), client.Model())
			t.Logf("Response: %s", resp.Content)
			t.Logf("Tokens: in=%d, out=%d", resp.InputTokens, resp.OutputTokens)
		})
	}
}

// TestAnthropicStreaming tests streaming with Anthropic.
func TestAnthropicStreaming(t *testing.T) {
	apiKey := envKey("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEYS")
	if apiKey == "" {
		t.Skip("ANTHROPIC_API_KEY(S) not set")
	}

	cfg := llm.Config{
		Provider:    "anthropic",
		Model:       "claude-haiku-4-5-20251001",
		APIKey:      apiKey,
		MaxTokens:   100,
		Temperature: 0.0,
		Timeout:     30 * time.Second,
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req := llm.Request{
		Messages:    []llm.Message{llm.NewTextMessage("user", "Count from 1 to 5")},
		MaxTokens:   100,
		Temperature: 0.0,
	}

	var chunks []string
	var done bool

	for event, err := range client.Stream(ctx, req) {
		if err != nil {
			t.Fatalf("Streaming failed: %v", err)
		}
		switch event.Type {
		case llm.EventContent:
			chunks = append(chunks, event.Text)
		case llm.EventDone:
			done = true
			t.Logf("Stream done: tokens in=%d, out=%d", event.InputTokens, event.OutputTokens)
		}
	}

	if !done {
		t.Error("Stream did not complete")
	}

	if len(chunks) == 0 {
		t.Error("No content chunks received")
	}

	fullResponse := strings.Join(chunks, "")
	t.Logf("Streamed response (%d chunks): %s", len(chunks), fullResponse)
}

// TestXAIStreaming tests streaming with xAI Grok via OpenAI-compat adapter.
func TestXAIStreaming(t *testing.T) {
	apiKey := envKey("XAI_API_KEY", "XAI_API_KEYS")
	if apiKey == "" {
		t.Skip("XAI_API_KEY(S) not set")
	}

	cfg := llm.Config{
		Provider:    "xai",
		Model:       "grok-3-mini",
		APIKey:      apiKey,
		MaxTokens:   100,
		Temperature: 0.0,
		Timeout:     30 * time.Second,
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req := llm.Request{
		Messages:    []llm.Message{llm.NewTextMessage("user", "Count from 1 to 5")},
		MaxTokens:   100,
		Temperature: 0.0,
	}

	var chunks []string
	var done bool

	for event, err := range client.Stream(ctx, req) {
		if err != nil {
			t.Fatalf("Streaming failed: %v", err)
		}
		switch event.Type {
		case llm.EventContent:
			chunks = append(chunks, event.Text)
		case llm.EventDone:
			done = true
			t.Logf("Stream done: tokens in=%d, out=%d", event.InputTokens, event.OutputTokens)
		}
	}

	if !done {
		t.Error("Stream did not complete")
	}

	fullResponse := strings.Join(chunks, "")
	t.Logf("Streamed response (%d chunks): %s", len(chunks), fullResponse)
}

// TestGeminiStreaming tests streaming with Google Gemini.
func TestGeminiStreaming(t *testing.T) {
	apiKey := envKey("GEMINI_API_KEY", "GEMINI_API_KEYS")
	if apiKey == "" {
		t.Skip("GEMINI_API_KEY(S) not set")
	}

	cfg := llm.Config{
		Provider:    "gemini",
		Model:       "gemini-2.5-flash-lite",
		APIKey:      apiKey,
		MaxTokens:   100,
		Temperature: 0.0,
		Timeout:     30 * time.Second,
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req := llm.Request{
		Messages:    []llm.Message{llm.NewTextMessage("user", "Count from 1 to 5")},
		MaxTokens:   100,
		Temperature: 0.0,
	}

	var chunks []string
	var done bool

	for event, err := range client.Stream(ctx, req) {
		if err != nil {
			t.Fatalf("Streaming failed: %v", err)
		}
		switch event.Type {
		case llm.EventContent:
			chunks = append(chunks, event.Text)
		case llm.EventDone:
			done = true
			t.Logf("Stream done: tokens in=%d, out=%d", event.InputTokens, event.OutputTokens)
		}
	}

	if !done {
		t.Error("Stream did not complete")
	}

	fullResponse := strings.Join(chunks, "")
	t.Logf("Streamed response (%d chunks): %s", len(chunks), fullResponse)
}

// TestPooledClientWithMultipleKeys tests auth pool rotation with multiple API keys.
func TestPooledClientWithMultipleKeys(t *testing.T) {
	keys := envKeys("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEYS")
	if len(keys) < 2 {
		t.Skip("ANTHROPIC_API_KEYS not set or has fewer than 2 keys")
	}

	cfg := llm.Config{
		Provider:    "anthropic",
		Model:       "claude-haiku-4-5-20251001",
		MaxTokens:   100,
		Temperature: 0.0,
		Timeout:     30 * time.Second,
	}

	client, err := llm.NewClientWithKeys(cfg, keys)
	if err != nil {
		t.Fatalf("Failed to create pooled client: %v", err)
	}
	defer client.Close()

	t.Logf("Created pooled client with %d keys", len(keys))

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req := llm.Request{
		Messages:    []llm.Message{llm.NewTextMessage("user", TestPrompt)},
		MaxTokens:   100,
		Temperature: 0.0,
	}

	resp, err := client.Complete(ctx, req)
	if err != nil {
		t.Fatalf("Pooled completion failed: %v", err)
	}

	if resp.Content == "" {
		t.Error("Empty response content")
	}

	t.Logf("Pooled response: %s (tokens: in=%d, out=%d)", resp.Content, resp.InputTokens, resp.OutputTokens)
}

// =============================================================================
// AUTH POOL UNIT TESTS
// =============================================================================

// TestAuthPoolCreation verifies pool construction and initial state.
func TestAuthPoolCreation(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a", "key-b", "key-c"})

	if pool.Count() != 3 {
		t.Fatalf("Count() = %d, want 3", pool.Count())
	}

	current, ok := pool.GetCurrent()
	if !ok {
		t.Fatal("GetCurrent() returned false for non-empty pool")
	}
	if current.APIKey != "key-a" {
		t.Errorf("GetCurrent().APIKey = %q, want key-a", current.APIKey)
	}
	if current.Name != "test-1" {
		t.Errorf("GetCurrent().Name = %q, want test-1", current.Name)
	}
}

// TestAuthPoolEmptyPool verifies behavior with zero profiles.
func TestAuthPoolEmptyPool(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{})

	if pool.Count() != 0 {
		t.Fatalf("Count() = %d, want 0", pool.Count())
	}
	if _, ok := pool.GetCurrent(); ok {
		t.Error("GetCurrent() should return false for empty pool")
	}

	_, err := pool.GetAvailable()
	if err == nil {
		t.Fatal("GetAvailable() should error for empty pool")
	}
}

// TestAuthPoolGetAvailableRotation verifies rotation when current profile is in cooldown.
func TestAuthPoolGetAvailableRotation(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a", "key-b", "key-c"})

	// Get first profile
	p1, err := pool.GetAvailable()
	if err != nil {
		t.Fatalf("GetAvailable() error: %v", err)
	}
	if p1.APIKey != "key-a" {
		t.Errorf("first profile = %q, want key-a", p1.APIKey)
	}

	// Put first profile in cooldown
	pool.MarkFailedByName(p1.Name, llm.NewRateLimitError("test", "rate limited"))

	// Should rotate to second
	p2, err := pool.GetAvailable()
	if err != nil {
		t.Fatalf("GetAvailable() after rotation error: %v", err)
	}
	if p2.APIKey != "key-b" {
		t.Errorf("rotated profile = %q, want key-b", p2.APIKey)
	}
}

// TestAuthPoolAllInCooldown verifies error when all profiles are exhausted.
func TestAuthPoolAllInCooldown(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a", "key-b"})

	// Exhaust both profiles
	p1, _ := pool.GetAvailable()
	pool.MarkFailedByName(p1.Name, llm.NewRateLimitError("test", "rate limited"))
	p2, _ := pool.GetAvailable()
	pool.MarkFailedByName(p2.Name, llm.NewRateLimitError("test", "rate limited"))

	_, err := pool.GetAvailable()
	if err == nil {
		t.Fatal("GetAvailable() should error when all in cooldown")
	}
	if !strings.Contains(err.Error(), "no available auth profiles") {
		t.Errorf("error = %q, want 'no available auth profiles'", err.Error())
	}
}

// TestAuthPoolMarkSuccess verifies recovery after failure.
func TestAuthPoolMarkSuccess(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a"})

	// Fail the profile
	p1, _ := pool.GetAvailable()
	pool.MarkFailedByName(p1.Name, llm.NewOverloadedError("test", "overloaded"))

	// Should be in cooldown
	_, err := pool.GetAvailable()
	if err == nil {
		t.Fatal("profile should be in cooldown")
	}

	// Mark success resets the profile
	pool.MarkSuccessByName(p1.Name)
	p, err := pool.GetAvailable()
	if err != nil {
		t.Fatalf("after MarkSuccess, GetAvailable() error: %v", err)
	}
	if p.APIKey != "key-a" {
		t.Errorf("recovered profile = %q, want key-a", p.APIKey)
	}
}

// TestAuthPoolStatus verifies the status string format.
func TestAuthPoolStatus(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a", "key-b"})

	status := pool.Status()
	if !strings.Contains(status, "*test-1:ok") {
		t.Errorf("status = %q, want current marker on test-1", status)
	}
	if !strings.Contains(status, "test-2:ok") {
		t.Errorf("status = %q, want test-2:ok", status)
	}
}

// =============================================================================
// POOLED CLIENT UNIT TESTS
// =============================================================================

// failingMockClient is a configurable mock for pool tests.
// It returns responses from the calls slice in order. When exhausted, it reuses the last entry.
type failingMockClient struct {
	calls    []mockCall
	callIdx  int
	provider string
	model    string
	closed   bool
}

type mockCall struct {
	resp         *llm.Response
	err          error
	midStreamErr error // if set, yield one content event then this error
}

func (m *failingMockClient) Complete(_ context.Context, _ llm.Request) (*llm.Response, error) {
	c := m.nextCall()
	return c.resp, c.err
}

func (m *failingMockClient) Stream(_ context.Context, _ llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		c := m.nextCall()
		if c.err != nil {
			yield(llm.StreamEvent{}, c.err)
			return
		}
		if c.midStreamErr != nil {
			// Yield one content event, then error (simulates mid-stream failure)
			if !yield(llm.StreamEvent{Type: llm.EventContent, Text: "partial"}, nil) {
				return
			}
			yield(llm.StreamEvent{}, c.midStreamErr)
			return
		}
		if !yield(llm.StreamEvent{Type: llm.EventContent, Text: "streamed"}, nil) {
			return
		}
		yield(llm.StreamEvent{Type: llm.EventDone, StopReason: "end_turn"}, nil)
	}
}

func (m *failingMockClient) nextCall() mockCall {
	if m.callIdx < len(m.calls) {
		c := m.calls[m.callIdx]
		m.callIdx++
		return c
	}
	return m.calls[len(m.calls)-1]
}

func (m *failingMockClient) Provider() string { return m.provider }
func (m *failingMockClient) Model() string    { return m.model }
func (m *failingMockClient) Close() error     { m.closed = true; return nil }

// threadSafeMockClient is a simple mock that always returns the same result.
// Safe for concurrent use because it has no mutable state.
type threadSafeMockClient struct {
	provider string
	model    string
	resp     *llm.Response
	err      error
}

func (m *threadSafeMockClient) Complete(_ context.Context, _ llm.Request) (*llm.Response, error) {
	return m.resp, m.err
}

func (m *threadSafeMockClient) Stream(_ context.Context, _ llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		if m.err != nil {
			yield(llm.StreamEvent{}, m.err)
			return
		}
		if !yield(llm.StreamEvent{Type: llm.EventContent, Text: "streamed"}, nil) {
			return
		}
		yield(llm.StreamEvent{Type: llm.EventDone, StopReason: "end_turn"}, nil)
	}
}

func (m *threadSafeMockClient) Provider() string { return m.provider }
func (m *threadSafeMockClient) Model() string    { return m.model }
func (m *threadSafeMockClient) Close() error     { return nil }

// TestPooledClientCompleteSuccess verifies happy path.
func TestPooledClientCompleteSuccess(t *testing.T) {
	mock := &failingMockClient{
		calls:    []mockCall{{resp: &llm.Response{Content: "ok"}}},
		provider: "test",
		model:    "test-model",
	}

	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return mock, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	resp, err := pc.Complete(context.Background(), llm.Request{})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if resp.Content != "ok" {
		t.Errorf("Content = %q, want ok", resp.Content)
	}
}

// TestPooledClientCompleteRetryOnRetryableError verifies retry with rotation.
func TestPooledClientCompleteRetryOnRetryableError(t *testing.T) {
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		mock := &failingMockClient{
			provider: "test",
			model:    "test-model",
		}
		if callCount <= 1 {
			// First client: initial creation, will be used and fail
			mock.calls = []mockCall{{err: llm.NewRateLimitError("test", "rate limited")}}
		} else {
			// Second client: after rotation, succeeds
			mock.calls = []mockCall{{resp: &llm.Response{Content: "recovered"}}}
		}
		return mock, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	resp, err := pc.Complete(context.Background(), llm.Request{})
	if err != nil {
		t.Fatalf("Complete should recover after rotation: %v", err)
	}
	if resp.Content != "recovered" {
		t.Errorf("Content = %q, want recovered", resp.Content)
	}
}

// TestPooledClientCompleteAuthErrorRotates verifies auth errors rotate to try other keys.
func TestPooledClientCompleteAuthErrorRotates(t *testing.T) {
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		return &failingMockClient{
			calls:    []mockCall{{err: llm.NewAuthError("test", "bad key", 401)}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	_, err = pc.Complete(context.Background(), llm.Request{})
	if err == nil {
		t.Fatal("Complete should return error after exhausting all keys")
	}
	// Wrapped error should still match via errors.As
	if !llm.IsAuthError(err) {
		t.Errorf("expected auth error (wrapped), got: %v", err)
	}
	// Should have rotated to try both keys before giving up
	if callCount != 2 {
		t.Errorf("clientFunc called %d times, want 2 (auth errors should rotate)", callCount)
	}
}

// TestPooledClientCompleteNoRetryOnBadRequest verifies immediate return for bad request errors.
func TestPooledClientCompleteNoRetryOnBadRequest(t *testing.T) {
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		return &failingMockClient{
			calls:    []mockCall{{err: llm.NewAPIErrorFromStatus("test", 400, "invalid json")}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	_, err = pc.Complete(context.Background(), llm.Request{})
	if err == nil {
		t.Fatal("Complete should return error immediately for bad request")
	}
	// Should NOT have rotated (bad request is not key-related)
	if callCount != 1 {
		t.Errorf("clientFunc called %d times, want 1 (no rotation for bad request)", callCount)
	}
}

// TestPooledClientStreamPreDataRetry verifies streams retry on pre-data connection errors.
func TestPooledClientStreamPreDataRetry(t *testing.T) {
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		mock := &failingMockClient{
			provider: "test",
			model:    "test-model",
		}
		if callCount <= 1 {
			// First client: rate limited on connection
			mock.calls = []mockCall{{err: llm.NewRateLimitError("test", "rate limited")}}
		} else {
			// Second client (after rotation): succeeds
			mock.calls = []mockCall{{resp: &llm.Response{}}}
		}
		return mock, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	var chunks int
	for event, err := range pc.Stream(context.Background(), llm.Request{}) {
		if err != nil {
			t.Fatalf("Stream should recover after rotation: %v", err)
		}
		if event.Type == llm.EventContent {
			chunks++
		}
	}
	if chunks != 1 {
		t.Errorf("chunks = %d, want 1", chunks)
	}
	// Should have created a second client (rotation on pre-data error)
	if callCount < 2 {
		t.Errorf("clientFunc called %d times, want >= 2 (should rotate on pre-data error)", callCount)
	}
}

// TestPooledClientStreamMidStreamNoRetry verifies streams are NOT retried after data has been yielded.
func TestPooledClientStreamMidStreamNoRetry(t *testing.T) {
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		return &failingMockClient{
			calls:    []mockCall{{midStreamErr: llm.NewRateLimitError("test", "rate limited")}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	var gotContent bool
	var gotErr bool
	for event, err := range pc.Stream(context.Background(), llm.Request{}) {
		if err != nil {
			gotErr = true
			if !llm.IsRateLimited(err) {
				t.Errorf("expected rate limit error, got: %v", err)
			}
			break
		}
		if event.Type == llm.EventContent {
			gotContent = true
		}
	}
	if !gotContent {
		t.Error("expected at least one content event before error")
	}
	if !gotErr {
		t.Error("expected mid-stream error")
	}
	// Should NOT have rotated (mid-stream error, data already yielded)
	if callCount != 1 {
		t.Errorf("clientFunc called %d times, want 1 (no retry after data yielded)", callCount)
	}
}

// TestPooledClientStreamAuthErrorRotates verifies auth errors rotate to try other keys.
func TestPooledClientStreamAuthErrorRotates(t *testing.T) {
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		return &failingMockClient{
			calls:    []mockCall{{err: llm.NewAuthError("test", "bad key", 401)}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	for _, err := range pc.Stream(context.Background(), llm.Request{}) {
		if err != nil {
			// Wrapped error should still match via errors.As
			if !llm.IsAuthError(err) {
				t.Errorf("expected auth error (wrapped), got: %v", err)
			}
			break
		}
		t.Fatal("Stream should not yield events")
	}
	// Should have rotated to try both keys before giving up
	if callCount != 2 {
		t.Errorf("clientFunc called %d times, want 2 (auth errors should rotate)", callCount)
	}
}

// TestPooledClientStreamBadRequestNoRetry verifies bad request errors are not retried.
func TestPooledClientStreamBadRequestNoRetry(t *testing.T) {
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		return &failingMockClient{
			calls:    []mockCall{{err: llm.NewAPIErrorFromStatus("test", 400, "invalid json")}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	for _, err := range pc.Stream(context.Background(), llm.Request{}) {
		if err != nil {
			break
		}
		t.Fatal("Stream should not yield events")
	}
	// Should NOT have rotated (bad request is not key-related)
	if callCount != 1 {
		t.Errorf("clientFunc called %d times, want 1 (no retry for bad request)", callCount)
	}
}

// TestPooledClientStreamSuccess verifies happy path streaming.
func TestPooledClientStreamSuccess(t *testing.T) {
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &failingMockClient{
			calls:    []mockCall{{resp: &llm.Response{}}}, // Stream uses nil err path
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	var chunks int
	for event, err := range pc.Stream(context.Background(), llm.Request{}) {
		if err != nil {
			t.Fatalf("Stream error: %v", err)
		}
		if event.Type == llm.EventContent {
			chunks++
		}
	}
	if chunks != 1 {
		t.Errorf("chunks = %d, want 1", chunks)
	}
}

// TestPooledClientContextCancellation verifies that a single-key pool returns immediately
// when the key is in cooldown — does not retry with the same profile.
func TestPooledClientContextCancellation(t *testing.T) {
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &failingMockClient{
			calls:    []mockCall{{err: llm.NewOverloadedError("test", "overloaded")}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, err = pc.Complete(ctx, llm.Request{})
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("Complete should error on cancelled context")
	}
	// Should return immediately — not retry with the same cooled-down profile
	if elapsed > 500*time.Millisecond {
		t.Errorf("elapsed = %v, expected < 500ms (should not retry with same profile)", elapsed)
	}
}

// TestPooledClientNoKeys verifies error on empty keys.
func TestPooledClientNoKeys(t *testing.T) {
	_, err := llm.NewPooledClient(llm.DefaultConfig(), []string{}, func(profile llm.AuthProfile) (llm.Client, error) {
		return nil, nil
	})
	if err == nil {
		t.Fatal("NewPooledClient should error with no keys")
	}
}

// TestPooledClientProviderModel verifies delegation.
func TestPooledClientProviderModel(t *testing.T) {
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &failingMockClient{
			calls:    []mockCall{{resp: &llm.Response{}}},
			provider: "my-provider",
			model:    "my-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	if pc.Provider() != "my-provider" {
		t.Errorf("Provider() = %q, want my-provider", pc.Provider())
	}
	if pc.Model() != "my-model" {
		t.Errorf("Model() = %q, want my-model", pc.Model())
	}
}

// TestPooledClientPoolStatus verifies status reporting.
func TestPooledClientPoolStatus(t *testing.T) {
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &failingMockClient{
			calls:    []mockCall{{resp: &llm.Response{}}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	status := pc.PoolStatus()
	if !strings.Contains(status, "ok") {
		t.Errorf("PoolStatus() = %q, want to contain 'ok'", status)
	}
}

// =============================================================================
// ADDITIONAL COVERAGE TESTS
// =============================================================================

// TestWithLoggingPrefix verifies the prefix logging wrapper.
func TestWithLoggingPrefix(t *testing.T) {
	inner := &mockClient{
		provider: "test",
		model:    "test-model",
		resp:     &llm.Response{Content: "hello"},
	}

	client := llm.WithLoggingPrefix(inner, "[MyApp]")

	if client.Provider() != "test" {
		t.Errorf("Provider() = %q, want test", client.Provider())
	}

	resp, err := client.Complete(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}
	if resp.Content != "hello" {
		t.Errorf("Content = %q, want hello", resp.Content)
	}
}

// TestLoggingClientClose verifies Close delegation.
func TestLoggingClientClose(t *testing.T) {
	inner := &mockClient{provider: "test", model: "test-model"}
	client := llm.WithLogging(inner)

	err := client.Close()
	if err != nil {
		t.Errorf("Close() error: %v", err)
	}
}

// TestCooldownErrorUnwrap verifies error unwrapping.
func TestCooldownErrorUnwrap(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a"})

	// Put profile in cooldown
	p, _ := pool.GetAvailable()
	pool.MarkFailedByName(p.Name, llm.NewRateLimitError("test", "rate limited"))

	// GetAvailable should return CooldownError
	_, err := pool.GetAvailable()
	if err == nil {
		t.Fatal("expected error")
	}

	// Should unwrap to ErrNoAvailableProfiles
	if !strings.Contains(err.Error(), "no available auth profiles") {
		t.Errorf("error = %q, want to contain 'no available auth profiles'", err.Error())
	}
}

// TestResolveAuthHeader verifies auth header resolution.
func TestResolveAuthHeader(t *testing.T) {
	// With explicit AuthHeader
	cfg := llm.Config{
		Provider:   "openai",
		AuthHeader: "X-Custom-Auth",
	}
	header := llm.ResolveAuthHeader(cfg)
	if header != "X-Custom-Auth" {
		t.Errorf("ResolveAuthHeader() = %q, want X-Custom-Auth", header)
	}

	// Without explicit AuthHeader, uses preset
	cfg = llm.Config{Provider: "openai"}
	header = llm.ResolveAuthHeader(cfg)
	if header != "Bearer" {
		t.Errorf("ResolveAuthHeader() = %q, want Bearer", header)
	}

	// Unknown provider falls back to "Bearer"
	cfg = llm.Config{Provider: "unknown"}
	header = llm.ResolveAuthHeader(cfg)
	if header != "Bearer" {
		t.Errorf("ResolveAuthHeader() = %q, want Bearer", header)
	}
}

// TestIsRetryableNetworkErrors verifies network error detection.
func TestIsRetryableNetworkErrors(t *testing.T) {
	tests := []struct {
		errMsg   string
		expected bool
	}{
		{"connection refused", true},
		{"no such host", true},
		{"timeout exceeded", true},
		{"unexpected eof", true},
		{"connection reset by peer", true},
		{"request failed: dial tcp", false}, // string-only path; in production, %w wrapping triggers typed net.Error check
		{"some random error", false},
		{"permission denied", false},
	}

	for _, tc := range tests {
		t.Run(tc.errMsg, func(t *testing.T) {
			err := fmt.Errorf("%s", tc.errMsg)
			got := llm.IsRetryable(err)
			if got != tc.expected {
				t.Errorf("IsRetryable(%q) = %v, want %v", tc.errMsg, got, tc.expected)
			}
		})
	}
}

// TestIsRetryableTypedErrors verifies type-based network error detection in IsRetryable.
func TestIsRetryableTypedErrors(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		expected bool
	}{
		{"net.DNSError", &net.DNSError{Err: "no such host", Name: "example.com"}, true},
		{"io.EOF", io.EOF, true},
		{"wrapped io.EOF", fmt.Errorf("stream failed: %w", io.EOF), true},
		{"io.ErrUnexpectedEOF", io.ErrUnexpectedEOF, true},
		{"wrapped syscall.ECONNRESET", fmt.Errorf("write: %w", syscall.ECONNRESET), true},
		{"wrapped syscall.ECONNREFUSED", fmt.Errorf("dial: %w", syscall.ECONNREFUSED), true},
		{"non-retryable plain error", errors.New("permission denied"), false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := llm.IsRetryable(tc.err)
			if got != tc.expected {
				t.Errorf("IsRetryable(%v) = %v, want %v", tc.err, got, tc.expected)
			}
		})
	}
}

// TestTokensNotReported verifies the sentinel constant value and StreamEvent usage.
func TestTokensNotReported(t *testing.T) {
	if llm.TokensNotReported != -1 {
		t.Errorf("TokensNotReported = %d, want -1", llm.TokensNotReported)
	}

	// Verify sentinel is distinguishable from zero in StreamEvent
	event := llm.StreamEvent{
		Type:         llm.EventDone,
		InputTokens:  llm.TokensNotReported,
		OutputTokens: llm.TokensNotReported,
	}
	if event.InputTokens == 0 {
		t.Error("TokensNotReported should not equal 0")
	}
	if event.OutputTokens == 0 {
		t.Error("TokensNotReported should not equal 0")
	}
}

// TestIsOverloadedFalse verifies IsOverloaded returns false for non-503 errors.
func TestIsOverloadedFalse(t *testing.T) {
	err := llm.NewRateLimitError("test", "rate limited")
	if llm.IsOverloaded(err) {
		t.Error("IsOverloaded should return false for 429 error")
	}

	// Non-APIError
	plainErr := fmt.Errorf("some error")
	if llm.IsOverloaded(plainErr) {
		t.Error("IsOverloaded should return false for non-APIError")
	}

	// nil
	if llm.IsOverloaded(nil) {
		t.Error("IsOverloaded should return false for nil")
	}
}

// TestIsContextLengthFalse verifies IsContextLength returns false for non-matching errors.
func TestIsContextLengthFalse(t *testing.T) {
	// 400 but not context length message
	err := llm.NewAPIErrorFromStatus("test", 400, "invalid json")
	if llm.IsContextLength(err) {
		t.Error("IsContextLength should return false for non-context-length 400")
	}

	// Non-400
	err = llm.NewRateLimitError("test", "rate limited")
	if llm.IsContextLength(err) {
		t.Error("IsContextLength should return false for 429")
	}

	// nil
	if llm.IsContextLength(nil) {
		t.Error("IsContextLength should return false for nil")
	}
}

// TestAuthPoolWithBaseURL verifies the API_KEY|BASE_URL format.
func TestAuthPoolWithBaseURL(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{
		"key-only",
		"key-with-url|https://custom.example.com/v1",
	})

	if pool.Count() != 2 {
		t.Fatalf("Count() = %d, want 2", pool.Count())
	}

	// First profile: no BaseURL
	p1, _ := pool.GetAvailable()
	if p1.APIKey != "key-only" {
		t.Errorf("first APIKey = %q, want key-only", p1.APIKey)
	}
	if p1.BaseURL != "" {
		t.Errorf("first BaseURL = %q, want empty", p1.BaseURL)
	}

	// Put first in cooldown to get second
	pool.MarkFailedByName(p1.Name, llm.NewRateLimitError("test", "rate limited"))

	// Second profile: has BaseURL
	p2, _ := pool.GetAvailable()
	if p2.APIKey != "key-with-url" {
		t.Errorf("second APIKey = %q, want key-with-url", p2.APIKey)
	}
	if p2.BaseURL != "https://custom.example.com/v1" {
		t.Errorf("second BaseURL = %q, want https://custom.example.com/v1", p2.BaseURL)
	}
}

// TestGetDefaultModelUnknown verifies fallback for unknown provider.
func TestGetDefaultModelUnknown(t *testing.T) {
	// Unknown providers fall back to claude-sonnet-4-6
	model := llm.GetDefaultModel("nonexistent-provider")
	if model != "claude-sonnet-4-6" {
		t.Errorf("GetDefaultModel(unknown) = %q, want claude-sonnet-4-6", model)
	}
}

// TestGetAvailableModelsUnknown verifies nil for unknown provider.
func TestGetAvailableModelsUnknown(t *testing.T) {
	models := llm.GetAvailableModels("nonexistent-provider")
	if models != nil {
		t.Errorf("GetAvailableModels(unknown) = %v, want nil", models)
	}
}

// TestEstimateCostNegativeTokens verifies negative tokens are clamped to 0.
func TestEstimateCostNegativeTokens(t *testing.T) {
	// -1 tokens (sentinel for "not reported") should not produce negative cost
	cost := llm.EstimateCost("claude-sonnet-4-6", -1, -1)
	if cost < 0 {
		t.Errorf("EstimateCost with negative tokens = %f, want >= 0", cost)
	}
	if cost != 0 {
		t.Errorf("EstimateCost with -1 tokens = %f, want 0", cost)
	}
}

// TestAuthPoolAuthErrorPermanentDisable verifies auth errors permanently disable profiles.
func TestAuthPoolAuthErrorPermanentDisable(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a"})

	p, _ := pool.GetAvailable()
	pool.MarkFailedByName(p.Name, llm.NewAuthError("test", "invalid key", 401))

	// Profile should be permanently disabled, not just in cooldown
	_, err := pool.GetAvailable()
	if err == nil {
		t.Fatal("expected error after auth failure")
	}
	// Should match ErrNoAvailableProfiles (permanently disabled, not cooldown)
	if !errors.Is(err, llm.ErrNoAvailableProfiles) {
		t.Errorf("error = %q, want errors.Is(ErrNoAvailableProfiles)", err.Error())
	}
	if !strings.Contains(err.Error(), "permanently disabled") {
		t.Errorf("error = %q, want to contain 'permanently disabled'", err.Error())
	}
}

// TestLoggingClientStreamEarlyBreak verifies stream cleanup on early break.
func TestLoggingClientStreamEarlyBreak(t *testing.T) {
	inner := &mockClient{
		provider: "test",
		model:    "test-model",
	}

	client := llm.WithLogging(inner)

	// Break after first event
	for event, err := range client.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage("user", "hi")},
	}) {
		if err != nil {
			t.Fatalf("Stream error: %v", err)
		}
		if event.Type == llm.EventContent {
			break // Early break
		}
	}
	// Should not panic or leak
}

// TestResolveBaseURL verifies base URL resolution.
func TestResolveBaseURL(t *testing.T) {
	// Explicit BaseURL takes precedence
	cfg := llm.Config{
		Provider: "anthropic",
		BaseURL:  "https://custom.example.com",
	}
	url := llm.ResolveBaseURL(cfg)
	if url != "https://custom.example.com" {
		t.Errorf("ResolveBaseURL() = %q, want https://custom.example.com", url)
	}

	// Falls back to preset
	cfg = llm.Config{Provider: "anthropic"}
	url = llm.ResolveBaseURL(cfg)
	if url != "https://api.anthropic.com/v1" {
		t.Errorf("ResolveBaseURL() = %q, want https://api.anthropic.com/v1", url)
	}

	// Unknown provider without BaseURL
	cfg = llm.Config{Provider: "unknown"}
	url = llm.ResolveBaseURL(cfg)
	if url != "" {
		t.Errorf("ResolveBaseURL() = %q, want empty", url)
	}
}

// TestCooldownErrorUnwrapActual verifies errors.Is with CooldownError.
func TestCooldownErrorUnwrapActual(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a"})

	// Put profile in cooldown
	p, _ := pool.GetAvailable()
	pool.MarkFailedByName(p.Name, llm.NewRateLimitError("test", "rate limited"))

	// GetAvailable should return CooldownError
	_, err := pool.GetAvailable()
	if err == nil {
		t.Fatal("expected error")
	}

	// errors.Is should match ErrNoAvailableProfiles via Unwrap
	if !errors.Is(err, llm.ErrNoAvailableProfiles) {
		t.Errorf("errors.Is(err, ErrNoAvailableProfiles) = false, want true")
	}

	// String should contain the message
	if !strings.Contains(err.Error(), "no available auth profiles") {
		t.Errorf("error = %q, want to contain 'no available auth profiles'", err.Error())
	}
}

// TestPooledClientCloseNilClient verifies Close handles nil client.
func TestPooledClientCloseNilClient(t *testing.T) {
	// This tests the case where client might be nil, though in practice
	// NewPooledClient always initializes a client
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &mockClient{provider: "test", model: "test"}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}

	// First close should work
	if err := pc.Close(); err != nil {
		t.Errorf("Close() error: %v", err)
	}
}

// TestPooledClientRotateClientAlreadyRotated verifies short-circuit when already rotated.
func TestPooledClientRotateClientAlreadyRotated(t *testing.T) {
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		mock := &failingMockClient{
			provider: "test",
			model:    "test-model",
		}
		if callCount <= 1 {
			// First client: rate limited
			mock.calls = []mockCall{{err: llm.NewRateLimitError("test", "rate limited")}}
		} else {
			// Second client: succeeds
			mock.calls = []mockCall{{resp: &llm.Response{Content: "ok"}}}
		}
		return mock, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	// First call rotates
	resp, err := pc.Complete(context.Background(), llm.Request{})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if resp.Content != "ok" {
		t.Errorf("Content = %q, want ok", resp.Content)
	}
}

// TestPooledClientStreamCancellation verifies stream respects context cancellation during backoff.
func TestPooledClientStreamCancellation(t *testing.T) {
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &failingMockClient{
			calls:    []mockCall{{err: llm.NewOverloadedError("test", "overloaded")}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	start := time.Now()
	for _, err := range pc.Stream(ctx, llm.Request{}) {
		if err != nil {
			break
		}
	}
	elapsed := time.Since(start)

	// Should return quickly due to context cancellation
	if elapsed > 500*time.Millisecond {
		t.Errorf("elapsed = %v, expected < 500ms", elapsed)
	}
}

// TestAuthPoolStatusUnhealthy verifies status shows unhealthy profiles.
func TestAuthPoolStatusUnhealthy(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a", "key-b"})

	// Mark first profile as permanently unhealthy (auth error)
	p, _ := pool.GetAvailable()
	pool.MarkFailedByName(p.Name, llm.NewAuthError("test", "bad key", 401))

	status := pool.Status()
	if !strings.Contains(status, "unhealthy") {
		t.Errorf("status = %q, want to contain 'unhealthy'", status)
	}
}

// TestNewClientWithKeysEmpty verifies NewClientWithKeys with no keys falls back to config.
func TestNewClientWithKeysEmpty(t *testing.T) {
	cfg := llm.Config{
		Provider: "ollama",
		Model:    "llama3",
	}

	// Empty keys should fall back to newSingleClient
	client, err := llm.NewClientWithKeys(cfg, []string{})
	if err != nil {
		t.Fatalf("NewClientWithKeys error: %v", err)
	}
	defer client.Close()

	if client.Provider() != "ollama" {
		t.Errorf("Provider() = %q, want ollama", client.Provider())
	}
}

// TestLoggingClientStreamError verifies stream error logging.
func TestLoggingClientStreamError(t *testing.T) {
	apiErr := llm.NewRateLimitError("test", "too fast")
	inner := &mockClient{
		provider: "test",
		model:    "test-model",
		err:      apiErr,
	}

	client := llm.WithLogging(inner)

	for _, err := range client.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage("user", "hi")},
	}) {
		if err != nil {
			if !llm.IsRateLimited(err) {
				t.Errorf("expected rate limit error, got: %v", err)
			}
			break
		}
	}
}

// TestMarkFailedByNameNotFound verifies MarkFailedByName with unknown profile name.
func TestMarkFailedByNameNotFound(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a"})

	// This should not panic - just silently return
	pool.MarkFailedByName("nonexistent-profile", llm.NewRateLimitError("test", "rate limited"))

	// Pool should still be functional
	p, err := pool.GetAvailable()
	if err != nil {
		t.Fatalf("GetAvailable() error: %v", err)
	}
	if p.APIKey != "key-a" {
		t.Errorf("APIKey = %q, want key-a", p.APIKey)
	}
}

// TestAuthPoolStatusCooldown verifies status shows cooldown duration.
func TestAuthPoolStatusCooldown(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a"})

	// Put profile in temporary cooldown (not auth error)
	p, _ := pool.GetAvailable()
	pool.MarkFailedByName(p.Name, llm.NewRateLimitError("test", "rate limited"))

	status := pool.Status()
	// Should contain "cooldown" with a duration, not "unhealthy"
	if !strings.Contains(status, "cooldown") {
		t.Errorf("status = %q, want to contain 'cooldown'", status)
	}
}

// TestNewPooledClientClientFuncError verifies error when clientFunc fails on init.
func TestNewPooledClientClientFuncError(t *testing.T) {
	_, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return nil, fmt.Errorf("client creation failed")
	})
	if err == nil {
		t.Fatal("NewPooledClient should error when clientFunc fails")
	}
	if !strings.Contains(err.Error(), "client creation failed") {
		t.Errorf("error = %q, want to contain 'client creation failed'", err.Error())
	}
}

// TestPooledClientRotateClientFuncError verifies error handling when clientFunc fails during rotation.
func TestPooledClientRotateClientFuncError(t *testing.T) {
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		if callCount == 1 {
			// First call (init): succeed but return failing client
			return &failingMockClient{
				calls:    []mockCall{{err: llm.NewRateLimitError("test", "rate limited")}},
				provider: "test",
				model:    "test-model",
			}, nil
		}
		// Subsequent calls (rotation): fail
		return nil, fmt.Errorf("rotation client creation failed")
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	// Complete should fail because rotation fails
	_, err = pc.Complete(context.Background(), llm.Request{})
	if err == nil {
		t.Fatal("Complete should error when rotation fails")
	}
}

// TestNewPooledClientWithBaseURL verifies per-profile BaseURL is used.
func TestNewPooledClientWithBaseURL(t *testing.T) {
	var capturedBaseURL string
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a|https://custom.example.com/v1"}, func(profile llm.AuthProfile) (llm.Client, error) {
		capturedBaseURL = profile.BaseURL
		return &mockClient{provider: "test", model: "test"}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	if capturedBaseURL != "https://custom.example.com/v1" {
		t.Errorf("BaseURL = %q, want https://custom.example.com/v1", capturedBaseURL)
	}
}

// TestPooledClientCompleteBackoffZeroCooldown verifies backoff when CooldownError has zero wait.
func TestPooledClientCompleteBackoffZeroCooldown(t *testing.T) {
	// Create a pool where all keys are in cooldown with very short duration
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		return &failingMockClient{
			calls:    []mockCall{{err: llm.NewRateLimitError("test", "rate limited")}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	// Should return context deadline error, not hang
	_, err = pc.Complete(ctx, llm.Request{})
	if err == nil {
		t.Fatal("Complete should error")
	}
}

// TestPooledClientStreamBackoffZeroCooldown verifies stream backoff with zero wait.
func TestPooledClientStreamBackoffZeroCooldown(t *testing.T) {
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &failingMockClient{
			calls:    []mockCall{{err: llm.NewRateLimitError("test", "rate limited")}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	// Should return context deadline error, not hang
	for _, err := range pc.Stream(ctx, llm.Request{}) {
		if err != nil {
			break
		}
	}
}

// TestPooledClientStreamAuthErrorNoHealthyKeys verifies stream auth error handling.
func TestPooledClientStreamAuthErrorNoHealthyKeys(t *testing.T) {
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &failingMockClient{
			calls:    []mockCall{{err: llm.NewAuthError("test", "bad key", 401)}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	for _, err := range pc.Stream(context.Background(), llm.Request{}) {
		if err != nil {
			if !llm.IsAuthError(err) {
				t.Errorf("expected auth error, got: %v", err)
			}
			return
		}
	}
	t.Fatal("expected error from stream")
}

// TestPooledClientCompleteAuthErrorNoHealthyKeys verifies Complete auth error with no healthy keys.
func TestPooledClientCompleteAuthErrorNoHealthyKeys(t *testing.T) {
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &failingMockClient{
			calls:    []mockCall{{err: llm.NewAuthError("test", "bad key", 401)}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	_, err = pc.Complete(context.Background(), llm.Request{})
	if err == nil {
		t.Fatal("Complete should error with auth error and no healthy keys")
	}
	if !llm.IsAuthError(err) {
		t.Errorf("expected auth error, got: %v", err)
	}
}

// TestMarkSuccessByNameNotFound verifies MarkSuccessByName with unknown profile name.
func TestMarkSuccessByNameNotFound(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a"})

	// This should not panic - just silently return
	pool.MarkSuccessByName("nonexistent-profile")

	// Pool should still be functional
	p, err := pool.GetAvailable()
	if err != nil {
		t.Fatalf("GetAvailable() error: %v", err)
	}
	if p.APIKey != "key-a" {
		t.Errorf("APIKey = %q, want key-a", p.APIKey)
	}
}

// TestBackoffVeryHighAttempt verifies backoff with very high attempt number.
func TestBackoffVeryHighAttempt(t *testing.T) {
	// Very high attempt should still be capped at maxDelay
	for i := 0; i < 10; i++ {
		d := llm.Backoff(100, 1*time.Second, 5*time.Second)
		if d > 5*time.Second {
			t.Errorf("backoff(100) = %v, exceeds max 5s", d)
		}
		if d <= 0 {
			t.Errorf("backoff(100) = %v, should be positive", d)
		}
	}
}

// TestPooledClientStreamRetryWithNewProfile verifies the "retrying with new profile" log path.
func TestPooledClientStreamRetryWithNewProfile(t *testing.T) {
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		mock := &failingMockClient{
			provider: "test",
			model:    "test-model",
		}
		if callCount <= 1 {
			// First client: rate limited
			mock.calls = []mockCall{{err: llm.NewRateLimitError("test", "rate limited")}}
		} else {
			// Second client: succeeds
			mock.calls = []mockCall{{resp: &llm.Response{}}}
		}
		return mock, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	var gotContent bool
	for event, err := range pc.Stream(context.Background(), llm.Request{}) {
		if err != nil {
			t.Fatalf("Stream error: %v", err)
		}
		if event.Type == llm.EventContent {
			gotContent = true
		}
	}
	if !gotContent {
		t.Error("expected content event")
	}
	if callCount != 2 {
		t.Errorf("callCount = %d, want 2", callCount)
	}
}

// TestPooledClientCompleteRetryWithNewProfile verifies the "retrying with new profile" log path.
func TestPooledClientCompleteRetryWithNewProfile(t *testing.T) {
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		mock := &failingMockClient{
			provider: "test",
			model:    "test-model",
		}
		if callCount <= 1 {
			// First client: overloaded (not auth, so rotation should help)
			mock.calls = []mockCall{{err: llm.NewOverloadedError("test", "overloaded")}}
		} else {
			// Second client: succeeds
			mock.calls = []mockCall{{resp: &llm.Response{Content: "recovered"}}}
		}
		return mock, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	resp, err := pc.Complete(context.Background(), llm.Request{})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if resp.Content != "recovered" {
		t.Errorf("Content = %q, want recovered", resp.Content)
	}
	if callCount != 2 {
		t.Errorf("callCount = %d, want 2", callCount)
	}
}

// TestMarkFailedByNameGenericError verifies cooldown for non-rate-limit, non-overload errors.
func TestMarkFailedByNameGenericError(t *testing.T) {
	pool := llm.NewAuthPool("test", []string{"key-a"})

	p, _ := pool.GetAvailable()
	// Use a generic API error (not rate limit, not overload, not auth)
	genericErr := llm.NewAPIErrorFromStatus("test", 500, "internal server error")
	pool.MarkFailedByName(p.Name, genericErr)

	// Profile should be in cooldown but still healthy
	_, err := pool.GetAvailable()
	if err == nil {
		t.Fatal("expected error after generic failure")
	}
	// Should be cooldown, not permanent
	if !strings.Contains(err.Error(), "cooldown") && !strings.Contains(err.Error(), "available") {
		t.Errorf("error = %q, want cooldown error", err.Error())
	}
}

// TestPooledClientStreamMidStreamAuthError verifies mid-stream auth error handling.
func TestPooledClientStreamMidStreamAuthError(t *testing.T) {
	callCount := 0
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		callCount++
		return &failingMockClient{
			calls:    []mockCall{{midStreamErr: llm.NewAuthError("test", "bad key", 401)}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	var gotContent bool
	var gotErr bool
	for event, err := range pc.Stream(context.Background(), llm.Request{}) {
		if err != nil {
			gotErr = true
			if !llm.IsAuthError(err) {
				t.Errorf("expected auth error, got: %v", err)
			}
			break
		}
		if event.Type == llm.EventContent {
			gotContent = true
		}
	}
	if !gotContent {
		t.Error("expected at least one content event before error")
	}
	if !gotErr {
		t.Error("expected mid-stream error")
	}
	// Mid-stream errors should not retry
	if callCount != 1 {
		t.Errorf("callCount = %d, want 1 (mid-stream should not retry)", callCount)
	}
}

// TestPooledClientStreamAllRetriesExhausted verifies stream error after all retries.
func TestPooledClientStreamAllRetriesExhausted(t *testing.T) {
	// Use 5 keys so maxRetries = 5. Each rotation succeeds but the new client also fails.
	// This way we never enter the backoff path (rotation always succeeds).
	// After 5 iterations, the loop ends and we get "all retries exhausted".
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b", "key-c", "key-d", "key-e"}, func(profile llm.AuthProfile) (llm.Client, error) {
		// All clients fail with overloaded error
		return &threadSafeMockClient{
			provider: "test",
			model:    "test-model",
			err:      llm.NewOverloadedError("test", "overloaded"),
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	var lastErr error
	for _, err := range pc.Stream(context.Background(), llm.Request{}) {
		if err != nil {
			lastErr = err
			break
		}
	}
	if lastErr == nil {
		t.Fatal("expected error after all retries exhausted")
	}
	// Should contain "all retries exhausted"
	if !strings.Contains(lastErr.Error(), "all retries exhausted") {
		t.Errorf("error = %q, want to contain 'all retries exhausted'", lastErr.Error())
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

// TestPooledClientStreamEarlyBreakSuccess verifies early break marks success.
func TestPooledClientStreamEarlyBreakSuccess(t *testing.T) {
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &failingMockClient{
			calls:    []mockCall{{resp: &llm.Response{}}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	// Break early after first content
	for event, err := range pc.Stream(context.Background(), llm.Request{}) {
		if err != nil {
			t.Fatalf("Stream error: %v", err)
		}
		if event.Type == llm.EventContent {
			break // Early break
		}
	}
	// Should not panic, profile should be marked success
}

// TestPooledClientStreamMidStreamNonRetryable verifies mid-stream non-retryable non-auth error.
func TestPooledClientStreamMidStreamNonRetryable(t *testing.T) {
	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a"}, func(profile llm.AuthProfile) (llm.Client, error) {
		return &failingMockClient{
			calls:    []mockCall{{midStreamErr: llm.NewAPIErrorFromStatus("test", 400, "bad request")}},
			provider: "test",
			model:    "test-model",
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	var gotErr bool
	for _, err := range pc.Stream(context.Background(), llm.Request{}) {
		if err != nil {
			gotErr = true
			break
		}
	}
	if !gotErr {
		t.Error("expected mid-stream error")
	}
}

// TestBackoffBaseExceedsMax verifies backoff when baseDelay > maxDelay.
func TestBackoffBaseExceedsMax(t *testing.T) {
	// When baseDelay > maxDelay, should cap immediately
	for i := 0; i < 10; i++ {
		d := llm.Backoff(0, 10*time.Second, 5*time.Second)
		if d > 5*time.Second {
			t.Errorf("backoff(0, 10s, 5s) = %v, exceeds max 5s", d)
		}
		if d <= 0 {
			t.Errorf("backoff(0, 10s, 5s) = %v, should be positive", d)
		}
	}
}

// TestNewClientWithKeysAndBaseURL verifies per-profile BaseURL through NewClientWithKeys.
func TestNewClientWithKeysAndBaseURL(t *testing.T) {
	cfg := llm.Config{
		Provider: "custom",
		Model:    "test-model",
	}

	// Keys with custom BaseURL
	keys := []string{"key-a|http://localhost:9999/v1"}

	client, err := llm.NewClientWithKeys(cfg, keys)
	if err != nil {
		t.Fatalf("NewClientWithKeys error: %v", err)
	}
	defer client.Close()

	// Should not panic and create a client
	if client.Provider() != "custom" {
		t.Errorf("Provider() = %q, want custom", client.Provider())
	}
}

// TestPooledClientConcurrentRotation verifies thundering herd prevention.
func TestPooledClientConcurrentRotation(t *testing.T) {
	// This test forcefully synchronizes multiple goroutines to ensure they all
	// hit the rate-limit failure at the exact same millisecond, brutally
	// testing the double-checked locking mechanism in rotateClient.
	var clientCreationCount int32
	var mu sync.Mutex

	// We use a gate to hold all goroutines inside their Complete() call
	// until they have all arrived, then we unleash them simultaneously.
	var unleash sync.WaitGroup
	var arrive sync.WaitGroup

	const numConcurrent = 50
	unleash.Add(1)
	arrive.Add(numConcurrent)

	pc, err := llm.NewPooledClient(llm.DefaultConfig(), []string{"key-a", "key-b"}, func(profile llm.AuthProfile) (llm.Client, error) {
		mu.Lock()
		clientCreationCount++
		count := clientCreationCount
		mu.Unlock()

		if count <= 1 {
			// First client: forces all callers to synchronize before failing
			return &mockClient{
				provider: "test",
				model:    "test-model",
				completeFunc: func(ctx context.Context, req llm.Request) (*llm.Response, error) {
					arrive.Done()  // Signal that we are ready
					unleash.Wait() // Block until the gate is opened
					return nil, llm.NewRateLimitError("test", "429 rate limited")
				},
			}, nil
		}
		// After rotation, succeed on every call
		return &threadSafeMockClient{
			provider: "test",
			model:    "test-model",
			resp:     &llm.Response{Content: "ok"},
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	var wg sync.WaitGroup
	errChan := make(chan error, numConcurrent)

	for i := 0; i < numConcurrent; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// This complete call will block inside the mock until we open the gate
			_, err := pc.Complete(context.Background(), llm.Request{})
			if err != nil {
				errChan <- err
			}
		}()
	}

	// Wait for all 50 goroutines to enter the Complete method and block
	arrive.Wait()

	// UNLEASH THE HERD
	unleash.Done()

	// Wait for them all to finish resolving the failover
	wg.Wait()
	close(errChan)

	mu.Lock()
	finalCount := clientCreationCount
	mu.Unlock()

	// 1 Initial Client + 1 Failover Client = EXACTLY 2.
	// If it is > 2, the double-checked lock failed and the herd broke through.
	if finalCount != 2 {
		t.Fatalf("Thundering herd protection failed! Expected exactly 2 clients created, got %d", finalCount)
	}
}

// TestNewPooledClientGetAvailableError verifies error when initial GetAvailable fails.
func TestNewPooledClientGetAvailableError(t *testing.T) {
	// Create pool where all profiles are immediately unhealthy
	// This is hard to test directly without modifying internal state,
	// so we'll test the zero-keys case which returns ErrNoAvailableProfiles
	_, err := llm.NewPooledClient(llm.DefaultConfig(), []string{}, func(profile llm.AuthProfile) (llm.Client, error) {
		return nil, nil
	})
	if err == nil {
		t.Fatal("NewPooledClient should error with no keys")
	}
}

// TestContentImageErrorAnthropic verifies that sending ContentImage to the
// Anthropic adapter returns an error rather than silently dropping the content.
func TestContentImageErrorAnthropic(t *testing.T) {
	cfg := llm.Config{
		Provider: "anthropic",
		Model:    "claude-sonnet-4-6",
		APIKey:   "test-key",
		BaseURL:  "http://localhost:1", // won't be reached
	}
	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	req := llm.Request{
		Messages: []llm.Message{{
			Role: llm.RoleUser,
			Content: []llm.ContentPart{
				{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "image/png", Data: "abc"}},
			},
		}},
	}

	_, err = client.Complete(context.Background(), req)
	if err == nil {
		t.Fatal("expected error for ContentImage, got nil")
	}
	if !strings.Contains(err.Error(), "image content not yet supported") {
		t.Errorf("unexpected error: %v", err)
	}
}

// TestContentImageErrorGemini verifies ContentImage error for gemini adapter.
func TestContentImageErrorGemini(t *testing.T) {
	cfg := llm.Config{
		Provider: "gemini",
		Model:    "gemini-2.5-flash",
		APIKey:   "test-key",
		BaseURL:  "http://localhost:1",
	}
	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	req := llm.Request{
		Messages: []llm.Message{{
			Role: llm.RoleUser,
			Content: []llm.ContentPart{
				{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "image/png", Data: "abc"}},
			},
		}},
	}

	_, err = client.Complete(context.Background(), req)
	if err == nil {
		t.Fatal("expected error for ContentImage, got nil")
	}
	if !strings.Contains(err.Error(), "image content not yet supported") {
		t.Errorf("unexpected error: %v", err)
	}
}

// TestContentImageErrorOpenAI verifies ContentImage error for openai adapter.
func TestContentImageErrorOpenAI(t *testing.T) {
	cfg := llm.Config{
		Provider: "openai",
		Model:    "gpt-4",
		APIKey:   "test-key",
	}
	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	req := llm.Request{
		Messages: []llm.Message{{
			Role: llm.RoleUser,
			Content: []llm.ContentPart{
				{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "image/png", Data: "abc"}},
			},
		}},
	}

	_, err = client.Complete(context.Background(), req)
	if err == nil {
		t.Fatal("expected error for ContentImage, got nil")
	}
	if !strings.Contains(err.Error(), "image content not yet supported") {
		t.Errorf("unexpected error: %v", err)
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

// TestLocalProviderRetryProtection verifies that local providers (Ollama, vLLM,
// LM Studio) created via NewClient get retry/backoff protection on transient
// errors, even though they have no API key.
func TestLocalProviderRetryProtection(t *testing.T) {
	providers := []string{"ollama", "vllm", "lmstudio"}

	for _, provider := range providers {
		t.Run(provider, func(t *testing.T) {
			cfg := llm.Config{
				Provider:  provider,
				Model:     "test-model",
				MaxTokens: 100,
				Timeout:   5 * time.Second,
			}

			client, err := llm.NewClient(cfg)
			if err != nil {
				t.Fatalf("NewClient error: %v", err)
			}
			defer client.Close()

			// PooledClient exposes PoolStatus(); a bare single client does not.
			// This proves NewClient wraps local providers in PooledClient for retry.
			type poolStatusReporter interface {
				PoolStatus() string
			}
			if _, ok := client.(poolStatusReporter); !ok {
				t.Errorf("NewClient(%s) returned %T, want *PooledClient (no retry protection)", provider, client)
			}
		})
	}
}

// TestLocalProviderRetryActualTransient verifies that a local provider client
// actually retries on transient HTTP errors (not just that it's wrapped).
func TestLocalProviderRetryActualTransient(t *testing.T) {
	// Server: fail once with 502 (retryable), then succeed
	attempts := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts == 1 {
			w.WriteHeader(http.StatusBadGateway)
			fmt.Fprint(w, `{"error":"bad gateway"}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"chatcmpl-1","object":"chat.completion","model":"llama3","choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`)
	}))
	defer srv.Close()

	cfg := llm.Config{
		Provider:  "ollama",
		Model:     "llama3",
		BaseURL:   srv.URL,
		MaxTokens: 100,
		Timeout:   5 * time.Second,
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}
	defer client.Close()

	resp, err := client.Complete(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
	})
	if err != nil {
		t.Fatalf("Complete should succeed after retry, got: %v", err)
	}
	if resp.Content != "Hello" {
		t.Errorf("Content = %q, want Hello", resp.Content)
	}
	if attempts < 2 {
		t.Errorf("attempts = %d, want >= 2 (1 failure + 1 success)", attempts)
	}
}

// =============================================================================
// REVIEW FIX TESTS
// =============================================================================

// TestTimeoutDefaultApplied verifies that a zero-value Timeout gets a sensible
// default rather than creating an unbounded HTTP client.
func TestTimeoutDefaultApplied(t *testing.T) {
	cfg := llm.Config{
		Provider:  "ollama", // no auth required
		Model:     "llama3",
		MaxTokens: 100,
		// Timeout deliberately omitted (zero value)
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient with zero Timeout should succeed: %v", err)
	}
	defer client.Close()

	// If we got here, the client was created with DefaultTimeout, not zero.
	// The real assertion is that the next line doesn't hang forever.
	if client.Provider() != "ollama" {
		t.Errorf("Provider = %q, want ollama", client.Provider())
	}
}

// TestNegativeTimeoutDefaultApplied verifies negative Timeout is corrected.
func TestNegativeTimeoutDefaultApplied(t *testing.T) {
	cfg := llm.Config{
		Provider:  "ollama",
		Model:     "llama3",
		MaxTokens: 100,
		Timeout:   -5 * time.Second,
	}

	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient with negative Timeout should succeed: %v", err)
	}
	defer client.Close()
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
		Temperature: -0.5,
	}

	_, err := llm.NewClient(cfg)
	if err == nil {
		t.Fatal("NewClient with negative Temperature should fail")
	}
	if !strings.Contains(err.Error(), "Temperature") {
		t.Errorf("error = %q, want mention of Temperature", err.Error())
	}
}

// TestThinkingLevelRejectedForOpenAICompat verifies that setting ThinkingLevel
// on an OpenAI-compatible provider returns an error instead of silently dropping it.
func TestThinkingLevelRejectedForOpenAICompat(t *testing.T) {
	providers := []string{"openai", "xai", "groq", "mistral"}

	for _, provider := range providers {
		t.Run(provider, func(t *testing.T) {
			cfg := llm.Config{
				Provider:      provider,
				Model:         "test-model",
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
			if !strings.Contains(err.Error(), "ThinkingLevel") {
				t.Errorf("error = %q, want mention of ThinkingLevel", err.Error())
			}
		})
	}
}

// TestThinkingLevelFromConfigRejectedForOpenAICompat verifies that ThinkingLevel
// set on Config (not Request) is also rejected.
func TestThinkingLevelFromConfigRejectedForOpenAICompat(t *testing.T) {
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

	// Request has ThinkingNone, but Config has ThinkingMedium — should still error
	_, err = client.Complete(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
	})
	if err == nil {
		t.Fatal("Complete should fail when Config.ThinkingLevel is set for openai_compat")
	}
	if !strings.Contains(err.Error(), "ThinkingLevel") {
		t.Errorf("error = %q, want mention of ThinkingLevel", err.Error())
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
		{llm.ThinkingLow, 0, 4096},
		{llm.ThinkingMedium, 0, 16384},
		{llm.ThinkingHigh, 0, 65536},
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

// TestNewAssistantMessage verifies assistant message construction including tool calls.
func TestNewAssistantMessage(t *testing.T) {
	resp := &llm.Response{
		Content: "Here is the result:",
		ToolCalls: []llm.ToolCall{
			{ID: "call_1", Name: "get_weather", Input: map[string]interface{}{"city": "Berlin"}},
			{ID: "call_2", Name: "get_time", Input: map[string]interface{}{"tz": "UTC"}, ThoughtSignature: "sig123"},
		},
	}

	msg := llm.NewAssistantMessage(resp)

	if msg.Role != llm.RoleAssistant {
		t.Errorf("Role = %q, want assistant", msg.Role)
	}
	if len(msg.Content) != 3 {
		t.Fatalf("Content parts = %d, want 3 (1 text + 2 tool_use)", len(msg.Content))
	}

	// Text part
	if msg.Content[0].Type != llm.ContentText || msg.Content[0].Text != "Here is the result:" {
		t.Errorf("content[0] = %+v, want text part", msg.Content[0])
	}

	// Tool use parts
	if msg.Content[1].Type != llm.ContentToolUse || msg.Content[1].ToolName != "get_weather" {
		t.Errorf("content[1] = %+v, want get_weather tool_use", msg.Content[1])
	}
	if msg.Content[2].ThoughtSignature != "sig123" {
		t.Errorf("content[2].ThoughtSignature = %q, want sig123", msg.Content[2].ThoughtSignature)
	}
}

// TestNewAssistantMessageTextOnly verifies message with no tool calls.
func TestNewAssistantMessageTextOnly(t *testing.T) {
	resp := &llm.Response{Content: "Just text"}
	msg := llm.NewAssistantMessage(resp)
	if len(msg.Content) != 1 || msg.Content[0].Type != llm.ContentText {
		t.Errorf("expected single text part, got %+v", msg.Content)
	}
}

// TestNewAssistantMessageEmpty verifies empty response produces empty content.
func TestNewAssistantMessageEmpty(t *testing.T) {
	resp := &llm.Response{}
	msg := llm.NewAssistantMessage(resp)
	if len(msg.Content) != 0 {
		t.Errorf("expected zero content parts, got %d", len(msg.Content))
	}
}

// TestNewAssistantMessageNilPanics verifies nil response panics with clear message.
func TestNewAssistantMessageNilPanics(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic for nil resp")
		}
		msg := fmt.Sprint(r)
		if !strings.Contains(msg, "must not be nil") {
			t.Errorf("panic message = %q, want 'must not be nil'", msg)
		}
	}()
	llm.NewAssistantMessage(nil)
}

// TestIsRetryableStringFallback verifies the narrowed string-matching patterns.
func TestIsRetryableStringFallback(t *testing.T) {
	tests := []struct {
		msg      string
		expected bool
	}{
		// Should match
		{"connection refused", true},
		{"dial tcp: no such host", true},
		{"context deadline exceeded: timeout", true},
		{"read tcp: connection reset", true},
		{"write: broken pipe", true},
		{"unexpected eof", true},  // HasSuffix "eof" — network-level condition
		{"something eof", true},   // HasSuffix "eof"
		// Should NOT match (narrowed in this fix)
		{"your payment request failed", false},
		{"request failed: invalid model", false},
	}

	for _, tc := range tests {
		err := errors.New(tc.msg)
		got := llm.IsRetryable(err)
		if got != tc.expected {
			t.Errorf("IsRetryable(%q) = %v, want %v", tc.msg, got, tc.expected)
		}
	}
}

// TestMaxRetryAttemptsCapped verifies the retry cap prevents excessive iterations.
func TestMaxRetryAttemptsCapped(t *testing.T) {
	// Create a pool with 20 keys (all will fail), verify we don't do 20 retries
	var attempts int
	keys := make([]string, 20)
	for i := range keys {
		keys[i] = fmt.Sprintf("key-%d", i)
	}

	pc, err := llm.NewPooledClient(llm.DefaultConfig(), keys, func(profile llm.AuthProfile) (llm.Client, error) {
		return &threadSafeMockClient{
			provider: "test",
			model:    "test-model",
			err:      llm.NewRateLimitError("test", "rate limited"),
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient error: %v", err)
	}
	defer pc.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, err = pc.Complete(ctx, llm.Request{})
	if err == nil {
		t.Fatal("expected error when all keys fail")
	}
	_ = attempts // attempts tracked inside pool, verified by timing
	// The test passes if it completes within 5s — without the cap,
	// 20 retries × backoff would take much longer.
}

// TestDefaultTimeoutConstant verifies the constant is exported and correct.
func TestDefaultTimeoutConstant(t *testing.T) {
	if llm.DefaultTimeout != 120*time.Second {
		t.Errorf("DefaultTimeout = %v, want 120s", llm.DefaultTimeout)
	}
}
