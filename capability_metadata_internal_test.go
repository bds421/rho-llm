package llm

import (
	"strings"
	"testing"
)

func TestCapabilityValidationFailsClosedForUnknownModel(t *testing.T) {
	err := RequireCapabilities(Config{Provider: "vllm", Model: "tenant/model"}, CapabilityChat)
	if err == nil || !strings.Contains(err.Error(), "no reviewed capability metadata") {
		t.Fatalf("RequireCapabilities() error = %v", err)
	}
}

// Cloud openai_compat catalogs must not lose vision/PDF/structured support when
// callers use default built-in metadata (v0.5→0.6 regression guard).
func TestDefaultCloudCatalogCapabilitiesMatchDocumentedMultimodalSupport(t *testing.T) {
	cases := []struct {
		provider string
		model    string
		want     []Capability
	}{
		{"xai", "grok-4.5", []Capability{CapabilityChat, CapabilityVision, CapabilityStructuredOutput}},
		{"mistral", "mistral-large-4-0", []Capability{CapabilityChat, CapabilityVision, CapabilityDocumentInput, CapabilityStructuredOutput}},
		{"meta", "muse-spark-1.2", []Capability{CapabilityChat, CapabilityVision, CapabilityDocumentInput, CapabilityStructuredOutput}},
		// GPT-5.x auto-selects openai_responses, which encodes vision/structured/batch
		// but not PDF (document fails closed at the protocol envelope — by design).
		{"openai", "gpt-5.6-sol", []Capability{CapabilityChat, CapabilityVision, CapabilityStructuredOutput, CapabilityBatch}},
		{"openai", "gpt-4.1", []Capability{CapabilityChat, CapabilityVision, CapabilityDocumentInput, CapabilityStructuredOutput}},
		{"anthropic", "claude-sonnet-5", []Capability{CapabilityChat, CapabilityVision, CapabilityDocumentInput, CapabilityBatch}},
		{"gemini", "gemini-3.6-flash", []Capability{CapabilityChat, CapabilityVision, CapabilityDocumentInput, CapabilityBatch}},
		{"gemini", "gemini-embedding-001", []Capability{CapabilityEmbeddings}},
		{"groq", "llama-3.3-70b-versatile", []Capability{CapabilityChat, CapabilityStructuredOutput, CapabilityVision}},
	}
	for _, tc := range cases {
		cfg := Config{Provider: tc.provider, Model: tc.model}
		if err := RequireCapabilities(cfg, tc.want...); err != nil {
			t.Fatalf("%s/%s: %v", tc.provider, tc.model, err)
		}
	}
	// xAI Chat Completions takes no files (live 2026-11-18): a PDF fails closed.
	if err := RequireCapabilities(Config{Provider: "xai", Model: "grok-4.7"}, CapabilityDocumentInput); err == nil {
		t.Fatal("xai claims document input; its Chat Completions rejects files")
	}
	// Local hosts stay chat-focused so undeclared vision fails closed.
	if err := RequireCapabilities(Config{Provider: "ollama", Model: "qwen3:8b"}, CapabilityVision); err == nil {
		t.Fatal("ollama default catalog should not claim vision without an explicit declaration")
	}
}

func TestRegisterModelZeroCapabilitiesAppliesChatDefaults(t *testing.T) {
	const model = "test/register-model-default-caps"
	if err := RegisterModel(ModelInfo{ID: model, Provider: "vllm"}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}
	info, ok := GetModelInfo(model)
	if !ok || !info.Capabilities.Supports(CapabilityChat, CapabilityStream) {
		t.Fatalf("RegisterModel zero Capabilities did not apply chat defaults: %+v", info)
	}
	if err := RequireCapabilities(Config{Provider: "vllm", Model: model}, CapabilityChat); err != nil {
		t.Fatalf("dispatch capability check after zero-cap RegisterModel: %v", err)
	}
	// Still fail closed for undeclared extras on local hosts.
	if err := RequireCapabilities(Config{Provider: "vllm", Model: model}, CapabilityVision); err == nil {
		t.Fatal("vllm default chat caps should not invent vision")
	}
}

func TestCompatibleLocalModelCanDeclareReviewedCapabilities(t *testing.T) {
	const model = "test/vllm-reviewed-capabilities"
	if err := RegisterModel(ModelInfo{
		ID: model, Provider: "vllm",
		Capabilities: Capabilities(CapabilityChat, CapabilityStream, CapabilityTools),
	}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}
	cfg := Config{Provider: "vllm", Model: model}
	if err := RequireCapabilities(cfg, CapabilityChat, CapabilityTools); err != nil {
		t.Fatalf("RequireCapabilities: %v", err)
	}
	if err := RequireCapabilities(cfg, CapabilityDocumentInput); err == nil {
		t.Fatal("undeclared document input capability was accepted")
	}
}

func TestDeploymentScopedCapabilitiesOverrideGlobalRegistryWithoutMutation(t *testing.T) {
	const model = "test/shared-deployment-model"
	if err := RegisterModel(ModelInfo{
		ID: model, Provider: "vllm",
		Capabilities: Capabilities(CapabilityChat, CapabilityTools),
	}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}

	embeddingDeployment := Config{
		Provider: "vllm", Model: model,
		ModelCapabilities: Capabilities(CapabilityEmbeddings),
	}
	chatDeployment := Config{
		Provider: "vllm", Model: model,
		ModelCapabilities: Capabilities(CapabilityChat),
	}
	if err := RequireCapabilities(embeddingDeployment, CapabilityEmbeddings); err != nil {
		t.Fatalf("deployment-scoped embeddings rejected: %v", err)
	}
	if err := RequireCapabilities(embeddingDeployment, CapabilityChat); err == nil {
		t.Fatal("global chat metadata overrode deployment-scoped capabilities")
	}
	if err := RequireCapabilities(chatDeployment, CapabilityChat); err != nil {
		t.Fatalf("second deployment chat rejected: %v", err)
	}
	if err := RequireCapabilities(chatDeployment, CapabilityEmbeddings); err == nil {
		t.Fatal("first deployment capabilities leaked into second deployment")
	}
	if err := RequireCapabilities(embeddingDeployment, CapabilityEmbeddings); err != nil {
		t.Fatalf("second deployment evaluation mutated the first: %v", err)
	}
}

func TestDeploymentScopedCapabilitiesAreBoundToExactConfigModel(t *testing.T) {
	cfg := Config{
		Provider: "vllm", Model: "tenant/exact-model",
		ModelCapabilities: Capabilities(CapabilityChat, CapabilityBatch),
	}
	if err := ValidateRequestCapabilities(cfg, Request{Model: "tenant/other-model"}, false); err == nil ||
		!strings.Contains(err.Error(), "cannot authorize") {
		t.Fatalf("different request model error = %v", err)
	}
	if err := RequireCapabilitiesForModel(cfg, "tenant/other-model", CapabilityBatch); err == nil {
		t.Fatal("different batch item model was authorized")
	}
	if err := ValidateRequestCapabilities(cfg, Request{Model: cfg.Model}, false); err != nil {
		t.Fatalf("exact request model rejected: %v", err)
	}
}

func TestRequestCapabilitiesCoversEveryChatFeature(t *testing.T) {
	temperature := 0.2
	req := Request{
		Tools:          []Tool{{Name: "lookup"}},
		ResponseFormat: &ResponseFormat{Type: ResponseFormatJSONObject},
		Temperature:    &temperature,
		Messages: []Message{{Role: RoleUser, Content: []ContentPart{
			{Type: ContentImage, Source: &ImageSource{}},
			{Type: ContentDocument, Document: &DocumentSource{}},
		}}},
	}
	got := Capabilities(RequestCapabilities(req, true)...)
	want := Capabilities(
		CapabilityChat, CapabilityStream, CapabilityTools,
		CapabilityStructuredOutput, CapabilityVision, CapabilityDocumentInput,
		CapabilityTemperature,
	)
	if got != want {
		t.Fatalf("RequestCapabilities() = %b, want %b", got, want)
	}
}

func TestTemperatureCapabilityIsRequiredOnlyWhenExplicit(t *testing.T) {
	temperature := 0.2
	if got := Capabilities(RequestCapabilities(Request{}, false)...); got.Supports(CapabilityTemperature) {
		t.Fatal("nil temperature required sampling capability")
	}
	if got := Capabilities(RequestCapabilities(Request{Temperature: &temperature}, false)...); !got.Supports(CapabilityTemperature) {
		t.Fatal("explicit temperature did not require sampling capability")
	}

	for _, model := range []string{"gemini-3.6-flash", "gemini-3.5-flash-lite"} {
		cfg := Config{
			Provider: "gemini", Model: model,
			ModelCapabilities: Capabilities(CapabilityChat, CapabilityTemperature),
		}
		if err := ValidateRequestCapabilities(cfg, Request{}, false); err != nil {
			t.Fatalf("model %q rejected provider-default sampling: %v", model, err)
		}
		if err := ValidateRequestCapabilities(cfg, Request{Temperature: &temperature}, false); err == nil ||
			!strings.Contains(err.Error(), "sampling_temperature") {
			t.Fatalf("model %q explicit temperature error = %v", model, err)
		}
	}
	if err := ValidateRequestCapabilities(
		Config{Provider: "gemini", Model: "gemini-3.1-flash-lite"},
		Request{Temperature: &temperature}, false,
	); err != nil {
		t.Fatalf("reviewed temperature-capable Gemini rejected explicit sampling: %v", err)
	}
}

func TestAnthropicReasoningRejectsIncompatibleExplicitTemperature(t *testing.T) {
	cfg := Config{
		Provider: "anthropic", Model: "tenant/claude", AnthropicVersion: AnthropicVersion20230601,
		ModelCapabilities: Capabilities(CapabilityChat, CapabilityReasoning, CapabilityTemperature),
	}
	for _, tc := range []struct {
		name         string
		configLevel  ThinkingLevel
		requestLevel ThinkingLevel
		temperature  *float64
		wantError    bool
	}{
		{name: "provider default", requestLevel: ThinkingLow},
		{name: "exact required value", requestLevel: ThinkingLow, temperature: float64Pointer(1)},
		{name: "incompatible request value", requestLevel: ThinkingLow, temperature: float64Pointer(0.2), wantError: true},
		{name: "incompatible config reasoning fallback", configLevel: ThinkingLow, temperature: float64Pointer(0.2), wantError: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			caseCfg := cfg
			caseCfg.ThinkingLevel = tc.configLevel
			err := ValidateRequestCapabilities(caseCfg, Request{
				ThinkingLevel: tc.requestLevel,
				Temperature:   tc.temperature,
			}, false)
			if tc.wantError {
				if err == nil || !strings.Contains(err.Error(), "requires sampling temperature 1") {
					t.Fatalf("ValidateRequestCapabilities() error = %v", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("ValidateRequestCapabilities() error = %v", err)
			}
		})
	}
}

func float64Pointer(value float64) *float64 { return &value }

func TestDedicatedModelCannotCrossModalities(t *testing.T) {
	cfg := Config{Provider: "openai", Model: "whisper-1"}
	if err := RequireCapabilities(cfg, CapabilityTranscription); err != nil {
		t.Fatalf("transcription rejected: %v", err)
	}
	if err := RequireCapabilities(cfg, CapabilityChat); err == nil {
		t.Fatal("transcription-only model admitted chat")
	}
}

func TestReasoningCapabilityMeansEffortIsEnforceable(t *testing.T) {
	if err := ValidateRequestCapabilities(
		Config{Provider: "openai", Model: "gpt-5.6-sol"},
		Request{ThinkingLevel: ThinkingLow}, false,
	); err != nil {
		t.Fatalf("Responses reasoning effort rejected: %v", err)
	}
	// A thinking model on an OpenAI-compatible host with a documented control
	// (Ollama's reasoning_effort) is enforceable since P2.
	if err := ValidateRequestCapabilities(Config{Provider: "ollama", Model: "qwen3:8b"}, Request{ThinkingLevel: ThinkingLow}, false); err != nil {
		t.Fatalf("ollama reasoning effort rejected: %v", err)
	}
	for _, cfg := range []Config{
		{Provider: "openai", Model: "gpt-4.1-mini"},
		{Provider: "minimax", Model: "MiniMax-M3.1-Flash"}, // thinking, but no control mapped
		{Provider: "gemini", Model: "gemini-2.5-flash"},
	} {
		if err := ValidateRequestCapabilities(cfg, Request{ThinkingLevel: ThinkingLow}, false); err == nil {
			t.Fatalf("provider=%q model=%q admitted unenforceable reasoning effort", cfg.Provider, cfg.Model)
		}
	}
}

// SamplingParams is merged at the top level of the body, where Gemini has no
// sampling fields: a top-level "temperature" is a 400 on every Gemini model
// (live, gemini-2.5-flash-lite, 2026-11-18). Request.Temperature is refused for
// models without sampling controls through CapabilityTemperature.
func TestGeminiSamplingParamsRefused(t *testing.T) {
	// The Interactions protocol answers a top-level "temperature" with a 400
	// "Unknown parameter" as well (live, gemini-3.8-flash, 2026-11-18).
	for _, key := range []string{"temperature", "top_p", "top_k", "topP", "topK"} {
		cfg := Config{Provider: "gemini-interactions", Model: "gemini-3.8-flash", APIKey: "k"}
		err := ValidateRequestCapabilities(cfg, Request{SamplingParams: map[string]any{key: 0.5}}, false)
		if err == nil || !strings.Contains(err.Error(), key) {
			t.Errorf("gemini-interactions SamplingParams[%s]: err = %v, want a refusal naming the key", key, err)
		}
	}
	for _, model := range []string{"gemini-4-hydrogen", "gemini-4-carbon", "gemini-3.8-flash"} {
		cfg := Config{Provider: "gemini", Model: model, APIKey: "k"}
		for _, key := range []string{"temperature", "top_p", "top_k", "topP", "topK"} {
			req := Request{SamplingParams: map[string]any{key: 0.5}}
			err := ValidateRequestCapabilities(cfg, req, false)
			if err == nil || !strings.Contains(err.Error(), key) {
				t.Errorf("%s SamplingParams[%s]: err = %v, want a refusal naming the key", model, key, err)
			}
		}
		temp := 5.0
		if err := ValidateRequestCapabilities(cfg, Request{Temperature: &temp}, false); err == nil {
			t.Errorf("%s: Temperature accepted", model)
		}
		for _, sp := range []map[string]any{nil, {}, {"seed": 1}} {
			if err := ValidateRequestCapabilities(cfg, Request{SamplingParams: sp}, true); err != nil {
				t.Errorf("%s: SamplingParams %v refused: %v", model, sp, err)
			}
		}
	}
	// A model that takes a temperature gets it typed; the top-level key is
	// still a 400 there, so it is refused too.
	old := Config{Provider: "gemini", Model: "gemini-2.5-flash", APIKey: "k"}
	if err := ValidateRequestCapabilities(old, Request{SamplingParams: map[string]any{"top_k": 3}}, false); err == nil {
		t.Error("gemini-2.5-flash accepted a top-level top_k (a 400 on the wire)")
	}
	temp := 0.3
	if err := ValidateRequestCapabilities(old, Request{Temperature: &temp, SamplingParams: map[string]any{"seed": 1}}, false); err != nil {
		t.Errorf("gemini-2.5-flash refused typed Temperature with seed: %v", err)
	}
	// Other protocols keep their top-level sampling keys.
	oa := Config{Provider: "openai", Model: "gpt-4.1", APIKey: "k"}
	if err := ValidateRequestCapabilities(oa, Request{SamplingParams: map[string]any{"top_p": 0.9}}, false); err != nil {
		t.Errorf("openai gpt-4.1 refused top_p: %v", err)
	}
	// The per-request model is what counts.
	if err := ValidateRequestCapabilities(old, Request{Model: "gemini-4-hydrogen", SamplingParams: map[string]any{"top_p": 1}}, false); err == nil {
		t.Error("Request.Model override to gemini-4-hydrogen accepted top_p")
	}
}
