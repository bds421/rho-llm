package llm

import (
	"fmt"
	"strings"
)

// Capability is one provider-neutral operation or request feature. Capability
// checks are deliberately separate from transport routing: a protocol being able
// to encode a field does not prove that a selected model implements it.
type Capability uint64

const (
	CapabilityChat Capability = 1 << iota
	CapabilityStream
	CapabilityTools
	CapabilityStructuredOutput
	CapabilityVision
	CapabilityDocumentInput
	CapabilityEmbeddings
	CapabilityBatch
	CapabilityImageGeneration
	CapabilitySpeechSynthesis
	CapabilityTranscription
	// CapabilityReasoning means the adapter/model combination can enforce a
	// requested ThinkingLevel. Intrinsic reasoning output alone is insufficient.
	CapabilityReasoning
	// CapabilityTemperature means the adapter/model combination can encode and
	// honor an explicitly requested sampling temperature. Nil uses the provider
	// default and does not require this capability.
	CapabilityTemperature
)

const allCapabilities = CapabilitySet(
	CapabilityChat | CapabilityStream | CapabilityTools | CapabilityStructuredOutput |
		CapabilityVision | CapabilityDocumentInput | CapabilityEmbeddings | CapabilityBatch |
		CapabilityImageGeneration | CapabilitySpeechSynthesis | CapabilityTranscription |
		CapabilityReasoning | CapabilityTemperature,
)

// CapabilitySet is a compact set of Capability values.
type CapabilitySet uint64

// Capabilities constructs a capability set.
func Capabilities(values ...Capability) CapabilitySet {
	var set CapabilitySet
	for _, value := range values {
		set |= CapabilitySet(value)
	}
	return set
}

// Supports reports whether every requested capability is present.
func (set CapabilitySet) Supports(required ...Capability) bool {
	for _, capability := range required {
		if set&CapabilitySet(capability) == 0 {
			return false
		}
	}
	return true
}

func (capability Capability) String() string {
	switch capability {
	case CapabilityChat:
		return "chat"
	case CapabilityStream:
		return "stream"
	case CapabilityTools:
		return "tools"
	case CapabilityStructuredOutput:
		return "structured_output"
	case CapabilityVision:
		return "vision"
	case CapabilityDocumentInput:
		return "document_input"
	case CapabilityEmbeddings:
		return "embeddings"
	case CapabilityBatch:
		return "batch"
	case CapabilityImageGeneration:
		return "image_generation"
	case CapabilitySpeechSynthesis:
		return "speech_synthesis"
	case CapabilityTranscription:
		return "transcription"
	case CapabilityReasoning:
		return "reasoning"
	case CapabilityTemperature:
		return "sampling_temperature"
	default:
		return fmt.Sprintf("capability(%d)", capability)
	}
}

// CapabilityProfile is the reviewed model metadata intersected with the
// selected wire protocol's implementation envelope.
type CapabilityProfile struct {
	Provider             string
	Model                string
	Protocol             string
	ProviderCapabilities CapabilitySet
	ModelCapabilities    CapabilitySet
	Capabilities         CapabilitySet
}

// ResolveCapabilityProfile resolves reviewed capability metadata for cfg.
// Config.ModelCapabilities is deployment-scoped authority for the exact
// Config.Model and takes precedence over the process-global registry. Unknown
// models without that declaration fail closed.
func ResolveCapabilityProfile(cfg Config) (CapabilityProfile, error) {
	model := configuredModel(cfg)
	modelCapabilities := cfg.ModelCapabilities
	if modelCapabilities != 0 {
		if model == "" || modelCapabilities&^allCapabilities != 0 {
			return CapabilityProfile{}, fmt.Errorf("llm: deployment-scoped model capability metadata is invalid")
		}
	} else {
		info, ok := GetModelInfo(model)
		if !ok || info.Capabilities == 0 {
			// A retired ID is a distinct, actionable failure: the provider no
			// longer serves it, so point the caller at the replacement rather
			// than suggesting they register the dead model.
			if replacement, retired := RetiredModelReplacement(model); retired {
				return CapabilityProfile{}, fmt.Errorf(
					"llm: model %q was retired by its provider; use %q instead",
					model, replacement,
				)
			}
			return CapabilityProfile{}, fmt.Errorf(
				"llm: model %q has no reviewed capability metadata (register it or set Config.ModelCapabilities)",
				model,
			)
		}
		if !sameProviderFamily(info.Provider, cfg.Provider) {
			return CapabilityProfile{}, fmt.Errorf(
				"llm: model %q is registered for provider %q, not %q",
				model, info.Provider, cfg.Provider,
			)
		}
		modelCapabilities = info.Capabilities
	}
	protocol := ResolveProtocol(Config{
		Provider: cfg.Provider, Model: model, ModelCapabilities: cfg.ModelCapabilities,
	})
	providerCapabilities := protocolCapabilityEnvelope(protocol)
	providerCapabilities &= modelProtocolCapabilityEnvelope(cfg.Provider, model)
	return CapabilityProfile{
		Provider:             canonicalProvider(cfg.Provider),
		Model:                model,
		Protocol:             protocol,
		ProviderCapabilities: providerCapabilities,
		ModelCapabilities:    modelCapabilities,
		Capabilities:         modelCapabilities & providerCapabilities,
	}, nil
}

func modelProtocolCapabilityEnvelope(provider, model string) CapabilitySet {
	envelope := allCapabilities
	info, ok := GetModelInfo(model)
	if !ok || !sameProviderFamily(info.Provider, provider) {
		// Unknown local/custom models remain governed by their exact deployment
		// capabilities and the generic protocol envelope.
		return envelope
	}
	if !supportsSamplingTemperature(info) {
		envelope &^= CapabilitySet(CapabilityTemperature)
	}
	return envelope
}

func configuredModel(cfg Config) string {
	if cfg.ModelCapabilities != 0 {
		return strings.TrimSpace(cfg.Model)
	}
	return ResolveModelAlias(cfg.Model)
}

// RequireCapabilitiesForModel validates capabilities for a requested model.
// A deployment-scoped declaration is bound to cfg.Model and cannot authorize a
// different model carried by a request or batch item.
func RequireCapabilitiesForModel(cfg Config, requestedModel string, required ...Capability) error {
	requestedModel = strings.TrimSpace(requestedModel)
	if requestedModel != "" {
		if cfg.ModelCapabilities != 0 && requestedModel != strings.TrimSpace(cfg.Model) {
			return fmt.Errorf(
				"llm: deployment-scoped capabilities for model %q cannot authorize model %q",
				cfg.Model, requestedModel,
			)
		}
		cfg.Model = requestedModel
	}
	return RequireCapabilities(cfg, required...)
}

// RequireCapabilities validates a reviewed provider/model combination before a
// dispatch can occur.
func RequireCapabilities(cfg Config, required ...Capability) error {
	profile, err := ResolveCapabilityProfile(cfg)
	if err != nil {
		return err
	}
	for _, capability := range required {
		if !profile.Capabilities.Supports(capability) {
			return fmt.Errorf(
				"llm: provider %q model %q does not support %s via %s",
				cfg.Provider, profile.Model, capability, profile.Protocol,
			)
		}
	}
	return nil
}

// RequestCapabilities derives all capabilities needed to encode and execute a
// chat request. It does not inspect prompt text.
func RequestCapabilities(req Request, stream bool) []Capability {
	required := []Capability{CapabilityChat}
	if stream {
		required = append(required, CapabilityStream)
	}
	if len(req.Tools) > 0 || len(req.ServerTools) > 0 {
		required = append(required, CapabilityTools)
	}
	if req.ResponseFormat != nil {
		required = append(required, CapabilityStructuredOutput)
	}
	if req.ThinkingLevel != ThinkingNone {
		required = append(required, CapabilityReasoning)
	}
	if req.Temperature != nil {
		required = append(required, CapabilityTemperature)
	}
	for _, message := range req.Messages {
		for _, part := range message.Content {
			switch part.Type {
			case ContentImage:
				required = appendCapability(required, CapabilityVision)
			case ContentDocument:
				required = appendCapability(required, CapabilityDocumentInput)
			}
		}
	}
	return required
}

// ValidateRequestCapabilities validates the exact request against reviewed
// model metadata before an adapter opens a network connection.
func ValidateRequestCapabilities(cfg Config, req Request, stream bool) error {
	effectiveReq := req
	if effectiveReq.ThinkingLevel == ThinkingNone && cfg.ThinkingLevel != ThinkingNone {
		effectiveReq.ThinkingLevel = cfg.ThinkingLevel
	}
	if err := RequireCapabilitiesForModel(
		cfg, req.Model, RequestCapabilities(effectiveReq, stream)...,
	); err != nil {
		return err
	}

	effectiveCfg := cfg
	if strings.TrimSpace(req.Model) != "" {
		effectiveCfg.Model = req.Model
	}
	profile, err := ResolveCapabilityProfile(effectiveCfg)
	if err != nil {
		return err
	}
	if profile.Protocol == "anthropic" {
		if err := AnthropicSamplingError(cfg, req); err != nil {
			return err
		}
	}
	if profile.Protocol == "anthropic" &&
		effectiveReq.ThinkingLevel != ThinkingNone &&
		req.Temperature != nil && *req.Temperature != 1.0 {
		return fmt.Errorf(
			"llm: provider %q requires sampling temperature 1 when reasoning is enabled; requested %g",
			profile.Provider, *req.Temperature,
		)
	}
	return nil
}

func appendCapability(values []Capability, value Capability) []Capability {
	for _, existing := range values {
		if existing == value {
			return values
		}
	}
	return append(values, value)
}

func protocolCapabilityEnvelope(protocol string) CapabilitySet {
	switch protocol {
	case "anthropic":
		return Capabilities(CapabilityChat, CapabilityStream, CapabilityTools, CapabilityStructuredOutput, CapabilityVision, CapabilityDocumentInput, CapabilityReasoning, CapabilityTemperature, CapabilityBatch)
	case "gemini":
		return Capabilities(
			CapabilityChat, CapabilityStream, CapabilityTools, CapabilityStructuredOutput,
			CapabilityVision, CapabilityDocumentInput, CapabilityReasoning, CapabilityTemperature,
			CapabilityEmbeddings, CapabilityImageGeneration, CapabilityBatch, CapabilityTranscription,
		)
	case "openai_responses":
		return Capabilities(CapabilityChat, CapabilityStream, CapabilityTools, CapabilityStructuredOutput, CapabilityVision, CapabilityDocumentInput, CapabilityBatch, CapabilityReasoning)
	case "openai_compat":
		return Capabilities(
			CapabilityChat, CapabilityStream, CapabilityTools, CapabilityStructuredOutput,
			CapabilityVision, CapabilityDocumentInput, CapabilityEmbeddings,
			CapabilityBatch, CapabilityImageGeneration, CapabilitySpeechSynthesis,
			CapabilityTranscription, CapabilityTemperature,
			// Reasoning goes out in each host's dialect; a host without a
			// documented control refuses the level in the adapter (P2).
			CapabilityReasoning,
		)
	default:
		return 0
	}
}

func canonicalProvider(provider string) string {
	switch strings.ToLower(strings.TrimSpace(provider)) {
	case "claude":
		return "anthropic"
	case "google":
		return "gemini"
	case "grok":
		return "xai"
	case "qwen", "dashscope-cn", "qwen-cn":
		return "dashscope"
	case "z-ai", "glm":
		return "zai"
	case "kimi", "moonshot-cn", "kimi-cn":
		return "moonshot"
	case "openai_responses":
		return "openai"
	default:
		return strings.ToLower(strings.TrimSpace(provider))
	}
}

func sameProviderFamily(left, right string) bool {
	return canonicalProvider(left) == canonicalProvider(right)
}

func defaultChatCapabilities(info ModelInfo) CapabilitySet {
	set := Capabilities(CapabilityChat, CapabilityStream)
	if !info.NoToolSupport {
		set |= CapabilitySet(CapabilityTools)
	}
	if info.SupportsThinking || info.ResponsesAPI || (info.Thinking && compatReasoningHosts[canonicalProvider(info.Provider)]) {
		set |= CapabilitySet(CapabilityReasoning)
	}
	if supportsSamplingTemperature(info) {
		set |= CapabilitySet(CapabilityTemperature)
	}
	switch canonicalProvider(info.Provider) {
	case "anthropic":
		// Native PDF + vision, JSON-schema output (output_config.format) on every
		// current Claude model; Message Batches for chat.
		set |= Capabilities(CapabilityStructuredOutput, CapabilityVision, CapabilityDocumentInput, CapabilityBatch)
	case "gemini":
		// Multimodal chat + batch; dedicated embedding/image models set their own bits.
		// Gemini chat models accept audio natively, so they transcribe via generateContent.
		set |= Capabilities(CapabilityStructuredOutput, CapabilityVision, CapabilityDocumentInput, CapabilityBatch, CapabilityTranscription)
	case "openai":
		// Chat Completions / Responses encode vision + structured output; PDF rides as
		// an image_url data URI on openai_compat (document capability is model-side).
		// Batch is first-party OpenAI only.
		set |= Capabilities(CapabilityStructuredOutput, CapabilityVision, CapabilityDocumentInput, CapabilityBatch)
	case "xai", "meta":
		// Multimodal cloud APIs on the openai_compat surface (PDF as data URI).
		set |= Capabilities(CapabilityStructuredOutput, CapabilityVision, CapabilityDocumentInput)
	case "groq", "mistral", "openrouter", "together", "fireworks", "deepinfra", "nvidia",
		"cerebras", "deepseek", "cohere", "dashscope", "zai", "minimax", "moonshot", "perplexity":
		// Cloud openai_compat hosts: the adapter can encode vision/PDF/structured
		// output; hosts that reject a given model still fail at the wire.
		set |= Capabilities(CapabilityStructuredOutput, CapabilityVision, CapabilityDocumentInput)
		// Local ollama/vllm/lmstudio stay chat/stream/tools-only — deployments that
		// actually run vision or JSON-schema models declare Capabilities explicitly.
	}
	return set
}

func supportsSamplingTemperature(info ModelInfo) bool {
	// Claude 4.7+ reject temperature/top_p/top_k (Sonnet 5.5 and Haiku 5.5
	// reject non-default values), so no explicit temperature can be honored.
	// This is the model side; API version 2026-10-21 rejects them on every
	// model (ValidateRequestCapabilities checks the version).
	if _, denied := anthropicWithoutSamplingControls[info.ID]; denied {
		return false
	}
	// OpenAI/xAI reasoning endpoints reject or ignore custom temperature.
	if (canonicalProvider(info.Provider) == "openai" || canonicalProvider(info.Provider) == "xai") &&
		(info.Thinking || info.ResponsesAPI) {
		return false
	}
	// Kimi K3 models fix temperature, top_p, n and the penalties ("omit them from
	// requests", platform.kimi.ai/docs/guide/kimi-k3-1-pro-quickstart, 2026-11-10).
	if _, denied := moonshotWithoutSamplingControls[info.ID]; denied {
		return false
	}
	// Google deprecated and ignores sampling controls beginning with the July
	// 2026 GA models, and later generations reject them outright. An explicit
	// list is used rather than a version comparison so a newly registered model
	// is a deliberate decision: adding a Gemini row without deciding this is a
	// bug, which is why registryGeminiSamplingDecided pins it.
	_, denied := geminiWithoutSamplingControls[info.ID]
	return !denied
}

// anthropicWithoutSamplingControls are the Claude IDs whose API rejects
// sampling parameters (platform.claude.com model docs, 2026-10-31). It must
// stay in step with the anthropic adapter's adaptiveThinkingModels — the
// adapter's tests pin that.
// moonshotWithoutSamplingControls lists the Moonshot models whose sampling
// parameters are fixed by the API.
var moonshotWithoutSamplingControls = map[string]struct{}{
	"kimi-k3.1-pro": {},
	"kimi-k3":       {},
}

var anthropicWithoutSamplingControls = map[string]struct{}{
	"claude-fable-5-2":  {},
	"claude-fable-5-1":  {},
	"claude-fable-5":    {},
	"claude-opus-5-5":   {},
	"claude-opus-5":     {},
	"claude-opus-4-8":   {},
	"claude-opus-4-7":   {},
	"claude-sonnet-5-5": {},
	"claude-sonnet-5":   {},
	"claude-haiku-5-5":  {},
}

// geminiWithoutSamplingControls are the Gemini IDs whose API rejects or ignores
// temperature/top_p/top_k. Google removed the controls from the July 2026 GA
// models onward, so every Gemini model registered after gemini-3.5-flash-lite
// belongs here unless Google documents otherwise.
var geminiWithoutSamplingControls = map[string]struct{}{
	"gemini-4-argon":        {},
	"gemini-3.5-flash-lite": {},
	"gemini-3.6-flash":      {},
	"gemini-3.7-flash":      {},
	"gemini-3.8-flash":      {},
}

// errAnthropicSampling is the error for a sampling setting that Anthropic API
// version 2026-10-21 rejects on every model.
func errAnthropicSampling(field string) error {
	return fmt.Errorf(
		"llm: Anthropic API version %s rejects %s on every model; leave it unset, "+
			"or pin Config.AnthropicVersion = llm.AnthropicVersion20230601 (Claude 4.6 and 4.5 accept it there)",
		AnthropicVersion20261021, field,
	)
}

// AnthropicSamplingError returns the error for a sampling setting (Temperature,
// or temperature / top_p / top_k in SamplingParams) under cfg's Anthropic API
// version, or nil when that version still accepts it.
func AnthropicSamplingError(cfg Config, req Request) error {
	if !cfg.AnthropicStrictVersion() {
		return nil
	}
	if req.Temperature != nil {
		return errAnthropicSampling("Temperature")
	}
	for _, key := range []string{"temperature", "top_p", "top_k"} {
		if _, ok := req.SamplingParams[key]; ok {
			return errAnthropicSampling("SamplingParams[" + key + "]")
		}
	}
	return nil
}

// compatReasoningHosts are the OpenAI-compatible hosts with a documented
// reasoning control (the adapter's dialect table is provider/openaicompat
// compat.go; keep the two in step). A Thinking model there gets
// CapabilityReasoning by default (P2).
var compatReasoningHosts = map[string]bool{
	"deepseek": true, "moonshot": true, "zai": true, "dashscope": true,
	"openrouter": true, "together": true, "fireworks": true, "cerebras": true,
	"deepinfra": true, "groq": true, "nvidia": true, "vllm": true,
	"ollama": true, "lmstudio": true, "openai": true,
}
