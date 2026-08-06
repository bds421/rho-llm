package llm

// ProviderPreset holds the default endpoint and auth configuration for a provider.
type ProviderPreset struct {
	BaseURL    string // Default API base URL
	AuthHeader string // Auth header prefix ("Bearer", "", etc.)
	Protocol   string // Wire protocol: "anthropic", "gemini", "openai_compat"
	// SupportsBatch advertises that the provider implements the OpenAI Files + Batches
	// REST API at BaseURL (NewBatchClient gates on it). Only true for first-party
	// OpenAI today — most openai_compat resellers do NOT expose /v1/batches.
	SupportsBatch bool
}

// presets maps provider names to their default configuration.
// Immutable after init — no mutex needed.
var presets = map[string]ProviderPreset{
	// Native protocols
	"anthropic": {BaseURL: "https://api.anthropic.com/v1", Protocol: "anthropic", SupportsBatch: true},
	"claude":    {BaseURL: "https://api.anthropic.com/v1", Protocol: "anthropic", SupportsBatch: true},
	"gemini":    {BaseURL: "https://generativelanguage.googleapis.com/v1beta/models", Protocol: "gemini", SupportsBatch: true},
	"google":    {BaseURL: "https://generativelanguage.googleapis.com/v1beta/models", Protocol: "gemini", SupportsBatch: true},

	// OpenAI Responses API (explicit provider selection)
	"openai_responses": {BaseURL: "https://api.openai.com/v1", AuthHeader: "Bearer", Protocol: "openai_responses", SupportsBatch: true},

	// OpenAI-compatible: cloud providers
	"openai": {BaseURL: "https://api.openai.com/v1", AuthHeader: "Bearer", Protocol: "openai_compat", SupportsBatch: true},
	"xai":    {BaseURL: "https://api.x.ai/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"grok":   {BaseURL: "https://api.x.ai/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	// Meta Model API (Muse Spark) — OpenAI-compatible surface at api.meta.ai.
	"meta":       {BaseURL: "https://api.meta.ai/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"groq":       {BaseURL: "https://api.groq.com/openai/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"cerebras":   {BaseURL: "https://api.cerebras.ai/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"mistral":    {BaseURL: "https://api.mistral.ai/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"openrouter": {BaseURL: "https://openrouter.ai/api/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	// DashScope / Qwen — intl default; dashscope-cn for mainland compatible-mode.
	"dashscope":    {BaseURL: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"qwen":         {BaseURL: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"dashscope-cn": {BaseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"qwen-cn":      {BaseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	// DeepSeek chat is OpenAI-compatible at the host root (paths are /chat/completions).
	// No first-party embeddings API — use another host for vectors.
	"deepseek":   {BaseURL: "https://api.deepseek.com", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"cohere":     {BaseURL: "https://api.cohere.ai/compatibility/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"together":   {BaseURL: "https://api.together.xyz/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"fireworks":  {BaseURL: "https://api.fireworks.ai/inference/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"nvidia":     {BaseURL: "https://integrate.api.nvidia.com/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"perplexity": {BaseURL: "https://api.perplexity.ai", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"deepinfra":  {BaseURL: "https://api.deepinfra.com/v1/openai", AuthHeader: "Bearer", Protocol: "openai_compat"},
	// Z.ai / GLM — OpenAI-compatible chat (+ tools/vision where the model admits it).
	"zai":  {BaseURL: "https://api.z.ai/api/openai/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"z-ai": {BaseURL: "https://api.z.ai/api/openai/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"glm":  {BaseURL: "https://api.z.ai/api/openai/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	// MiniMax global OpenAI-compatible chat. Proprietary speech/video APIs are not
	// registered as rho modalities (non-OpenAI wire).
	"minimax": {BaseURL: "https://api.minimax.io/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	// Moonshot / Kimi — global .ai and mainland .cn compatible bases.
	"moonshot":    {BaseURL: "https://api.moonshot.ai/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"kimi":        {BaseURL: "https://api.moonshot.ai/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"moonshot-cn": {BaseURL: "https://api.moonshot.cn/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},
	"kimi-cn":     {BaseURL: "https://api.moonshot.cn/v1", AuthHeader: "Bearer", Protocol: "openai_compat"},

	// Providers requiring auth that the preset model can't express are intentionally
	// NOT listed: Amazon Bedrock (AWS SigV4) and Google Vertex AI (GCP service-account
	// OAuth) need request signing outside this stdlib-only library — front them with a
	// gateway, or use the gemini protocol + a custom BaseURL/bearer token for Vertex.
	// Azure OpenAI and Cloudflare/Vercel gateways are OpenAI-compatible but
	// account/deployment-specific: configure them with Config.BaseURL (+ AuthHeader).

	// OpenAI-compatible: local providers (no auth)
	"ollama":   {BaseURL: "http://localhost:11434/v1", AuthHeader: "", Protocol: "openai_compat"},
	"vllm":     {BaseURL: "http://localhost:8000/v1", AuthHeader: "", Protocol: "openai_compat"},
	"lmstudio": {BaseURL: "http://localhost:1234/v1", AuthHeader: "", Protocol: "openai_compat"},
}

// noAuthProviders lists providers that do not require API keys.
var noAuthProviders = map[string]bool{
	"ollama":   true,
	"vllm":     true,
	"lmstudio": true,
}

// IsNoAuthProvider returns true if the provider does not require an API key.
func IsNoAuthProvider(provider string) bool {
	return noAuthProviders[provider]
}

// PresetFor returns the preset for a provider, or a zero value if not found.
func PresetFor(provider string) (ProviderPreset, bool) {
	p, ok := presets[provider]
	return p, ok
}

// ResolveProtocol determines the wire protocol for a Config.
// Known providers use their preset protocol; unknown providers default
// to openai_compat (the most common wire format).
//
// Auto-detection: when provider is "openai" and the model has ResponsesAPI: true
// in the registry, the protocol is automatically upgraded to "openai_responses".
// The Responses API is the proper endpoint for GPT-5 family models — it provides
// reasoning effort control and avoids wasting tokens on hidden reasoning.
// Users can also explicitly set Provider: "openai_responses".
func ResolveProtocol(cfg Config) string {
	// Explicit provider override
	if cfg.Provider == "openai_responses" {
		return "openai_responses"
	}

	// Auto-detect: openai provider + ResponsesAPI model → always use Responses API
	if cfg.Provider == "openai" {
		model := configuredModel(cfg)
		if info, ok := GetModelInfo(model); ok && info.ResponsesAPI {
			return "openai_responses"
		}
	}

	if preset, ok := presets[cfg.Provider]; ok {
		return preset.Protocol
	}
	return "openai_compat"
}

// ResolveBaseURL returns the effective base URL for a Config.
// Config.BaseURL takes precedence over the provider preset.
func ResolveBaseURL(cfg Config) string {
	if cfg.BaseURL != "" {
		return cfg.BaseURL
	}
	if preset, ok := presets[cfg.Provider]; ok {
		return preset.BaseURL
	}
	return ""
}

// ResolveAuthHeader returns the effective auth header for a Config.
// Config.AuthHeader takes precedence over the provider preset.
func ResolveAuthHeader(cfg Config) string {
	if cfg.AuthHeader != "" {
		return cfg.AuthHeader
	}
	if preset, ok := presets[cfg.Provider]; ok {
		return preset.AuthHeader
	}
	return "Bearer"
}
