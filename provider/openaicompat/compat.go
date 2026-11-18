package openaicompat

import (
	"fmt"
	"strings"

	llm "github.com/bds421/rho-llm"
)

// Per-host behaviour of the OpenAI-compatible Chat Completions dialect, kept as
// data (P3): hosts agree on the request shape but each spells reasoning
// differently. Values come from each host's own docs (2026-11-12) and pi-ai's
// compat detection (openai-completions.ts detectCompat).

// thinkingFormat is how a host takes a reasoning request.
type thinkingFormat int

const (
	// thinkingUnsupported: a ThinkingLevel is refused (no documented control).
	thinkingUnsupported thinkingFormat = iota
	// thinkingReasoningEffort: top-level "reasoning_effort": <level>.
	thinkingReasoningEffort
	// thinkingDeepSeek: "thinking": {"type": "enabled"} plus reasoning_effort.
	thinkingDeepSeek
	// thinkingKimi: Kimi K3 models take reasoning_effort only (thinking is
	// always on and the thinking parameter is not supported); older Kimi
	// models take "thinking": {"type": "enabled"}.
	thinkingKimi
	// thinkingZai: "thinking": {"type": "enabled"}.
	thinkingZai
	// thinkingQwen: top-level "enable_thinking": true (DashScope).
	thinkingQwen
	// thinkingOpenRouter: "reasoning": {"effort": <level>}.
	thinkingOpenRouter
	// thinkingTogether: "reasoning": {"enabled": true}.
	thinkingTogether
)

// reasoningFields are the response fields that carry reasoning text, in the
// order they are read, and the only names reasoning is replayed under.
var reasoningFields = []string{"reasoning_content", "reasoning"}

// documentFormat is how a host takes a PDF in a Chat Completions message.
type documentFormat int

const (
	// documentFile: {"type":"file","file":{"filename","file_data":<data URL>}},
	// the OpenAI Chat Completions shape (also OpenRouter). The default.
	documentFile documentFormat = iota
	// documentURL: {"type":"document_url","document_url":<data URL>} (Mistral).
	documentURL
	// documentUnsupported: the host's Chat Completions takes no files; the PDF
	// is refused before the request (xAI: "File content is not supported on
	// /v1/chat/completions. Please use /v1/responses instead.").
	documentUnsupported
)

// hostCompat is the per-host data the adapter consults.
type hostCompat struct {
	thinking thinkingFormat
	// replayReasoning: send reasoning from earlier assistant turns back under
	// the field it arrived in. Only hosts that document taking it back: a
	// strict host (Groq) rejects an unknown message field.
	replayReasoning bool
	// emptyReasoningOnAssistant: every replayed assistant message carries
	// reasoning_content, "" when it has none. DeepSeek answers a 400 when a
	// tool conversation does not pass reasoning_content back.
	emptyReasoningOnAssistant bool
	// documents: how a PDF part goes out. Every host rejects a PDF sent as an
	// image_url data URI (live 2026-11-18: OpenAI 400 invalid_image_format,
	// xAI 400 invalid_image, Mistral 422).
	documents documentFormat
}

// compatByProvider maps a preset (and its aliases) to its compat data.
var compatByProvider = map[string]hostCompat{
	"openai":       {thinking: thinkingReasoningEffort},
	"gpt":          {thinking: thinkingReasoningEffort},
	"deepseek":     {thinking: thinkingDeepSeek, replayReasoning: true, emptyReasoningOnAssistant: true},
	"moonshot":     {thinking: thinkingKimi, replayReasoning: true},
	"kimi":         {thinking: thinkingKimi, replayReasoning: true},
	"moonshot-cn":  {thinking: thinkingKimi, replayReasoning: true},
	"kimi-cn":      {thinking: thinkingKimi, replayReasoning: true},
	"zai":          {thinking: thinkingZai, replayReasoning: true},
	"z-ai":         {thinking: thinkingZai, replayReasoning: true},
	"glm":          {thinking: thinkingZai, replayReasoning: true},
	"dashscope":    {thinking: thinkingQwen, replayReasoning: true},
	"qwen":         {thinking: thinkingQwen, replayReasoning: true},
	"dashscope-cn": {thinking: thinkingQwen, replayReasoning: true},
	"qwen-cn":      {thinking: thinkingQwen, replayReasoning: true},
	"openrouter":   {thinking: thinkingOpenRouter, replayReasoning: true},
	"together":     {thinking: thinkingTogether, replayReasoning: true},
	"fireworks":    {thinking: thinkingReasoningEffort, replayReasoning: true},
	"cerebras":     {thinking: thinkingReasoningEffort},
	"deepinfra":    {thinking: thinkingReasoningEffort},
	"groq":         {thinking: thinkingReasoningEffort},
	"nvidia":       {thinking: thinkingReasoningEffort},
	"vllm":         {thinking: thinkingReasoningEffort},
	"ollama":       {thinking: thinkingReasoningEffort},
	"lmstudio":     {thinking: thinkingReasoningEffort},
	// MiniMax, Mistral, Cohere, xAI, Perplexity, Meta: no documented control
	// that maps onto ThinkingLevel here; the level is refused, not ignored.
	// MiniMax documents taking reasoning_content back.
	"minimax": {replayReasoning: true},
	"mistral": {documents: documentURL},
	"xai":     {documents: documentUnsupported},
	"grok":    {documents: documentUnsupported},
}

// compatByHost recognises a custom provider name by its BaseURL.
var compatByHost = []struct {
	host   string
	compat string
}{
	{"api.deepseek.com", "deepseek"},
	{"api.moonshot.", "moonshot"},
	{"api.kimi.", "moonshot"},
	{"api.z.ai", "zai"},
	{"open.bigmodel.cn", "zai"},
	{"dashscope", "dashscope"},
	{"openrouter.ai", "openrouter"},
	{"api.together.", "together"},
	{"fireworks.ai", "fireworks"},
	{"api.minimax.", "minimax"},
	{"api.openai.com", "openai"},
	{"api.mistral.ai", "mistral"},
	{"api.x.ai", "xai"},
}

// detectCompat returns the compat data for a provider name, falling back to
// the BaseURL for an unknown name, and to "no reasoning control" otherwise.
func detectCompat(provider, baseURL string) hostCompat {
	if c, ok := compatByProvider[strings.ToLower(provider)]; ok {
		return c
	}
	lower := strings.ToLower(baseURL)
	for _, h := range compatByHost {
		if strings.Contains(lower, h.host) {
			return compatByProvider[h.compat]
		}
	}
	return hostCompat{}
}

// applyThinking sets the host's reasoning fields for level on apiReq, or
// reports false when the host has no documented control.
func (h hostCompat) applyThinking(apiReq *openaiRequest, level llm.ThinkingLevel, model string) bool {
	effort := string(level)
	switch h.thinking {
	case thinkingReasoningEffort:
		apiReq.ReasoningEffort = effort
	case thinkingDeepSeek:
		apiReq.Thinking = &openaiThinking{Type: "enabled"}
		apiReq.ReasoningEffort = effort
	case thinkingKimi:
		if strings.HasPrefix(strings.ToLower(model), "kimi-k3") {
			apiReq.ReasoningEffort = effort
		} else {
			apiReq.Thinking = &openaiThinking{Type: "enabled"}
		}
	case thinkingZai:
		apiReq.Thinking = &openaiThinking{Type: "enabled"}
	case thinkingQwen:
		on := true
		apiReq.EnableThinking = &on
	case thinkingOpenRouter:
		apiReq.Reasoning = &openaiReasoning{Effort: effort}
	case thinkingTogether:
		on := true
		apiReq.Reasoning = &openaiReasoning{Enabled: &on}
	default:
		return false
	}
	return true
}

// replayReasoning copies the reasoning parts of an assistant message into the
// wire fields they arrived in. Reasoning without a recorded field (another
// provider's, or Mistral's in-content chunks) is not sent. With empty, a
// message without reasoning still carries reasoning_content "".
func replayReasoning(msg *openaiMessage, parts []llm.ContentPart, empty bool) {
	byField := map[string][]string{}
	for _, p := range parts {
		if p.Type != llm.ContentThinking || p.Thinking == "" {
			continue
		}
		for _, f := range reasoningFields {
			if p.ReasoningField == f {
				byField[f] = append(byField[f], p.Thinking)
			}
		}
	}
	if v, ok := byField["reasoning_content"]; ok {
		joined := strings.Join(v, "\n")
		msg.ReasoningContent = &joined
	}
	if v, ok := byField["reasoning"]; ok {
		joined := strings.Join(v, "\n")
		msg.Reasoning = &joined
	}
	if empty && msg.ReasoningContent == nil {
		blank := ""
		msg.ReasoningContent = &blank
	}
}

// documentPart encodes the i-th PDF of a message in the host's format.
func documentPart(format documentFormat, doc llm.ContentPart, i int) map[string]any {
	dataURI := fmt.Sprintf("data:%s;base64,%s", doc.Document.MediaType, doc.Document.Data)
	if format == documentURL {
		return map[string]any{"type": "document_url", "document_url": dataURI}
	}
	filename := doc.Title
	if filename == "" {
		filename = fmt.Sprintf("document-%d.pdf", i+1)
	}
	return map[string]any{
		"type": "file",
		"file": map[string]string{"filename": filename, "file_data": dataURI},
	}
}
