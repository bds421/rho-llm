// Package llm provides a multi-provider LLM client with streaming, tool use,
// extended thinking, and auth pool rotation. Supports Anthropic, Google Gemini,
// and all OpenAI-compatible providers (xAI, OpenAI, Groq, Cerebras, Mistral,
// OpenRouter, Ollama, vLLM, LM Studio).
package llm

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"iter"
	"log/slog"
	"strings"
	"time"
	"unicode/utf8"
)

// redactSecret removes the literal API key from a free-text message (e.g. a
// provider error body that echoed the request key) so it can't leak into logs
// or serialized state. The library knows its own key, so this is an exact,
// reliable scrub — not heuristic pattern-matching. The rules, by key length:
//
//   - 8+ bytes: replaced wherever it occurs (substring), as in v0.9.4.
//   - 4–7 bytes (test, local or proxy keys): replaced only where it stands as
//     a whole token — not adjacent to another letter, digit, '-' or '_' — so it
//     is scrubbed from "Bearer abc1" or `"key":"abc1"` without mangling words
//     that merely contain it. Well-known placeholder values that local servers
//     accept in place of a key ("none", "EMPTY", "ollama", "dummy", …; see
//     isPlaceholderKey) are not secrets and are left alone, so "must be none"
//     is not rewritten to "must be REDACTED".
//   - 0–3 bytes: left alone. Such a key has no entropy worth protecting and
//     cannot be told apart from ordinary words ("a", "id", "key").
func redactSecret(msg, key string) string {
	if msg == "" || len(key) < minWholeTokenSecretLen {
		return msg
	}
	if len(key) >= 8 {
		return strings.ReplaceAll(msg, key, "REDACTED")
	}
	if isPlaceholderKey(key) {
		return msg
	}
	var b strings.Builder
	pos := 0 // msg[:pos] is already copied to b
	for from := 0; ; {
		i := strings.Index(msg[from:], key)
		if i < 0 {
			break
		}
		i += from
		end := i + len(key)
		if (i == 0 || !isTokenByte(msg[i-1])) && (end == len(msg) || !isTokenByte(msg[end])) {
			b.WriteString(msg[pos:i])
			b.WriteString("REDACTED")
			pos = end
			from = end // never match inside an already-redacted span
			continue
		}
		from = i + 1
	}
	if pos == 0 {
		return msg
	}
	b.WriteString(msg[pos:])
	return b.String()
}

// minWholeTokenSecretLen is the shortest key redactSecret scrubs.
const minWholeTokenSecretLen = 4

// placeholderKeys are values local/self-hosted servers (Ollama, vLLM, LM
// Studio, llama.cpp, …) accept in place of a real key. They are compared
// case-insensitively and only matter below 8 bytes; longer ones (e.g.
// "sk-no-key-required") keep the v0.9.4 substring scrub, which is harmless.
var placeholderKeys = map[string]struct{}{
	"none": {}, "empty": {}, "null": {}, "nil": {}, "ollama": {}, "dummy": {},
	"test": {}, "unused": {}, "nokey": {}, "no-key": {}, "apikey": {},
	"api-key": {}, "api_key": {}, "token": {}, "local": {}, "vllm": {},
	"lmstudio": {}, "sk-none": {}, "sk-xxx": {}, "sk-1234": {}, "changeme": {},
}

func isPlaceholderKey(key string) bool {
	_, ok := placeholderKeys[strings.ToLower(key)]
	return ok
}

func isTokenByte(c byte) bool {
	return c == '-' || c == '_' || c >= '0' && c <= '9' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z'
}

// =============================================================================
// NAMED STRING TYPES
// =============================================================================

// Role represents a message role.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleSystem    Role = "system"
)

// ContentType represents the type of a content part.
type ContentType string

const (
	ContentText       ContentType = "text"
	ContentImage      ContentType = "image"
	ContentDocument   ContentType = "document"
	ContentThinking   ContentType = "thinking"
	ContentToolUse    ContentType = "tool_use"
	ContentToolResult ContentType = "tool_result"

	// ContentServerToolUse is a call the provider ran itself (ServerTool);
	// ContentServerToolResult is its result, paired by ToolResultID. Both
	// carry ProviderBlock and are replayed only to the provider that made them.
	ContentServerToolUse    ContentType = "server_tool_use"
	ContentServerToolResult ContentType = "server_tool_result"
	// ContentProviderBlock is a provider content block the neutral model does
	// not describe, kept in ProviderBlock (BlockType names it) so a turn can be
	// sent back unchanged to the same provider.
	ContentProviderBlock ContentType = "provider_block"
)

// EventType represents the type of a streaming event.
type EventType string

const (
	EventContent  EventType = "content"
	EventToolUse  EventType = "tool_use"
	EventThinking EventType = "thinking"
	EventDone     EventType = "done"
	EventError    EventType = "error"

	// EventServerTool carries a server tool call (when its input is complete)
	// or result in Part. EventCitation carries one citation for the text
	// streamed so far in Citation.
	EventServerTool EventType = "server_tool"
	EventCitation   EventType = "citation"
)

// ThinkingLevel represents the extended thinking budget level.
type ThinkingLevel string

const (
	ThinkingNone    ThinkingLevel = ""
	ThinkingMinimal ThinkingLevel = "minimal" // OpenAI: minimal reasoning effort
	ThinkingLow     ThinkingLevel = "low"
	ThinkingMedium  ThinkingLevel = "medium"
	ThinkingHigh    ThinkingLevel = "high"
	ThinkingXHigh   ThinkingLevel = "xhigh" // OpenAI: maximum reasoning effort
	// ThinkingMax is the deepest level: Claude 4.7+ adaptive effort "max". Providers
	// without a "max" level get their highest one (OpenAI "xhigh"; Gemini and
	// budget-based Claude models a token budget clamped to the model limit).
	ThinkingMax ThinkingLevel = "max"
)

// Effort is a provider effort level (see Request.Effort).
type Effort string

// Effort levels.
const (
	EffortLow    Effort = "low"
	EffortMedium Effort = "medium"
	EffortHigh   Effort = "high"
	EffortXHigh  Effort = "xhigh"
	EffortMax    Effort = "max"
)

// Valid reports whether e is empty or one of the effort levels.
func (e Effort) Valid() bool {
	switch e {
	case "", EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax:
		return true
	}
	return false
}

// ThinkingBinding says what Anthropic does with a replayed thinking block
// whose conversation changed after the block was produced (an edited system
// prompt, tool set or earlier message). Newer Claude models bind each block to
// the conversation that produced it and answer a 400 for such a block.
type ThinkingBinding string

const (
	// ThinkingBindingDropBlock (the default) asks Anthropic to drop the stale
	// block and every thinking block after it, and answer without that
	// reasoning. Each drop is reported in Response.InputTransformations.
	ThinkingBindingDropBlock ThinkingBinding = "drop_block"
	// ThinkingBindingError keeps the 400, for tests that must catch an edit.
	ThinkingBindingError ThinkingBinding = "error"
	// ThinkingBindingOff sends no binding control: the account's default
	// applies, and no InputTransformations are reported.
	ThinkingBindingOff ThinkingBinding = "off"
)

// InputTransformation is one change the provider made to the request before
// the model read it (Anthropic input_transformations). Type is
// "thinking_dropped" or "thinking_mismatch_allowed"; Path indexes the
// messages as sent ("messages.3.content.0"); Reason is
// "prefix_binding_mismatch" (the history changed), "model_binding_mismatch"
// (another model wrote the block) or "organization_binding_mismatch" (another
// account did). Later API versions may add values.
type InputTransformation struct {
	Type   string `json:"type"`
	Path   string `json:"path,omitempty"`
	Reason string `json:"reason,omitempty"`
}

// ReasoningSummary controls whether reasoning summary text is included in responses.
type ReasoningSummary string

const (
	ReasoningSummaryNone     ReasoningSummary = ""         // omit from request
	ReasoningSummaryAuto     ReasoningSummary = "auto"     // provider decides
	ReasoningSummaryDetailed ReasoningSummary = "detailed" // detailed summary
	ReasoningSummaryConcise  ReasoningSummary = "concise"  // concise summary
	// ReasoningSummaryOmitted asks for no thinking text: Anthropic display
	// "omitted" (thinking blocks with empty text, billed the same); OpenAI
	// Responses sends no summary.
	ReasoningSummaryOmitted ReasoningSummary = "omitted"
)

// ThinkingBudgetTokens returns the default token budget for a thinking level.
// If customBudget > 0, it overrides the level default.
func ThinkingBudgetTokens(level ThinkingLevel, customBudget int) int {
	if customBudget > 0 {
		return customBudget
	}
	switch level {
	case ThinkingMinimal:
		return 1024
	case ThinkingLow:
		return 4096
	case ThinkingMedium:
		return 16384
	case ThinkingHigh:
		return 65536
	case ThinkingXHigh, ThinkingMax:
		return 128000
	default:
		return 0
	}
}

// ClampThinkingBudget clamps a thinking budget to the model's max output tokens.
// Returns the clamped value and logs a warning if clamping occurred.
func ClampThinkingBudget(provider, model string, budget, maxTokens int) int {
	if maxTokens > 0 && budget > maxTokens {
		slog.Debug("clamping thinking budget to model max_tokens",
			"provider", provider, "model", model,
			"requested", budget, "max", maxTokens)
		return maxTokens
	}
	return budget
}

// TokensNotReported is the sentinel value for token counts when the provider
// did not report usage (e.g. stream ended before usage chunk arrived).
// Callers can distinguish "not reported" (-1) from "zero tokens" (0).
const TokensNotReported = -1

// Stop reasons — the unified vocabulary adapters normalize provider-native
// values into (Gemini "STOP", OpenAI "length", Anthropic "stop_sequence", …).
// Compare Response.StopReason / StreamEvent.StopReason against these constants;
// provider-specific reasons with no unified equivalent pass through verbatim.
const (
	StopEndTurn   = "end_turn"   // the model finished its turn normally
	StopToolUse   = "tool_use"   // the model is waiting for tool results
	StopMaxTokens = "max_tokens" // output truncated at the token limit
	StopError     = "error"      // the turn failed — NormalizeForProvider drops it on replay
	StopAborted   = "aborted"    // the turn was aborted by the caller — dropped on replay
	// StopPauseTurn: the provider paused a long server-tool turn. Send the
	// paused assistant message back unchanged as the next request, with the
	// same tools, to let it continue (ContinuePausedTurn, Session.Continue).
	StopPauseTurn = "pause_turn"
	// StopRefusal: the provider's safety classifiers declined the request.
	StopRefusal = "refusal"
)

// =============================================================================
// MESSAGE TYPES
// =============================================================================

// Message represents a conversation message.
type Message struct {
	Role    Role          `json:"role"`    // user, assistant, system
	Content []ContentPart `json:"content"` // Content parts (text, images, tool results)

	// Provenance — set on assistant messages so a stored conversation knows which
	// provider/model produced each turn. This drives cross-provider handoff
	// decisions (e.g. "same provider → replay thinking signatures verbatim").
	// All omitempty, so older serialized messages remain valid.
	Provider   string `json:"provider,omitempty"`    // producing provider (e.g. "anthropic")
	Model      string `json:"model,omitempty"`       // producing model id
	StopReason string `json:"stop_reason,omitempty"` // end_turn, tool_use, max_tokens, error
	// ContainerID is the provider container the turn ran in (Anthropic code
	// execution, also behind dynamic web search filtering). A pause_turn
	// continuation sends it back so the turn keeps its container.
	ContainerID string `json:"container_id,omitempty"`
}

// ContentPart represents a part of message content.
type ContentPart struct {
	Type ContentType `json:"type"` // text, image, tool_use, tool_result

	// Text content
	Text string `json:"text,omitempty"`

	// Image content
	Source *ImageSource `json:"source,omitempty"`

	// Document content: a base64 PDF, or plain text (Anthropic). Title and
	// Context are optional and passed to the model alongside the document
	// (Anthropic); other providers ignore them.
	Document *DocumentSource `json:"document,omitempty"`
	Title    string          `json:"title,omitempty"`
	Context  string          `json:"context,omitempty"`

	// Thinking / extended-reasoning content (from assistant). Carried in the
	// neutral model so a conversation can replay it to the SAME provider on the
	// next turn (Anthropic requires the thinking block + signature to be returned
	// when extended thinking is active). ThinkingSignature is the provider-opaque
	// signature that authenticates the block for replay; Redacted marks an
	// encrypted (redacted_thinking) block. On a cross-provider handoff a thinking
	// block without a matching signature is dropped — see NormalizeForProvider.
	Thinking          string `json:"thinking,omitempty"`
	ThinkingSignature string `json:"thinking_signature,omitempty"`
	Redacted          bool   `json:"redacted,omitempty"`

	// Tool use (from assistant)
	ToolUseID        string `json:"id,omitempty"`
	ToolName         string `json:"name,omitempty"`
	ToolInput        any    `json:"input,omitempty"`
	ThoughtSignature string `json:"thought_signature,omitempty"` // Gemini 3: preserved for tool result round-trip

	// ReasoningField (thinking parts, openai_compat) is the response field
	// the reasoning arrived in ("reasoning_content" or "reasoning"). A host
	// that takes reasoning back (DeepSeek, Kimi, Qwen, …) gets it on later
	// turns under the same name; normalization keeps such a part for the
	// same provider instead of dropping it as unsigned.
	ReasoningField string `json:"reasoning_field,omitempty"`

	// Tool result (from user)
	ToolResultID      string `json:"tool_use_id,omitempty"`
	ToolResultContent string `json:"content,omitempty"`
	IsError           bool   `json:"is_error,omitempty"`

	// ToolResultParts carries a rich tool result — text and/or image content
	// blocks — for tools that return images. Anthropic serializes these as a
	// native content-block array; providers without image-tool-result support
	// receive only the text (see ToolResultText). When set, it takes precedence
	// over ToolResultContent.
	ToolResultParts []ContentPart `json:"tool_result_parts,omitempty"`

	// Caching (Anthropic): mark this content block as cacheable
	CacheControl bool `json:"cache_control,omitempty"`
	// CacheTTL is the cache lifetime when CacheControl is set: CacheTTL1h, or
	// empty / CacheTTL5m for the default five minutes.
	CacheTTL string `json:"cache_ttl,omitempty"`

	// Citations on a text part (Anthropic: web search, documents, search
	// results). Replayed to the producing provider, stripped for others.
	Citations []Citation `json:"citations,omitempty"`

	// Server tools and provider blocks. A server tool call uses ToolUseID,
	// ToolName and ToolInput; its result uses ToolResultID, BlockType (the
	// provider's block type, e.g. "web_search_tool_result") and, for web
	// search, WebSearchResults or ServerToolError.
	BlockType        string            `json:"block_type,omitempty"`
	WebSearchResults []WebSearchResult `json:"web_search_results,omitempty"`
	ServerToolError  *ServerToolError  `json:"server_tool_error,omitempty"`

	// ProviderBlock is the block exactly as the provider sent it (compact
	// JSON). Adapters replay it verbatim to the same provider, so opaque
	// fields such as encrypted_content survive unchanged; the typed fields
	// above are for reading. Set on ContentServerToolUse,
	// ContentServerToolResult and ContentProviderBlock parts.
	ProviderBlock json.RawMessage `json:"provider_block,omitempty"`
}

// ImageSource represents an image source.
type ImageSource struct {
	Type      string `json:"type"`       // base64
	MediaType string `json:"media_type"` // image/jpeg, image/png, etc.
	Data      string `json:"data"`       // base64 encoded data
}

// validImageMediaTypes lists the MIME types accepted for image content.
var validImageMediaTypes = map[string]bool{
	"image/jpeg": true,
	"image/png":  true,
	"image/gif":  true,
	"image/webp": true,
}

// ValidateImageSource checks that a ContentImage part has a valid, non-empty
// source with a supported media type. Called by adapter buildRequest methods
// before serializing image content to the wire format.
func ValidateImageSource(part ContentPart) error {
	if part.Source == nil {
		return fmt.Errorf("image source must not be nil")
	}
	if part.Source.Data == "" {
		return fmt.Errorf("image data must not be empty")
	}
	if !validImageMediaTypes[part.Source.MediaType] {
		return fmt.Errorf("unsupported image media type: %s", part.Source.MediaType)
	}
	if part.Source.Type != "base64" {
		return fmt.Errorf("unsupported image source type: %s", part.Source.Type)
	}
	decoded, err := decodeCanonicalBase64(part.Source.Data)
	if err != nil {
		return fmt.Errorf("invalid image base64 data: %w", err)
	}
	if actual := imageMediaTypeFromSignature(decoded); actual == "" {
		return fmt.Errorf("image data has no supported signature")
	} else if actual != part.Source.MediaType {
		return fmt.Errorf("image data media type %s does not match declared %s", actual, part.Source.MediaType)
	}
	return nil
}

// NewImageMessage creates a single-image message (parallel to NewTextMessage).
func NewImageMessage(role Role, mediaType, base64Data string) Message {
	return Message{
		Role: role,
		Content: []ContentPart{
			{
				Type: ContentImage,
				Source: &ImageSource{
					Type:      "base64",
					MediaType: mediaType,
					Data:      base64Data,
				},
			},
		},
	}
}

// DocumentSource represents an inline document: a base64 PDF
// (DocumentSourceBase64, application/pdf) or plain text (DocumentSourceText,
// text/plain, Data holds the text itself; Anthropic only).
type DocumentSource struct {
	Type      string `json:"type"`       // DocumentSourceBase64 or DocumentSourceText
	MediaType string `json:"media_type"` // application/pdf or text/plain
	Data      string `json:"data"`       // base64 PDF bytes, or the text
}

// Document source types.
const (
	DocumentSourceBase64 = "base64"
	DocumentSourceText   = "text"
)

// IsTextDocument reports whether part is a plain-text document.
func IsTextDocument(part ContentPart) bool {
	return part.Type == ContentDocument && part.Document != nil && part.Document.Type == DocumentSourceText
}

// NewTextDocument returns a plain-text document part. title may be empty.
func NewTextDocument(title, text string) ContentPart {
	return ContentPart{
		Type:     ContentDocument,
		Document: &DocumentSource{Type: DocumentSourceText, MediaType: "text/plain", Data: text},
		Title:    title,
	}
}

// validDocumentMediaTypes lists the MIME types accepted for document content.
// Gemini and Anthropic natively parse PDFs; OpenAI-compatible providers receive
// them as image_url data URIs (vision models).
var validDocumentMediaTypes = map[string]bool{
	"application/pdf": true,
}

// ValidateDocumentSource checks that a ContentDocument part has a valid,
// non-empty document with a supported media type. Called by adapter
// buildRequest methods before serializing document content to the wire format.
func ValidateDocumentSource(part ContentPart) error {
	if part.Document == nil {
		return fmt.Errorf("document source must not be nil")
	}
	if part.Document.Type == DocumentSourceText {
		return validateTextDocument(part.Document)
	}
	if part.Document.Data == "" {
		return fmt.Errorf("document data must not be empty")
	}
	if !validDocumentMediaTypes[part.Document.MediaType] {
		return fmt.Errorf("unsupported document media type: %s", part.Document.MediaType)
	}
	if part.Document.Type != "base64" {
		return fmt.Errorf("unsupported document source type: %s", part.Document.Type)
	}
	decoded, err := decodeCanonicalBase64(part.Document.Data)
	if err != nil {
		return fmt.Errorf("invalid document base64 data: %w", err)
	}
	if !bytes.HasPrefix(decoded, []byte("%PDF-")) {
		return fmt.Errorf("document data does not have a PDF signature")
	}
	return nil
}

func decodeCanonicalBase64(encoded string) ([]byte, error) {
	decoded, err := base64.StdEncoding.Strict().DecodeString(encoded)
	if err != nil {
		return nil, err
	}
	if base64.StdEncoding.EncodeToString(decoded) != encoded {
		return nil, fmt.Errorf("encoding is not canonical RFC 4648 base64")
	}
	return decoded, nil
}

func imageMediaTypeFromSignature(data []byte) string {
	switch {
	case len(data) >= 8 && bytes.Equal(data[:8], []byte("\x89PNG\r\n\x1a\n")):
		return "image/png"
	case len(data) >= 3 && data[0] == 0xff && data[1] == 0xd8 && data[2] == 0xff:
		return "image/jpeg"
	case len(data) >= 6 && (bytes.Equal(data[:6], []byte("GIF87a")) || bytes.Equal(data[:6], []byte("GIF89a"))):
		return "image/gif"
	case len(data) >= 12 && bytes.Equal(data[:4], []byte("RIFF")) && bytes.Equal(data[8:12], []byte("WEBP")):
		return "image/webp"
	default:
		return ""
	}
}

// NewDocumentMessage creates a single-document message (parallel to NewImageMessage).
func NewDocumentMessage(role Role, mediaType, base64Data string) Message {
	return Message{
		Role: role,
		Content: []ContentPart{
			{
				Type: ContentDocument,
				Document: &DocumentSource{
					Type:      "base64",
					MediaType: mediaType,
					Data:      base64Data,
				},
			},
		},
	}
}

// NewTextMessage creates a simple text message.
func NewTextMessage(role Role, text string) Message {
	return Message{
		Role: role,
		Content: []ContentPart{
			{Type: ContentText, Text: text},
		},
	}
}

// NewToolResultMessage creates a tool result message.
func NewToolResultMessage(toolUseID, result string, isError bool) Message {
	return Message{
		Role: RoleUser,
		Content: []ContentPart{
			{
				Type:              ContentToolResult,
				ToolResultID:      toolUseID,
				ToolResultContent: result,
				IsError:           isError,
			},
		},
	}
}

// NewToolResultParts creates a tool-result message whose content is a list of
// parts (text and/or image), for tools that return images. Providers that don't
// support image tool results (everything except Anthropic) receive only the text.
func NewToolResultParts(toolUseID string, isError bool, parts ...ContentPart) Message {
	return Message{
		Role: RoleUser,
		Content: []ContentPart{
			{
				Type:            ContentToolResult,
				ToolResultID:    toolUseID,
				ToolResultParts: parts,
				IsError:         isError,
			},
		},
	}
}

// ToolResultText returns the textual content of a tool-result part: ToolResultContent
// if set, otherwise the concatenated text of ToolResultParts (images dropped).
// Adapters without image-tool-result support call this to degrade gracefully.
func (p ContentPart) ToolResultText() string {
	if p.ToolResultContent != "" {
		return p.ToolResultContent
	}
	out := ""
	for _, sub := range p.ToolResultParts {
		if sub.Type == ContentText && sub.Text != "" {
			if out != "" {
				out += "\n"
			}
			out += sub.Text
		}
	}
	return out
}

// NewSystemMessage creates a system message with text content.
func NewSystemMessage(text string) Message {
	return Message{
		Role: RoleSystem,
		Content: []ContentPart{
			{Type: ContentText, Text: text},
		},
	}
}

// NewAssistantMessage creates an assistant message from a Response, preserving
// both text content and tool_use blocks. Use this instead of NewTextMessage
// when the response contains tool calls — NewTextMessage would lose them.
// When resp.Parts is set it is the message content as is (provider order,
// server tool blocks and citations included), and the flat fields are not
// consulted. Panics if resp is nil (caller bug — matches Go conventions for nil receivers).
func NewAssistantMessage(resp *Response) Message {
	if resp == nil {
		panic("llm.NewAssistantMessage: resp must not be nil")
	}
	msg := Message{
		Role:        RoleAssistant,
		Model:       resp.Model,
		StopReason:  resp.StopReason,
		ContainerID: resp.ContainerID,
	}
	if len(resp.Parts) > 0 {
		msg.Content = append([]ContentPart(nil), resp.Parts...)
		return msg
	}
	// Thinking first — Anthropic requires the thinking block to precede tool_use
	// when extended thinking is active, and to carry its signature for replay.
	if resp.Thinking != "" || resp.ThinkingRedacted {
		msg.Content = append(msg.Content, ContentPart{
			Type:              ContentThinking,
			Thinking:          resp.Thinking,
			ThinkingSignature: resp.ThinkingSignature,
			Redacted:          resp.ThinkingRedacted,
			ReasoningField:    resp.ReasoningField,
		})
	}
	if resp.Content != "" {
		msg.Content = append(msg.Content, ContentPart{
			Type: ContentText,
			Text: resp.Content,
		})
	}
	for _, tc := range resp.ToolCalls {
		msg.Content = append(msg.Content, ContentPart{
			Type:             ContentToolUse,
			ToolUseID:        tc.ID,
			ToolName:         tc.Name,
			ToolInput:        tc.Input,
			ThoughtSignature: tc.ThoughtSignature,
		})
	}
	return msg
}

// =============================================================================
// TOOL TYPES
// =============================================================================

// Tool represents a tool/function the LLM can call.
type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`

	// Caching (Anthropic): mark this tool definition as cacheable
	CacheControl bool `json:"cache_control,omitempty"`
	// CacheTTL: CacheTTL1h, or empty for the default five minutes.
	CacheTTL string `json:"cache_ttl,omitempty"`
}

// ToolChoiceMode selects how the model may use the tools in a Request.
type ToolChoiceMode string

const (
	// ToolChoiceAuto lets the model decide whether to call a tool. This is every
	// provider's default, so a nil *ToolChoice and ToolChoiceAuto behave alike.
	ToolChoiceAuto ToolChoiceMode = "auto"
	// ToolChoiceNone forbids tool calls; the model must answer with text.
	ToolChoiceNone ToolChoiceMode = "none"
	// ToolChoiceRequired forces the model to call some tool of its choosing.
	ToolChoiceRequired ToolChoiceMode = "required"
	// ToolChoiceTool forces the model to call one named tool (see ToolChoice.Name).
	ToolChoiceTool ToolChoiceMode = "tool"
)

// ToolChoice constrains which tool (if any) the model may call. Nil means
// "provider default", which is Auto everywhere.
//
// Each adapter translates this to its own wire shape — OpenAI's "required" vs
// Anthropic's {"type":"any"} vs Gemini's function_calling_config.mode "ANY" —
// so callers express the intent once and it travels across a provider handoff.
//
// A ToolChoice that constrains tool use (anything but Auto) is only meaningful
// alongside Tools; adapters drop it when Tools is empty rather than sending a
// constraint the provider would reject.
type ToolChoice struct {
	Mode ToolChoiceMode `json:"mode"`
	// Name is the tool to force, required when Mode is ToolChoiceTool and
	// ignored otherwise.
	Name string `json:"name,omitempty"`
}

// ToolChoiceAutoChoice, ToolChoiceNoneChoice and ToolChoiceRequiredChoice are
// ready-made values for the modes that take no name.
func NewToolChoice(mode ToolChoiceMode) *ToolChoice { return &ToolChoice{Mode: mode} }

// ForceTool returns a ToolChoice that forces a call to the named tool.
func ForceTool(name string) *ToolChoice {
	return &ToolChoice{Mode: ToolChoiceTool, Name: name}
}

// Validate reports whether the ToolChoice is coherent. A nil receiver is valid
// (it means "provider default"). Adapters call this before building a request so
// a malformed choice fails loudly here instead of as an opaque provider 400.
func (tc *ToolChoice) Validate() error {
	if tc == nil {
		return nil
	}
	switch tc.Mode {
	case ToolChoiceAuto, ToolChoiceNone, ToolChoiceRequired:
		if tc.Name != "" {
			return fmt.Errorf("llm: ToolChoice.Name is only valid with Mode %q, got Mode %q", ToolChoiceTool, tc.Mode)
		}
		return nil
	case ToolChoiceTool:
		if tc.Name == "" {
			return fmt.Errorf("llm: ToolChoice.Mode %q requires a tool Name", ToolChoiceTool)
		}
		return nil
	case "":
		return fmt.Errorf("llm: ToolChoice.Mode is empty (use %q, %q, %q or %q)",
			ToolChoiceAuto, ToolChoiceNone, ToolChoiceRequired, ToolChoiceTool)
	default:
		return fmt.Errorf("llm: unknown ToolChoice.Mode %q", tc.Mode)
	}
}

// ToolCall represents a tool invocation by the LLM.
type ToolCall struct {
	ID               string `json:"id"`
	Name             string `json:"name"`
	Input            any    `json:"input"`
	ThoughtSignature string `json:"thought_signature,omitempty"` // Gemini 3: round-trip through tool results
}

// =============================================================================
// REQUEST/RESPONSE TYPES
// =============================================================================

// Request represents an LLM completion request.
type Request struct {
	Model            string           `json:"model"`
	Messages         []Message        `json:"messages"`
	System           string           `json:"system,omitempty"`
	MaxTokens        int              `json:"max_tokens"`
	Temperature      *float64         `json:"temperature,omitempty"`
	Tools            []Tool           `json:"tools,omitempty"`
	ToolChoice       *ToolChoice      `json:"tool_choice,omitempty"`       // constrain tool use; nil = provider default (auto)
	ThinkingLevel    ThinkingLevel    `json:"thinking_level,omitempty"`    // low, medium, high (zero value = none)
	ThinkingBudget   int              `json:"thinking_budget,omitempty"`   // custom token budget; overrides ThinkingLevel default when > 0
	ReasoningSummary ReasoningSummary `json:"reasoning_summary,omitempty"` // OpenAI Responses: auto, detailed, concise. Anthropic adaptive thinking: unset/detailed/concise = display "summarized", omitted = "omitted", auto = the API default
	// Effort sets the provider's effort without asking for thinking: Anthropic
	// output_config.effort, OpenAI Responses reasoning.effort. With a
	// ThinkingLevel on a model where the level already sets the effort, the two
	// must agree. Adapters without an effort control refuse it.
	Effort        Effort   `json:"effort,omitempty"`
	StopSequences []string `json:"stop_sequences,omitempty"`

	// ServerTools are tools the provider runs itself (Anthropic web search).
	// They are sent after Tools. Adapters without server tools return an
	// error when any is set.
	ServerTools []ServerTool `json:"server_tools,omitempty"`
	// ContainerID reuses a provider container (Anthropic "container").
	ContainerID string `json:"container_id,omitempty"`

	// Caching
	SystemCacheControl bool   `json:"system_cache_control,omitempty"` // Anthropic: cache the system prompt
	SystemCacheTTL     string `json:"system_cache_ttl,omitempty"`     // Anthropic: CacheTTL1h, or empty for 5 minutes
	CachedContent      string `json:"cached_content,omitempty"`       // Gemini: pre-created cache name

	// ResponseFormat requests structured output (JSON mode / JSON schema). Wired
	// for OpenAI-compatible providers (response_format), OpenAI Responses
	// (text.format), Gemini (responseMimeType/responseSchema) and Anthropic
	// (output_config.format, json_schema only: json_object is refused). Nil =
	// free text.
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`

	// SamplingParams passes provider-specific request-body fields straight through
	// to the wire, for knobs the neutral Request does not model: OpenRouter's
	// "provider" routing object, vLLM's "priority", llama.cpp/SGLang thinking
	// budgets, "top_p"/"top_k"/"seed", and so on.
	//
	// Keys are merged into the adapter's JSON body at the TOP level, after the
	// adapter has built it. Merging is deliberately restricted: a key that the
	// adapter already set — the structural fields that define the request, such as
	// "model", "messages" and "tools" — is REJECTED with an error rather than
	// silently overriding it, so a stray key cannot redirect a request to another
	// model or rewrite its history. Use the typed Request fields for those.
	//
	// Values must marshal to JSON. This is an escape hatch, not the neutral
	// currency: anything expressible as a typed Request field belongs there.
	//
	// Deprecated for Anthropic effort: SamplingParams["output_config"] still
	// passes through when the adapter builds no output_config, but use the typed
	// Effort. On Anthropic API version 2026-10-21 "temperature", "top_p" and
	// "top_k" are refused.
	SamplingParams map[string]any `json:"sampling_params,omitempty"`
}

// ResponseFormat selects structured output for a Request.
type ResponseFormat struct {
	// Type is "json_object" (any JSON) or "json_schema" (constrained to Schema).
	Type string `json:"type"`
	// Name is the schema name (required by OpenAI for json_schema).
	Name string `json:"name,omitempty"`
	// Schema is a JSON Schema, used when Type == "json_schema".
	Schema map[string]any `json:"schema,omitempty"`
}

// Structured-output type constants.
const (
	ResponseFormatJSONObject = "json_object"
	ResponseFormatJSONSchema = "json_schema"
)

// Response represents an LLM completion response.
type Response struct {
	ID                string     `json:"id"`
	Model             string     `json:"model"`
	Content           string     `json:"content"`                      // Extracted text content
	ToolCalls         []ToolCall `json:"tool_calls"`                   // Tool use requests
	Thinking          string     `json:"thinking"`                     // Extended thinking content
	ThinkingSignature string     `json:"thinking_signature,omitempty"` // Anthropic: authenticates the thinking block for replay on the next turn
	ThinkingRedacted  bool       `json:"thinking_redacted,omitempty"`  // Anthropic: thinking block is redacted (encrypted)
	ReasoningField    string     `json:"reasoning_field,omitempty"`    // openai_compat: the field Thinking arrived in (see ContentPart.ReasoningField)
	StopReason        string     `json:"stop_reason"`                  // end_turn, tool_use, max_tokens
	RawStopReason     string     `json:"raw_stop_reason,omitempty"`    // the provider's own stop reason before normalization (e.g. "STOP", "length", "stop_sequence")
	// InputTokens is the UNCACHED input only. Cache-token contract (identical
	// for every adapter): InputTokens, CacheReadTokens and CacheCreationTokens
	// are disjoint, and the full prompt is their sum. Anthropic reports them
	// that way natively; the Gemini adapter subtracts cachedContentTokenCount
	// from promptTokenCount, and the OpenAI-style adapters subtract
	// cached_tokens from prompt_tokens / input_tokens (both totals include the
	// cached share). EstimateCost relies on this to price each token once.
	InputTokens    int `json:"input_tokens"`
	OutputTokens   int `json:"output_tokens"`
	ThinkingTokens int `json:"thinking_tokens,omitempty"` // Gemini, openai_responses: reasoning tokens, separate from OutputTokens (bill OutputTokens+ThinkingTokens as output)

	// Cache token usage — disjoint from InputTokens (see above).
	CacheCreationTokens int `json:"cache_creation_input_tokens,omitempty"` // tokens written to cache (Anthropic)
	CacheReadTokens     int `json:"cache_read_input_tokens,omitempty"`     // tokens read from cache (Anthropic, Gemini, OpenAI-style)
	// CacheCreation1hTokens is the share of CacheCreationTokens written with
	// the 1-hour TTL (Anthropic usage.cache_creation), priced at the 1-hour
	// write rate; the rest of CacheCreationTokens is 5-minute writes.
	CacheCreation1hTokens int `json:"cache_creation_1h_input_tokens,omitempty"`

	// StopDetails explains a refusal (Anthropic stop_details); nil otherwise.
	StopDetails *StopDetails `json:"stop_details,omitempty"`

	// ServerToolUse counts billed server tool calls (Anthropic
	// usage.server_tool_use). Iterations is Anthropic's usage.iterations, the
	// per-attempt billing record when present (see UsageIteration).
	ServerToolUse ServerToolUsage  `json:"server_tool_use,omitzero"`
	Iterations    []UsageIteration `json:"iterations,omitempty"`

	// Parts is the assistant content in provider order, one part per
	// provider block (text with citations, thinking with its own signature,
	// tool and server tool calls, server tool results, unknown blocks). Set by
	// adapters whose content the flat fields above cannot carry faithfully
	// (Anthropic); NewAssistantMessage uses it when set, so a turn — a paused
	// one included — goes back to the provider unchanged.
	Parts []ContentPart `json:"parts,omitempty"`

	// ContainerID is the provider container the response ran in, if any.
	ContainerID string `json:"container_id,omitempty"`

	// InputTransformations lists what the provider changed in the request
	// before the model read it: replayed thinking blocks dropped or let
	// through (see ThinkingBinding). Empty when nothing changed.
	InputTransformations []InputTransformation `json:"input_transformations,omitempty"`
}

// =============================================================================
// STREAMING TYPES
// =============================================================================

// StreamEvent represents a streaming event.
type StreamEvent struct {
	Type EventType `json:"type"` // content, tool_use, thinking, usage, done, error

	// Content event
	Text string `json:"text,omitempty"`

	// Tool use event
	ToolCall *ToolCall `json:"tool_call,omitempty"`

	// Thinking event
	Thinking string `json:"thinking,omitempty"`

	// Thinking signature — emitted on EventDone (Anthropic) so a streamed
	// assistant turn can be reassembled with a replayable thinking block.
	ThinkingSignature string `json:"thinking_signature,omitempty"`
	ThinkingRedacted  bool   `json:"thinking_redacted,omitempty"`
	// ReasoningField (EventDone, openai_compat): the field the streamed
	// reasoning arrived in, as on Response.
	ReasoningField string `json:"reasoning_field,omitempty"`

	// Usage event. InputTokens is the uncached input only — same disjoint
	// cache-token contract as Response.InputTokens.
	InputTokens    int `json:"input_tokens,omitempty"`
	OutputTokens   int `json:"output_tokens,omitempty"`
	ThinkingTokens int `json:"thinking_tokens,omitempty"` // Gemini, openai_responses: reasoning tokens, separate from OutputTokens

	// Done event: the model that answered (Anthropic message_start; with a
	// refusal fallback, the fallback model).
	Model string `json:"model,omitempty"`

	// Done event
	StopReason    string `json:"stop_reason,omitempty"`
	RawStopReason string `json:"raw_stop_reason,omitempty"` // the provider's own stop reason before normalization

	// Cache token usage (reported in EventDone), disjoint from InputTokens.
	// CacheReadTokens: Anthropic, Gemini, OpenAI-style; CacheCreationTokens: Anthropic.
	CacheCreationTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadTokens     int `json:"cache_read_input_tokens,omitempty"`

	// Done event: the 1-hour share of CacheCreationTokens and refusal details,
	// as on Response.
	CacheCreation1hTokens int          `json:"cache_creation_1h_input_tokens,omitempty"`
	StopDetails           *StopDetails `json:"stop_details,omitempty"`

	// Done event: request changes the provider reported, as on Response.
	InputTransformations []InputTransformation `json:"input_transformations,omitempty"`

	// Done event: server tool usage, usage iterations, the assembled
	// assistant content and the container, as on Response.
	ServerToolUse ServerToolUsage  `json:"server_tool_use,omitzero"`
	Iterations    []UsageIteration `json:"iterations,omitempty"`
	Parts         []ContentPart    `json:"parts,omitempty"`
	ContainerID   string           `json:"container_id,omitempty"`

	// EventServerTool: the server tool call or result.
	Part *ContentPart `json:"part,omitempty"`
	// EventCitation: a citation on the current text block.
	Citation *Citation `json:"citation,omitempty"`

	// Error event
	Error string `json:"error,omitempty"`
}

// =============================================================================
// CLIENT INTERFACE
// =============================================================================

// Client defines the interface for LLM providers.
type Client interface {
	// Complete generates a completion (non-streaming).
	Complete(ctx context.Context, req Request) (*Response, error)

	// Requires Go 1.23+ for iter.Seq2
	// Stream returns an iterator of streaming events.
	// The caller controls the stream via for-range; breaking stops iteration
	// and cleans up the underlying HTTP connection.
	//
	//   for event, err := range client.Stream(ctx, req) {
	//       if err != nil { break }
	//       fmt.Print(event.Text)
	//   }
	Stream(ctx context.Context, req Request) iter.Seq2[StreamEvent, error]

	// Provider returns the provider name.
	Provider() string

	// Model returns the default model name.
	Model() string

	// Close releases resources.
	Close() error
}

// =============================================================================
// AUTH PROFILE (failover pattern)
// =============================================================================

// AuthProfile represents an authentication profile for rotation.
type AuthProfile struct {
	Name      string    `json:"name"`
	APIKey    string    `json:"api_key"`
	BaseURL   string    `json:"base_url,omitempty"`
	IsHealthy bool      `json:"is_healthy"`
	LastUsed  time.Time `json:"last_used"` // Last time this profile was attempted (not necessarily successful)
	LastError string    `json:"last_error,omitempty"`
	Cooldown  time.Time `json:"cooldown_until,omitempty"`
}

// IsAvailable checks if the profile is available for use.
func (p *AuthProfile) IsAvailable() bool {
	if !p.IsHealthy {
		return false
	}
	if !p.Cooldown.IsZero() && time.Now().Before(p.Cooldown) {
		return false
	}
	return true
}

// MarkUsed marks the profile as used.
func (p *AuthProfile) MarkUsed() {
	p.LastUsed = time.Now()
}

// MarkFailed marks the profile as failed with cooldown. The error text is
// scrubbed of this profile's own API key AND any credentials embedded in a
// per-key BaseURL override first, so a server response that echoed either can't
// lodge it in LastError (which is logged and serialized).
func (p *AuthProfile) MarkFailed(err error, cooldownDuration time.Duration) {
	if err != nil {
		p.LastError = redactProfileSecrets(err.Error(), p.APIKey, p.BaseURL)
	}
	p.Cooldown = time.Now().Add(cooldownDuration)
}

// MarkHealthy marks the profile as healthy.
func (p *AuthProfile) MarkHealthy() {
	p.IsHealthy = true
	p.LastError = ""
	p.Cooldown = time.Time{}
}

// MarshalJSON implements json.Marshaler. Redacts APIKey to prevent accidental
// secret leakage when AuthProfile is serialized for logging or debugging — and
// scrubs the same key out of LastError (a free-text field that may have captured
// a server response echoing the key) before redacting APIKey itself.
func (p AuthProfile) MarshalJSON() ([]byte, error) {
	return json.Marshal(p.redacted())
}

// redactedProfile is AuthProfile without its methods (see redactedConfig).
type redactedProfile AuthProfile

// redacted returns p with APIKey, BaseURL credentials and any echo of them in
// LastError scrubbed.
func (p AuthProfile) redacted() redactedProfile {
	tmp := redactedProfile(p)
	tmp.LastError = redactProfileSecrets(tmp.LastError, tmp.APIKey, tmp.BaseURL)
	if tmp.APIKey != "" {
		tmp.APIKey = "REDACTED"
	}
	tmp.BaseURL = redactURLCredentials(tmp.BaseURL)
	return tmp
}

// String redacts the key, as MarshalJSON does (B32).
func (p AuthProfile) String() string { return fmt.Sprintf("%+v", p.redacted()) }

// GoString is the %#v form of String.
func (p AuthProfile) GoString() string { return fmt.Sprintf("%#v", p.redacted()) }

// LogValue redacts the key for log/slog.
func (p AuthProfile) LogValue() slog.Value { return slog.AnyValue(p.redacted()) }

func validateTextDocument(doc *DocumentSource) error {
	if doc.MediaType != "text/plain" {
		return fmt.Errorf("text document media type must be text/plain, got %q", doc.MediaType)
	}
	if strings.TrimSpace(doc.Data) == "" {
		return fmt.Errorf("text document data must not be empty")
	}
	if !utf8.ValidString(doc.Data) {
		return fmt.Errorf("text document data is not valid UTF-8")
	}
	return nil
}
