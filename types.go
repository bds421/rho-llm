// Package llm provides a multi-provider LLM client with streaming, tool use,
// extended thinking, and auth pool rotation. Supports Anthropic, Google Gemini,
// and all OpenAI-compatible providers (xAI, OpenAI, Groq, Cerebras, Mistral,
// OpenRouter, Ollama, vLLM, LM Studio).
package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"log/slog"
	"strings"
	"time"
)

// redactSecret removes the literal API key from a free-text message (e.g. a
// provider error body that echoed the request key) so it can't leak into logs
// or serialized state. The library knows its own key, so this is an exact,
// reliable scrub — not heuristic pattern-matching. A short key is left alone to
// avoid mangling unrelated text (real keys are long); an empty key is a no-op.
func redactSecret(msg, key string) string {
	if len(key) < 8 || msg == "" {
		return msg
	}
	return strings.ReplaceAll(msg, key, "REDACTED")
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
)

// EventType represents the type of a streaming event.
type EventType string

const (
	EventContent  EventType = "content"
	EventToolUse  EventType = "tool_use"
	EventThinking EventType = "thinking"
	EventDone     EventType = "done"
	EventError    EventType = "error"
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
)

// ReasoningSummary controls whether reasoning summary text is included in responses.
type ReasoningSummary string

const (
	ReasoningSummaryNone     ReasoningSummary = ""         // omit from request
	ReasoningSummaryAuto     ReasoningSummary = "auto"     // provider decides
	ReasoningSummaryDetailed ReasoningSummary = "detailed" // detailed summary
	ReasoningSummaryConcise  ReasoningSummary = "concise"  // concise summary
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
	case ThinkingXHigh:
		return 128000
	default:
		return 0
	}
}

// ClampThinkingBudget clamps a thinking budget to the model's max output tokens.
// Returns the clamped value and logs a warning if clamping occurred.
func ClampThinkingBudget(provider, model string, budget, maxTokens int) int {
	if maxTokens > 0 && budget > maxTokens {
		slog.Warn("clamping thinking budget to model max_tokens",
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
}

// ContentPart represents a part of message content.
type ContentPart struct {
	Type ContentType `json:"type"` // text, image, tool_use, tool_result

	// Text content
	Text string `json:"text,omitempty"`

	// Image content
	Source *ImageSource `json:"source,omitempty"`

	// Document content (e.g. PDFs passed inline as base64)
	Document *DocumentSource `json:"document,omitempty"`

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

// DocumentSource represents an inline document (e.g. a PDF) passed as base64.
type DocumentSource struct {
	Type      string `json:"type"`       // base64
	MediaType string `json:"media_type"` // application/pdf, etc.
	Data      string `json:"data"`       // base64 encoded data
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
	if part.Document.Data == "" {
		return fmt.Errorf("document data must not be empty")
	}
	if !validDocumentMediaTypes[part.Document.MediaType] {
		return fmt.Errorf("unsupported document media type: %s", part.Document.MediaType)
	}
	if part.Document.Type != "base64" {
		return fmt.Errorf("unsupported document source type: %s", part.Document.Type)
	}
	return nil
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
// Panics if resp is nil (caller bug — matches Go conventions for nil receivers).
func NewAssistantMessage(resp *Response) Message {
	if resp == nil {
		panic("llm.NewAssistantMessage: resp must not be nil")
	}
	msg := Message{
		Role:       RoleAssistant,
		Model:      resp.Model,
		StopReason: resp.StopReason,
	}
	// Thinking first — Anthropic requires the thinking block to precede tool_use
	// when extended thinking is active, and to carry its signature for replay.
	if resp.Thinking != "" || resp.ThinkingRedacted {
		msg.Content = append(msg.Content, ContentPart{
			Type:              ContentThinking,
			Thinking:          resp.Thinking,
			ThinkingSignature: resp.ThinkingSignature,
			Redacted:          resp.ThinkingRedacted,
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
	ThinkingLevel    ThinkingLevel    `json:"thinking_level,omitempty"`    // low, medium, high (zero value = none)
	ThinkingBudget   int              `json:"thinking_budget,omitempty"`   // custom token budget; overrides ThinkingLevel default when > 0
	ReasoningSummary ReasoningSummary `json:"reasoning_summary,omitempty"` // OpenAI Responses API: auto, detailed, concise
	StopSequences    []string         `json:"stop_sequences,omitempty"`

	// Caching
	SystemCacheControl bool   `json:"system_cache_control,omitempty"` // Anthropic: cache the system prompt
	CachedContent      string `json:"cached_content,omitempty"`       // Gemini: pre-created cache name

	// ResponseFormat requests structured output (JSON mode / JSON schema). Wired
	// for OpenAI-compatible providers (response_format) and Gemini
	// (responseMimeType/responseSchema). Anthropic has no native JSON mode (use a
	// tool or prompt); the Responses adapter does not wire it. Nil = free text.
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
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
	StopReason        string     `json:"stop_reason"`                  // end_turn, tool_use, max_tokens
	InputTokens       int        `json:"input_tokens"`
	OutputTokens      int        `json:"output_tokens"`
	ThinkingTokens    int        `json:"thinking_tokens,omitempty"` // Gemini: tokens consumed by thinking (separate from OutputTokens)

	// Cache token usage (Anthropic)
	CacheCreationTokens int `json:"cache_creation_input_tokens,omitempty"` // tokens written to cache
	CacheReadTokens     int `json:"cache_read_input_tokens,omitempty"`     // tokens read from cache
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

	// Usage event
	InputTokens    int `json:"input_tokens,omitempty"`
	OutputTokens   int `json:"output_tokens,omitempty"`
	ThinkingTokens int `json:"thinking_tokens,omitempty"` // Gemini: tokens consumed by thinking

	// Done event
	StopReason string `json:"stop_reason,omitempty"`

	// Cache token usage (reported in EventDone, Anthropic only)
	CacheCreationTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadTokens     int `json:"cache_read_input_tokens,omitempty"`

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
// scrubbed of this profile's own API key first, so a server response that echoed
// the key can't lodge it in LastError (which is logged and serialized).
func (p *AuthProfile) MarkFailed(err error, cooldownDuration time.Duration) {
	if err != nil {
		p.LastError = redactSecret(err.Error(), p.APIKey)
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
	type profileAlias AuthProfile // break recursion
	tmp := profileAlias(p)
	if tmp.APIKey != "" {
		tmp.LastError = redactSecret(tmp.LastError, tmp.APIKey)
		tmp.APIKey = "REDACTED"
	}
	tmp.BaseURL = redactURLCredentials(tmp.BaseURL)
	return json.Marshal(tmp)
}
