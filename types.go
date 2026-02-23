// Package llm provides a multi-provider LLM client with streaming, tool use,
// extended thinking, and auth pool rotation. Supports Anthropic, Google Gemini,
// and all OpenAI-compatible providers (xAI, OpenAI, Groq, Cerebras, Mistral,
// OpenRouter, Ollama, vLLM, LM Studio).
package llm

import (
	"context"
	"encoding/json"
	"iter"
	"time"
)

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
	ThinkingNone   ThinkingLevel = ""
	ThinkingLow    ThinkingLevel = "low"
	ThinkingMedium ThinkingLevel = "medium"
	ThinkingHigh   ThinkingLevel = "high"
)

// ThinkingBudgetTokens returns the default token budget for a thinking level.
// If customBudget > 0, it overrides the level default.
func ThinkingBudgetTokens(level ThinkingLevel, customBudget int) int {
	if customBudget > 0 {
		return customBudget
	}
	switch level {
	case ThinkingLow:
		return 4096
	case ThinkingMedium:
		return 16384
	case ThinkingHigh:
		return 65536
	default:
		return 4096
	}
}

// TokensNotReported is the sentinel value for token counts when the provider
// did not report usage (e.g. stream ended before usage chunk arrived).
// Callers can distinguish "not reported" (-1) from "zero tokens" (0).
const TokensNotReported = -1

// =============================================================================
// MESSAGE TYPES
// =============================================================================

// Message represents a conversation message.
type Message struct {
	Role    Role          `json:"role"`    // user, assistant, system
	Content []ContentPart `json:"content"` // Content parts (text, images, tool results)
}

// ContentPart represents a part of message content.
type ContentPart struct {
	Type ContentType `json:"type"` // text, image, tool_use, tool_result

	// Text content
	Text string `json:"text,omitempty"`

	// Image content
	Source *ImageSource `json:"source,omitempty"`

	// Tool use (from assistant)
	ToolUseID        string `json:"id,omitempty"`
	ToolName         string `json:"name,omitempty"`
	ToolInput        any    `json:"input,omitempty"`
	ThoughtSignature string `json:"thought_signature,omitempty"` // Gemini 3: preserved for tool result round-trip

	// Tool result (from user)
	ToolResultID      string `json:"tool_use_id,omitempty"`
	ToolResultContent string `json:"content,omitempty"`
	IsError           bool   `json:"is_error,omitempty"`
}

// ImageSource represents an image source.
type ImageSource struct {
	Type      string `json:"type"`       // base64
	MediaType string `json:"media_type"` // image/jpeg, image/png, etc.
	Data      string `json:"data"`       // base64 encoded data
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

// NewAssistantMessage creates an assistant message from a Response, preserving
// both text content and tool_use blocks. Use this instead of NewTextMessage
// when the response contains tool calls — NewTextMessage would lose them.
func NewAssistantMessage(resp *Response) Message {
	msg := Message{Role: RoleAssistant}
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
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"input_schema"`
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
	Model         string        `json:"model"`
	Messages      []Message     `json:"messages"`
	System        string        `json:"system,omitempty"`
	MaxTokens     int           `json:"max_tokens"`
	Temperature   float64       `json:"temperature"`
	Tools         []Tool        `json:"tools,omitempty"`
	ThinkingLevel  ThinkingLevel `json:"thinking_level,omitempty"`  // low, medium, high (zero value = none)
	ThinkingBudget int           `json:"thinking_budget,omitempty"` // custom token budget; overrides ThinkingLevel default when > 0
	StopSequences  []string      `json:"stop_sequences,omitempty"`
}

// Response represents an LLM completion response.
type Response struct {
	ID           string     `json:"id"`
	Model        string     `json:"model"`
	Content      string     `json:"content"`     // Extracted text content
	ToolCalls    []ToolCall `json:"tool_calls"`  // Tool use requests
	Thinking     string     `json:"thinking"`    // Extended thinking content
	StopReason   string     `json:"stop_reason"` // end_turn, tool_use, max_tokens
	InputTokens  int        `json:"input_tokens"`
	OutputTokens int        `json:"output_tokens"`
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

	// Usage event
	InputTokens  int `json:"input_tokens,omitempty"`
	OutputTokens int `json:"output_tokens,omitempty"`

	// Done event
	StopReason string `json:"stop_reason,omitempty"`

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

// MarkFailed marks the profile as failed with cooldown.
func (p *AuthProfile) MarkFailed(err error, cooldownDuration time.Duration) {
	if err != nil {
		p.LastError = err.Error()
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
// secret leakage when AuthProfile is serialized for logging or debugging.
func (p AuthProfile) MarshalJSON() ([]byte, error) {
	type profileAlias AuthProfile // break recursion
	tmp := profileAlias(p)
	if tmp.APIKey != "" {
		tmp.APIKey = "REDACTED"
	}
	return json.Marshal(tmp)
}
