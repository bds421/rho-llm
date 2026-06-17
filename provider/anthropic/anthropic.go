// Package anthropic implements the Anthropic Claude API adapter.
package anthropic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"net/http"
	"strings"

	"github.com/bds421/rho-llm"
)

const defaultAnthropicBase = "https://api.anthropic.com/v1"

func init() {
	llm.RegisterProvider("anthropic", func(cfg llm.Config) (llm.Client, error) {
		return New(cfg)
	})
}

// Client implements the Claude API with streaming and tool use.
// Auth rotation is handled by PooledClient (pool.go), not here.
type Client struct {
	config       llm.Config
	endpoint     string // resolved messages endpoint (cfg.BaseURL or default)
	httpClient   *http.Client
	providerName string
}

// New creates a new Anthropic client.
func New(cfg llm.Config) (*Client, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("API key is required")
	}

	base := llm.ResolveBaseURL(cfg)
	if base == "" {
		base = defaultAnthropicBase
	}

	providerName := cfg.ProviderName
	if providerName == "" {
		providerName = "anthropic"
	}

	return &Client{
		config:       cfg,
		endpoint:     base + "/messages",
		httpClient:   llm.SafeHTTPClient(cfg.Timeout),
		providerName: providerName,
	}, nil
}

// Provider returns the provider name.
func (c *Client) Provider() string {
	return c.providerName
}

// Model returns the default model name.
func (c *Client) Model() string {
	return c.config.Model
}

// Close releases resources. Drains idle connections from the HTTP transport
// to prevent connection pool leakage during auth pool rotation.
func (c *Client) Close() error {
	c.httpClient.CloseIdleConnections()
	return nil
}

// Complete generates a non-streaming completion.
func (c *Client) Complete(ctx context.Context, req llm.Request) (*llm.Response, error) {
	return c.doRequest(ctx, req, false)
}

// Stream returns an iterator of streaming events.
func (c *Client) Stream(ctx context.Context, req llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		c.doStreamRequest(ctx, req, yield)
	}
}

// =============================================================================
// INTERNAL REQUEST HANDLING
// =============================================================================

// anthropicRequest is the Anthropic API request format.
// System is any to support both plain string and structured content blocks
// (required for cache_control on system prompts).
type anthropicRequest struct {
	Model         string             `json:"model"`
	Messages      []anthropicMessage `json:"messages"`
	System        any                `json:"system,omitempty"`
	MaxTokens     int                `json:"max_tokens"`
	Temperature   *float64           `json:"temperature,omitempty"`
	Tools         []anthropicTool    `json:"tools,omitempty"`
	Stream        bool               `json:"stream,omitempty"`
	Thinking      *anthropicThinking `json:"thinking,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content []any  `json:"content"`
}

type anthropicTool struct {
	Name         string                 `json:"name"`
	Description  string                 `json:"description"`
	InputSchema  map[string]any         `json:"input_schema"`
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

// anthropicCacheControl represents the cache_control annotation.
// Anthropic currently only supports type "ephemeral".
type anthropicCacheControl struct {
	Type string `json:"type"`
}

type anthropicThinking struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

// anthropicResponse is the Anthropic API response format.
type anthropicResponse struct {
	ID      string `json:"id"`
	Type    string `json:"type"`
	Role    string `json:"role"`
	Model   string `json:"model"`
	Content []struct {
		Type      string `json:"type"`
		Text      string `json:"text,omitempty"`
		ID        string `json:"id,omitempty"`
		Name      string `json:"name,omitempty"`
		Input     any    `json:"input,omitempty"`
		Thinking  string `json:"thinking,omitempty"`
		Signature string `json:"signature,omitempty"` // thinking-block signature (for replay)
		Data      string `json:"data,omitempty"`      // redacted_thinking payload
	} `json:"content"`
	StopReason string `json:"stop_reason"`
	Usage      struct {
		InputTokens         int `json:"input_tokens"`
		OutputTokens        int `json:"output_tokens"`
		CacheCreationTokens int `json:"cache_creation_input_tokens,omitempty"`
		CacheReadTokens     int `json:"cache_read_input_tokens,omitempty"`
	} `json:"usage"`
}

// setBetaHeader sets the anthropic-beta header from config when thinking is active.
func (c *Client) setBetaHeader(httpReq *http.Request, req llm.Request) {
	if req.ThinkingLevel == llm.ThinkingNone {
		return
	}
	if len(c.config.BetaFeatures) > 0 {
		httpReq.Header.Set("anthropic-beta", strings.Join(c.config.BetaFeatures, ","))
	}
}

func (c *Client) doRequest(ctx context.Context, req llm.Request, stream bool) (*llm.Response, error) {
	// Fall back to config-level ThinkingLevel if not set on the request
	if req.ThinkingLevel == llm.ThinkingNone && c.config.ThinkingLevel != llm.ThinkingNone {
		req.ThinkingLevel = c.config.ThinkingLevel
	}

	apiReq, err := c.buildRequest(req, stream)
	if err != nil {
		return nil, err
	}

	body, err := json.Marshal(apiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", c.config.APIKey)
	httpReq.Header.Set("anthropic-version", c.config.EffectiveAnthropicVersion())

	// Beta features from config (e.g., extended thinking).
	c.setBetaHeader(httpReq, req)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, llm.ErrorFromResponse("anthropic", resp, c.config)
	}

	var apiResp anthropicResponse
	if err := llm.DecodeJSONResponse(resp, c.config, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return c.parseResponse(&apiResp), nil
}

func (c *Client) doStreamRequest(ctx context.Context, req llm.Request, yield func(llm.StreamEvent, error) bool) {
	// Fall back to config-level ThinkingLevel if not set on the request
	if req.ThinkingLevel == llm.ThinkingNone && c.config.ThinkingLevel != llm.ThinkingNone {
		req.ThinkingLevel = c.config.ThinkingLevel
	}

	apiReq, err := c.buildRequest(req, true)
	if err != nil {
		yield(llm.StreamEvent{}, err)
		return
	}

	body, err := json.Marshal(apiReq)
	if err != nil {
		yield(llm.StreamEvent{}, fmt.Errorf("failed to marshal request: %w", err))
		return
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.endpoint, bytes.NewReader(body))
	if err != nil {
		yield(llm.StreamEvent{}, fmt.Errorf("failed to create request: %w", err))
		return
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", c.config.APIKey)
	httpReq.Header.Set("anthropic-version", c.config.EffectiveAnthropicVersion())
	httpReq.Header.Set("Accept", "text/event-stream")

	// Beta features from config.
	c.setBetaHeader(httpReq, req)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		yield(llm.StreamEvent{}, fmt.Errorf("request failed: %w", err))
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		yield(llm.StreamEvent{}, llm.ErrorFromResponse("anthropic", resp, c.config))
		return
	}

	c.parseStream(resp.Body, yield)
}

func (c *Client) buildRequest(req llm.Request, stream bool) (anthropicRequest, error) {
	apiReq := anthropicRequest{
		Model:       req.Model,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature, // nil = omit from wire
		Stream:      stream,
	}

	if apiReq.Model == "" {
		apiReq.Model = c.config.Model
	}
	if apiReq.MaxTokens == 0 {
		apiReq.MaxTokens = c.config.MaxTokens
	}

	// Build system prompt. Collect all system text first (from req.System and RoleSystem messages).
	var systemTexts []string
	if req.System != "" {
		systemTexts = append(systemTexts, req.System)
	}

	// Convert messages
	for _, msg := range req.Messages {
		if msg.Role == llm.RoleSystem {
			// System messages go to the top-level "system" field, not the messages array.
			// Anthropic rejects role="system" in the messages array.
			for _, part := range msg.Content {
				if part.Type == llm.ContentText && part.Text != "" {
					systemTexts = append(systemTexts, part.Text)
				}
			}
			continue
		}

		apiMsg := anthropicMessage{Role: string(msg.Role)}
		for _, part := range msg.Content {
			switch part.Type {
			case llm.ContentText:
				if part.Text != "" {
					block := map[string]any{
						"type": "text",
						"text": part.Text,
					}
					if part.CacheControl {
						block["cache_control"] = map[string]string{"type": "ephemeral"}
					}
					apiMsg.Content = append(apiMsg.Content, block)
				}
			case llm.ContentImage:
				if err := llm.ValidateImageSource(part); err != nil {
					return anthropicRequest{}, err
				}
				block := map[string]any{
					"type": "image",
					"source": map[string]any{
						"type":       part.Source.Type,
						"media_type": part.Source.MediaType,
						"data":       part.Source.Data,
					},
				}
				if part.CacheControl {
					block["cache_control"] = map[string]string{"type": "ephemeral"}
				}
				apiMsg.Content = append(apiMsg.Content, block)
			case llm.ContentDocument:
				// Anthropic parses PDFs natively via the "document" content block.
				if err := llm.ValidateDocumentSource(part); err != nil {
					return anthropicRequest{}, err
				}
				block := map[string]any{
					"type": "document",
					"source": map[string]any{
						"type":       part.Document.Type,
						"media_type": part.Document.MediaType,
						"data":       part.Document.Data,
					},
				}
				if part.CacheControl {
					block["cache_control"] = map[string]string{"type": "ephemeral"}
				}
				apiMsg.Content = append(apiMsg.Content, block)
			case llm.ContentThinking:
				// Replay extended-thinking blocks so multi-turn tool use stays
				// valid (Anthropic requires the thinking block + its signature on
				// the next turn). A block without an Anthropic signature — e.g.
				// produced by another provider — is skipped rather than risk an
				// API rejection; NormalizeForProvider degrades those to text.
				//
				// Thinking blocks are ONLY valid when extended thinking is enabled
				// for this request — Anthropic rejects them otherwise. So when
				// thinking is off this turn, drop historical thinking blocks.
				if req.ThinkingLevel == llm.ThinkingNone {
					continue
				}
				if part.Redacted && part.Thinking != "" {
					apiMsg.Content = append(apiMsg.Content, map[string]any{
						"type": "redacted_thinking",
						"data": part.Thinking,
					})
				} else if part.ThinkingSignature != "" && part.Thinking != "" {
					apiMsg.Content = append(apiMsg.Content, map[string]any{
						"type":      "thinking",
						"thinking":  part.Thinking,
						"signature": part.ThinkingSignature,
					})
				}
			case llm.ContentToolUse:
				input := part.ToolInput
				if input == nil {
					input = map[string]any{} // Anthropic requires input to be an object, not null
				}
				apiMsg.Content = append(apiMsg.Content, map[string]any{
					"type":  "tool_use",
					"id":    part.ToolUseID,
					"name":  part.ToolName,
					"input": input,
				})
			case llm.ContentToolResult:
				block := map[string]any{
					"type":        "tool_result",
					"tool_use_id": part.ToolResultID,
					"is_error":    part.IsError,
				}
				// Rich tool results (text + image blocks) serialize as a content
				// array; plain results stay a string.
				if len(part.ToolResultParts) > 0 {
					var blocks []any
					for _, sub := range part.ToolResultParts {
						switch sub.Type {
						case llm.ContentText:
							if sub.Text != "" {
								blocks = append(blocks, map[string]any{"type": "text", "text": sub.Text})
							}
						case llm.ContentImage:
							if err := llm.ValidateImageSource(sub); err != nil {
								return anthropicRequest{}, err
							}
							blocks = append(blocks, map[string]any{
								"type": "image",
								"source": map[string]any{
									"type":       sub.Source.Type,
									"media_type": sub.Source.MediaType,
									"data":       sub.Source.Data,
								},
							})
						}
					}
					block["content"] = blocks
				} else {
					block["content"] = part.ToolResultContent
				}
				apiMsg.Content = append(apiMsg.Content, block)
			}
		}
		apiReq.Messages = append(apiReq.Messages, apiMsg)
	}

	// Set system field: structured blocks (with cache_control) or plain string
	if req.SystemCacheControl && len(systemTexts) > 0 {
		// Send as array of content blocks with cache_control on the last block
		var blocks []any
		for i, text := range systemTexts {
			block := map[string]any{
				"type": "text",
				"text": text,
			}
			if i == len(systemTexts)-1 {
				block["cache_control"] = map[string]string{"type": "ephemeral"}
			}
			blocks = append(blocks, block)
		}
		apiReq.System = blocks
	} else if len(systemTexts) > 0 {
		apiReq.System = strings.Join(systemTexts, "\n")
	}

	// Convert tools (omit empty array — matches Gemini and OpenAI adapters)
	if len(req.Tools) > 0 {
		for _, tool := range req.Tools {
			schema := tool.InputSchema
			if schema == nil {
				schema = map[string]any{"type": "object"} // required object, not null
			}
			at := anthropicTool{
				Name:        tool.Name,
				Description: tool.Description,
				InputSchema: schema,
			}
			if tool.CacheControl {
				at.CacheControl = &anthropicCacheControl{Type: "ephemeral"}
			}
			apiReq.Tools = append(apiReq.Tools, at)
		}
	}

	// Configure stop sequences
	if len(req.StopSequences) > 0 {
		apiReq.StopSequences = req.StopSequences
	}

	// Configure thinking
	if req.ThinkingLevel != llm.ThinkingNone {
		budget := llm.ThinkingBudgetTokens(req.ThinkingLevel, req.ThinkingBudget)
		// Clamp budget to model's max output tokens — budget_tokens cannot
		// exceed max_tokens, which itself cannot exceed the model's limit.
		if info, ok := llm.GetModelInfo(apiReq.Model); ok && info.MaxTokens > 0 {
			budget = llm.ClampThinkingBudget("anthropic", apiReq.Model, budget, info.MaxTokens)
		}
		apiReq.Thinking = &anthropicThinking{
			Type:         "enabled",
			BudgetTokens: budget,
		}
		// Anthropic requires temperature = 1.0 when extended thinking is enabled
		one := 1.0
		if req.Temperature != nil && *req.Temperature != 1.0 {
			slog.Warn("overriding temperature to 1.0 (required by Anthropic extended thinking)",
				"requested_temperature", *req.Temperature)
		}
		apiReq.Temperature = &one
	}

	return apiReq, nil
}

// normalizeStopReason maps Anthropic stop reasons onto the unified vocabulary
// (llm.StopEndTurn, …). Anthropic already uses the unified names for the
// common cases; a configured stop sequence is a normal end of turn. Reasons
// with no unified equivalent (e.g. "refusal", "pause_turn") pass through.
func normalizeStopReason(reason string) string {
	if reason == "stop_sequence" {
		return llm.StopEndTurn
	}
	return reason
}

func (c *Client) parseResponse(apiResp *anthropicResponse) *llm.Response {
	resp := &llm.Response{
		ID:                  apiResp.ID,
		Model:               apiResp.Model,
		StopReason:          normalizeStopReason(apiResp.StopReason),
		InputTokens:         apiResp.Usage.InputTokens,
		OutputTokens:        apiResp.Usage.OutputTokens,
		CacheCreationTokens: apiResp.Usage.CacheCreationTokens,
		CacheReadTokens:     apiResp.Usage.CacheReadTokens,
	}

	for _, block := range apiResp.Content {
		switch block.Type {
		case "text":
			resp.Content += block.Text
		case "tool_use":
			resp.ToolCalls = append(resp.ToolCalls, llm.ToolCall{
				ID:    block.ID,
				Name:  block.Name,
				Input: block.Input,
			})
		case "thinking":
			resp.Thinking += block.Thinking
			if block.Signature != "" {
				resp.ThinkingSignature = block.Signature
			}
		case "redacted_thinking":
			resp.Thinking += block.Data
			resp.ThinkingRedacted = true
		}
	}

	return resp
}

func (c *Client) parseStream(body io.Reader, yield func(llm.StreamEvent, error) bool) {
	maxToolInput := c.config.EffectiveMaxToolInputBytes()
	scanner := bufio.NewScanner(body)
	scanner.Buffer(nil, c.config.EffectiveMaxSSELineBytes())

	var currentToolCall *llm.ToolCall
	var inputBuffer strings.Builder
	// Token counts initialize to "not reported". If the stream ends before
	// message_start/message_delta events, callers can distinguish "not reported"
	// from "zero tokens" (0).
	var inputTokens = llm.TokensNotReported
	var cacheCreationTokens, cacheReadTokens int
	var thinkingSignature string
	doneEmitted := false

	// flushToolCall emits any accumulated tool call and clears the buffer.
	// Returns false if the caller broke iteration. Called both on
	// content_block_stop and defensively on a new content_block_start, so a
	// spec-violating stream that omits the stop can't silently drop a tool call.
	flushToolCall := func() bool {
		if currentToolCall == nil {
			return true
		}
		var input any
		raw := inputBuffer.String()
		if err := json.Unmarshal([]byte(raw), &input); err != nil {
			slog.Warn("failed to parse tool input JSON", "provider", "anthropic", "tool", currentToolCall.Name, "error", err)
			input = raw
		}
		currentToolCall.Input = input
		tc := currentToolCall
		currentToolCall = nil
		inputBuffer.Reset()
		return yield(llm.StreamEvent{Type: llm.EventToolUse, ToolCall: tc}, nil)
	}

	for scanner.Scan() {
		data, ok := llm.SSEData(scanner.Text())
		if !ok {
			continue
		}
		if data == "[DONE]" {
			break
		}

		var event struct {
			Type  string `json:"type"`
			Index int    `json:"index"`
			Delta struct {
				Type        string `json:"type"`
				Text        string `json:"text"`
				PartialJSON string `json:"partial_json"`
				Thinking    string `json:"thinking"`
				Signature   string `json:"signature"`
				StopReason  string `json:"stop_reason"`
			} `json:"delta"`
			ContentBlock struct {
				Type  string `json:"type"`
				ID    string `json:"id"`
				Name  string `json:"name"`
				Input any    `json:"input"`
			} `json:"content_block"`
			Message struct {
				StopReason string `json:"stop_reason"`
				Usage      struct {
					InputTokens         int `json:"input_tokens"`
					OutputTokens        int `json:"output_tokens"`
					CacheCreationTokens int `json:"cache_creation_input_tokens,omitempty"`
					CacheReadTokens     int `json:"cache_read_input_tokens,omitempty"`
				} `json:"usage"`
			} `json:"message"`
			// message_delta puts usage at top level (not inside message)
			Usage struct {
				OutputTokens int `json:"output_tokens"`
			} `json:"usage"`
		}

		if err := json.Unmarshal([]byte(data), &event); err != nil {
			if !yield(llm.StreamEvent{}, fmt.Errorf("malformed SSE event from anthropic: %w", err)) {
				return
			}
			continue
		}

		switch event.Type {
		case "content_block_start":
			if event.ContentBlock.Type == "tool_use" {
				// Flush a still-open tool call first (defends against a missing
				// content_block_stop), then begin the new one.
				if !flushToolCall() {
					return
				}
				currentToolCall = &llm.ToolCall{
					ID:   event.ContentBlock.ID,
					Name: event.ContentBlock.Name,
				}
				inputBuffer.Reset()
			}

		case "content_block_delta":
			switch event.Delta.Type {
			case "text_delta":
				if !yield(llm.StreamEvent{Type: llm.EventContent, Text: event.Delta.Text}, nil) {
					return
				}
			case "thinking_delta":
				if !yield(llm.StreamEvent{Type: llm.EventThinking, Thinking: event.Delta.Thinking}, nil) {
					return
				}
			case "signature_delta":
				// Authenticates the thinking block; surfaced on EventDone so a
				// streamed turn can be reassembled with a replayable thinking block.
				thinkingSignature += event.Delta.Signature
			case "input_json_delta":
				if inputBuffer.Len()+len(event.Delta.PartialJSON) > maxToolInput {
					yield(llm.StreamEvent{}, fmt.Errorf("tool input exceeded %d bytes", maxToolInput))
					return // stop parsing — continuing would corrupt the tool call
				}
				inputBuffer.WriteString(event.Delta.PartialJSON)
			}

		case "content_block_stop":
			if !flushToolCall() {
				return
			}

		case "message_delta":
			// Flush a still-open tool call before completing. If the final
			// tool_use block's content_block_stop was dropped (truncated /
			// spec-violating stream) and message_delta arrives directly, the
			// accumulated call would otherwise be silently lost while Done still
			// reports a clean tool_use turn. The new-content_block_start flush
			// only covers a *following* block; this covers the terminal one.
			if !flushToolCall() {
				return
			}
			// The terminal event: emit Done and stop. Returning here means any
			// trailing bytes the server sends after the turn is complete
			// (including malformed lines) can't surface as a spurious error that
			// would mask the completed turn.
			doneEmitted = true
			yield(llm.StreamEvent{
				Type:                llm.EventDone,
				StopReason:          normalizeStopReason(event.Delta.StopReason),
				InputTokens:         inputTokens,
				OutputTokens:        event.Usage.OutputTokens,
				ThinkingSignature:   thinkingSignature,
				CacheCreationTokens: cacheCreationTokens,
				CacheReadTokens:     cacheReadTokens,
			}, nil)
			return

		case "message_stop":
			// Final event

		case "message_start":
			// Capture input tokens and cache usage (reported once at stream start)
			inputTokens = event.Message.Usage.InputTokens
			cacheCreationTokens = event.Message.Usage.CacheCreationTokens
			cacheReadTokens = event.Message.Usage.CacheReadTokens
		}
	}

	// Only report scanner errors if the stream did not already complete
	// successfully. A trailing read error after EventDone is noise.
	if err := scanner.Err(); err != nil && !doneEmitted {
		yield(llm.StreamEvent{}, fmt.Errorf("stream error: %w", err))
		return
	}

	// Clean EOF without message_delta: the server truncated the turn. Yield an
	// explicit error — otherwise a half-accumulated tool call or missing stop
	// reason would be indistinguishable from a complete turn. Wrapping
	// io.ErrUnexpectedEOF lets the pool classify the failure as retryable.
	if !doneEmitted {
		yield(llm.StreamEvent{}, fmt.Errorf("anthropic: stream ended without completion: %w", io.ErrUnexpectedEOF))
	}
}
