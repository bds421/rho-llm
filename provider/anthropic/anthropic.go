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

const (
	defaultAnthropicBase = "https://api.anthropic.com/v1"
	anthropicVersion     = "2023-06-01"
)

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
// System is interface{} to support both plain string and structured content blocks
// (required for cache_control on system prompts).
type anthropicRequest struct {
	Model         string             `json:"model"`
	Messages      []anthropicMessage `json:"messages"`
	System        interface{}        `json:"system,omitempty"`
	MaxTokens     int                `json:"max_tokens"`
	Temperature   float64            `json:"temperature"`
	Tools         []anthropicTool    `json:"tools,omitempty"`
	Stream        bool               `json:"stream,omitempty"`
	Thinking      *anthropicThinking `json:"thinking,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
}

type anthropicMessage struct {
	Role    string        `json:"role"`
	Content []interface{} `json:"content"`
}

type anthropicTool struct {
	Name         string                 `json:"name"`
	Description  string                 `json:"description"`
	InputSchema  map[string]interface{} `json:"input_schema"`
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
		Type     string `json:"type"`
		Text     string `json:"text,omitempty"`
		ID       string `json:"id,omitempty"`
		Name     string `json:"name,omitempty"`
		Input    any    `json:"input,omitempty"`
		Thinking string `json:"thinking,omitempty"`
	} `json:"content"`
	StopReason string `json:"stop_reason"`
	Usage      struct {
		InputTokens         int `json:"input_tokens"`
		OutputTokens        int `json:"output_tokens"`
		CacheCreationTokens int `json:"cache_creation_input_tokens,omitempty"`
		CacheReadTokens     int `json:"cache_read_input_tokens,omitempty"`
	} `json:"usage"`
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
	httpReq.Header.Set("anthropic-version", anthropicVersion)

	// Extended thinking requires a beta feature flag. This header will become
	// unnecessary when Anthropic promotes interleaved thinking to GA.
	// Track: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
	if req.ThinkingLevel != llm.ThinkingNone {
		httpReq.Header.Set("anthropic-beta", "interleaved-thinking-2025-05-14")
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, readErr := io.ReadAll(io.LimitReader(resp.Body, llm.MaxErrorBodyBytes))
		if readErr != nil {
			slog.Warn("failed to read error response body", "provider", "anthropic", "error", readErr)
		}
		return nil, llm.NewAPIErrorFromStatus("anthropic", resp.StatusCode, string(body))
	}

	var apiResp anthropicResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, llm.MaxResponseBodyBytes)).Decode(&apiResp); err != nil {
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
	httpReq.Header.Set("anthropic-version", anthropicVersion)
	httpReq.Header.Set("Accept", "text/event-stream")

	// Extended thinking requires a beta feature flag (see doRequest for details).
	if req.ThinkingLevel != llm.ThinkingNone {
		httpReq.Header.Set("anthropic-beta", "interleaved-thinking-2025-05-14")
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		yield(llm.StreamEvent{}, fmt.Errorf("request failed: %w", err))
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, readErr := io.ReadAll(io.LimitReader(resp.Body, llm.MaxErrorBodyBytes))
		if readErr != nil {
			slog.Warn("failed to read error response body", "provider", "anthropic", "error", readErr)
		}
		yield(llm.StreamEvent{}, llm.NewAPIErrorFromStatus("anthropic", resp.StatusCode, string(body)))
		return
	}

	c.parseStream(resp.Body, yield)
}

func (c *Client) buildRequest(req llm.Request, stream bool) (anthropicRequest, error) {
	apiReq := anthropicRequest{
		Model:       req.Model,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
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
					block := map[string]interface{}{
						"type": "text",
						"text": part.Text,
					}
					if part.CacheControl {
						block["cache_control"] = map[string]string{"type": "ephemeral"}
					}
					apiMsg.Content = append(apiMsg.Content, block)
				}
			case llm.ContentImage:
				return anthropicRequest{}, fmt.Errorf("image content not yet supported by %s adapter", c.providerName)
			case llm.ContentToolUse:
				apiMsg.Content = append(apiMsg.Content, map[string]interface{}{
					"type":  "tool_use",
					"id":    part.ToolUseID,
					"name":  part.ToolName,
					"input": part.ToolInput,
				})
			case llm.ContentToolResult:
				apiMsg.Content = append(apiMsg.Content, map[string]interface{}{
					"type":        "tool_result",
					"tool_use_id": part.ToolResultID,
					"content":     part.ToolResultContent,
					"is_error":    part.IsError,
				})
			}
		}
		apiReq.Messages = append(apiReq.Messages, apiMsg)
	}

	// Set system field: structured blocks (with cache_control) or plain string
	if req.SystemCacheControl && len(systemTexts) > 0 {
		// Send as array of content blocks with cache_control on the last block
		var blocks []interface{}
		for i, text := range systemTexts {
			block := map[string]interface{}{
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

	// Convert tools
	for _, tool := range req.Tools {
		at := anthropicTool{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: tool.InputSchema,
		}
		if tool.CacheControl {
			at.CacheControl = &anthropicCacheControl{Type: "ephemeral"}
		}
		apiReq.Tools = append(apiReq.Tools, at)
	}

	// Configure stop sequences
	if len(req.StopSequences) > 0 {
		apiReq.StopSequences = req.StopSequences
	}

	// Configure thinking
	if req.ThinkingLevel != llm.ThinkingNone {
		budget := llm.ThinkingBudgetTokens(req.ThinkingLevel, req.ThinkingBudget)
		apiReq.Thinking = &anthropicThinking{
			Type:         "enabled",
			BudgetTokens: budget,
		}
		// Anthropic requires temperature = 1.0 when extended thinking is enabled
		if req.Temperature != 1.0 {
			slog.Debug("overriding temperature to 1.0 (required by Anthropic extended thinking)",
				"requested_temperature", req.Temperature)
		}
		apiReq.Temperature = 1.0
	}

	return apiReq, nil
}

func (c *Client) parseResponse(apiResp *anthropicResponse) *llm.Response {
	resp := &llm.Response{
		ID:                  apiResp.ID,
		Model:               apiResp.Model,
		StopReason:          apiResp.StopReason,
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
		}
	}

	return resp
}

func (c *Client) parseStream(body io.Reader, yield func(llm.StreamEvent, error) bool) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(nil, llm.MaxSSELineBytes)

	var currentToolCall *llm.ToolCall
	var inputBuffer strings.Builder
	// Token counts initialize to "not reported". If the stream ends before
	// message_start/message_delta events, callers can distinguish "not reported"
	// from "zero tokens" (0).
	var inputTokens = llm.TokensNotReported
	var cacheCreationTokens, cacheReadTokens int
	doneEmitted := false

	for scanner.Scan() {
		line := scanner.Text()

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
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
			case "input_json_delta":
				if inputBuffer.Len()+len(event.Delta.PartialJSON) > llm.MaxToolInputBytes {
					if !yield(llm.StreamEvent{}, fmt.Errorf("tool input exceeded %d bytes", llm.MaxToolInputBytes)) {
						return
					}
					continue
				}
				inputBuffer.WriteString(event.Delta.PartialJSON)
			}

		case "content_block_stop":
			if currentToolCall != nil {
				// Parse accumulated input
				var input any
				raw := inputBuffer.String()
				if err := json.Unmarshal([]byte(raw), &input); err != nil {
					slog.Warn("failed to parse tool input JSON", "provider", "anthropic", "tool", currentToolCall.Name, "error", err)
					input = raw
				}
				currentToolCall.Input = input
				if !yield(llm.StreamEvent{Type: llm.EventToolUse, ToolCall: currentToolCall}, nil) {
					return
				}
				currentToolCall = nil
			}

		case "message_delta":
			doneEmitted = true
			if !yield(llm.StreamEvent{
				Type:                llm.EventDone,
				StopReason:          event.Delta.StopReason,
				InputTokens:         inputTokens,
				OutputTokens:        event.Usage.OutputTokens,
				CacheCreationTokens: cacheCreationTokens,
				CacheReadTokens:     cacheReadTokens,
			}, nil) {
				return
			}

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
	}
}
