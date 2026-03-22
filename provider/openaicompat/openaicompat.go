// Package openaicompat implements the OpenAI-compatible chat completions API adapter.
// Works with: OpenAI, xAI/Grok, Groq, Cerebras, Mistral, OpenRouter,
// Ollama, vLLM, LM Studio, and any provider speaking the same wire protocol.
package openaicompat

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

func init() {
	llm.RegisterProvider("openai_compat", func(cfg llm.Config) (llm.Client, error) {
		return New(cfg)
	})
}

// Client implements the OpenAI-compatible chat completions API.
type Client struct {
	config       llm.Config
	httpClient   *http.Client
	baseURL      string // Resolved endpoint (e.g., "https://api.x.ai/v1")
	authHeader   string // Auth prefix (e.g., "Bearer") or "" for no auth
	providerName string // What Provider() returns
}

// New creates a new OpenAI-compatible client.
func New(cfg llm.Config) (*Client, error) {
	// Validate: cloud providers require an API key; local ones do not
	if cfg.APIKey == "" && !llm.IsNoAuthProvider(cfg.Provider) {
		return nil, fmt.Errorf("%s API key is required", cfg.Provider)
	}

	baseURL := llm.ResolveBaseURL(cfg)
	if baseURL == "" {
		return nil, fmt.Errorf("no base URL configured for provider %s (set BaseURL in config)", cfg.Provider)
	}

	authHeader := cfg.AuthHeader
	if authHeader == "" {
		if preset, ok := llm.PresetFor(cfg.Provider); ok {
			authHeader = preset.AuthHeader
		}
	}

	providerName := cfg.ProviderName
	if providerName == "" {
		providerName = cfg.Provider
	}

	return &Client{
		config:       cfg,
		httpClient:   llm.SafeHTTPClient(cfg.Timeout),
		baseURL:      baseURL,
		authHeader:   authHeader,
		providerName: providerName,
	}, nil
}

// Provider returns the provider name.
func (c *Client) Provider() string {
	return c.providerName
}

// Model returns the model name.
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
	// Use the Responses API for GPT-5 family models when reasoning effort is requested.
	// These models support reasoning but only via the Responses API's reasoning parameter,
	// not via the Chat Completions API.
	if c.useResponsesAPI(req) {
		return c.completeViaResponsesAPI(ctx, req)
	}

	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		return nil, err
	}

	body, err := json.Marshal(apiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := c.baseURL + "/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if c.authHeader != "" && c.config.APIKey != "" {
		httpReq.Header.Set("Authorization", c.authHeader+" "+c.config.APIKey)
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, readErr := io.ReadAll(io.LimitReader(resp.Body, llm.MaxErrorBodyBytes))
		if readErr != nil {
			slog.Warn("failed to read error response body", "provider", c.providerName, "error", readErr)
		}
		return nil, llm.NewAPIErrorFromStatus(c.providerName, resp.StatusCode, string(body))
	}

	var apiResp openaiResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, llm.MaxResponseBodyBytes)).Decode(&apiResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return c.parseResponse(&apiResp), nil
}

// Stream returns an iterator of streaming events.
func (c *Client) Stream(ctx context.Context, req llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
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

		url := c.baseURL + "/chat/completions"
		httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
		if err != nil {
			yield(llm.StreamEvent{}, fmt.Errorf("failed to create request: %w", err))
			return
		}

		httpReq.Header.Set("Content-Type", "application/json")
		if c.authHeader != "" && c.config.APIKey != "" {
			httpReq.Header.Set("Authorization", c.authHeader+" "+c.config.APIKey)
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
				slog.Warn("failed to read error response body", "provider", c.providerName, "error", readErr)
			}
			yield(llm.StreamEvent{}, llm.NewAPIErrorFromStatus(c.providerName, resp.StatusCode, string(body)))
			return
		}

		c.parseStream(resp.Body, yield)
	}
}

// =============================================================================
// OPENAI-COMPATIBLE WIRE FORMAT
// =============================================================================

type openaiRequest struct {
	Model               string               `json:"model"`
	Messages            []openaiMessage      `json:"messages"`
	MaxTokens           int                  `json:"max_tokens,omitempty"`
	MaxCompletionTokens int                  `json:"max_completion_tokens,omitempty"`
	Temperature         *float64             `json:"temperature,omitempty"`
	Stream              bool                 `json:"stream,omitempty"`
	StreamOptions       *openaiStreamOptions `json:"stream_options,omitempty"`
	Tools               []openaiTool         `json:"tools,omitempty"`
	Stop                []string             `json:"stop,omitempty"`
	Reasoning           *openaiReasoning     `json:"reasoning,omitempty"`
}

type openaiReasoning struct {
	Effort string `json:"effort"`
}

type openaiStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type openaiMessage struct {
	Role             string           `json:"role"`
	Content          interface{}      `json:"content"`                     // string or array
	ReasoningContent string           `json:"reasoning_content,omitempty"` // thinking/reasoning output (OpenAI, DeepSeek, etc.)
	Reasoning        string           `json:"reasoning,omitempty"`         // thinking/reasoning output (Ollama)
	ToolCalls        []openaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string           `json:"tool_call_id,omitempty"`
}

type openaiTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string                 `json:"name"`
		Description string                 `json:"description"`
		Parameters  map[string]interface{} `json:"parameters"`
	} `json:"function"`
}

type openaiToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type openaiResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int           `json:"index"`
		Message      openaiMessage `json:"message"`
		FinishReason string        `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// normalizeStopReason maps OpenAI finish reasons to the unified set
// (end_turn, tool_use, max_tokens) used by the library interface.
func normalizeStopReason(reason string) string {
	switch reason {
	case "stop":
		return "end_turn"
	case "tool_calls":
		return "tool_use"
	case "length":
		return "max_tokens"
	default:
		return reason
	}
}

func (c *Client) buildRequest(req llm.Request, stream bool) (openaiRequest, error) {
	model := req.Model
	if model == "" {
		model = c.config.Model
	}

	maxTok := req.MaxTokens
	if maxTok == 0 {
		maxTok = c.config.MaxTokens
	}

	apiReq := openaiRequest{
		Model:  model,
		Stream: stream,
	}

	// OpenAI and xAI reasoning models require max_completion_tokens instead
	// of max_tokens, and reject custom temperature. Other providers (Mistral,
	// Groq, Ollama) use standard max_tokens even for reasoning models.
	info, _ := llm.GetModelInfo(model)
	if info.Thinking && (info.Provider == "openai" || info.Provider == "xai") {
		apiReq.MaxCompletionTokens = maxTok
		// Omit temperature entirely — these reasoning models only accept default (1).
	} else {
		apiReq.MaxTokens = maxTok
		apiReq.Temperature = req.Temperature // nil = omit from wire (provider default)
	}

	// Request usage stats in streaming responses (required since May 2024)
	if stream {
		apiReq.StreamOptions = &openaiStreamOptions{IncludeUsage: true}
	}

	// Add system message if provided
	if req.System != "" {
		apiReq.Messages = append(apiReq.Messages, openaiMessage{
			Role:    "system",
			Content: req.System,
		})
	}

	// Convert messages
	for _, msg := range req.Messages {
		// Handle tool results - each one becomes a separate "tool" message in OpenAI format.
		// Anthropic allows multiple tool_result parts in one user message; OpenAI requires
		// each to be a separate message with role="tool".
		if msg.Role == llm.RoleUser {
			var textParts []string
			var imageParts []llm.ContentPart
			hasToolResults := false

			for _, part := range msg.Content {
				switch part.Type {
				case llm.ContentToolResult:
					hasToolResults = true
					apiReq.Messages = append(apiReq.Messages, openaiMessage{
						Role:       "tool",
						Content:    part.ToolResultContent,
						ToolCallID: part.ToolResultID,
					})
				case llm.ContentText:
					if part.Text != "" {
						textParts = append(textParts, part.Text)
					}
				case llm.ContentImage:
					if err := llm.ValidateImageSource(part); err != nil {
						return openaiRequest{}, err
					}
					imageParts = append(imageParts, part)
				}
			}

			// If images are present, build a multipart content array
			if len(imageParts) > 0 {
				var contentArray []interface{}
				if len(textParts) > 0 {
					contentArray = append(contentArray, map[string]interface{}{
						"type": "text",
						"text": strings.Join(textParts, "\n"),
					})
				}
				for _, img := range imageParts {
					dataURI := fmt.Sprintf("data:%s;base64,%s", img.Source.MediaType, img.Source.Data)
					contentArray = append(contentArray, map[string]interface{}{
						"type":      "image_url",
						"image_url": map[string]string{"url": dataURI},
					})
				}
				apiReq.Messages = append(apiReq.Messages, openaiMessage{
					Role:    "user",
					Content: contentArray,
				})
				continue
			}

			// If there was also text content alongside tool results, emit it as a user message
			if len(textParts) > 0 {
				apiReq.Messages = append(apiReq.Messages, openaiMessage{
					Role:    "user",
					Content: strings.Join(textParts, "\n"),
				})
			}

			if hasToolResults || len(textParts) > 0 {
				continue
			}
		}

		oaiMsg := openaiMessage{Role: string(msg.Role)}

		// Build content from text parts
		var textParts []string
		for _, part := range msg.Content {
			if part.Type == llm.ContentText && part.Text != "" {
				textParts = append(textParts, part.Text)
			}
		}
		if len(textParts) > 0 {
			oaiMsg.Content = strings.Join(textParts, "\n")
		}

		// Check for tool calls in assistant messages
		if msg.Role == llm.RoleAssistant {
			for _, part := range msg.Content {
				if part.Type == llm.ContentToolUse {
					inputJSON, err := json.Marshal(part.ToolInput)
					if err != nil {
						slog.Warn("failed to marshal tool input", "provider", c.providerName, "tool", part.ToolName, "error", err)
						inputJSON = []byte("{}")
					}
					oaiMsg.ToolCalls = append(oaiMsg.ToolCalls, openaiToolCall{
						ID:   part.ToolUseID,
						Type: "function",
						Function: struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						}{
							Name:      part.ToolName,
							Arguments: string(inputJSON),
						},
					})
				}
			}
		}

		apiReq.Messages = append(apiReq.Messages, oaiMsg)
	}

	// Map ThinkingLevel to OpenAI's reasoning.effort for reasoning models.
	// For non-reasoning models or providers that don't support it, the field
	// is silently omitted (omitempty).
	thinkingLevel := req.ThinkingLevel
	if thinkingLevel == llm.ThinkingNone && c.config.ThinkingLevel != llm.ThinkingNone {
		thinkingLevel = c.config.ThinkingLevel
	}
	// Map ThinkingLevel to reasoning.effort for o-series models (o3, o4-mini, etc.).
	// GPT-5 family models reason internally but don't support the reasoning parameter —
	// they use max_completion_tokens to cap total output including reasoning tokens.
	if thinkingLevel != llm.ThinkingNone && info.Thinking && info.SupportsReasoningEffort {
		apiReq.Reasoning = &openaiReasoning{Effort: string(thinkingLevel)}
	}

	// Configure stop sequences
	if len(req.StopSequences) > 0 {
		apiReq.Stop = req.StopSequences
	}

	// Convert tools (skip if model doesn't support tool calling)
	if len(req.Tools) > 0 {
		model := apiReq.Model
		info, _ := llm.GetModelInfo(model)
		if !info.NoToolSupport {
			for _, tool := range req.Tools {
				apiReq.Tools = append(apiReq.Tools, openaiTool{
					Type: "function",
					Function: struct {
						Name        string                 `json:"name"`
						Description string                 `json:"description"`
						Parameters  map[string]interface{} `json:"parameters"`
					}{
						Name:        tool.Name,
						Description: tool.Description,
						Parameters:  tool.InputSchema,
					},
				})
			}
		}
	}

	return apiReq, nil
}

func (c *Client) parseResponse(apiResp *openaiResponse) *llm.Response {
	resp := &llm.Response{
		ID:           apiResp.ID,
		Model:        apiResp.Model,
		InputTokens:  apiResp.Usage.PromptTokens,
		OutputTokens: apiResp.Usage.CompletionTokens,
	}

	if len(apiResp.Choices) > 0 {
		choice := apiResp.Choices[0]
		resp.StopReason = normalizeStopReason(choice.FinishReason)

		// Extract content
		if content, ok := choice.Message.Content.(string); ok {
			resp.Content = content
		}

		// Extract reasoning/thinking content.
		// Most providers use "reasoning_content"; Ollama uses "reasoning".
		reasoning := choice.Message.ReasoningContent
		if reasoning == "" {
			reasoning = choice.Message.Reasoning
		}
		if reasoning != "" {
			resp.Thinking = reasoning
		}

		// Extract tool calls
		for _, tc := range choice.Message.ToolCalls {
			var input any
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &input); err != nil {
				slog.Warn("failed to parse tool input JSON", "provider", c.providerName, "tool", tc.Function.Name, "error", err)
				input = tc.Function.Arguments
			}
			resp.ToolCalls = append(resp.ToolCalls, llm.ToolCall{
				ID:    tc.ID,
				Name:  tc.Function.Name,
				Input: input,
			})
		}
	}

	return resp
}

func (c *Client) parseStream(body io.Reader, yield func(llm.StreamEvent, error) bool) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(nil, llm.MaxSSELineBytes)

	var currentToolCall *llm.ToolCall
	var inputBuffer strings.Builder

	// OpenAI sends finish_reason and usage in SEPARATE chunks when stream_options is set:
	//   Chunk 1: choices[0].finish_reason = "stop", usage = {}
	//   Chunk 2: choices = [], usage = {prompt_tokens: N, completion_tokens: M}
	// We must track state across chunks to emit a complete EventDone.
	//
	// Token counts initialize to "not reported". If the stream ends without
	// a usage chunk (connection dropped, provider doesn't support it), callers
	// can distinguish "not reported" from "zero tokens" (0).
	var finishReason string
	var inputTokens, outputTokens = llm.TokensNotReported, llm.TokensNotReported

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
			ID      string `json:"id"`
			Choices []struct {
				Index int `json:"index"`
				Delta struct {
					Content          string           `json:"content"`
					ReasoningContent string           `json:"reasoning_content"` // OpenAI, DeepSeek, etc.
					Reasoning        string           `json:"reasoning"`         // Ollama
					ToolCalls        []openaiToolCall `json:"tool_calls"`
				} `json:"delta"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}

		if err := json.Unmarshal([]byte(data), &event); err != nil {
			if !yield(llm.StreamEvent{}, fmt.Errorf("malformed SSE event from %s: %w", c.providerName, err)) {
				return
			}
			continue
		}

		// Capture usage if present (may arrive in a separate chunk with empty choices).
		// Usage is a pointer so we can distinguish "field absent" (nil) from "zero tokens".
		if event.Usage != nil {
			inputTokens = event.Usage.PromptTokens
			outputTokens = event.Usage.CompletionTokens
		}

		if len(event.Choices) > 0 {
			choice := event.Choices[0]

			// Reasoning/thinking delta.
			// Most providers use "reasoning_content"; Ollama uses "reasoning".
			reasoningDelta := choice.Delta.ReasoningContent
			if reasoningDelta == "" {
				reasoningDelta = choice.Delta.Reasoning
			}
			if reasoningDelta != "" {
				if !yield(llm.StreamEvent{Type: llm.EventThinking, Thinking: reasoningDelta}, nil) {
					return
				}
			}

			// Content delta
			if choice.Delta.Content != "" {
				if !yield(llm.StreamEvent{Type: llm.EventContent, Text: choice.Delta.Content}, nil) {
					return
				}
			}

			// Tool call deltas
			for _, tc := range choice.Delta.ToolCalls {
				if tc.ID != "" {
					// New tool call starting
					if currentToolCall != nil {
						// Finish previous one
						var input any
						raw := inputBuffer.String()
						if err := json.Unmarshal([]byte(raw), &input); err != nil {
							slog.Warn("failed to parse tool input JSON", "provider", c.providerName, "tool", currentToolCall.Name, "error", err)
							input = raw
						}
						currentToolCall.Input = input
						if !yield(llm.StreamEvent{Type: llm.EventToolUse, ToolCall: currentToolCall}, nil) {
							return
						}
					}
					currentToolCall = &llm.ToolCall{
						ID:   tc.ID,
						Name: tc.Function.Name,
					}
					inputBuffer.Reset()
				}
				if tc.Function.Arguments != "" {
					if inputBuffer.Len()+len(tc.Function.Arguments) > llm.MaxToolInputBytes {
						yield(llm.StreamEvent{}, fmt.Errorf("tool input exceeded %d bytes", llm.MaxToolInputBytes))
						return // stop parsing — continuing would corrupt the tool call
					}
					inputBuffer.WriteString(tc.Function.Arguments)
				}
			}

			// Capture finish reason (usage may arrive in next chunk)
			if choice.FinishReason != "" {
				finishReason = normalizeStopReason(choice.FinishReason)
				if currentToolCall != nil {
					var input any
					raw := inputBuffer.String()
					if err := json.Unmarshal([]byte(raw), &input); err != nil {
						slog.Warn("failed to parse tool input JSON", "provider", c.providerName, "tool", currentToolCall.Name, "error", err)
						input = raw
					}
					currentToolCall.Input = input
					if !yield(llm.StreamEvent{Type: llm.EventToolUse, ToolCall: currentToolCall}, nil) {
						return
					}
					currentToolCall = nil
				}
			}
		}
	}

	// Flush any uncommitted tool call. This can happen if the stream ends
	// without a finish_reason (network drop, spec-violating server like Ollama).
	// Without this, the accumulated tool call JSON is silently lost.
	if currentToolCall != nil {
		var input any
		raw := inputBuffer.String()
		if err := json.Unmarshal([]byte(raw), &input); err != nil {
			slog.Warn("failed to parse tool input JSON", "provider", c.providerName, "tool", currentToolCall.Name, "error", err)
			input = raw
		}
		currentToolCall.Input = input
		if !yield(llm.StreamEvent{Type: llm.EventToolUse, ToolCall: currentToolCall}, nil) {
			return
		}
	}

	// Emit EventDone after all chunks processed (finish_reason + usage now combined)
	if finishReason != "" {
		if !yield(llm.StreamEvent{
			Type:         llm.EventDone,
			StopReason:   finishReason,
			InputTokens:  inputTokens,
			OutputTokens: outputTokens,
		}, nil) {
			return
		}
	} else if scanner.Err() == nil {
		// Stream ended cleanly but server never sent a finish_reason —
		// protocol violation by the provider. Log so operators can investigate.
		slog.Warn("stream ended without finish_reason", "provider", c.providerName)
	}

	// Only report scanner errors if the stream did not already complete
	// successfully. A trailing read error after EventDone is noise.
	if err := scanner.Err(); err != nil && finishReason == "" {
		yield(llm.StreamEvent{}, fmt.Errorf("stream error: %w", err))
	}
}

// =============================================================================
// OPENAI RESPONSES API (for GPT-5 family reasoning effort control)
// =============================================================================

// responsesRequest is the wire format for OpenAI's Responses API (/v1/responses).
// Used for GPT-5 family models that support reasoning effort only via this API.
type responsesRequest struct {
	Model           string                `json:"model"`
	Input           []responsesInputMsg   `json:"input"`
	Reasoning       *responsesReasoning   `json:"reasoning,omitempty"`
	MaxOutputTokens int                   `json:"max_output_tokens,omitempty"`
	Temperature     *float64              `json:"temperature,omitempty"`
	Tools           []responsesTool       `json:"tools,omitempty"`
	Store           bool                  `json:"store"`
}

type responsesReasoning struct {
	Effort string `json:"effort"`
}

type responsesInputMsg struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // string or array of content parts
}

type responsesTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string                 `json:"name"`
		Description string                 `json:"description"`
		Parameters  map[string]interface{} `json:"parameters"`
	} `json:"function"`
}

// responsesResponse is the wire format for the Responses API response.
type responsesResponse struct {
	ID                string                 `json:"id"`
	Model             string                 `json:"model"`
	Status            string                 `json:"status"` // completed, incomplete, failed
	Output            []responsesOutputItem  `json:"output"`
	Usage             responsesUsage         `json:"usage"`
	IncompleteDetails *responsesIncomplete   `json:"incomplete_details"`
}

type responsesOutputItem struct {
	Type    string                    `json:"type"` // message, function_call
	ID      string                    `json:"id"`
	Role    string                    `json:"role,omitempty"`
	Content []responsesContentBlock   `json:"content,omitempty"`
	// function_call fields
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type responsesContentBlock struct {
	Type string `json:"type"` // output_text
	Text string `json:"text"`
}

type responsesUsage struct {
	InputTokens     int `json:"input_tokens"`
	OutputTokens    int `json:"output_tokens"`
	ReasoningTokens int `json:"reasoning_tokens"`
}

type responsesIncomplete struct {
	Reason string `json:"reason"` // max_output_tokens, content_filter
}

// useResponsesAPI returns true when the request should use the Responses API
// instead of Chat Completions. This applies to GPT-5 family models that reason
// internally (info.Thinking == true) but don't support the reasoning parameter
// in Chat Completions (info.SupportsReasoningEffort == false), AND when a
// thinking level is actually requested.
func (c *Client) useResponsesAPI(req llm.Request) bool {
	model := req.Model
	if model == "" {
		model = c.config.Model
	}
	info, ok := llm.GetModelInfo(model)
	if !ok {
		return false
	}

	thinkingLevel := req.ThinkingLevel
	if thinkingLevel == llm.ThinkingNone {
		thinkingLevel = c.config.ThinkingLevel
	}

	// Use Responses API when: model reasons internally, doesn't support
	// reasoning effort via Chat Completions, and caller wants reasoning control.
	return thinkingLevel != llm.ThinkingNone && info.Thinking && !info.SupportsReasoningEffort
}

// completeViaResponsesAPI sends the request to /v1/responses and maps the
// response back to the standard llm.Response.
func (c *Client) completeViaResponsesAPI(ctx context.Context, req llm.Request) (*llm.Response, error) {
	apiReq, err := c.buildResponsesRequest(req)
	if err != nil {
		return nil, err
	}

	body, err := json.Marshal(apiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal responses request: %w", err)
	}

	url := c.baseURL + "/responses"
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if c.authHeader != "" && c.config.APIKey != "" {
		httpReq.Header.Set("Authorization", c.authHeader+" "+c.config.APIKey)
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errBody, readErr := io.ReadAll(io.LimitReader(resp.Body, llm.MaxErrorBodyBytes))
		if readErr != nil {
			slog.Warn("failed to read error response body", "provider", c.providerName, "error", readErr)
		}
		return nil, llm.NewAPIErrorFromStatus(c.providerName, resp.StatusCode, string(errBody))
	}

	var apiResp responsesResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, llm.MaxResponseBodyBytes)).Decode(&apiResp); err != nil {
		return nil, fmt.Errorf("failed to decode responses API response: %w", err)
	}

	return c.parseResponsesResponse(&apiResp), nil
}

// buildResponsesRequest converts an llm.Request into the Responses API wire format.
func (c *Client) buildResponsesRequest(req llm.Request) (responsesRequest, error) {
	model := req.Model
	if model == "" {
		model = c.config.Model
	}

	maxTok := req.MaxTokens
	if maxTok == 0 {
		maxTok = c.config.MaxTokens
	}

	thinkingLevel := req.ThinkingLevel
	if thinkingLevel == llm.ThinkingNone {
		thinkingLevel = c.config.ThinkingLevel
	}

	apiReq := responsesRequest{
		Model:           model,
		MaxOutputTokens: maxTok,
		Store:           false, // Don't persist responses server-side
	}

	// Set reasoning effort
	if thinkingLevel != llm.ThinkingNone {
		apiReq.Reasoning = &responsesReasoning{Effort: string(thinkingLevel)}
	}

	// Add system message
	if req.System != "" {
		apiReq.Input = append(apiReq.Input, responsesInputMsg{
			Role:    "system",
			Content: req.System,
		})
	}

	// Convert messages
	for _, msg := range req.Messages {
		// Handle tool results — each becomes a separate message in the input array
		if msg.Role == llm.RoleUser {
			var textParts []string
			var imageParts []llm.ContentPart
			hasToolResults := false

			for _, part := range msg.Content {
				switch part.Type {
				case llm.ContentToolResult:
					hasToolResults = true
					apiReq.Input = append(apiReq.Input, responsesInputMsg{
						Role:    "tool",
						Content: part.ToolResultContent,
					})
				case llm.ContentText:
					if part.Text != "" {
						textParts = append(textParts, part.Text)
					}
				case llm.ContentImage:
					if err := llm.ValidateImageSource(part); err != nil {
						return responsesRequest{}, err
					}
					imageParts = append(imageParts, part)
				}
			}

			// Build multipart content for images
			if len(imageParts) > 0 {
				var contentArray []interface{}
				if len(textParts) > 0 {
					contentArray = append(contentArray, map[string]interface{}{
						"type": "input_text",
						"text": strings.Join(textParts, "\n"),
					})
				}
				for _, img := range imageParts {
					dataURI := fmt.Sprintf("data:%s;base64,%s", img.Source.MediaType, img.Source.Data)
					contentArray = append(contentArray, map[string]interface{}{
						"type":      "input_image",
						"image_url": dataURI,
					})
				}
				apiReq.Input = append(apiReq.Input, responsesInputMsg{
					Role:    "user",
					Content: contentArray,
				})
				continue
			}

			if len(textParts) > 0 {
				apiReq.Input = append(apiReq.Input, responsesInputMsg{
					Role:    "user",
					Content: strings.Join(textParts, "\n"),
				})
			}

			if hasToolResults || len(textParts) > 0 {
				continue
			}
		}

		inputMsg := responsesInputMsg{Role: string(msg.Role)}

		var textParts []string
		for _, part := range msg.Content {
			if part.Type == llm.ContentText && part.Text != "" {
				textParts = append(textParts, part.Text)
			}
		}
		if len(textParts) > 0 {
			inputMsg.Content = strings.Join(textParts, "\n")
		}

		apiReq.Input = append(apiReq.Input, inputMsg)
	}

	// Convert tools
	info, _ := llm.GetModelInfo(model)
	if len(req.Tools) > 0 && !info.NoToolSupport {
		for _, tool := range req.Tools {
			apiReq.Tools = append(apiReq.Tools, responsesTool{
				Type: "function",
				Function: struct {
					Name        string                 `json:"name"`
					Description string                 `json:"description"`
					Parameters  map[string]interface{} `json:"parameters"`
				}{
					Name:        tool.Name,
					Description: tool.Description,
					Parameters:  tool.InputSchema,
				},
			})
		}
	}

	return apiReq, nil
}

// parseResponsesResponse maps the Responses API wire format to llm.Response.
func (c *Client) parseResponsesResponse(apiResp *responsesResponse) *llm.Response {
	resp := &llm.Response{
		ID:           apiResp.ID,
		Model:        apiResp.Model,
		InputTokens:  apiResp.Usage.InputTokens,
		OutputTokens: apiResp.Usage.OutputTokens,
	}

	// Map reasoning_tokens to ThinkingTokens for cost tracking
	if apiResp.Usage.ReasoningTokens > 0 {
		resp.ThinkingTokens = apiResp.Usage.ReasoningTokens
	}

	// Map status to stop reason
	switch apiResp.Status {
	case "completed":
		resp.StopReason = "end_turn"
	case "incomplete":
		if apiResp.IncompleteDetails != nil {
			switch apiResp.IncompleteDetails.Reason {
			case "max_output_tokens":
				resp.StopReason = "max_tokens"
			default:
				resp.StopReason = apiResp.IncompleteDetails.Reason
			}
		} else {
			resp.StopReason = "max_tokens"
		}
	case "failed":
		resp.StopReason = "error"
	default:
		resp.StopReason = apiResp.Status
	}

	// Extract output items
	var contentParts []string
	for _, item := range apiResp.Output {
		switch item.Type {
		case "function_call":
			var input any
			if err := json.Unmarshal([]byte(item.Arguments), &input); err != nil {
				slog.Warn("failed to parse tool input JSON", "provider", c.providerName, "tool", item.Name, "error", err)
				input = item.Arguments
			}
			resp.ToolCalls = append(resp.ToolCalls, llm.ToolCall{
				ID:    item.CallID,
				Name:  item.Name,
				Input: input,
			})
			// When tool calls are present, the stop reason should indicate tool use
			resp.StopReason = "tool_use"
		case "message":
			for _, block := range item.Content {
				if block.Type == "output_text" && block.Text != "" {
					contentParts = append(contentParts, block.Text)
				}
			}
		}
	}

	if len(contentParts) > 0 {
		resp.Content = strings.Join(contentParts, "\n")
	}

	return resp
}
