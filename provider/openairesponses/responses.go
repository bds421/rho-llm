// Package openairesponses implements the OpenAI Responses API (/v1/responses) adapter.
// Used for GPT-5 family models that support reasoning effort control only via the
// Responses API, not the Chat Completions API.
package openairesponses

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
	llm.RegisterProvider("openai_responses", func(cfg llm.Config) (llm.Client, error) {
		return New(cfg)
	})
}

// Client implements the OpenAI Responses API for GPT-5 family models.
type Client struct {
	config       llm.Config
	httpClient   *http.Client
	baseURL      string
	authHeader   string
	providerName string
}

// New creates a new OpenAI Responses API client.
func New(cfg llm.Config) (*Client, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("openai API key is required for Responses API")
	}

	baseURL := llm.ResolveBaseURL(cfg)
	if baseURL == "" {
		return nil, fmt.Errorf("no base URL configured for provider %s", cfg.Provider)
	}

	authHeader := cfg.AuthHeader
	if authHeader == "" {
		if preset, ok := llm.PresetFor(cfg.Provider); ok {
			authHeader = preset.AuthHeader
		}
	}

	providerName := cfg.ProviderName
	if providerName == "" {
		providerName = "openai_responses"
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

// Close releases resources.
func (c *Client) Close() error {
	c.httpClient.CloseIdleConnections()
	return nil
}

// Complete generates a non-streaming completion via the Responses API.
func (c *Client) Complete(ctx context.Context, req llm.Request) (*llm.Response, error) {
	apiReq, err := c.buildRequest(req, false)
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

	return c.parseResponse(&apiResp), nil
}

// Stream returns an iterator of streaming events via the Responses API SSE stream.
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

		url := c.baseURL + "/responses"
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
			errBody, readErr := io.ReadAll(io.LimitReader(resp.Body, llm.MaxErrorBodyBytes))
			if readErr != nil {
				slog.Warn("failed to read error response body", "provider", c.providerName, "error", readErr)
			}
			yield(llm.StreamEvent{}, llm.NewAPIErrorFromStatus(c.providerName, resp.StatusCode, string(errBody)))
			return
		}

		c.parseStream(resp.Body, yield)
	}
}

// parseStream reads SSE events from the Responses API stream and yields
// llm.StreamEvent values to the caller's iterator.
func (c *Client) parseStream(body io.Reader, yield func(llm.StreamEvent, error) bool) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(nil, llm.MaxSSELineBytes)

	// Per-tool-call accumulation: the Responses API sends argument deltas
	// followed by a single "done" event with the full arguments, name, and call_id.
	// We accumulate deltas as a safety net but prefer the done event's values.
	var inputBuffer strings.Builder
	var completed bool

	for scanner.Scan() {
		line := scanner.Text()

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		// Parse the event type first to decide how to handle it.
		var envelope struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal([]byte(data), &envelope); err != nil {
			if !yield(llm.StreamEvent{}, fmt.Errorf("malformed SSE event from %s: %w", c.providerName, err)) {
				return
			}
			continue
		}

		switch envelope.Type {
		case "response.output_text.delta":
			var ev struct {
				Delta string `json:"delta"`
			}
			if err := json.Unmarshal([]byte(data), &ev); err != nil {
				if !yield(llm.StreamEvent{}, fmt.Errorf("malformed text delta from %s: %w", c.providerName, err)) {
					return
				}
				continue
			}
			if ev.Delta != "" {
				if !yield(llm.StreamEvent{Type: llm.EventContent, Text: ev.Delta}, nil) {
					return
				}
			}

		case "response.function_call_arguments.delta":
			var ev struct {
				Delta string `json:"delta"`
			}
			if err := json.Unmarshal([]byte(data), &ev); err != nil {
				if !yield(llm.StreamEvent{}, fmt.Errorf("malformed function call delta from %s: %w", c.providerName, err)) {
					return
				}
				continue
			}
			if inputBuffer.Len()+len(ev.Delta) > llm.MaxToolInputBytes {
				yield(llm.StreamEvent{}, fmt.Errorf("tool input exceeded %d bytes", llm.MaxToolInputBytes))
				return
			}
			inputBuffer.WriteString(ev.Delta)

		case "response.function_call_arguments.done":
			var ev struct {
				Name      string `json:"name"`
				CallID    string `json:"call_id"`
				Arguments string `json:"arguments"`
			}
			if err := json.Unmarshal([]byte(data), &ev); err != nil {
				if !yield(llm.StreamEvent{}, fmt.Errorf("malformed function call done from %s: %w", c.providerName, err)) {
					return
				}
				continue
			}
			var input any
			args := ev.Arguments
			if args == "" {
				args = inputBuffer.String()
			}
			if err := json.Unmarshal([]byte(args), &input); err != nil {
				slog.Warn("failed to parse tool input JSON", "provider", c.providerName, "tool", ev.Name, "error", err)
				input = args
			}
			if !yield(llm.StreamEvent{
				Type: llm.EventToolUse,
				ToolCall: &llm.ToolCall{
					ID:    ev.CallID,
					Name:  ev.Name,
					Input: input,
				},
			}, nil) {
				return
			}
			inputBuffer.Reset()

		case "response.reasoning_summary_text.delta":
			var ev struct {
				Delta string `json:"delta"`
			}
			if err := json.Unmarshal([]byte(data), &ev); err != nil {
				if !yield(llm.StreamEvent{}, fmt.Errorf("malformed reasoning delta from %s: %w", c.providerName, err)) {
					return
				}
				continue
			}
			if ev.Delta != "" {
				if !yield(llm.StreamEvent{Type: llm.EventThinking, Thinking: ev.Delta}, nil) {
					return
				}
			}

		case "response.completed":
			var ev struct {
				Response struct {
					ID     string         `json:"id"`
					Status string         `json:"status"`
					Usage  responsesUsage `json:"usage"`
				} `json:"response"`
			}
			if err := json.Unmarshal([]byte(data), &ev); err != nil {
				if !yield(llm.StreamEvent{}, fmt.Errorf("malformed completed event from %s: %w", c.providerName, err)) {
					return
				}
				continue
			}
			stopReason := "end_turn"
			if ev.Response.Status == "incomplete" {
				stopReason = "max_tokens"
			} else if ev.Response.Status == "failed" {
				stopReason = "error"
			}
			completed = true
			if !yield(llm.StreamEvent{
				Type:           llm.EventDone,
				StopReason:     stopReason,
				InputTokens:    ev.Response.Usage.InputTokens,
				OutputTokens:   ev.Response.Usage.OutputTokens,
				ThinkingTokens: ev.Response.Usage.ReasoningTokens,
			}, nil) {
				return
			}

		case "error":
			var ev struct {
				Error struct {
					Message string `json:"message"`
					Code    string `json:"code"`
				} `json:"error"`
			}
			if err := json.Unmarshal([]byte(data), &ev); err != nil {
				yield(llm.StreamEvent{}, fmt.Errorf("malformed error event from %s: %w", c.providerName, err))
				return
			}
			msg := ev.Error.Message
			if msg == "" {
				msg = "unknown error"
			}
			if ev.Error.Code != "" {
				msg = ev.Error.Code + ": " + msg
			}
			yield(llm.StreamEvent{}, fmt.Errorf("stream error from %s: %s", c.providerName, msg))
			return

		default:
			// Skip events we don't need to handle (response.created,
			// response.output_item.added, response.output_item.done, etc.)
		}
	}

	if !completed && scanner.Err() == nil {
		slog.Warn("stream ended without response.completed event", "provider", c.providerName)
	}

	if err := scanner.Err(); err != nil && !completed {
		yield(llm.StreamEvent{}, fmt.Errorf("stream error: %w", err))
	}
}

// =============================================================================
// WIRE FORMAT TYPES
// =============================================================================

type responsesRequest struct {
	Model           string  `json:"model"`
	Input           []any   `json:"input"` // mix of responsesInputMsg, responsesFunctionCall, responsesFunctionCallOutput
	Reasoning       *responsesReasoning `json:"reasoning,omitempty"`
	MaxOutputTokens int                 `json:"max_output_tokens,omitempty"`
	Temperature     *float64            `json:"temperature,omitempty"`
	Tools           []responsesTool     `json:"tools,omitempty"`
	Store           bool                `json:"store"`
	Stream          bool                `json:"stream,omitempty"`
}

type responsesReasoning struct {
	Effort  string `json:"effort"`
	Summary string `json:"summary,omitempty"`
}

// responsesInputMsg represents a conversation message (user, assistant, system).
type responsesInputMsg struct {
	Role    string `json:"role"`
	Content any    `json:"content"` // string or array of content parts
}

// responsesFunctionCall represents an assistant-issued function call re-submitted as context.
type responsesFunctionCall struct {
	Type      string `json:"type"`      // "function_call"
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// responsesFunctionCallOutput represents a tool result for the Responses API.
type responsesFunctionCallOutput struct {
	Type   string `json:"type"`    // "function_call_output"
	CallID string `json:"call_id"`
	Output string `json:"output"`
}

type responsesTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string                 `json:"name"`
		Description string                 `json:"description"`
		Parameters  map[string]interface{} `json:"parameters"`
	} `json:"function"`
}

type responsesResponse struct {
	ID                string               `json:"id"`
	Model             string               `json:"model"`
	Status            string               `json:"status"` // completed, incomplete, failed
	Output            []responsesOutputItem `json:"output"`
	Usage             responsesUsage       `json:"usage"`
	IncompleteDetails *responsesIncomplete `json:"incomplete_details"`
}

type responsesOutputItem struct {
	Type    string                  `json:"type"` // message, function_call, reasoning
	ID      string                  `json:"id"`
	Role    string                  `json:"role,omitempty"`
	Content []responsesContentBlock `json:"content,omitempty"`
	// function_call fields
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	// reasoning fields
	Summary []responsesSummaryText `json:"summary,omitempty"`
}

type responsesSummaryText struct {
	Type string `json:"type"` // summary_text
	Text string `json:"text"`
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

// =============================================================================
// REQUEST BUILDING
// =============================================================================

func (c *Client) buildRequest(req llm.Request, stream bool) (responsesRequest, error) {
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
		Store:           false,
		Stream:          stream,
	}

	// Temperature: all ResponsesAPI models are reasoning models that reject
	// custom temperature (they only accept the default). Omit it entirely,
	// matching the openaicompat adapter's behavior for reasoning models.
	if req.Temperature != nil {
		slog.Warn("ignoring Temperature for reasoning model (Responses API)",
			"provider", c.providerName, "model", model)
	}

	// Set reasoning effort and optional summary
	if thinkingLevel != llm.ThinkingNone {
		reasoning := &responsesReasoning{Effort: string(thinkingLevel)}
		if req.ReasoningSummary != llm.ReasoningSummaryNone {
			reasoning.Summary = string(req.ReasoningSummary)
		}
		apiReq.Reasoning = reasoning
	} else if req.ReasoningSummary != llm.ReasoningSummaryNone {
		// Summary can be set even without an explicit effort level
		apiReq.Reasoning = &responsesReasoning{Summary: string(req.ReasoningSummary)}
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
		switch msg.Role {
		case llm.RoleUser:
			if err := c.buildUserMessage(&apiReq, msg); err != nil {
				return responsesRequest{}, err
			}

		case llm.RoleAssistant:
			c.buildAssistantMessage(&apiReq, msg)

		default:
			// System or other roles passed as plain messages
			var textParts []string
			for _, part := range msg.Content {
				if part.Type == llm.ContentText && part.Text != "" {
					textParts = append(textParts, part.Text)
				}
			}
			if len(textParts) > 0 {
				apiReq.Input = append(apiReq.Input, responsesInputMsg{
					Role:    string(msg.Role),
					Content: strings.Join(textParts, "\n"),
				})
			}
		}
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

// buildUserMessage handles user messages: text, images, and tool results.
// Tool results are emitted as function_call_output items (Responses API format).
func (c *Client) buildUserMessage(apiReq *responsesRequest, msg llm.Message) error {
	var textParts []string
	var imageParts []llm.ContentPart

	for _, part := range msg.Content {
		switch part.Type {
		case llm.ContentToolResult:
			// Responses API uses function_call_output items, not role="tool" messages
			apiReq.Input = append(apiReq.Input, responsesFunctionCallOutput{
				Type:   "function_call_output",
				CallID: part.ToolResultID,
				Output: part.ToolResultContent,
			})
		case llm.ContentText:
			if part.Text != "" {
				textParts = append(textParts, part.Text)
			}
		case llm.ContentImage:
			if err := llm.ValidateImageSource(part); err != nil {
				return err
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
		return nil
	}

	if len(textParts) > 0 {
		apiReq.Input = append(apiReq.Input, responsesInputMsg{
			Role:    "user",
			Content: strings.Join(textParts, "\n"),
		})
	}
	return nil
}

// buildAssistantMessage handles assistant messages: text content and tool-use
// parts. Tool-use parts are emitted as function_call items (Responses API format).
func (c *Client) buildAssistantMessage(apiReq *responsesRequest, msg llm.Message) {
	var textParts []string

	for _, part := range msg.Content {
		switch part.Type {
		case llm.ContentText:
			if part.Text != "" {
				textParts = append(textParts, part.Text)
			}
		case llm.ContentToolUse:
			// Emit text accumulated so far before the function call
			if len(textParts) > 0 {
				apiReq.Input = append(apiReq.Input, responsesInputMsg{
					Role:    "assistant",
					Content: strings.Join(textParts, "\n"),
				})
				textParts = nil
			}
			// Serialize tool input to JSON string
			inputJSON, err := json.Marshal(part.ToolInput)
			if err != nil {
				slog.Warn("failed to marshal tool input", "provider", c.providerName, "tool", part.ToolName, "error", err)
				inputJSON = []byte("{}")
			}
			apiReq.Input = append(apiReq.Input, responsesFunctionCall{
				Type:      "function_call",
				CallID:    part.ToolUseID,
				Name:      part.ToolName,
				Arguments: string(inputJSON),
			})
		}
	}

	// Emit any remaining text
	if len(textParts) > 0 {
		apiReq.Input = append(apiReq.Input, responsesInputMsg{
			Role:    "assistant",
			Content: strings.Join(textParts, "\n"),
		})
	}
}

// =============================================================================
// RESPONSE PARSING
// =============================================================================

func (c *Client) parseResponse(apiResp *responsesResponse) *llm.Response {
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
	var thinkingParts []string
	for _, item := range apiResp.Output {
		switch item.Type {
		case "reasoning":
			for _, s := range item.Summary {
				if s.Type == "summary_text" && s.Text != "" {
					thinkingParts = append(thinkingParts, s.Text)
				}
			}
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

	if len(thinkingParts) > 0 {
		resp.Thinking = strings.Join(thinkingParts, "\n")
	}
	if len(contentParts) > 0 {
		resp.Content = strings.Join(contentParts, "\n")
	}

	return resp
}
