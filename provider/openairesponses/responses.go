// Package openairesponses implements the OpenAI Responses API (/v1/responses) adapter.
// Used for GPT-5 family models that support reasoning effort control only via the
// Responses API, not the Chat Completions API. Streaming is not supported.
package openairesponses

import (
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
	apiReq, err := c.buildRequest(req)
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

// Stream is not supported by the Responses API provider and returns an explicit error.
func (c *Client) Stream(ctx context.Context, req llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		yield(llm.StreamEvent{}, fmt.Errorf(
			"streaming is not supported by the OpenAI Responses API provider (%s); use Complete() instead",
			c.providerName,
		))
	}
}

// =============================================================================
// WIRE FORMAT TYPES
// =============================================================================

type responsesRequest struct {
	Model           string              `json:"model"`
	Input           []responsesInputMsg `json:"input"`
	Reasoning       *responsesReasoning `json:"reasoning,omitempty"`
	MaxOutputTokens int                 `json:"max_output_tokens,omitempty"`
	Temperature     *float64            `json:"temperature,omitempty"`
	Tools           []responsesTool     `json:"tools,omitempty"`
	Store           bool                `json:"store"`
}

type responsesReasoning struct {
	Effort  string `json:"effort"`
	Summary string `json:"summary,omitempty"`
}

type responsesInputMsg struct {
	Role       string      `json:"role"`
	Content    interface{} `json:"content"`                  // string or array of content parts
	ToolCallID string      `json:"tool_call_id,omitempty"`   // Required for tool result messages
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

func (c *Client) buildRequest(req llm.Request) (responsesRequest, error) {
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
		if msg.Role == llm.RoleUser {
			var textParts []string
			var imageParts []llm.ContentPart
			hasToolResults := false

			for _, part := range msg.Content {
				switch part.Type {
				case llm.ContentToolResult:
					hasToolResults = true
					// Responses API tool results require tool_call_id
					apiReq.Input = append(apiReq.Input, responsesInputMsg{
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
