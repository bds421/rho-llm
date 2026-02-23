// Package gemini implements the Google Gemini API adapter.
package gemini

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

	"gitlab2024.bds421-cloud.com/bds421/rho/llm"
)

const (
	defaultGeminiBase = "https://generativelanguage.googleapis.com/v1beta/models"
)

func init() {
	llm.RegisterProvider("gemini", func(cfg llm.Config) (llm.Client, error) {
		return New(cfg)
	})
}

// Client implements the Google Gemini API.
type Client struct {
	config       llm.Config
	baseURL      string // resolved base URL (cfg.BaseURL or default)
	httpClient   *http.Client
	providerName string
}

// New creates a new Gemini client.
func New(cfg llm.Config) (*Client, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("Gemini API key is required")
	}

	if cfg.Model == "" {
		cfg.Model = "gemini-2.0-flash" // Default to Gemini 2.0 Flash
	}

	base := llm.ResolveBaseURL(cfg)
	if base == "" {
		base = defaultGeminiBase
	}

	providerName := cfg.ProviderName
	if providerName == "" {
		providerName = "gemini"
	}

	return &Client{
		config:       cfg,
		baseURL:      base,
		httpClient:   llm.SafeHTTPClient(cfg.Timeout),
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
	return nil
}

// Complete generates a non-streaming completion.
func (c *Client) Complete(ctx context.Context, req llm.Request) (*llm.Response, error) {
	model := req.Model
	if model == "" {
		model = c.config.Model
	}
	url := fmt.Sprintf("%s/%s:generateContent", c.baseURL, model)

	apiReq := c.buildRequest(req)

	body, err := json.Marshal(apiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-goog-api-key", c.config.APIKey)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, readErr := io.ReadAll(io.LimitReader(resp.Body, llm.MaxErrorBodyBytes))
		if readErr != nil {
			slog.Warn("failed to read error response body", "provider", "gemini", "error", readErr)
		}
		return nil, llm.NewAPIErrorFromStatus("gemini", resp.StatusCode, string(body))
	}

	var apiResp geminiResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, llm.MaxResponseBodyBytes)).Decode(&apiResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return c.parseResponse(&apiResp, req.Model), nil
}

// Stream returns an iterator of streaming events.
func (c *Client) Stream(ctx context.Context, req llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		model := req.Model
		if model == "" {
			model = c.config.Model
		}
		url := fmt.Sprintf("%s/%s:streamGenerateContent?alt=sse", c.baseURL, model)

		apiReq := c.buildRequest(req)

		body, err := json.Marshal(apiReq)
		if err != nil {
			yield(llm.StreamEvent{}, fmt.Errorf("failed to marshal request: %w", err))
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
		if err != nil {
			yield(llm.StreamEvent{}, fmt.Errorf("failed to create request: %w", err))
			return
		}

		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("x-goog-api-key", c.config.APIKey)

		resp, err := c.httpClient.Do(httpReq)
		if err != nil {
			yield(llm.StreamEvent{}, fmt.Errorf("request failed: %w", err))
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, readErr := io.ReadAll(io.LimitReader(resp.Body, llm.MaxErrorBodyBytes))
			if readErr != nil {
				slog.Warn("failed to read error response body", "provider", "gemini", "error", readErr)
			}
			yield(llm.StreamEvent{}, llm.NewAPIErrorFromStatus("gemini", resp.StatusCode, string(body)))
			return
		}

		c.parseStream(resp.Body, yield)
	}
}

// Gemini API types
type geminiRequest struct {
	Contents          []geminiContent         `json:"contents"`
	SystemInstruction *geminiContent          `json:"systemInstruction,omitempty"`
	GenerationConfig  *geminiGenerationConfig `json:"generationConfig,omitempty"`
	Tools             []geminiTool            `json:"tools,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text             string                  `json:"text,omitempty"`
	FunctionCall     *geminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResponse `json:"functionResponse,omitempty"`
	ThoughtSignature string                  `json:"thoughtSignature,omitempty"` // Gemini 3: part-level thought signature
}

type geminiFunctionCall struct {
	Name string `json:"name"`
	Args any    `json:"args"`
}

type geminiFunctionResponse struct {
	Name     string `json:"name"`
	Response any    `json:"response"`
}

type geminiGenerationConfig struct {
	Temperature     float64  `json:"temperature"`
	MaxOutputTokens int      `json:"maxOutputTokens,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
}

type geminiTool struct {
	FunctionDeclarations []geminiFunctionDeclaration `json:"functionDeclarations,omitempty"`
}

type geminiFunctionDeclaration struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

type geminiResponse struct {
	Candidates []struct {
		Content       geminiContent `json:"content"`
		FinishReason  string        `json:"finishReason"`
		SafetyRatings []struct {
			Category    string `json:"category"`
			Probability string `json:"probability"`
		} `json:"safetyRatings"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
	ModelVersion string `json:"modelVersion"`
}

func (c *Client) buildRequest(req llm.Request) geminiRequest {
	apiReq := geminiRequest{
		GenerationConfig: &geminiGenerationConfig{
			Temperature:     req.Temperature,
			MaxOutputTokens: req.MaxTokens,
		},
	}

	if apiReq.GenerationConfig.MaxOutputTokens == 0 {
		apiReq.GenerationConfig.MaxOutputTokens = c.config.MaxTokens
	}

	// System instruction
	if req.System != "" {
		apiReq.SystemInstruction = &geminiContent{
			Parts: []geminiPart{{Text: req.System}},
		}
	}

	// Convert messages
	for _, msg := range req.Messages {
		content := geminiContent{}

		// Map roles
		switch msg.Role {
		case llm.RoleUser:
			content.Role = "user"
		case llm.RoleAssistant:
			content.Role = "model"
		case llm.RoleSystem:
			// System messages go to systemInstruction
			if apiReq.SystemInstruction == nil {
				apiReq.SystemInstruction = &geminiContent{}
			}
			for _, part := range msg.Content {
				if part.Type == llm.ContentText {
					apiReq.SystemInstruction.Parts = append(apiReq.SystemInstruction.Parts, geminiPart{Text: part.Text})
				}
			}
			continue
		default:
			content.Role = string(msg.Role)
		}

		// Convert content parts
		for _, part := range msg.Content {
			switch part.Type {
			case llm.ContentText:
				content.Parts = append(content.Parts, geminiPart{Text: part.Text})

			case llm.ContentToolUse:
				gp := geminiPart{
					FunctionCall: &geminiFunctionCall{
						Name: part.ToolName,
						Args: part.ToolInput,
					},
				}
				// Gemini 3: thoughtSignature is at the Part level, sibling to functionCall
				if part.ThoughtSignature != "" {
					gp.ThoughtSignature = part.ThoughtSignature
				}
				content.Parts = append(content.Parts, gp)

			case llm.ContentToolResult:
				funcName := resolveToolName(part.ToolResultID)
				content.Parts = append(content.Parts, geminiPart{
					FunctionResponse: &geminiFunctionResponse{
						Name: funcName,
						Response: map[string]string{
							"result": part.ToolResultContent,
						},
					},
				})
			}
		}

		if len(content.Parts) > 0 {
			apiReq.Contents = append(apiReq.Contents, content)
		}
	}

	// Convert tools
	if len(req.Tools) > 0 {
		tool := geminiTool{}
		for _, t := range req.Tools {
			tool.FunctionDeclarations = append(tool.FunctionDeclarations, geminiFunctionDeclaration{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			})
		}
		apiReq.Tools = []geminiTool{tool}
	}

	return apiReq
}

func (c *Client) parseResponse(apiResp *geminiResponse, requestModel string) *llm.Response {
	model := requestModel
	if model == "" {
		model = c.config.Model
	}

	resp := &llm.Response{
		Model:        model,
		InputTokens:  apiResp.UsageMetadata.PromptTokenCount,
		OutputTokens: apiResp.UsageMetadata.CandidatesTokenCount,
	}

	if len(apiResp.Candidates) > 0 {
		candidate := apiResp.Candidates[0]
		resp.StopReason = normalizeStopReason(candidate.FinishReason)

		callIndex := 0
		for _, part := range candidate.Content.Parts {
			if part.Text != "" {
				resp.Content += part.Text
			}
			if part.FunctionCall != nil {
				tc := llm.ToolCall{
					ID:    makeToolCallID(callIndex, part.FunctionCall.Name),
					Name:  part.FunctionCall.Name,
					Input: part.FunctionCall.Args,
				}
				callIndex++
				// Gemini 3: thoughtSignature is at the Part level
				if part.ThoughtSignature != "" {
					tc.ThoughtSignature = part.ThoughtSignature
				}
				resp.ToolCalls = append(resp.ToolCalls, tc)
			}
		}
	}

	return resp
}

func (c *Client) parseStream(body io.Reader, yield func(llm.StreamEvent, error) bool) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(nil, llm.MaxSSELineBytes)

	callIndex := 0
	for scanner.Scan() {
		line := scanner.Text()

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")

		var event geminiResponse
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			if !yield(llm.StreamEvent{}, fmt.Errorf("malformed SSE event from gemini: %w", err)) {
				return
			}
			continue
		}

		if len(event.Candidates) > 0 {
			candidate := event.Candidates[0]

			for _, part := range candidate.Content.Parts {
				if part.Text != "" {
					if !yield(llm.StreamEvent{Type: llm.EventContent, Text: part.Text}, nil) {
						return
					}
				}
				if part.FunctionCall != nil {
					tc := &llm.ToolCall{
						ID:    makeToolCallID(callIndex, part.FunctionCall.Name),
						Name:  part.FunctionCall.Name,
						Input: part.FunctionCall.Args,
					}
					callIndex++
					// Gemini 3: thoughtSignature is at the Part level
					if part.ThoughtSignature != "" {
						tc.ThoughtSignature = part.ThoughtSignature
					}
					if !yield(llm.StreamEvent{
						Type:     llm.EventToolUse,
						ToolCall: tc,
					}, nil) {
						return
					}
				}
			}

			if candidate.FinishReason != "" {
				if !yield(llm.StreamEvent{
					Type:         llm.EventDone,
					StopReason:   normalizeStopReason(candidate.FinishReason),
					InputTokens:  event.UsageMetadata.PromptTokenCount,
					OutputTokens: event.UsageMetadata.CandidatesTokenCount,
				}, nil) {
					return
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		yield(llm.StreamEvent{}, fmt.Errorf("stream error: %w", err))
	}
}

// normalizeStopReason maps Gemini finish reasons to the unified set
// (end_turn, tool_use, max_tokens) used by the library interface.
func normalizeStopReason(reason string) string {
	switch reason {
	case "STOP":
		return "end_turn"
	case "FUNCTION_CALLING":
		return "tool_use"
	case "MAX_TOKENS":
		return "max_tokens"
	default:
		return reason
	}
}

// Gemini doesn't provide tool call IDs like OpenAI/Anthropic. We generate synthetic
// IDs so the library interface stays consistent. The format must be invertible so
// tool results can recover the original function name.
//
// Format: "call_<index>_<name>" where index ensures uniqueness for parallel calls.

// makeToolCallID generates a synthetic tool call ID. See resolveToolName for inverse.
func makeToolCallID(index int, name string) string {
	return fmt.Sprintf("call_%d_%s", index, name)
}

// resolveToolName recovers the function name from a synthetic tool_use_id.
// Inverse of makeToolCallID. Also handles legacy "call_<name>" format.
func resolveToolName(toolUseID string) string {
	if !strings.HasPrefix(toolUseID, "call_") {
		return toolUseID
	}
	rest := strings.TrimPrefix(toolUseID, "call_")

	// Current format: "call_<digits>_<name>" — strip numeric index prefix
	if idx := strings.IndexByte(rest, '_'); idx > 0 {
		allDigits := true
		for _, c := range rest[:idx] {
			if c < '0' || c > '9' {
				allDigits = false
				break
			}
		}
		if allDigits {
			return rest[idx+1:]
		}
	}

	// Legacy format: "call_<name>"
	return rest
}
