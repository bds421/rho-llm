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

	"github.com/bds421/rho-llm"
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

// Close releases resources. Drains idle connections from the HTTP transport
// to prevent connection pool leakage during auth pool rotation.
func (c *Client) Close() error {
	c.httpClient.CloseIdleConnections()
	return nil
}

// Complete generates a non-streaming completion.
func (c *Client) Complete(ctx context.Context, req llm.Request) (*llm.Response, error) {
	model := req.Model
	if model == "" {
		model = c.config.Model
	}
	url := fmt.Sprintf("%s/%s:generateContent", c.baseURL, model)

	apiReq, err := c.buildRequest(req)
	if err != nil {
		return nil, err
	}

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
		return nil, llm.ErrorFromResponse("gemini", resp, c.config)
	}

	var apiResp geminiResponse
	if err := llm.DecodeJSONResponse(resp, c.config, &apiResp); err != nil {
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

		apiReq, err := c.buildRequest(req)
		if err != nil {
			yield(llm.StreamEvent{}, err)
			return
		}

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
			yield(llm.StreamEvent{}, llm.ErrorFromResponse("gemini", resp, c.config))
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
	CachedContent     string                  `json:"cachedContent,omitempty"` // pre-created cache resource name
}

type geminiThinkingConfig struct {
	ThinkingBudget int    `json:"thinkingBudget,omitempty"` // token count (mutually exclusive with ThinkingLevel)
	ThinkingLevel  string `json:"thinkingLevel,omitempty"`  // Gemini 3: "low", "medium", "high"
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text             string                  `json:"text,omitempty"`
	Thought          bool                    `json:"thought,omitempty"`
	InlineData       *geminiInlineData       `json:"inlineData,omitempty"`
	FunctionCall     *geminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResponse `json:"functionResponse,omitempty"`
	ThoughtSignature string                  `json:"thoughtSignature,omitempty"` // Gemini 3: part-level thought signature
}

type geminiInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
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
	Temperature      *float64              `json:"temperature,omitempty"`
	MaxOutputTokens  int                   `json:"maxOutputTokens,omitempty"`
	StopSequences    []string              `json:"stopSequences,omitempty"`
	ThinkingConfig   *geminiThinkingConfig `json:"thinkingConfig,omitempty"`
	ResponseMimeType string                `json:"responseMimeType,omitempty"`
	ResponseSchema   map[string]any        `json:"responseSchema,omitempty"`
}

type geminiTool struct {
	FunctionDeclarations []geminiFunctionDeclaration `json:"functionDeclarations,omitempty"`
}

type geminiFunctionDeclaration struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
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
	// Pointer so "usage absent" (nil) is distinguishable from "zero tokens" —
	// the TokensNotReported sentinel depends on it.
	UsageMetadata *geminiUsageMetadata `json:"usageMetadata"`
	ModelVersion  string               `json:"modelVersion"`
}

type geminiUsageMetadata struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	TotalTokenCount         int `json:"totalTokenCount"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount,omitempty"`
	CachedContentTokenCount int `json:"cachedContentTokenCount,omitempty"`
}

func (c *Client) buildRequest(req llm.Request) (geminiRequest, error) {
	apiReq := geminiRequest{
		GenerationConfig: &geminiGenerationConfig{
			Temperature:     req.Temperature, // nil = omit from wire (provider default)
			MaxOutputTokens: req.MaxTokens,
		},
		CachedContent: req.CachedContent,
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
				if part.Type == llm.ContentText && part.Text != "" {
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
				if part.Text != "" {
					content.Parts = append(content.Parts, geminiPart{Text: part.Text})
				}

			case llm.ContentImage:
				if err := llm.ValidateImageSource(part); err != nil {
					return geminiRequest{}, err
				}
				content.Parts = append(content.Parts, geminiPart{
					InlineData: &geminiInlineData{
						MimeType: part.Source.MediaType,
						Data:     part.Source.Data,
					},
				})

			case llm.ContentDocument:
				// Gemini parses PDFs natively via inlineData (same wire shape as images).
				if err := llm.ValidateDocumentSource(part); err != nil {
					return geminiRequest{}, err
				}
				content.Parts = append(content.Parts, geminiPart{
					InlineData: &geminiInlineData{
						MimeType: part.Document.MediaType,
						Data:     part.Document.Data,
					},
				})

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
							"result": part.ToolResultText(),
						},
					},
				})
			}
		}

		if len(content.Parts) > 0 {
			apiReq.Contents = append(apiReq.Contents, content)
		}
	}

	// Configure stop sequences
	if len(req.StopSequences) > 0 {
		apiReq.GenerationConfig.StopSequences = req.StopSequences
	}

	// Structured output (JSON mode / JSON schema).
	if rf := req.ResponseFormat; rf != nil && rf.Type != "" {
		apiReq.GenerationConfig.ResponseMimeType = "application/json"
		if rf.Type == llm.ResponseFormatJSONSchema && len(rf.Schema) > 0 {
			apiReq.GenerationConfig.ResponseSchema = rf.Schema
		}
	}

	// Configure thinking
	thinkingLevel := req.ThinkingLevel
	if thinkingLevel == llm.ThinkingNone && c.config.ThinkingLevel != llm.ThinkingNone {
		thinkingLevel = c.config.ThinkingLevel
	}

	model := req.Model
	if model == "" {
		model = c.config.Model
	}
	info, hasInfo := llm.GetModelInfo(model)

	if thinkingLevel != llm.ThinkingNone && !(hasInfo && info.Thinking) {
		// Gemini 3+ models accept thinkingConfig inside generationConfig.
		// Gemini 2.5 models (info.Thinking == true) think natively and
		// reject thinkingConfig — skip for those.
		tc := &geminiThinkingConfig{}

		// Prefer string-based thinkingLevel for standard levels (Gemini 3);
		// fall back to thinkingBudget for custom budgets or non-standard levels.
		if req.ThinkingBudget > 0 {
			budget := req.ThinkingBudget
			if hasInfo && info.MaxTokens > 0 {
				budget = llm.ClampThinkingBudget(c.providerName, model, budget, info.MaxTokens)
			}
			tc.ThinkingBudget = budget
		} else {
			switch thinkingLevel {
			case llm.ThinkingLow, llm.ThinkingMedium, llm.ThinkingHigh:
				tc.ThinkingLevel = string(thinkingLevel)
			default:
				// ThinkingMinimal, ThinkingXHigh — no matching Gemini string,
				// fall back to token budget.
				budget := llm.ThinkingBudgetTokens(thinkingLevel, 0)
				if hasInfo && info.MaxTokens > 0 {
					budget = llm.ClampThinkingBudget(c.providerName, model, budget, info.MaxTokens)
				}
				tc.ThinkingBudget = budget
			}
		}
		apiReq.GenerationConfig.ThinkingConfig = tc
	}

	// Gemini 2.5 models think by default but do NOT support thinkingConfig —
	// the API rejects it. Thinking tokens silently consume maxOutputTokens.
	// Pad maxOutputTokens so the caller's intended output budget isn't starved.
	if hasInfo && info.Thinking {
		cur := apiReq.GenerationConfig.MaxOutputTokens
		if cur > 0 {
			padded := cur + llm.ThinkingBudgetTokens(llm.ThinkingLow, 0)
			if info.MaxTokens > 0 && padded > info.MaxTokens {
				padded = info.MaxTokens
			}
			slog.Warn("padding maxOutputTokens for native thinking model",
				"provider", c.providerName, "model", model,
				"original", cur, "padded", padded)
			apiReq.GenerationConfig.MaxOutputTokens = padded
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

	return apiReq, nil
}

func (c *Client) parseResponse(apiResp *geminiResponse, requestModel string) *llm.Response {
	model := requestModel
	if model == "" {
		model = c.config.Model
	}

	resp := &llm.Response{
		Model:        model,
		InputTokens:  llm.TokensNotReported,
		OutputTokens: llm.TokensNotReported,
	}
	if u := apiResp.UsageMetadata; u != nil {
		resp.InputTokens = u.PromptTokenCount
		resp.OutputTokens = u.CandidatesTokenCount
		resp.ThinkingTokens = u.ThoughtsTokenCount
		resp.CacheReadTokens = u.CachedContentTokenCount
	}

	if len(apiResp.Candidates) > 0 {
		candidate := apiResp.Candidates[0]
		resp.StopReason = normalizeStopReason(candidate.FinishReason)

		callIndex := 0
		for _, part := range candidate.Content.Parts {
			if part.Text != "" {
				if part.Thought {
					resp.Thinking += part.Text
				} else {
					resp.Content += part.Text
				}
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
	maxToolInput := c.config.EffectiveMaxToolInputBytes()
	scanner := bufio.NewScanner(body)
	scanner.Buffer(nil, c.config.EffectiveMaxSSELineBytes())

	callIndex := 0
	doneEmitted := false

	// Usage may arrive on any chunk (often only the final one carries it).
	// Track the latest seen, initialized to "not reported" so a stream that
	// never reports usage is distinguishable from a zero-token one.
	inputTokens, outputTokens := llm.TokensNotReported, llm.TokensNotReported
	var thinkingTokens, cacheReadTokens int

	for scanner.Scan() {
		data, ok := llm.SSEData(scanner.Text())
		if !ok {
			continue
		}

		var event geminiResponse
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			if !yield(llm.StreamEvent{}, fmt.Errorf("malformed SSE event from gemini: %w", err)) {
				return
			}
			continue
		}

		if u := event.UsageMetadata; u != nil {
			inputTokens = u.PromptTokenCount
			outputTokens = u.CandidatesTokenCount
			thinkingTokens = u.ThoughtsTokenCount
			cacheReadTokens = u.CachedContentTokenCount
		}

		if len(event.Candidates) > 0 {
			candidate := event.Candidates[0]

			for _, part := range candidate.Content.Parts {
				if part.Text != "" {
					if part.Thought {
						if !yield(llm.StreamEvent{Type: llm.EventThinking, Thinking: part.Text}, nil) {
							return
						}
					} else {
						if !yield(llm.StreamEvent{Type: llm.EventContent, Text: part.Text}, nil) {
							return
						}
					}
				}
				if part.FunctionCall != nil {
					argsJSON, _ := json.Marshal(part.FunctionCall.Args)
					if len(argsJSON) > maxToolInput {
						yield(llm.StreamEvent{}, fmt.Errorf("tool input exceeded %d bytes", maxToolInput))
						return
					}
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
				// The terminal event: emit Done and stop. Returning here means
				// trailing bytes after the turn is complete (including malformed
				// lines) can't surface as a spurious error masking the turn.
				doneEmitted = true
				yield(llm.StreamEvent{
					Type:            llm.EventDone,
					StopReason:      normalizeStopReason(candidate.FinishReason),
					InputTokens:     inputTokens,
					OutputTokens:    outputTokens,
					ThinkingTokens:  thinkingTokens,
					CacheReadTokens: cacheReadTokens,
				}, nil)
				return
			}
		}
	}

	// Only report scanner errors if the stream did not already complete
	// successfully. A trailing read error after EventDone is noise.
	if err := scanner.Err(); err != nil && !doneEmitted {
		yield(llm.StreamEvent{}, fmt.Errorf("stream error: %w", err))
		return
	}

	// Clean EOF without a finishReason: the server truncated the turn. Yield
	// an explicit error instead of ending silently (see the anthropic adapter
	// for the rationale; io.ErrUnexpectedEOF makes it pool-retryable).
	if !doneEmitted {
		yield(llm.StreamEvent{}, fmt.Errorf("gemini: stream ended without completion: %w", io.ErrUnexpectedEOF))
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
