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
	"net/url"
	"strings"

	llm "github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/internal/toolinput"
	"github.com/bds421/rho-llm/internal/urlerr"
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
	config     llm.Config
	baseURL    string // resolved base URL (cfg.BaseURL or default)
	httpClient *http.Client
	// streamClient serves Stream: same transport, no whole-body Timeout
	// (H3). See llm.NewStreamingHTTPClient.
	streamClient *http.Client
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
	httpClient, err := llm.NewSafeHTTPClient(cfg)
	if err != nil {
		return nil, err
	}

	return &Client{
		config:       cfg,
		baseURL:      base,
		httpClient:   httpClient,
		streamClient: llm.NewStreamingHTTPClient(httpClient, cfg),
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

// streamHTTPClient returns the streaming client, deriving it from httpClient
// for a Client value built without New (tests).
func (c *Client) streamHTTPClient() *http.Client {
	if c.streamClient != nil {
		return c.streamClient
	}
	return llm.NewStreamingHTTPClient(c.httpClient, c.config)
}

// Complete generates a non-streaming completion.
func (c *Client) Complete(ctx context.Context, req llm.Request) (*llm.Response, error) {
	model := req.Model
	if model == "" {
		model = c.config.Model
	}
	endpoint := fmt.Sprintf("%s/%s:generateContent", c.baseURL, url.PathEscape(model))

	apiReq, err := c.buildRequest(req)
	if err != nil {
		return nil, err
	}

	// Long decode (Gemini 4): a turn that needs more than one request's
	// output ends with finishReason CONTINUATION and a continuationToken. The
	// next hop is the same request plus that token, with no partial output
	// added to contents; the hops' parts and usage add up to one turn.
	var merged *geminiResponse
	for hop := 1; ; hop++ {
		apiResp, err := c.completeHop(ctx, endpoint, apiReq, req.SamplingParams)
		if err != nil {
			return nil, err
		}
		merged = mergeHop(merged, apiResp)
		token := apiResp.continuation()
		if token == "" || hop >= maxContinuationHops {
			break
		}
		apiReq.ContinuationToken = token
	}
	return c.parseResponse(merged, req.Model), nil
}

// maxContinuationHops bounds a long-decode turn. The server stops the turn at
// maxOutputTokens anyway (1,048,576 at most, 32,768 or more per hop); the
// bound only guards against a server that keeps answering CONTINUATION.
const maxContinuationHops = 64

// completeHop sends one generateContent request.
func (c *Client) completeHop(ctx context.Context, endpoint string, apiReq geminiRequest, sampling map[string]any) (*geminiResponse, error) {
	body, err := llm.MergeSamplingParams(apiReq, sampling)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-goog-api-key", c.config.APIKey)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", urlerr.Redact(err))
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, llm.ErrorFromResponse("gemini", resp, c.config)
	}

	var apiResp geminiResponse
	if err := llm.DecodeJSONResponse(resp, c.config, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}
	if apiResp.Error != nil && len(apiResp.Candidates) == 0 {
		return nil, c.apiError(apiResp.Error)
	}
	return &apiResp, nil
}

// continuation returns the token for the next long-decode hop, or "" when
// the turn is finished.
func (r *geminiResponse) continuation() string {
	if len(r.Candidates) == 0 || r.Candidates[0].FinishReason != finishContinuation {
		return ""
	}
	return r.Candidates[0].ContinuationToken
}

// mergeHop folds one long-decode hop into the turn so far: parts are
// appended, the finish reason and token are the latest hop's, and usage is
// summed (each hop reports, and bills, its own counts).
func mergeHop(acc, hop *geminiResponse) *geminiResponse {
	if acc == nil {
		return hop
	}
	acc.UsageMetadata = acc.UsageMetadata.add(hop.UsageMetadata)
	acc.PromptFeedback = hop.PromptFeedback
	if len(hop.Candidates) == 0 {
		// No candidate on a later hop (a blocked continuation): the turn
		// ends with what that hop reported.
		acc.Candidates = nil
		return acc
	}
	if len(acc.Candidates) == 0 {
		acc.Candidates = hop.Candidates[:1]
		return acc
	}
	cand := &acc.Candidates[0]
	next := hop.Candidates[0]
	cand.Content.Parts = append(cand.Content.Parts, next.Content.Parts...)
	cand.FinishReason = next.FinishReason
	cand.ContinuationToken = next.ContinuationToken
	return acc
}

// Stream returns an iterator of streaming events.
func (c *Client) Stream(ctx context.Context, req llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		model := req.Model
		if model == "" {
			model = c.config.Model
		}
		endpoint := fmt.Sprintf("%s/%s:streamGenerateContent?alt=sse", c.baseURL, url.PathEscape(model))

		apiReq, err := c.buildRequest(req)
		if err != nil {
			yield(llm.StreamEvent{}, err)
			return
		}

		// Long decode: see Complete. Each hop streams on; only the last one
		// ends with EventDone. A stream that drops mid-hop fails as before
		// (the events already yielded cannot be taken back, so resuming from
		// a mid-stream checkpoint is not attempted: that needs every event held back
		// until the next checkpoint, up to 5,000 tokens of added latency).
		st := &streamState{}
		for hop := 1; ; hop++ {
			token, more := c.streamHop(ctx, endpoint, apiReq, req.SamplingParams, st, hop < maxContinuationHops, yield)
			if !more {
				return
			}
			apiReq.ContinuationToken = token
		}
	}
}

// streamHop sends one streamGenerateContent request and streams its events.
// more reports that the hop ended in CONTINUATION and the turn goes on with
// token.
func (c *Client) streamHop(ctx context.Context, endpoint string, apiReq geminiRequest, sampling map[string]any,
	st *streamState, canContinue bool, yield func(llm.StreamEvent, error) bool) (token string, more bool) {
	body, err := llm.MergeSamplingParams(apiReq, sampling)
	if err != nil {
		yield(llm.StreamEvent{}, fmt.Errorf("failed to marshal request: %w", err))
		return "", false
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(body))
	if err != nil {
		yield(llm.StreamEvent{}, fmt.Errorf("failed to create request: %w", err))
		return "", false
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-goog-api-key", c.config.APIKey)

	resp, err := c.streamHTTPClient().Do(httpReq)
	if err != nil {
		yield(llm.StreamEvent{}, fmt.Errorf("request failed: %w", urlerr.Redact(err)))
		return "", false
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		yield(llm.StreamEvent{}, llm.ErrorFromResponse("gemini", resp, c.config))
		return "", false
	}

	return c.parseStreamHop(resp.Body, st, canContinue, yield)
}

// Gemini API types
type geminiRequest struct {
	Contents          []geminiContent         `json:"contents"`
	SystemInstruction *geminiContent          `json:"systemInstruction,omitempty"`
	GenerationConfig  *geminiGenerationConfig `json:"generationConfig,omitempty"`
	Tools             []geminiTool            `json:"tools,omitempty"`
	ToolConfig        *geminiToolConfig       `json:"toolConfig,omitempty"`
	CachedContent     string                  `json:"cachedContent,omitempty"` // pre-created cache resource name
	// ContinuationToken resumes a long-decode turn (Gemini 4); top level,
	// per the REST example at aistudio.google.com/docs/long-decode.
	ContinuationToken string `json:"continuationToken,omitempty"`
}

// geminiToolConfig carries Gemini's function-calling mode. Gemini spells tool
// choice differently from every other provider: an uppercase mode enum, with
// "force one named tool" expressed as mode ANY plus a one-element
// allowedFunctionNames list rather than a dedicated mode.
type geminiToolConfig struct {
	FunctionCallingConfig geminiFunctionCallingConfig `json:"functionCallingConfig"`
}

type geminiFunctionCallingConfig struct {
	Mode                 string   `json:"mode"`
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

type geminiThinkingConfig struct {
	ThinkingBudget  int    `json:"thinkingBudget,omitempty"`  // token count (mutually exclusive with ThinkingLevel)
	ThinkingLevel   string `json:"thinkingLevel,omitempty"`   // Gemini 3: "low", "medium", "high"
	IncludeThoughts bool   `json:"includeThoughts,omitempty"` // return thought summaries (parts with thought: true)
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text    string `json:"text,omitempty"`
	Thought bool   `json:"thought,omitempty"`
	// AudioTranscription carries the transcript from dedicated transcription
	// models (gemini-3.5-transcribe), which do not use the text field.
	AudioTranscription *geminiAudioTranscription `json:"audioTranscription,omitempty"`
	InlineData         *geminiInlineData         `json:"inlineData,omitempty"`
	FunctionCall       *geminiFunctionCall       `json:"functionCall,omitempty"`
	FunctionResponse   *geminiFunctionResponse   `json:"functionResponse,omitempty"`
	ThoughtSignature   string                    `json:"thoughtSignature,omitempty"` // Gemini 3: part-level thought signature
}

type geminiAudioTranscription struct {
	Text string `json:"text"`
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

// geminiFunctionDeclaration sends the tool schema as parametersJsonSchema,
// which takes full JSON Schema. The older "parameters" field takes only
// Gemini's OpenAPI subset and answers 400 "Unknown name" for keywords such as
// additionalProperties or $defs that the other adapters pass through.
type geminiFunctionDeclaration struct {
	Name                 string         `json:"name"`
	Description          string         `json:"description"`
	ParametersJSONSchema map[string]any `json:"parametersJsonSchema"`
}

type geminiCandidate struct {
	Content      geminiContent `json:"content"`
	FinishReason string        `json:"finishReason"`
	// ContinuationToken comes with finishReason CONTINUATION (and as a
	// checkpoint on some stream chunks); see Complete.
	ContinuationToken string `json:"continuationToken,omitempty"`
	SafetyRatings     []struct {
		Category    string `json:"category"`
		Probability string `json:"probability"`
	} `json:"safetyRatings"`
}

type geminiResponse struct {
	Candidates []geminiCandidate `json:"candidates"`
	// Pointer so "usage absent" (nil) is distinguishable from "zero tokens" —
	// the TokensNotReported sentinel depends on it.
	UsageMetadata *geminiUsageMetadata `json:"usageMetadata"`
	ModelVersion  string               `json:"modelVersion"`
	// A blocked prompt has no candidates, only promptFeedback.blockReason.
	PromptFeedback *struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback"`
	// An error after the 200, sent as a stream chunk.
	Error *geminiError `json:"error"`
}

// geminiError is the Google API error object ({"code","message","status"}).
type geminiError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
}

// blockReason returns the prompt's block reason when Gemini refused it
// before generating any candidate.
func (r *geminiResponse) blockReason() string {
	if len(r.Candidates) == 0 && r.PromptFeedback != nil {
		return r.PromptFeedback.BlockReason
	}
	return ""
}

// apiError turns an in-band error object into the *llm.APIError the same
// error would produce as an HTTP status, with the deployment's secrets
// scrubbed from the message.
func (c *Client) apiError(e *geminiError) error {
	status := e.Code
	if status < 400 || status > 599 {
		status = 500
	}
	message := e.Message
	if e.Status != "" {
		message = e.Status + ": " + message
	}
	message = llm.RedactSecrets(message, c.config)
	return llm.NewAPIErrorFromStatusWithLimit("gemini", status, message, c.config.EffectiveMaxErrorMessageLen())
}

type geminiUsageMetadata struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	TotalTokenCount         int `json:"totalTokenCount"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount,omitempty"`
	CachedContentTokenCount int `json:"cachedContentTokenCount,omitempty"`
}

// splitPrompt maps Gemini's usage onto the library's cache-token contract
// (see llm.Response): promptTokenCount is the TOTAL effective prompt and
// already includes cachedContentTokenCount ("When cachedContent is set, this
// is still the total effective prompt size meaning this includes the number of
// tokens in the cached content" — ai.google.dev/api/generate-content), so the
// cached share is subtracted to get the uncached InputTokens. Negative wire
// values count as 0, and a cached count above the prompt total is clamped to
// it, so InputTokens is never negative and the two never sum past the prompt.
// add sums two hops' usage. Nil means not reported; a sum with one
// reported side is that side. Negative counts are clamped to 0.
func (u *geminiUsageMetadata) add(v *geminiUsageMetadata) *geminiUsageMetadata {
	if u == nil {
		return v
	}
	if v == nil {
		return u
	}
	return &geminiUsageMetadata{
		PromptTokenCount:        max(u.PromptTokenCount, 0) + max(v.PromptTokenCount, 0),
		CandidatesTokenCount:    max(u.CandidatesTokenCount, 0) + max(v.CandidatesTokenCount, 0),
		TotalTokenCount:         max(u.TotalTokenCount, 0) + max(v.TotalTokenCount, 0),
		ThoughtsTokenCount:      max(u.ThoughtsTokenCount, 0) + max(v.ThoughtsTokenCount, 0),
		CachedContentTokenCount: max(u.CachedContentTokenCount, 0) + max(v.CachedContentTokenCount, 0),
	}
}

func (u *geminiUsageMetadata) splitPrompt() (input, cacheRead int) {
	prompt, cached := max(u.PromptTokenCount, 0), max(u.CachedContentTokenCount, 0)
	cached = min(cached, prompt)
	return prompt - cached, cached
}

func (c *Client) buildRequest(req llm.Request) (geminiRequest, error) {
	if err := req.ToolChoice.Validate(); err != nil {
		return geminiRequest{}, err
	}
	if err := llm.UnsupportedServerTools(c.Provider(), req); err != nil {
		return geminiRequest{}, err
	}
	reqModel := req.Model
	if reqModel == "" {
		reqModel = c.config.Model
	}
	traits := traitsFor(reqModel)
	if req.CachedContent != "" && traits.noExplicitCache {
		return geminiRequest{}, fmt.Errorf("gemini: %s does not support explicit caching (CachedContent); "+
			"it caches repeated prompt prefixes implicitly", reqModel)
	}
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

	// functionResponse.name must be the called function's name. Look it up
	// from the call in the same transcript: IDs are opaque once normalization
	// renamed a duplicate or another provider made the call (B6).
	toolNames := toolNamesByID(req.Messages)

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
				if llm.IsTextDocument(part) {
					return geminiRequest{}, fmt.Errorf("gemini: text documents are Anthropic-only; send the text as a text part")
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
						Args: toolinput.Object(part.ToolInput),
					},
				}
				// Gemini 3: thoughtSignature is at the Part level, sibling to functionCall
				if part.ThoughtSignature != "" {
					gp.ThoughtSignature = part.ThoughtSignature
				}
				content.Parts = append(content.Parts, gp)

			case llm.ContentToolResult:
				funcName, ok := toolNames[part.ToolResultID]
				if !ok {
					funcName = resolveToolName(part.ToolResultID)
				}
				// A failed tool goes under "error", the key Gemini's
				// FunctionResponse reference names for error details; IsError
				// was dropped, so the model read a failure as a result (B30).
				resultKey := "result"
				if part.IsError {
					resultKey = "error"
				}
				content.Parts = append(content.Parts, geminiPart{
					FunctionResponse: &geminiFunctionResponse{
						Name: funcName,
						Response: map[string]string{
							resultKey: part.ToolResultText(),
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
	if req.Effort != "" {
		return geminiRequest{}, fmt.Errorf("gemini: Effort is not supported; use ThinkingLevel")
	}
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
		//
		// includeThoughts asks for thought summaries. Without it Gemini
		// returns no thought parts, so a caller that set ThinkingLevel never
		// saw EventThinking or Response.Thinking (B25).
		tc := &geminiThinkingConfig{IncludeThoughts: true}

		// Prefer string-based thinkingLevel for standard levels (Gemini 3);
		// fall back to thinkingBudget for custom budgets or non-standard levels.
		if req.ThinkingBudget > 0 {
			tc.ThinkingBudget = c.clampThinkingBudget(model, req.ThinkingBudget, info, hasInfo)
		} else {
			switch thinkingLevel {
			case llm.ThinkingLow, llm.ThinkingMedium, llm.ThinkingHigh:
				tc.ThinkingLevel = string(thinkingLevel)
			case llm.ThinkingXHigh, llm.ThinkingMax:
				// "xhigh" exists on Gemini 4 Hydrogen only; elsewhere "high"
				// is the deepest level. A 128K token budget instead was
				// clamped to the model's 65,536 output limit, one above the
				// API's thinkingBudget maximum, and rejected with a 400.
				tc.ThinkingLevel = string(llm.ThinkingHigh)
				if traits.xhighThinking {
					tc.ThinkingLevel = string(llm.ThinkingXHigh)
				}
			case llm.ThinkingMinimal:
				if traits.noMinimalThinking {
					// Gemini 3.8 Flash and Gemini 4 answer "minimal" with an
					// error and warn on thinkingBudget; "low" is their
					// shallowest level.
					tc.ThinkingLevel = string(llm.ThinkingLow)
					break
				}
				budget := llm.ThinkingBudgetTokens(thinkingLevel, 0)
				tc.ThinkingBudget = c.clampThinkingBudget(model, budget, info, hasInfo)
			default:
				// ThinkingMinimal — no matching Gemini string, fall back to a
				// token budget.
				budget := llm.ThinkingBudgetTokens(thinkingLevel, 0)
				tc.ThinkingBudget = c.clampThinkingBudget(model, budget, info, hasInfo)
			}
		}
		apiReq.GenerationConfig.ThinkingConfig = tc
	}

	// Gemini 2.5 models think by default but do NOT support thinkingConfig —
	// the API rejects it. Gemini 3+ models (SupportsThinking) also think by
	// default, even without a ThinkingLevel. Either way thinking tokens
	// silently consume maxOutputTokens, so pad it so the caller's intended
	// output budget isn't starved.
	if hasInfo && (info.Thinking || info.SupportsThinking) {
		cur := apiReq.GenerationConfig.MaxOutputTokens
		if cur > 0 {
			padded := cur + llm.ThinkingBudgetTokens(llm.ThinkingLow, 0)
			// MaxTokens is one request's output limit. A long-decode model
			// continues past it, so the caller's total is never lowered.
			if info.MaxTokens > 0 && padded > info.MaxTokens && !traits.longDecode {
				padded = info.MaxTokens
			}
			slog.Debug("padding maxOutputTokens for native thinking model",
				"provider", c.providerName, "model", model,
				"original", cur, "padded", padded)
			apiReq.GenerationConfig.MaxOutputTokens = padded
		}
	}

	// Convert tools
	if len(req.Tools) > 0 {
		tool := geminiTool{}
		for _, t := range req.Tools {
			params := t.InputSchema
			if params == nil {
				params = map[string]any{"type": "object"} // valid object, not null
			}
			tool.FunctionDeclarations = append(tool.FunctionDeclarations, geminiFunctionDeclaration{
				Name:                 t.Name,
				Description:          t.Description,
				ParametersJSONSchema: params,
			})
		}
		apiReq.Tools = []geminiTool{tool}
		apiReq.ToolConfig = toolChoiceWire(req.ToolChoice)
	}

	return apiReq, nil
}

// toolChoiceWire maps the neutral ToolChoice onto Gemini's toolConfig. Gemini
// has no "required" mode: forcing a call is mode ANY, and forcing one specific
// tool is ANY narrowed by allowedFunctionNames. AUTO is the default, so it is
// left off the wire.
func toolChoiceWire(tc *llm.ToolChoice) *geminiToolConfig {
	if tc == nil {
		return nil
	}
	switch tc.Mode {
	case llm.ToolChoiceNone:
		return &geminiToolConfig{FunctionCallingConfig: geminiFunctionCallingConfig{Mode: "NONE"}}
	case llm.ToolChoiceRequired:
		return &geminiToolConfig{FunctionCallingConfig: geminiFunctionCallingConfig{Mode: "ANY"}}
	case llm.ToolChoiceTool:
		return &geminiToolConfig{FunctionCallingConfig: geminiFunctionCallingConfig{
			Mode:                 "ANY",
			AllowedFunctionNames: []string{tc.Name},
		}}
	default: // ToolChoiceAuto — omit, it is the default
		return nil
	}
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
		resp.InputTokens, resp.CacheReadTokens = u.splitPrompt()
		resp.OutputTokens = max(u.CandidatesTokenCount, 0)
		resp.ThinkingTokens = max(u.ThoughtsTokenCount, 0)
	}

	// A blocked prompt: no candidates, a block reason. It is a refusal, not
	// an empty successful answer.
	if reason := apiResp.blockReason(); reason != "" {
		resp.StopReason = llm.StopRefusal
		resp.RawStopReason = reason
		return resp
	}

	if len(apiResp.Candidates) > 0 {
		candidate := apiResp.Candidates[0]
		resp.StopReason = normalizeStopReason(candidate.FinishReason)
		resp.RawStopReason = candidate.FinishReason

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

		// Gemini has no distinct tool-calling finish reason: a turn that
		// requests a function call still reports "STOP". Infer tool_use from
		// the parts so callers can drive the standard agentic loop
		// (`for resp.StopReason == "tool_use"`). Only a normal completion is
		// reclassified — MAX_TOKENS and friends must survive, since a
		// truncated turn is not a usable tool call.
		if len(resp.ToolCalls) > 0 && resp.StopReason == "end_turn" {
			resp.StopReason = "tool_use"
		}
	}

	return resp
}

// streamState carries one streamed turn across its long-decode hops.
type streamState struct {
	callIndex int
	// Gemini reports finishReason "STOP" even for a turn that requests a
	// function call, so the terminal event must infer tool_use from whether
	// any tool call was actually emitted. See parseResponse for the rationale.
	sawToolCall bool
	// usage is the summed usage of the finished hops (nil: none reported).
	usage *geminiUsageMetadata
}

// parseStream parses a single-hop stream.
func (c *Client) parseStream(body io.Reader, yield func(llm.StreamEvent, error) bool) {
	c.parseStreamHop(body, &streamState{}, false, yield)
}

// parseStreamHop parses one hop of a streamed turn. When canContinue is set
// and the hop ends in CONTINUATION with a token, it yields no EventDone and
// returns the token with more = true.
func (c *Client) parseStreamHop(body io.Reader, st *streamState, canContinue bool, yield func(llm.StreamEvent, error) bool) (token string, more bool) {
	maxToolInput := c.config.EffectiveMaxToolInputBytes()
	scanner := bufio.NewScanner(body)
	scanner.Buffer(nil, c.config.EffectiveMaxSSELineBytes())

	doneEmitted := false

	// Usage may arrive on any chunk (often only the final one carries it).
	// Track this hop's latest; nil means "not reported", so a stream that
	// never reports usage is distinguishable from a zero-token one.
	var hopUsage *geminiUsageMetadata
	usage := func() (input, output, thinking, cacheRead int) {
		u := st.usage.add(hopUsage)
		if u == nil {
			return llm.TokensNotReported, llm.TokensNotReported, 0, 0
		}
		input, cacheRead = u.splitPrompt()
		return input, max(u.CandidatesTokenCount, 0), max(u.ThoughtsTokenCount, 0), cacheRead
	}

	for scanner.Scan() {
		data, ok := llm.SSEData(scanner.Text())
		if !ok {
			continue
		}

		var event geminiResponse
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			if !yield(llm.StreamEvent{}, fmt.Errorf("malformed SSE event from gemini: %w", err)) {
				return "", false
			}
			continue
		}

		if event.Error != nil {
			yield(llm.StreamEvent{}, c.apiError(event.Error))
			return "", false
		}

		if event.UsageMetadata != nil {
			hopUsage = event.UsageMetadata
		}

		// A blocked prompt ends the stream as a refusal; without this it ran
		// to EOF and surfaced as a retryable truncation, re-sending the prompt.
		if reason := event.blockReason(); reason != "" {
			inputTokens, outputTokens, thinkingTokens, cacheReadTokens := usage()
			yield(llm.StreamEvent{
				Type:            llm.EventDone,
				StopReason:      llm.StopRefusal,
				RawStopReason:   reason,
				InputTokens:     inputTokens,
				OutputTokens:    outputTokens,
				ThinkingTokens:  thinkingTokens,
				CacheReadTokens: cacheReadTokens,
			}, nil)
			return "", false
		}

		if len(event.Candidates) > 0 {
			candidate := event.Candidates[0]

			for _, part := range candidate.Content.Parts {
				if part.Text != "" {
					if part.Thought {
						if !yield(llm.StreamEvent{Type: llm.EventThinking, Thinking: part.Text}, nil) {
							return "", false
						}
					} else {
						if !yield(llm.StreamEvent{Type: llm.EventContent, Text: part.Text}, nil) {
							return "", false
						}
					}
				}
				if part.FunctionCall != nil {
					argsJSON, _ := json.Marshal(part.FunctionCall.Args)
					if len(argsJSON) > maxToolInput {
						yield(llm.StreamEvent{}, fmt.Errorf("tool input exceeded %d bytes", maxToolInput))
						return "", false
					}
					tc := &llm.ToolCall{
						ID:    makeToolCallID(st.callIndex, part.FunctionCall.Name),
						Name:  part.FunctionCall.Name,
						Input: part.FunctionCall.Args,
					}
					st.callIndex++
					// Gemini 3: thoughtSignature is at the Part level
					if part.ThoughtSignature != "" {
						tc.ThoughtSignature = part.ThoughtSignature
					}
					st.sawToolCall = true
					if !yield(llm.StreamEvent{
						Type:     llm.EventToolUse,
						ToolCall: tc,
					}, nil) {
						return "", false
					}
				}
			}

			if candidate.ContinuationToken != "" {
				// The final chunk's token, or a mid-stream checkpoint.
				token = candidate.ContinuationToken
			}

			if candidate.FinishReason == finishContinuation && canContinue && token != "" {
				// Not the end of the turn: the next hop continues it.
				st.usage = st.usage.add(hopUsage)
				return token, true
			}

			if candidate.FinishReason != "" {
				// The terminal event: emit Done and stop. Returning here means
				// trailing bytes after the turn is complete (including malformed
				// lines) can't surface as a spurious error masking the turn.
				doneEmitted = true
				stopReason := normalizeStopReason(candidate.FinishReason)
				// Only a normal completion is reclassified: MAX_TOKENS and
				// friends must survive, since a truncated turn is not a
				// usable tool call.
				if st.sawToolCall && stopReason == "end_turn" {
					stopReason = "tool_use"
				}
				inputTokens, outputTokens, thinkingTokens, cacheReadTokens := usage()
				yield(llm.StreamEvent{
					Type:            llm.EventDone,
					StopReason:      stopReason,
					RawStopReason:   candidate.FinishReason,
					InputTokens:     inputTokens,
					OutputTokens:    outputTokens,
					ThinkingTokens:  thinkingTokens,
					CacheReadTokens: cacheReadTokens,
				}, nil)
				return "", false
			}
		}
	}

	// Only report scanner errors if the stream did not already complete
	// successfully. A trailing read error after EventDone is noise.
	if err := scanner.Err(); err != nil && !doneEmitted {
		yield(llm.StreamEvent{}, fmt.Errorf("stream error: %w", err))
		return "", false
	}

	// Clean EOF without a finishReason: the server truncated the turn. Yield
	// an explicit error instead of ending silently (see the anthropic adapter
	// for the rationale; io.ErrUnexpectedEOF makes it pool-retryable).
	if !doneEmitted {
		yield(llm.StreamEvent{}, fmt.Errorf("gemini: stream ended without completion: %w", io.ErrUnexpectedEOF))
	}
	return "", false
}

// normalizeStopReason maps Gemini finish reasons to the unified set
// (end_turn, tool_use, max_tokens, refusal, error) used by the library
// interface. Content filters (safety, recitation, blocklist, personal data,
// image filters, ESCALATION: "an escalation rule filtered the request", and
// PUP_LIMITED_DISABLED: generation stopped because the account is restricted
// for a Prohibited Use Policy violation) are refusals: the model was barred
// from answering, nothing failed in transit. A failed turn (MALFORMED_FUNCTION_CALL and the
// other tool-call failures, OTHER, LANGUAGE, an unspecified reason) is
// StopError: before, the raw string passed through as if it were a normal
// stop, and a caller's tool loop ended on a turn that had failed.
// RawStopReason keeps the original value.
func normalizeStopReason(reason string) string {
	switch reason {
	case "STOP":
		return llm.StopEndTurn
	case "FUNCTION_CALLING":
		return llm.StopToolUse
	case "MAX_TOKENS", finishContinuation:
		// CONTINUATION reaches the caller only when maxContinuationHops ran
		// out: the turn is truncated, not failed.
		return llm.StopMaxTokens
	case "SAFETY", "PROHIBITED_CONTENT", "RECITATION", "BLOCKLIST", "SPII",
		"IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION",
		"ESCALATION", "PUP_LIMITED_DISABLED":
		return llm.StopRefusal
	case "":
		return ""
	default:
		return llm.StopError
	}
}

// Gemini doesn't provide tool call IDs like OpenAI/Anthropic. We generate synthetic
// IDs so the library interface stays consistent. The format must be invertible so
// tool results can recover the original function name.
//
// Format: "call_<index>_<name>" where index ensures uniqueness for parallel calls.

// toolNamesByID maps every tool-use ID in msgs to its function name.
func toolNamesByID(msgs []llm.Message) map[string]string {
	names := make(map[string]string)
	for _, m := range msgs {
		for _, p := range m.Content {
			if p.Type == llm.ContentToolUse && p.ToolUseID != "" && p.ToolName != "" {
				names[p.ToolUseID] = p.ToolName
			}
		}
	}
	return names
}

// makeToolCallID generates a synthetic tool call ID. See resolveToolName for inverse.
func makeToolCallID(index int, name string) string {
	return fmt.Sprintf("call_%d_%s", index, name)
}

// resolveToolName recovers the function name from a synthetic tool_use_id when
// the transcript does not carry the call (toolNamesByID finds no match).
// It is the strict inverse of makeToolCallID. Values outside the current
// synthetic-ID grammar pass through unchanged; callers must not reinterpret
// an older or provider-owned ID as a Gemini function name.
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

	return toolUseID
}

// maxThinkingBudget is the largest thinkingBudget the Gemini API accepts
// ("thinking_budget must be in the range [-1, 65535]") — one below the 65,536
// output limit of most Gemini models.
const maxThinkingBudget = 65535

// clampThinkingBudget clamps a thinking budget to the model's output limit and
// to the API's thinkingBudget maximum.
func (c *Client) clampThinkingBudget(model string, budget int, info llm.ModelInfo, hasInfo bool) int {
	if hasInfo && info.MaxTokens > 0 {
		budget = llm.ClampThinkingBudget(c.providerName, model, budget, info.MaxTokens)
	}
	return llm.ClampThinkingBudget(c.providerName, model, budget, maxThinkingBudget)
}
