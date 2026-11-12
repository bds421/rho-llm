// Package anthropic implements the Anthropic Claude API adapter.
package anthropic

import (
	"bufio"
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"net/http"
	"strings"

	llm "github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/internal/toolinput"
	"github.com/bds421/rho-llm/internal/urlerr"
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
	config     llm.Config
	endpoint   string // resolved messages endpoint (cfg.BaseURL or default)
	httpClient *http.Client
	// streamClient serves Stream: same transport, no whole-body Timeout
	// (H3). See llm.NewStreamingHTTPClient.
	streamClient *http.Client
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
	httpClient, err := llm.NewSafeHTTPClient(cfg)
	if err != nil {
		return nil, err
	}

	return &Client{
		config:       cfg,
		endpoint:     base + "/messages",
		httpClient:   httpClient,
		streamClient: llm.NewStreamingHTTPClient(httpClient, cfg),
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
	Model         string                 `json:"model"`
	Messages      []anthropicMessage     `json:"messages"`
	System        any                    `json:"system,omitempty"`
	MaxTokens     int                    `json:"max_tokens"`
	Temperature   *float64               `json:"temperature,omitempty"`
	Tools         []any                  `json:"tools,omitempty"` // anthropicTool, then server tools
	ToolChoice    any                    `json:"tool_choice,omitempty"`
	Stream        bool                   `json:"stream,omitempty"`
	Thinking      *anthropicThinking     `json:"thinking,omitempty"`
	OutputConfig  *anthropicOutputConfig `json:"output_config,omitempty"`
	StopSequences []string               `json:"stop_sequences,omitempty"`
	Container     string                 `json:"container,omitempty"`
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
// Anthropic currently only supports type "ephemeral"; TTL is "1h" or omitted
// (5 minutes).
type anthropicCacheControl struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitempty"`
}

// cacheControl returns the cache_control marker for a CacheTTL value.
func cacheControl(ttl string) (*anthropicCacheControl, error) {
	switch ttl {
	case "", llm.CacheTTL5m:
		return &anthropicCacheControl{Type: "ephemeral"}, nil
	case llm.CacheTTL1h:
		return &anthropicCacheControl{Type: "ephemeral", TTL: llm.CacheTTL1h}, nil
	}
	return nil, fmt.Errorf("anthropic: unsupported cache TTL %q (use %q or %q)", ttl, llm.CacheTTL5m, llm.CacheTTL1h)
}

type anthropicThinking struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitempty"`
	Display      string `json:"display,omitempty"`
}

type anthropicOutputConfig struct {
	Effort string                 `json:"effort,omitempty"`
	Format *anthropicOutputFormat `json:"format,omitempty"`
}

type anthropicOutputFormat struct {
	Type   string         `json:"type"`
	Schema map[string]any `json:"schema"`
}

// anthropicResponse is the Anthropic API response format.
type anthropicResponse struct {
	ID          string           `json:"id"`
	Type        string           `json:"type"`
	Role        string           `json:"role"`
	Model       string           `json:"model"`
	Content     []anthropicBlock `json:"content"`
	StopReason  string           `json:"stop_reason"`
	StopDetails json.RawMessage  `json:"stop_details,omitempty"`
	Usage       anthropicUsage   `json:"usage"`
	Container   *struct {
		ID string `json:"id"`
	} `json:"container,omitempty"`
}

// interleavedThinkingBeta is DefaultConfig's beta flag. It only does anything
// with budget thinking (Claude 4.5 and older; adaptive models interleave on
// their own), and it is deprecated, so it goes out only with such a request.
const interleavedThinkingBeta = "interleaved-thinking-2025-05-14"

// setBetaHeader sets the anthropic-beta header from Config.BetaFeatures on
// every request (server-side fallback and other betas do not depend on
// thinking), except the interleaved-thinking flag without budget thinking.
func (c *Client) setBetaHeader(httpReq *http.Request, apiReq anthropicRequest) {
	budgetThinking := apiReq.Thinking != nil && apiReq.Thinking.Type == "enabled"
	if betas := BetaFeatures(c.config.BetaFeatures, budgetThinking); len(betas) > 0 {
		httpReq.Header.Set("anthropic-beta", strings.Join(betas, ","))
	}
}

// BetaFeatures returns the anthropic-beta values to send: the configured
// features, with the interleaved-thinking beta only when a request uses
// budget thinking. The batch client sends the same set.
func BetaFeatures(features []string, budgetThinking bool) []string {
	var betas []string
	for _, beta := range features {
		if beta == interleavedThinkingBeta && !budgetThinking {
			continue
		}
		betas = append(betas, beta)
	}
	return betas
}

func (c *Client) doRequest(ctx context.Context, req llm.Request, stream bool) (*llm.Response, error) {
	// Fall back to config-level ThinkingLevel if not set on the request
	if req.ThinkingLevel == llm.ThinkingNone && c.config.ThinkingLevel != llm.ThinkingNone {
		req.ThinkingLevel = c.config.ThinkingLevel
	}

	if err := llm.ValidateRequestCapabilities(c.capabilityConfig(), req, stream); err != nil {
		return nil, err
	}
	apiReq, err := c.buildRequest(req, stream)
	if err != nil {
		return nil, err
	}

	body, err := llm.MergeSamplingParams(apiReq, req.SamplingParams)
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
	c.setBetaHeader(httpReq, apiReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", urlerr.Redact(err))
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

	if err := llm.ValidateRequestCapabilities(c.capabilityConfig(), req, true); err != nil {
		yield(llm.StreamEvent{}, err)
		return
	}
	apiReq, err := c.buildRequest(req, true)
	if err != nil {
		yield(llm.StreamEvent{}, err)
		return
	}

	body, err := llm.MergeSamplingParams(apiReq, req.SamplingParams)
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
	c.setBetaHeader(httpReq, apiReq)

	resp, err := c.streamHTTPClient().Do(httpReq)
	if err != nil {
		yield(llm.StreamEvent{}, fmt.Errorf("request failed: %w", urlerr.Redact(err)))
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		yield(llm.StreamEvent{}, llm.ErrorFromResponse("anthropic", resp, c.config))
		return
	}

	c.parseStream(resp.Body, yield)
}

// minThinkingBudgetTokens is Anthropic's documented floor for
// thinking.budget_tokens; the API rejects anything smaller.
const minThinkingBudgetTokens = 1024

// thinkingOff is how a model accepts ThinkingNone.
type thinkingOff int

const (
	thinkingOffDisabled     thinkingOff = iota // {"type":"disabled"}
	thinkingOffBetweenTools                    // {"type":"between_tools"} — Sonnet 5.5 rejects "disabled"
	thinkingOffOmit                            // always-on thinking: any explicit off is a 400
)

type adaptiveThinkingProfile struct {
	off                thinkingOff
	noForcedToolChoice bool // tool_choice "any"/"tool" returns a 400
	// disabledUpToHigh: {"type":"disabled"} is accepted only at effort low,
	// medium or high; with xhigh or max it returns a 400.
	disabledUpToHigh bool
}

// adaptiveThinkingModels are the Claude models that accept only adaptive
// thinking. Per platform.claude.com/docs/en/build-with-claude/extended-thinking
// (2026-10-31): "Claude 4.7 and later models do not support it [budget_tokens]
// and reject requests that use it, returning a 400 error." Every anthropic
// registry row must be classified here or listed as a budget model in the
// tests, so a new Claude model is a deliberate decision.
var adaptiveThinkingModels = map[string]adaptiveThinkingProfile{
	"claude-fable-5-2":  {off: thinkingOffOmit, noForcedToolChoice: true},
	"claude-fable-5-1":  {off: thinkingOffOmit, noForcedToolChoice: true},
	"claude-fable-5":    {off: thinkingOffOmit},
	"claude-opus-5-5":   {off: thinkingOffOmit, noForcedToolChoice: true},
	"claude-opus-5":     {off: thinkingOffDisabled, disabledUpToHigh: true},
	"claude-opus-4-8":   {off: thinkingOffDisabled},
	"claude-opus-4-7":   {off: thinkingOffDisabled},
	"claude-sonnet-5-5": {off: thinkingOffBetweenTools, noForcedToolChoice: true},
	"claude-sonnet-5":   {off: thinkingOffDisabled},
	"claude-haiku-5-5":  {off: thinkingOffDisabled, disabledUpToHigh: true},
}

func (c *Client) buildRequest(req llm.Request, stream bool) (anthropicRequest, error) {
	// Keep live and batch encoding consistent when reasoning is configured on
	// the client rather than on each individual request.
	if req.ThinkingLevel == llm.ThinkingNone {
		req.ThinkingLevel = c.config.ThinkingLevel
	}
	if err := req.ToolChoice.Validate(); err != nil {
		return anthropicRequest{}, err
	}
	if err := llm.AnthropicSamplingError(c.config, req); err != nil {
		return anthropicRequest{}, err
	}
	if c.config.AnthropicStrictVersion() {
		if err := strictVersionContent(req.Messages); err != nil {
			return anthropicRequest{}, err
		}
	}
	if err := llm.ValidateCacheMarkers(req); err != nil {
		return anthropicRequest{}, err
	}
	if req.CachedContent != "" {
		return anthropicRequest{}, fmt.Errorf("anthropic: CachedContent is a Gemini cache name; Anthropic caches by prefix, use CacheControl on the system prompt, tools or content")
	}
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
	// Thinking is on for this request unless the model takes {"type":"disabled"}
	// and no ThinkingLevel is set: always-on models omit the parameter, and
	// between_tools (Sonnet 5.5) still writes thinking blocks between tool
	// calls, which must be echoed back.
	profile, adaptive := adaptiveThinkingModels[llm.ResolveModelAlias(apiReq.Model)]
	thinkingOn := req.ThinkingLevel != llm.ThinkingNone ||
		(adaptive && (profile.off == thinkingOffOmit || profile.off == thinkingOffBetweenTools))

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
				// Anthropic rejects a text block that is empty or only
				// whitespace ("text content blocks must contain non-whitespace
				// text"); such a part carries nothing, so it is left out (B31).
				if strings.TrimSpace(part.Text) != "" {
					block := map[string]any{
						"type": "text",
						"text": part.Text,
					}
					if part.CacheControl {
						cc, err := cacheControl(part.CacheTTL)
						if err != nil {
							return anthropicRequest{}, err
						}
						block["cache_control"] = cc
					}
					if len(part.Citations) > 0 {
						citations, err := citationsWire(part.Citations)
						if err != nil {
							return anthropicRequest{}, err
						}
						block["citations"] = citations
					}
					apiMsg.Content = append(apiMsg.Content, block)
				}
			case llm.ContentServerToolUse, llm.ContentServerToolResult, llm.ContentProviderBlock:
				block, ok, err := replayBlock(part)
				if err != nil {
					return anthropicRequest{}, err
				}
				if ok {
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
					cc, err := cacheControl(part.CacheTTL)
					if err != nil {
						return anthropicRequest{}, err
					}
					block["cache_control"] = cc
				}
				apiMsg.Content = append(apiMsg.Content, block)
			case llm.ContentDocument:
				// A base64 PDF or a plain-text source; both share the source shape.
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
				if part.Title != "" {
					block["title"] = part.Title
				}
				if part.Context != "" {
					block["context"] = part.Context
				}
				if part.CacheControl {
					cc, err := cacheControl(part.CacheTTL)
					if err != nil {
						return anthropicRequest{}, err
					}
					block["cache_control"] = cc
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
				// Models whose thinking is always on count as enabled.
				if !thinkingOn {
					continue
				}
				// A signed block goes back unchanged even with empty text (the
				// default display "omitted" returns exactly that): dropping or
				// editing it breaks a continuation and the cached prefix.
				if part.ProviderBlock != nil && (part.Redacted || part.ThinkingSignature != "") {
					block, _, err := replayBlock(part)
					if err != nil {
						return anthropicRequest{}, err
					}
					apiMsg.Content = append(apiMsg.Content, block)
				} else if part.Redacted && part.Thinking != "" {
					apiMsg.Content = append(apiMsg.Content, map[string]any{
						"type": "redacted_thinking",
						"data": part.Thinking,
					})
				} else if part.ThinkingSignature != "" {
					apiMsg.Content = append(apiMsg.Content, map[string]any{
						"type":      "thinking",
						"thinking":  part.Thinking,
						"signature": part.ThinkingSignature,
					})
				}
			case llm.ContentToolUse:
				input := toolinput.Object(part.ToolInput) // Anthropic requires an object
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
							if strings.TrimSpace(sub.Text) != "" {
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
					// A rich result whose parts were all empty serialized as
					// "content": null; content is optional, so leave it out.
					if len(blocks) > 0 {
						block["content"] = blocks
					}
				} else {
					block["content"] = part.ToolResultContent
				}
				apiMsg.Content = append(apiMsg.Content, block)
			}
		}
		if len(apiMsg.Content) == 0 {
			// Everything filtered out (e.g. an empty-text part, or thinking dropped
			// when thinking is off). Skip it — Anthropic rejects an empty-content
			// message, and NormalizeForProvider drops empties the same way.
			continue
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
				cc, err := cacheControl(req.SystemCacheTTL)
				if err != nil {
					return anthropicRequest{}, err
				}
				block["cache_control"] = cc
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
				cc, err := cacheControl(tool.CacheTTL)
				if err != nil {
					return anthropicRequest{}, err
				}
				at.CacheControl = cc
			}
			apiReq.Tools = append(apiReq.Tools, at)
		}
	}
	for _, st := range req.ServerTools {
		wire, err := serverToolWire(st)
		if err != nil {
			return anthropicRequest{}, err
		}
		apiReq.Tools = append(apiReq.Tools, wire)
	}
	if len(apiReq.Tools) > 0 {
		// Anthropic rejects tool_choice without a tools array, so it only rides
		// along when tools were actually built.
		apiReq.ToolChoice = toolChoiceWire(req.ToolChoice)
	}
	apiReq.Container = req.ContainerID

	// Configure stop sequences
	if len(req.StopSequences) > 0 {
		apiReq.StopSequences = req.StopSequences
	}

	format, err := outputFormat(req.ResponseFormat)
	if err != nil {
		return anthropicRequest{}, err
	}
	if format != nil {
		apiReq.OutputConfig = &anthropicOutputConfig{Format: format}
	}

	// Configure thinking
	resolvedModel := llm.ResolveModelAlias(apiReq.Model)
	profile, adaptiveModel := adaptiveThinkingModels[resolvedModel]
	effort, err := requestEffort(req, resolvedModel, adaptiveModel)
	if err != nil {
		return anthropicRequest{}, err
	}
	if effort != "" {
		if apiReq.OutputConfig == nil {
			apiReq.OutputConfig = &anthropicOutputConfig{}
		}
		apiReq.OutputConfig.Effort = effort
	}
	if adaptiveModel {
		// Claude 4.7 and later reject thinking.type "enabled" + budget_tokens
		// with a 400; depth is set by output_config.effort instead.
		if apiReq.ToolChoice != nil && profile.noForcedToolChoice &&
			(req.ToolChoice.Mode == llm.ToolChoiceRequired || req.ToolChoice.Mode == llm.ToolChoiceTool) {
			return anthropicRequest{}, fmt.Errorf("anthropic: %s rejects forced tool_choice (%q); use ToolChoiceAuto and name the tool in the prompt", resolvedModel, req.ToolChoice.Mode)
		}
		if req.ThinkingLevel == llm.ThinkingNone {
			switch profile.off {
			case thinkingOffDisabled:
				if e := cmp.Or(effort, samplingEffort(req.SamplingParams)); profile.disabledUpToHigh && aboveHigh(e) {
					return anthropicRequest{}, fmt.Errorf("anthropic: %s rejects thinking disabled at effort %q; set a ThinkingLevel for adaptive thinking, or use effort high or lower", resolvedModel, e)
				}
				apiReq.Thinking = &anthropicThinking{Type: "disabled"}
			case thinkingOffBetweenTools:
				if e := cmp.Or(effort, samplingEffort(req.SamplingParams)); aboveHigh(e) {
					return anthropicRequest{}, fmt.Errorf("anthropic: %s accepts thinking between_tools only at effort high or lower, not %q; set a ThinkingLevel for adaptive thinking", resolvedModel, e)
				}
				apiReq.Thinking = &anthropicThinking{Type: "between_tools"}
			case thinkingOffOmit:
				// Thinking is always on and any explicit "off" is a 400:
				// leave the parameter out and let the model run adaptive.
			}
		} else {
			if req.ThinkingBudget > 0 {
				return anthropicRequest{}, fmt.Errorf("anthropic: %s does not support ThinkingBudget; use ThinkingLevel to control adaptive effort", resolvedModel)
			}
			apiReq.Thinking = &anthropicThinking{Type: "adaptive", Display: thinkingDisplay(req.ReasoningSummary)}
		}
	} else if req.ThinkingLevel != llm.ThinkingNone {
		budget := llm.ThinkingBudgetTokens(req.ThinkingLevel, req.ThinkingBudget)
		// Clamp budget to the model's max output tokens — budget_tokens cannot
		// exceed max_tokens, which itself cannot exceed the model's limit.
		if info, ok := llm.GetModelInfo(apiReq.Model); ok && info.MaxTokens > 0 {
			budget = llm.ClampThinkingBudget("anthropic", apiReq.Model, budget, info.MaxTokens)
		}
		// Then clamp to THIS request's max_tokens. The registry ceiling is the
		// model's limit, not the caller's budget: asking for ThinkingHigh
		// (65536) with MaxTokens 16000 otherwise sent budget_tokens 64000 and
		// Anthropic rejected it ("max_tokens must be greater than
		// thinking.budget_tokens"). Reserve a slice of the budget for the
		// visible answer, so enabling thinking never starves the output.
		if apiReq.MaxTokens > 0 && budget >= apiReq.MaxTokens {
			budget = llm.ClampThinkingBudget(
				"anthropic", apiReq.Model, budget, apiReq.MaxTokens-apiReq.MaxTokens/4,
			)
		}
		// Anthropic also enforces a floor: budget_tokens must be >= 1024. A
		// max_tokens small enough that the reserve drops below it cannot carry
		// thinking at all, so disable thinking rather than send a request the
		// API rejects from the other direction.
		if budget < minThinkingBudgetTokens {
			if apiReq.MaxTokens > minThinkingBudgetTokens {
				budget = minThinkingBudgetTokens
			} else {
				slog.Debug("disabling thinking: max_tokens too small for the minimum budget",
					"provider", "anthropic", "model", apiReq.Model,
					"max_tokens", apiReq.MaxTokens, "min_budget", minThinkingBudgetTokens)
				return apiReq, nil
			}
		}
		// Extended thinking accepts only tool_choice auto or none; "any" or a
		// named tool is a 400 on every model. The guard covered only the
		// adaptive models (B31).
		if apiReq.ToolChoice != nil &&
			(req.ToolChoice.Mode == llm.ToolChoiceRequired || req.ToolChoice.Mode == llm.ToolChoiceTool) {
			return anthropicRequest{}, fmt.Errorf("anthropic: extended thinking rejects forced tool_choice (%q); use ToolChoiceAuto, or turn thinking off for this request", req.ToolChoice.Mode)
		}
		apiReq.Thinking = &anthropicThinking{
			Type:         "enabled",
			BudgetTokens: budget,
		}
	}

	return apiReq, nil
}

// toolChoiceWire maps the neutral ToolChoice onto Anthropic's shape, which is an
// object rather than a bare string: {"type":"auto"|"any"|"none"} — note that
// Anthropic spells "force some tool" as "any", not "required" — or
// {"type":"tool","name":…} to force one named tool. Auto is the default and is
// left off the wire.
func toolChoiceWire(tc *llm.ToolChoice) any {
	if tc == nil {
		return nil
	}
	switch tc.Mode {
	case llm.ToolChoiceNone:
		return map[string]any{"type": "none"}
	case llm.ToolChoiceRequired:
		return map[string]any{"type": "any"}
	case llm.ToolChoiceTool:
		return map[string]any{"type": "tool", "name": tc.Name}
	default: // ToolChoiceAuto — omit, it is the default
		return nil
	}
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
		ID:                    apiResp.ID,
		Model:                 apiResp.Model,
		StopReason:            normalizeStopReason(apiResp.StopReason),
		RawStopReason:         apiResp.StopReason,
		InputTokens:           apiResp.Usage.InputTokens,
		OutputTokens:          apiResp.Usage.OutputTokens,
		CacheCreationTokens:   apiResp.Usage.CacheCreationTokens,
		CacheReadTokens:       apiResp.Usage.CacheReadTokens,
		ServerToolUse:         apiResp.Usage.serverToolUse(),
		CacheCreation1hTokens: apiResp.Usage.CacheCreation.Ephemeral1h,
		StopDetails:           llm.StopDetailsFromRaw(apiResp.StopDetails),
		Iterations:            apiResp.Usage.iterations(),
	}
	if apiResp.Container != nil {
		resp.ContainerID = apiResp.Container.ID
	}

	parts := make([]llm.ContentPart, 0, len(apiResp.Content))
	for _, block := range apiResp.Content {
		parts = append(parts, blockPart(block))
	}
	resp.SetParts(parts)
	return resp
}

func (c *Client) parseStream(body io.Reader, yield func(llm.StreamEvent, error) bool) {
	maxToolInput := c.config.EffectiveMaxToolInputBytes()
	scanner := bufio.NewScanner(body)
	scanner.Buffer(nil, c.config.EffectiveMaxSSELineBytes())

	var currentToolCall *llm.ToolCall
	currentToolIndex := -1
	var inputBuffer strings.Builder
	// Token counts initialize to "not reported". If the stream ends before
	// message_start/message_delta events, callers can distinguish "not reported"
	// from "zero tokens" (0).
	var inputTokens = llm.TokensNotReported
	var cacheCreationTokens, cacheReadTokens int
	var startUsage anthropicUsage
	var started bool
	var model string
	var containerID string
	var thinkingSignature string
	blocks := newStreamBlocks()

	// fail ends the stream with err. After message_start the error comes with
	// an EventError carrying the usage reported so far (input and cache tokens,
	// message_start's output count) and the content assembled so far, so a
	// caller can record what the broken turn cost.
	fail := func(err error) {
		if !started {
			yield(llm.StreamEvent{}, err)
			return
		}
		yield(llm.StreamEvent{
			Type:                  llm.EventError,
			Error:                 err.Error(),
			Model:                 model,
			InputTokens:           inputTokens,
			OutputTokens:          startUsage.OutputTokens,
			CacheCreationTokens:   cacheCreationTokens,
			CacheReadTokens:       cacheReadTokens,
			CacheCreation1hTokens: startUsage.CacheCreation.Ephemeral1h,
			ServerToolUse:         startUsage.serverToolUse(),
			Iterations:            startUsage.iterations(),
			Parts:                 blocks.parts(),
			ContainerID:           containerID,
		}, err)
	}

	// flushToolCall emits any accumulated tool call and clears the buffer.
	// Returns false if the caller broke iteration. Called both on
	// content_block_stop and defensively on a new content_block_start, so a
	// spec-violating stream that omits the stop can't silently drop a tool call.
	flushToolCall := func() bool {
		if currentToolCall == nil {
			return true
		}
		input, err := toolinput.Decode(inputBuffer.String())
		if err != nil {
			slog.Debug("failed to parse tool input JSON", "provider", "anthropic", "tool", currentToolCall.Name, "error", err)
		}
		currentToolCall.Input = input
		tc := currentToolCall
		blocks.setToolUse(currentToolIndex, *tc)
		currentToolCall = nil
		currentToolIndex = -1
		inputBuffer.Reset()
		return yield(llm.StreamEvent{Type: llm.EventToolUse, ToolCall: tc}, nil)
	}

	// flushServerToolUse completes a streamed server tool call and emits it.
	flushServerToolUse := func(index int) bool {
		part, ok := blocks.finishServerToolUse(index)
		if !ok {
			return true
		}
		return yield(llm.StreamEvent{Type: llm.EventServerTool, Part: &part}, nil)
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
				Type        string          `json:"type"`
				Text        string          `json:"text"`
				PartialJSON string          `json:"partial_json"`
				Thinking    string          `json:"thinking"`
				Signature   string          `json:"signature"`
				StopReason  string          `json:"stop_reason"`
				Citation    json.RawMessage `json:"citation"`
				StopDetails json.RawMessage `json:"stop_details"`
				Container   *struct {
					ID string `json:"id"`
				} `json:"container"`
			} `json:"delta"`
			ContentBlock json.RawMessage `json:"content_block"`
			Message      struct {
				Model      string         `json:"model"`
				StopReason string         `json:"stop_reason"`
				Usage      anthropicUsage `json:"usage"`
				Container  *struct {
					ID string `json:"id"`
				} `json:"container"`
			} `json:"message"`
			// message_delta puts usage at top level (not inside message)
			Usage anthropicUsage `json:"usage"`
			Error struct {
				Type    string `json:"type"`
				Message string `json:"message"`
			} `json:"error"`
		}

		if err := json.Unmarshal([]byte(data), &event); err != nil {
			if !yield(llm.StreamEvent{}, fmt.Errorf("malformed SSE event from anthropic: %w", err)) {
				return
			}
			continue
		}

		switch event.Type {
		case "content_block_start":
			var block anthropicBlock
			if len(event.ContentBlock) > 0 {
				if err := json.Unmarshal(event.ContentBlock, &block); err != nil {
					if !yield(llm.StreamEvent{}, fmt.Errorf("malformed content block from anthropic: %w", err)) {
						return
					}
					continue
				}
			}
			// Flush a still-open tool call first (defends against a missing
			// content_block_stop), then begin the new block.
			if !flushToolCall() || !flushServerToolUse(blocks.openServerToolUse()) {
				return
			}
			if result, done := blocks.start(event.Index, block); done && result.Type == llm.ContentServerToolResult {
				if !yield(llm.StreamEvent{Type: llm.EventServerTool, Part: &result}, nil) {
					return
				}
			}
			if block.Type == "tool_use" {
				currentToolCall = &llm.ToolCall{
					ID:   block.ID,
					Name: block.Name,
				}
				currentToolIndex = event.Index
				inputBuffer.Reset()
			}

		case "content_block_delta":
			switch event.Delta.Type {
			case "text_delta":
				blocks.appendText(event.Index, event.Delta.Text)
				if !yield(llm.StreamEvent{Type: llm.EventContent, Text: event.Delta.Text}, nil) {
					return
				}
			case "citations_delta":
				citation, ok := citationOf(event.Delta.Citation)
				if !ok {
					slog.Debug("dropping a citation that is not a JSON object", "provider", "anthropic")
					break
				}
				blocks.appendCitation(event.Index, citation)
				if !yield(llm.StreamEvent{Type: llm.EventCitation, Citation: &citation}, nil) {
					return
				}
			case "thinking_delta":
				blocks.appendThinking(event.Index, event.Delta.Thinking)
				if !yield(llm.StreamEvent{Type: llm.EventThinking, Thinking: event.Delta.Thinking}, nil) {
					return
				}
			case "signature_delta":
				// Authenticates the thinking block; surfaced on EventDone so a
				// streamed turn can be reassembled with a replayable thinking block.
				thinkingSignature += event.Delta.Signature
				blocks.appendSignature(event.Index, event.Delta.Signature)
			case "input_json_delta":
				if blocks.isServerToolUse(event.Index) {
					if !blocks.appendServerInput(event.Index, event.Delta.PartialJSON, maxToolInput) {
						fail(fmt.Errorf("tool input exceeded %d bytes", maxToolInput))
						return
					}
					break
				}
				if inputBuffer.Len()+len(event.Delta.PartialJSON) > maxToolInput {
					fail(fmt.Errorf("tool input exceeded %d bytes", maxToolInput))
					return // stop parsing — continuing would corrupt the tool call
				}
				inputBuffer.WriteString(event.Delta.PartialJSON)
			}

		case "content_block_stop":
			if !flushToolCall() || !flushServerToolUse(event.Index) {
				return
			}

		case "message_delta":
			// Flush a still-open tool call before completing. If the final
			// tool_use block's content_block_stop was dropped (truncated /
			// spec-violating stream) and message_delta arrives directly, the
			// accumulated call would otherwise be silently lost while Done still
			// reports a clean tool_use turn. The new-content_block_start flush
			// only covers a *following* block; this covers the terminal one.
			if !flushToolCall() || !flushServerToolUse(blocks.openServerToolUse()) {
				return
			}
			if event.Delta.Container != nil {
				containerID = event.Delta.Container.ID
			}
			// message_delta usage is cumulative; with server tools it carries the
			// input and cache counts of every sampling step, which message_start
			// (the first step only) under-reports.
			if event.Usage.InputTokens > 0 {
				inputTokens = event.Usage.InputTokens
			}
			if event.Usage.CacheCreationTokens > 0 {
				cacheCreationTokens = event.Usage.CacheCreationTokens
			}
			if event.Usage.CacheReadTokens > 0 {
				cacheReadTokens = event.Usage.CacheReadTokens
			}
			cache1h := startUsage.CacheCreation.Ephemeral1h
			if event.Usage.CacheCreation.Ephemeral1h > 0 {
				cache1h = event.Usage.CacheCreation.Ephemeral1h
			}
			serverToolUse := event.Usage.serverToolUse()
			if serverToolUse == (llm.ServerToolUsage{}) {
				serverToolUse = startUsage.serverToolUse()
			}
			iterations := event.Usage.iterations()
			if len(iterations) == 0 {
				iterations = startUsage.iterations()
			}
			// The terminal event: emit Done and stop. Returning here means any
			// trailing bytes the server sends after the turn is complete
			// (including malformed lines) can't surface as a spurious error that
			// would mask the completed turn.
			yield(llm.StreamEvent{
				Type:                  llm.EventDone,
				Model:                 model,
				StopReason:            normalizeStopReason(event.Delta.StopReason),
				RawStopReason:         event.Delta.StopReason,
				InputTokens:           inputTokens,
				OutputTokens:          event.Usage.OutputTokens,
				ThinkingSignature:     thinkingSignature,
				CacheCreationTokens:   cacheCreationTokens,
				CacheReadTokens:       cacheReadTokens,
				ServerToolUse:         serverToolUse,
				CacheCreation1hTokens: cache1h,
				StopDetails:           llm.StopDetailsFromRaw(event.Delta.StopDetails),
				Iterations:            iterations,
				Parts:                 blocks.parts(),
				ContainerID:           containerID,
			}, nil)
			return

		case "error":
			fail(streamError(event.Error.Type, event.Error.Message, c.config))
			return

		case "message_stop":
			// Final event

		case "message_start":
			// Capture input tokens and cache usage (reported once at stream start)
			inputTokens = event.Message.Usage.InputTokens
			cacheCreationTokens = event.Message.Usage.CacheCreationTokens
			cacheReadTokens = event.Message.Usage.CacheReadTokens
			startUsage = event.Message.Usage
			started = true
			model = event.Message.Model
			if event.Message.Container != nil {
				containerID = event.Message.Container.ID
			}
		}
	}

	// Reaching this point means no message_delta arrived (that case returns
	// right after EventDone, so a trailing read error after it is never seen).
	if err := scanner.Err(); err != nil {
		fail(fmt.Errorf("stream error: %w", err))
		return
	}

	// Clean EOF without message_delta: the server truncated the turn. Yield an
	// explicit error — otherwise a half-accumulated tool call or missing stop
	// reason would be indistinguishable from a complete turn. Wrapping
	// io.ErrUnexpectedEOF lets the pool classify the failure as retryable.
	fail(fmt.Errorf("anthropic: stream ended without completion: %w", io.ErrUnexpectedEOF))
}

// samplingEffort returns output_config.effort when the caller passes it
// through SamplingParams (the adapter sets output_config itself only with a
// ThinkingLevel), or "".
func samplingEffort(params map[string]any) string {
	raw, ok := params["output_config"]
	if !ok {
		return ""
	}
	data, err := json.Marshal(raw)
	if err != nil {
		return ""
	}
	var cfg struct {
		Effort string `json:"effort"`
	}
	if json.Unmarshal(data, &cfg) != nil {
		return ""
	}
	return cfg.Effort
}

// capabilityConfig is the config the capability check runs against. A
// client built directly with New may leave Provider empty; it is Anthropic.
func (c *Client) capabilityConfig() llm.Config {
	cfg := c.config
	if cfg.Provider == "" {
		cfg.Provider = "anthropic"
	}
	return cfg
}
