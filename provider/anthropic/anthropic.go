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
	Effort string `json:"effort"`
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
	var betas []string
	for _, beta := range c.config.BetaFeatures {
		if beta == interleavedThinkingBeta && !budgetThinking {
			continue
		}
		betas = append(betas, beta)
	}
	if len(betas) > 0 {
		httpReq.Header.Set("anthropic-beta", strings.Join(betas, ","))
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
				if part.Text != "" {
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

	// Configure thinking
	resolvedModel := llm.ResolveModelAlias(apiReq.Model)
	if profile, ok := adaptiveThinkingModels[resolvedModel]; ok {
		// Claude 4.7 and later reject thinking.type "enabled" + budget_tokens
		// with a 400; depth is set by output_config.effort instead.
		if apiReq.ToolChoice != nil && profile.noForcedToolChoice &&
			(req.ToolChoice.Mode == llm.ToolChoiceRequired || req.ToolChoice.Mode == llm.ToolChoiceTool) {
			return anthropicRequest{}, fmt.Errorf("anthropic: %s rejects forced tool_choice (%q); use ToolChoiceAuto and name the tool in the prompt", resolvedModel, req.ToolChoice.Mode)
		}
		if req.ThinkingLevel == llm.ThinkingNone {
			switch profile.off {
			case thinkingOffDisabled:
				if effort := samplingEffort(req.SamplingParams); profile.disabledUpToHigh && (effort == "xhigh" || effort == "max") {
					return anthropicRequest{}, fmt.Errorf("anthropic: %s rejects thinking disabled at effort %q (SamplingParams output_config); set a ThinkingLevel for adaptive thinking, or use effort high or lower", resolvedModel, effort)
				}
				apiReq.Thinking = &anthropicThinking{Type: "disabled"}
			case thinkingOffBetweenTools:
				apiReq.Thinking = &anthropicThinking{Type: "between_tools"}
			case thinkingOffOmit:
				// Thinking is always on and any explicit "off" is a 400:
				// leave the parameter out and let the model run adaptive.
			}
		} else {
			if req.ThinkingBudget > 0 {
				return anthropicRequest{}, fmt.Errorf("anthropic: %s does not support ThinkingBudget; use ThinkingLevel to control adaptive effort", resolvedModel)
			}
			effort := string(req.ThinkingLevel)
			if req.ThinkingLevel == llm.ThinkingMinimal {
				effort = "low"
			}
			apiReq.Thinking = &anthropicThinking{Type: "adaptive", Display: "summarized"}
			apiReq.OutputConfig = &anthropicOutputConfig{Effort: effort}
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
		Iterations:            apiResp.Usage.Iterations,
	}
	if apiResp.Container != nil {
		resp.ContainerID = apiResp.Container.ID
	}

	for _, block := range apiResp.Content {
		resp.Parts = append(resp.Parts, blockPart(block))
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
	currentToolIndex := -1
	var inputBuffer strings.Builder
	// Token counts initialize to "not reported". If the stream ends before
	// message_start/message_delta events, callers can distinguish "not reported"
	// from "zero tokens" (0).
	var inputTokens = llm.TokensNotReported
	var cacheCreationTokens, cacheReadTokens int
	var startUsage anthropicUsage
	var containerID string
	var thinkingSignature string
	blocks := newStreamBlocks()

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
			slog.Debug("failed to parse tool input JSON", "provider", "anthropic", "tool", currentToolCall.Name, "error", err)
			input = raw
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
				StopReason string         `json:"stop_reason"`
				Usage      anthropicUsage `json:"usage"`
				Container  *struct {
					ID string `json:"id"`
				} `json:"container"`
			} `json:"message"`
			// message_delta puts usage at top level (not inside message)
			Usage anthropicUsage `json:"usage"`
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
						yield(llm.StreamEvent{}, fmt.Errorf("tool input exceeded %d bytes", maxToolInput))
						return
					}
					break
				}
				if inputBuffer.Len()+len(event.Delta.PartialJSON) > maxToolInput {
					yield(llm.StreamEvent{}, fmt.Errorf("tool input exceeded %d bytes", maxToolInput))
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
			serverToolUse := event.Usage.serverToolUse()
			if serverToolUse == (llm.ServerToolUsage{}) {
				serverToolUse = startUsage.serverToolUse()
			}
			iterations := event.Usage.Iterations
			if len(iterations) == 0 {
				iterations = startUsage.Iterations
			}
			// The terminal event: emit Done and stop. Returning here means any
			// trailing bytes the server sends after the turn is complete
			// (including malformed lines) can't surface as a spurious error that
			// would mask the completed turn.
			yield(llm.StreamEvent{
				Type:                  llm.EventDone,
				StopReason:            normalizeStopReason(event.Delta.StopReason),
				RawStopReason:         event.Delta.StopReason,
				InputTokens:           inputTokens,
				OutputTokens:          event.Usage.OutputTokens,
				ThinkingSignature:     thinkingSignature,
				CacheCreationTokens:   cacheCreationTokens,
				CacheReadTokens:       cacheReadTokens,
				ServerToolUse:         serverToolUse,
				CacheCreation1hTokens: startUsage.CacheCreation.Ephemeral1h,
				StopDetails:           llm.StopDetailsFromRaw(event.Delta.StopDetails),
				Iterations:            iterations,
				Parts:                 blocks.parts(),
				ContainerID:           containerID,
			}, nil)
			return

		case "message_stop":
			// Final event

		case "message_start":
			// Capture input tokens and cache usage (reported once at stream start)
			inputTokens = event.Message.Usage.InputTokens
			cacheCreationTokens = event.Message.Usage.CacheCreationTokens
			cacheReadTokens = event.Message.Usage.CacheReadTokens
			startUsage = event.Message.Usage
			if event.Message.Container != nil {
				containerID = event.Message.Container.ID
			}
		}
	}

	// Reaching this point means no message_delta arrived (that case returns
	// right after EventDone, so a trailing read error after it is never seen).
	if err := scanner.Err(); err != nil {
		yield(llm.StreamEvent{}, fmt.Errorf("stream error: %w", err))
		return
	}

	// Clean EOF without message_delta: the server truncated the turn. Yield an
	// explicit error — otherwise a half-accumulated tool call or missing stop
	// reason would be indistinguishable from a complete turn. Wrapping
	// io.ErrUnexpectedEOF lets the pool classify the failure as retryable.
	yield(llm.StreamEvent{}, fmt.Errorf("anthropic: stream ended without completion: %w", io.ErrUnexpectedEOF))
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
