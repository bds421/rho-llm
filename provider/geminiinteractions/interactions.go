// Package geminiinteractions implements the Gemini Interactions API adapter
// (POST {base}/interactions), registered as the "gemini_interactions"
// protocol. Google calls generateContent "legacy" and points new work at this
// API (aistudio.google.com/llms.txt, read 2026-11-18). It is opt-in: the
// "gemini" preset keeps using generateContent.
//
// The adapter is stateless by default: every call sends the whole transcript
// as input steps with store:false, so nothing is retained server side. The wire
// facts below were read from aistudio.google.com/docs/interactions and
// api-reference/interactions-api/create-interaction and confirmed against the
// live API on 2026-11-18 (model gemini-3.8-flash).
package geminiinteractions

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"

	llm "github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/internal/toolinput"
	"github.com/bds421/rho-llm/internal/urlerr"
)

const (
	// Protocol is the wire-protocol name the adapter registers under.
	Protocol = "gemini_interactions"

	defaultProviderName = "gemini-interactions"
	defaultBase         = "https://generativelanguage.googleapis.com/v1beta"

	// skipThoughtSignature is Google's documented placeholder for a function
	// call that has no real thought signature (a call made by another
	// provider). Gemini 3 answers 400 "Request contains an invalid argument"
	// to a function_call step without a preceding thought step; the
	// interactions API accepted this placeholder (and arbitrary strings) live.
	skipThoughtSignature = "skip_thought_signature_validator"
)

func init() {
	llm.RegisterProvider(Protocol, func(cfg llm.Config) (llm.Client, error) {
		return New(cfg)
	})
}

// Client implements llm.Client over the Interactions API.
type Client struct {
	config       llm.Config
	endpoint     string
	httpClient   *http.Client
	streamClient *http.Client
	providerName string
}

// New creates an Interactions API client.
func New(cfg llm.Config) (*Client, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("Gemini API key is required")
	}
	base := llm.ResolveBaseURL(cfg)
	if base == "" {
		base = defaultBase
	}
	providerName := cfg.ProviderName
	if providerName == "" {
		providerName = defaultProviderName
	}
	httpClient, err := llm.NewSafeHTTPClient(cfg)
	if err != nil {
		return nil, err
	}
	return &Client{
		config:       cfg,
		endpoint:     strings.TrimRight(base, "/") + "/interactions",
		httpClient:   httpClient,
		streamClient: llm.NewStreamingHTTPClient(httpClient, cfg),
		providerName: providerName,
	}, nil
}

// Provider returns the provider name.
func (c *Client) Provider() string { return c.providerName }

// Model returns the configured model.
func (c *Client) Model() string { return c.config.Model }

// Close releases idle connections.
func (c *Client) Close() error {
	c.httpClient.CloseIdleConnections()
	return nil
}

func (c *Client) streamHTTPClient() *http.Client {
	if c.streamClient != nil {
		return c.streamClient
	}
	return llm.NewStreamingHTTPClient(c.httpClient, c.config)
}

// Complete sends one non-streaming interaction.
func (c *Client) Complete(ctx context.Context, req llm.Request) (*llm.Response, error) {
	body, model, err := c.encode(req, false)
	if err != nil {
		return nil, err
	}
	httpReq, err := c.newHTTPRequest(ctx, body)
	if err != nil {
		return nil, err
	}
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", urlerr.Redact(err))
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, llm.ErrorFromResponse(c.providerName, resp, c.config)
	}
	var in interaction
	if err := llm.DecodeJSONResponse(resp, c.config, &in); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}
	if in.Error != nil {
		return nil, c.apiError(in.Error)
	}
	return c.parseInteraction(&in, model), nil
}

// Stream sends one streaming interaction.
func (c *Client) Stream(ctx context.Context, req llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		body, model, err := c.encode(req, true)
		if err != nil {
			yield(llm.StreamEvent{}, err)
			return
		}
		httpReq, err := c.newHTTPRequest(ctx, body)
		if err != nil {
			yield(llm.StreamEvent{}, err)
			return
		}
		resp, err := c.streamHTTPClient().Do(httpReq)
		if err != nil {
			yield(llm.StreamEvent{}, fmt.Errorf("request failed: %w", urlerr.Redact(err)))
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			yield(llm.StreamEvent{}, llm.ErrorFromResponse(c.providerName, resp, c.config))
			return
		}
		c.parseStream(resp.Body, model, yield)
	}
}

func (c *Client) newHTTPRequest(ctx context.Context, body []byte) (*http.Request, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-goog-api-key", c.config.APIKey)
	return httpReq, nil
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

// interactionRequest is the POST /interactions body. Field names are the
// documented snake_case ones.
type interactionRequest struct {
	Model             string      `json:"model"`
	Input             []inputStep `json:"input"`
	SystemInstruction string      `json:"system_instruction,omitempty"`
	Tools             []toolDecl  `json:"tools,omitempty"`
	GenerationConfig  *genConfig  `json:"generation_config,omitempty"`
	Store             bool        `json:"store"` // always sent: the API default is true
	Stream            bool        `json:"stream,omitempty"`
}

// genConfig holds the documented generation_config fields in use.
type genConfig struct {
	Temperature       *float64 `json:"temperature,omitempty"`
	MaxOutputTokens   int      `json:"max_output_tokens,omitempty"`
	StopSequences     []string `json:"stop_sequences,omitempty"`
	ThinkingLevel     string   `json:"thinking_level,omitempty"`
	ThinkingSummaries string   `json:"thinking_summaries,omitempty"`
	// ToolChoice is a string ("auto", "any", "none") or, to force one tool,
	// {"allowed_tools":{"mode":"any","tools":[name]}} (confirmed live).
	ToolChoice any `json:"tool_choice,omitempty"`
}

type toolDecl struct {
	Type        string         `json:"type"`
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
}

// inputStep is one element of the input array. Type selects the fields:
// user_input and model_output carry content; thought carries a signature;
// function_call carries id/name/arguments; function_result carries
// call_id/name/result.
type inputStep struct {
	Type      string         `json:"type"`
	Content   []contentBlock `json:"content,omitempty"`
	Signature string         `json:"signature,omitempty"`
	ID        string         `json:"id,omitempty"`
	CallID    string         `json:"call_id,omitempty"`
	Name      string         `json:"name,omitempty"`
	Arguments any            `json:"arguments,omitempty"`
	Result    *string        `json:"result,omitempty"`
	IsError   bool           `json:"is_error,omitempty"`
}

type contentBlock struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	Data     string `json:"data,omitempty"`
	MimeType string `json:"mime_type,omitempty"`
}

// interaction is the response resource (and the "interaction" object of the
// created / completed stream events).
type interaction struct {
	ID     string    `json:"id"`
	Status string    `json:"status"`
	Model  string    `json:"model"`
	Usage  *usage    `json:"usage"`
	Steps  []outStep `json:"steps"`
	Error  *wireErr  `json:"error"`
}

// usage uses pointers' absence (nil *usage) as "not reported".
type usage struct {
	TotalInputTokens   int `json:"total_input_tokens"`
	TotalCachedTokens  int `json:"total_cached_tokens"`
	TotalOutputTokens  int `json:"total_output_tokens"`
	TotalThoughtTokens int `json:"total_thought_tokens"`
}

// split maps usage onto the library's disjoint cache-token contract. The
// Interactions docs do not say whether total_input_tokens includes
// total_cached_tokens; it is treated like generateContent's promptTokenCount
// (it does) and clamped, so InputTokens is never negative (UNVERIFIED: the
// live probes had 0 cached tokens).
func (u *usage) split() (input, cacheRead int) {
	in, cached := max(u.TotalInputTokens, 0), max(u.TotalCachedTokens, 0)
	cached = min(cached, in)
	return in - cached, cached
}

type outStep struct {
	Type      string          `json:"type"`
	Signature string          `json:"signature"`
	Summary   []contentBlock  `json:"summary"`
	Content   []contentBlock  `json:"content"`
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// wireErr is the error object, {"message","code"} with a string code such as
// "invalid_request" or "not_found" (confirmed live as HTTP bodies; the
// in-band form inside a 200 stream is UNVERIFIED and handled defensively).
type wireErr struct {
	Message string `json:"message"`
	Code    any    `json:"code"`
}

// streamEvent is one SSE data payload.
type streamEvent struct {
	EventType   string       `json:"event_type"`
	Status      string       `json:"status"`
	Interaction *interaction `json:"interaction"`
	Index       int          `json:"index"`
	Step        *outStep     `json:"step"`
	Delta       *streamDelta `json:"delta"`
	Error       *wireErr     `json:"error"`
}

type streamDelta struct {
	Type      string          `json:"type"`
	Text      string          `json:"text"`
	Signature string          `json:"signature"`
	Arguments json.RawMessage `json:"arguments"`
	Content   *contentBlock   `json:"content"`
}

// ---------------------------------------------------------------------------
// Request building
// ---------------------------------------------------------------------------

// encode builds and marshals the request body. SamplingParams merge at the
// top level like every adapter, with one addition: a boolean "store" key opts
// in to server-side storage (the adapter otherwise sends store:false). The
// effective model is returned for response bookkeeping.
func (c *Client) encode(req llm.Request, stream bool) ([]byte, string, error) {
	apiReq, err := c.buildRequest(req)
	if err != nil {
		return nil, "", err
	}
	apiReq.Stream = stream
	sampling := req.SamplingParams
	if v, ok := sampling["store"]; ok {
		b, isBool := v.(bool)
		if !isBool {
			return nil, "", fmt.Errorf("gemini-interactions: SamplingParams \"store\" must be a bool, got %T", v)
		}
		apiReq.Store = b
		sampling = make(map[string]any, len(req.SamplingParams))
		for k, val := range req.SamplingParams {
			if k != "store" {
				sampling[k] = val
			}
		}
	}
	body, err := llm.MergeSamplingParams(apiReq, sampling)
	if err != nil {
		return nil, "", fmt.Errorf("failed to marshal request: %w", err)
	}
	return body, apiReq.Model, nil
}

func (c *Client) buildRequest(req llm.Request) (interactionRequest, error) {
	if err := req.ToolChoice.Validate(); err != nil {
		return interactionRequest{}, err
	}
	if err := llm.UnsupportedServerTools(c.providerName, req); err != nil {
		return interactionRequest{}, err
	}
	if req.Effort != "" {
		return interactionRequest{}, fmt.Errorf("gemini-interactions: Effort is not supported; use ThinkingLevel")
	}
	if req.ThinkingBudget > 0 {
		return interactionRequest{}, fmt.Errorf("gemini-interactions: ThinkingBudget is not supported; use ThinkingLevel")
	}
	if req.CachedContent != "" {
		return interactionRequest{}, fmt.Errorf("gemini-interactions: explicit caching (CachedContent) is not supported by the Interactions API yet")
	}
	if req.ResponseFormat != nil && req.ResponseFormat.Type != "" {
		return interactionRequest{}, fmt.Errorf("gemini-interactions: ResponseFormat is not supported by this adapter yet")
	}

	model := req.Model
	if model == "" {
		model = c.config.Model
	}
	apiReq := interactionRequest{
		Model:            model,
		GenerationConfig: &genConfig{Temperature: req.Temperature, StopSequences: req.StopSequences},
	}
	g := apiReq.GenerationConfig
	g.MaxOutputTokens = req.MaxTokens
	if g.MaxOutputTokens == 0 {
		g.MaxOutputTokens = c.config.MaxTokens
	}

	var system []string
	if req.System != "" {
		system = append(system, req.System)
	}
	toolNames := toolNamesByID(req.Messages)
	for _, msg := range req.Messages {
		switch msg.Role {
		case llm.RoleSystem:
			for _, p := range msg.Content {
				if p.Type == llm.ContentText && p.Text != "" {
					system = append(system, p.Text)
				}
			}
		case llm.RoleUser:
			steps, err := userSteps(msg, toolNames)
			if err != nil {
				return interactionRequest{}, err
			}
			apiReq.Input = append(apiReq.Input, steps...)
		case llm.RoleAssistant:
			steps, err := assistantSteps(msg)
			if err != nil {
				return interactionRequest{}, err
			}
			apiReq.Input = append(apiReq.Input, steps...)
		default:
			return interactionRequest{}, fmt.Errorf("gemini-interactions: unsupported message role %q", msg.Role)
		}
	}
	if len(apiReq.Input) == 0 {
		return interactionRequest{}, fmt.Errorf("gemini-interactions: request has no input content")
	}
	apiReq.SystemInstruction = strings.Join(system, "\n\n")

	// Thinking.
	level := req.ThinkingLevel
	if level == llm.ThinkingNone && c.config.ThinkingLevel != llm.ThinkingNone {
		level = c.config.ThinkingLevel
	}
	t := traitsFor(model)
	if level != llm.ThinkingNone {
		wire, err := thinkingLevelWire(level, t)
		if err != nil {
			return interactionRequest{}, err
		}
		g.ThinkingLevel = wire
		// Without this no thought summary is returned, so a caller that set a
		// level would never see EventThinking / Response.Thinking.
		g.ThinkingSummaries = "auto"
	}
	// Thought tokens count against max_output_tokens (confirmed live: a
	// 60-token cap at thinking_level high ended "incomplete" after 54 thought
	// tokens and 2 output tokens). Pad as the generateContent adapter does.
	if info, ok := llm.GetModelInfo(model); ok && info.SupportsThinking && g.MaxOutputTokens > 0 {
		padded := g.MaxOutputTokens + llm.ThinkingBudgetTokens(llm.ThinkingLow, 0)
		if info.MaxTokens > 0 && padded > info.MaxTokens && !t.longDecode {
			padded = info.MaxTokens
		}
		g.MaxOutputTokens = max(padded, g.MaxOutputTokens)
	}

	// Tools.
	if len(req.Tools) > 0 {
		for _, tool := range req.Tools {
			params := tool.InputSchema
			if params == nil {
				params = map[string]any{"type": "object"}
			}
			apiReq.Tools = append(apiReq.Tools, toolDecl{Type: "function", Name: tool.Name, Description: tool.Description, Parameters: params})
		}
		g.ToolChoice = toolChoiceWire(req.ToolChoice)
	}
	return apiReq, nil
}

// toolChoiceWire maps the neutral ToolChoice. "any" forces some call; forcing
// one named tool is allowed_tools{mode any, tools [name]} (both confirmed
// live). Auto is the default and stays off the wire.
func toolChoiceWire(tc *llm.ToolChoice) any {
	if tc == nil {
		return nil
	}
	switch tc.Mode {
	case llm.ToolChoiceNone:
		return "none"
	case llm.ToolChoiceRequired:
		return "any"
	case llm.ToolChoiceTool:
		return map[string]any{"allowed_tools": map[string]any{"mode": "any", "tools": []string{tc.Name}}}
	default:
		return nil
	}
}

func thinkingLevelWire(level llm.ThinkingLevel, t modelTraits) (string, error) {
	switch level {
	case llm.ThinkingMinimal:
		if t.noMinimalThinking {
			return "low", nil // "minimal" answers 400 on these models (live)
		}
		return "minimal", nil
	case llm.ThinkingLow, llm.ThinkingMedium, llm.ThinkingHigh:
		return string(level), nil
	case llm.ThinkingXHigh, llm.ThinkingMax:
		if t.xhighThinking {
			return "xhigh", nil
		}
		return "high", nil // xhigh answers 400 elsewhere (live, gemini-3.8-flash)
	default:
		return "", fmt.Errorf("gemini-interactions: unsupported ThinkingLevel %q", level)
	}
}

// userSteps converts a user message. Tool results become function_result
// steps and everything else user_input steps, in part order, so a result
// always follows the call it answers.
func userSteps(msg llm.Message, toolNames map[string]string) ([]inputStep, error) {
	var steps []inputStep
	var pending []contentBlock
	flush := func() {
		if len(pending) > 0 {
			steps = append(steps, inputStep{Type: "user_input", Content: pending})
			pending = nil
		}
	}
	for _, p := range msg.Content {
		switch p.Type {
		case llm.ContentText:
			if p.Text != "" {
				pending = append(pending, contentBlock{Type: "text", Text: p.Text})
			}
		case llm.ContentImage:
			if err := llm.ValidateImageSource(p); err != nil {
				return nil, err
			}
			pending = append(pending, contentBlock{Type: "image", Data: p.Source.Data, MimeType: p.Source.MediaType})
		case llm.ContentDocument:
			if err := llm.ValidateDocumentSource(p); err != nil {
				return nil, err
			}
			if llm.IsTextDocument(p) {
				return nil, fmt.Errorf("gemini-interactions: text documents are Anthropic-only; send the text as a text part")
			}
			pending = append(pending, contentBlock{Type: "document", Data: p.Document.Data, MimeType: p.Document.MediaType})
		case llm.ContentToolResult:
			flush()
			name, ok := toolNames[p.ToolResultID]
			if !ok || name == "" {
				// The API rejects a function_result without a name (live:
				// "Name cannot be empty"), and the name is only known from the
				// call earlier in the transcript.
				return nil, fmt.Errorf("gemini-interactions: tool result %q has no matching tool call in the transcript", p.ToolResultID)
			}
			text := p.ToolResultText()
			steps = append(steps, inputStep{Type: "function_result", CallID: p.ToolResultID, Name: name, Result: &text, IsError: p.IsError})
		}
	}
	flush()
	return steps, nil
}

// assistantSteps converts an assistant message into thought / model_output /
// function_call steps. A signed thinking part replays as a thought step. A
// turn with tool calls but no signature (another provider made it) gets the
// documented placeholder signature, since Gemini 3 refuses a function_call
// without a thought step.
func assistantSteps(msg llm.Message) ([]inputStep, error) {
	var steps []inputStep
	var text []contentBlock
	flush := func() {
		if len(text) > 0 {
			steps = append(steps, inputStep{Type: "model_output", Content: text})
			text = nil
		}
	}
	hasThought := false
	for _, p := range msg.Content {
		if p.Type == llm.ContentThinking && p.ThinkingSignature != "" && !p.Redacted {
			hasThought = true
		}
	}
	needDummy := !hasThought
	for i, p := range msg.Content {
		switch p.Type {
		case llm.ContentThinking:
			if p.ThinkingSignature != "" && !p.Redacted {
				flush()
				steps = append(steps, inputStep{Type: "thought", Signature: p.ThinkingSignature})
			}
		case llm.ContentText:
			if p.Text != "" {
				text = append(text, contentBlock{Type: "text", Text: p.Text})
			}
		case llm.ContentToolUse:
			flush()
			if needDummy {
				steps = append(steps, inputStep{Type: "thought", Signature: skipThoughtSignature})
				needDummy = false
			}
			id := p.ToolUseID
			if id == "" {
				id = fmt.Sprintf("call_%d_%s", i, p.ToolName)
			}
			steps = append(steps, inputStep{Type: "function_call", ID: id, Name: p.ToolName, Arguments: toolinput.Object(p.ToolInput)})
		}
	}
	flush()
	return steps, nil
}

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

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

// normalizeStatus maps the interaction status to the unified stop reasons.
// Confirmed live: completed, requires_action (a function_call is waiting),
// incomplete (max_output_tokens hit), in_progress (stream). failed and
// cancelled are documented in the API reference's status list but were not
// provoked. RawStopReason keeps the status.
func normalizeStatus(status string, calls int) string {
	switch status {
	case "completed":
		if calls > 0 {
			return llm.StopToolUse
		}
		return llm.StopEndTurn
	case "requires_action":
		if calls > 0 {
			return llm.StopToolUse
		}
		return llm.StopError
	case "incomplete":
		return llm.StopMaxTokens
	case "":
		return ""
	default:
		return llm.StopError
	}
}

func (c *Client) parseInteraction(in *interaction, requestModel string) *llm.Response {
	model := in.Model
	if model == "" {
		model = requestModel
	}
	resp := &llm.Response{
		ID:           in.ID,
		Model:        model,
		InputTokens:  llm.TokensNotReported,
		OutputTokens: llm.TokensNotReported,
	}
	if u := in.Usage; u != nil {
		resp.InputTokens, resp.CacheReadTokens = u.split()
		resp.OutputTokens = max(u.TotalOutputTokens, 0)
		resp.ThinkingTokens = max(u.TotalThoughtTokens, 0)
	}
	for i, s := range in.Steps {
		switch s.Type {
		case "thought":
			for _, b := range s.Summary {
				resp.Thinking += b.Text
			}
			if s.Signature != "" {
				resp.ThinkingSignature = s.Signature
			}
		case "model_output":
			for _, b := range s.Content {
				if b.Type == "text" || b.Type == "" {
					resp.Content += b.Text
				}
			}
		case "function_call":
			id := s.ID
			if id == "" {
				id = fmt.Sprintf("call_%d_%s", i, s.Name)
			}
			input, _ := toolinput.Decode(string(s.Arguments))
			resp.ToolCalls = append(resp.ToolCalls, llm.ToolCall{ID: id, Name: s.Name, Input: input})
		}
	}
	resp.RawStopReason = in.Status
	resp.StopReason = normalizeStatus(in.Status, len(resp.ToolCalls))
	return resp
}

// apiError turns an in-band error object into the *llm.APIError the same
// error would produce as an HTTP status. The string codes seen on HTTP error
// bodies ("invalid_request", "not_found") are mapped to their statuses; the
// rest (UNVERIFIED) fall back to a numeric code or 500.
func (c *Client) apiError(e *wireErr) error {
	status := 500
	switch code := e.Code.(type) {
	case float64:
		if code >= 400 && code <= 599 {
			status = int(code)
		}
	case string:
		switch strings.ToLower(code) {
		case "invalid_request", "invalid_argument":
			status = 400
		case "unauthenticated":
			status = 401
		case "permission_denied":
			status = 403
		case "not_found":
			status = 404
		case "rate_limit_exceeded", "resource_exhausted", "rate_limited":
			status = 429
		case "unavailable", "overloaded":
			status = 503
		}
	}
	message := llm.RedactSecrets(e.Message, c.config)
	return llm.NewAPIErrorFromStatusWithLimit(c.providerName, status, message, c.config.EffectiveMaxErrorMessageLen())
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

// pendingCall accumulates one streaming function_call step.
type pendingCall struct {
	id, name string
	args     strings.Builder
}

// parseStream reads the SSE stream. Event shapes (confirmed live):
// interaction.created, interaction.status_update, step.start {index, step},
// step.delta {index, delta{type: text | thought_summary | thought_signature |
// arguments_delta}}, step.stop, interaction.completed {interaction{status,
// usage}}, then "data: [DONE]". A function_call's arguments arrive as
// arguments_delta strings and are decoded at step.stop.
func (c *Client) parseStream(body io.Reader, requestModel string, yield func(llm.StreamEvent, error) bool) {
	maxToolInput := c.config.EffectiveMaxToolInputBytes()
	scanner := bufio.NewScanner(body)
	scanner.Buffer(nil, c.config.EffectiveMaxSSELineBytes())

	calls := map[int]*pendingCall{}
	var signature string
	sawCall := 0
	model := requestModel
	done := false

	finish := func(status string, u *usage) {
		done = true
		ev := llm.StreamEvent{
			Type:              llm.EventDone,
			Model:             model,
			StopReason:        normalizeStatus(status, sawCall),
			RawStopReason:     status,
			ThinkingSignature: signature,
			InputTokens:       llm.TokensNotReported,
			OutputTokens:      llm.TokensNotReported,
		}
		if u != nil {
			ev.InputTokens, ev.CacheReadTokens = u.split()
			ev.OutputTokens = max(u.TotalOutputTokens, 0)
			ev.ThinkingTokens = max(u.TotalThoughtTokens, 0)
		}
		yield(ev, nil)
	}

	for scanner.Scan() {
		data, ok := llm.SSEData(scanner.Text())
		if !ok {
			continue
		}
		if data == "[DONE]" {
			break
		}
		var ev streamEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			if !yield(llm.StreamEvent{}, fmt.Errorf("malformed SSE event from gemini-interactions: %w", err)) {
				return
			}
			continue
		}
		if ev.Error != nil {
			yield(llm.StreamEvent{}, c.apiError(ev.Error))
			return
		}
		switch ev.EventType {
		case "interaction.created":
			if ev.Interaction != nil && ev.Interaction.Model != "" {
				model = ev.Interaction.Model
			}
		case "interaction.status_update":
			// A terminal failure may arrive as a status update.
			if ev.Status == "failed" || ev.Status == "cancelled" {
				finish(ev.Status, nil)
				return
			}
		case "step.start":
			if ev.Step != nil && ev.Step.Type == "function_call" {
				pc := &pendingCall{id: ev.Step.ID, name: ev.Step.Name}
				// Arguments arrive as deltas; a non-empty object here (a
				// complete call in one event) is taken as is.
				if a := strings.TrimSpace(string(ev.Step.Arguments)); a != "" && a != "{}" && a != "null" {
					pc.args.WriteString(a)
				}
				calls[ev.Index] = pc
			}
		case "step.delta":
			if ev.Delta == nil {
				continue
			}
			switch ev.Delta.Type {
			case "text":
				if ev.Delta.Text != "" && !yield(llm.StreamEvent{Type: llm.EventContent, Text: ev.Delta.Text}, nil) {
					return
				}
			case "thought_summary":
				text := ev.Delta.Text
				if ev.Delta.Content != nil {
					text = ev.Delta.Content.Text
				}
				if text != "" && !yield(llm.StreamEvent{Type: llm.EventThinking, Thinking: text}, nil) {
					return
				}
			case "thought_signature":
				if ev.Delta.Signature != "" {
					signature = ev.Delta.Signature
				}
			case "arguments_delta":
				pc := calls[ev.Index]
				if pc == nil {
					continue
				}
				var frag string
				if err := json.Unmarshal(ev.Delta.Arguments, &frag); err != nil {
					// Not a string: a complete object delivered as a delta.
					frag = string(ev.Delta.Arguments)
				}
				if pc.args.Len()+len(frag) > maxToolInput {
					yield(llm.StreamEvent{}, fmt.Errorf("tool input exceeded %d bytes", maxToolInput))
					return
				}
				pc.args.WriteString(frag)
			}
		case "step.stop":
			pc := calls[ev.Index]
			if pc == nil {
				continue
			}
			delete(calls, ev.Index)
			id := pc.id
			if id == "" {
				id = fmt.Sprintf("call_%d_%s", ev.Index, pc.name)
			}
			input, _ := toolinput.Decode(pc.args.String())
			sawCall++
			if !yield(llm.StreamEvent{Type: llm.EventToolUse, ToolCall: &llm.ToolCall{ID: id, Name: pc.name, Input: input}}, nil) {
				return
			}
		case "interaction.completed":
			var status string
			var u *usage
			if ev.Interaction != nil {
				status, u = ev.Interaction.Status, ev.Interaction.Usage
				if ev.Interaction.Model != "" {
					model = ev.Interaction.Model
				}
			}
			finish(status, u)
			return
		}
	}

	if err := scanner.Err(); err != nil && !done {
		yield(llm.StreamEvent{}, fmt.Errorf("stream error: %w", err))
		return
	}
	// EOF or [DONE] without interaction.completed: the server cut the turn
	// short. io.ErrUnexpectedEOF makes it pool-retryable.
	if !done {
		yield(llm.StreamEvent{}, fmt.Errorf("gemini-interactions: stream ended without completion: %w", io.ErrUnexpectedEOF))
	}
}

// ---------------------------------------------------------------------------
// Per-model rules
// ---------------------------------------------------------------------------

// modelTraits mirrors provider/gemini's rules that apply here too (read
// 2026-11-18; "minimal" and "xhigh" both re-confirmed live on 3.8 Flash).
type modelTraits struct {
	noMinimalThinking bool // thinking_level "minimal" is a 400 (3.8/3.7 Flash, Gemini 4)
	xhighThinking     bool // thinking_level "xhigh" exists (Gemini 4 Hydrogen, Neon)
	longDecode        bool // Gemini 4: a turn may exceed one request's output limit
}

func traitsFor(model string) modelTraits {
	id := llm.ResolveModelAlias(model)
	gemini4 := strings.HasPrefix(id, "gemini-4-")
	return modelTraits{
		noMinimalThinking: gemini4 || id == "gemini-3.8-flash" || id == "gemini-3.7-flash",
		xhighThinking:     id == "gemini-4-hydrogen" || id == "gemini-4-neon",
		longDecode:        gemini4,
	}
}
