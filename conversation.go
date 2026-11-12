package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"slices"
)

// ConversationSchemaVersion is stamped into every serialized Conversation so the
// persisted format can evolve without silently mis-reading old data. Bump it (and
// add handling in LoadConversation) on any breaking change to the stored shape.
//
// The serialized form is deliberately a plain, self-describing JSON document keyed
// by explicit json tags — NOT tied to Go type or package names — so conversations
// stay portable and debuggable across versions.
const ConversationSchemaVersion = 1

// Usage accumulates token counts and estimated cost across a conversation.
// Cost is summed per response using each response's own model, so the total stays
// correct even when a conversation is handed off between providers mid-session
// (different models have different pricing).
//
// InputTokens, CacheReadTokens and CacheCreationTokens follow the disjoint
// cache-token contract of Response: InputTokens counts uncached input only, so
// total prompt tokens = InputTokens + CacheReadTokens + CacheCreationTokens.
type Usage struct {
	InputTokens         int `json:"input_tokens"`
	OutputTokens        int `json:"output_tokens"`
	ThinkingTokens      int `json:"thinking_tokens,omitempty"`
	CacheCreationTokens int `json:"cache_creation_tokens,omitempty"`
	CacheReadTokens     int `json:"cache_read_tokens,omitempty"`
	// CacheCreation1hTokens is the 1-hour share of CacheCreationTokens.
	CacheCreation1hTokens int `json:"cache_creation_1h_tokens,omitempty"`
	// ServerToolUse sums the billed server tool calls (web searches, …).
	ServerToolUse ServerToolUsage `json:"server_tool_use,omitzero"`
	Cost          float64         `json:"cost,omitempty"` // estimated USD, summed per response
}

// AddResponse folds one response's usage into the running totals, estimating its
// cost from the response's own model (registry pricing) at the synchronous rate.
// Nil-safe.
//
// Token counts are clamped to >= 0 first: a streamed turn whose provider never
// reported usage carries the TokensNotReported (-1) sentinel, which must not
// pollute the accumulated totals or produce a negative cost.
func (u *Usage) AddResponse(resp *Response) {
	u.add(resp, false)
}

// AddBatchResponse is like AddResponse but prices the response at the Batch API
// rate (50% of synchronous). Token accumulation is identical — only the estimated
// cost differs. Use it when folding results from a BatchClient.
func (u *Usage) AddBatchResponse(resp *Response) {
	u.add(resp, true)
}

// add is the shared accumulator for AddResponse/AddBatchResponse, keeping the
// TokensNotReported (-1) clamp in exactly one place. batch selects batch pricing.
func (u *Usage) add(resp *Response, batch bool) {
	if resp == nil {
		return
	}
	in := nonNeg(resp.InputTokens)
	out := nonNeg(resp.OutputTokens)
	think := nonNeg(resp.ThinkingTokens)
	cacheW := nonNeg(resp.CacheCreationTokens)
	cacheR := nonNeg(resp.CacheReadTokens)

	// Saturate instead of wrapping: a running total near math.MaxInt must not
	// turn negative, and the running cost stops at math.MaxFloat64, never +Inf.
	u.InputTokens = saturatingAdd(u.InputTokens, in)
	u.OutputTokens = saturatingAdd(u.OutputTokens, out)
	u.ThinkingTokens = saturatingAdd(u.ThinkingTokens, think)
	u.CacheCreationTokens = saturatingAdd(u.CacheCreationTokens, cacheW)
	u.CacheReadTokens = saturatingAdd(u.CacheReadTokens, cacheR)
	u.CacheCreation1hTokens = saturatingAdd(u.CacheCreation1hTokens, nonNeg(resp.CacheCreation1hTokens))
	u.ServerToolUse.WebSearchRequests = saturatingAdd(u.ServerToolUse.WebSearchRequests, nonNeg(resp.ServerToolUse.WebSearchRequests))
	u.ServerToolUse.WebFetchRequests = saturatingAdd(u.ServerToolUse.WebFetchRequests, nonNeg(resp.ServerToolUse.WebFetchRequests))
	u.ServerToolUse.CodeExecutionRequests = saturatingAdd(u.ServerToolUse.CodeExecutionRequests, nonNeg(resp.ServerToolUse.CodeExecutionRequests))
	u.Cost = addCost(u.Cost, EstimateCost(CostInput{
		Model:               resp.Model,
		InputTokens:         in,
		OutputTokens:        out,
		ThinkingTokens:      think,
		CacheCreateTokens:   cacheW,
		CacheReadTokens:     cacheR,
		CacheCreate1hTokens: resp.CacheCreation1hTokens,
		Batch:               batch,
		WebSearchRequests:   resp.ServerToolUse.WebSearchRequests,
		Iterations:          resp.Iterations,
	}))
}

// nonNeg clamps a token count to >= 0 (filters the TokensNotReported sentinel and
// any stray negatives).
func nonNeg(n int) int {
	if n < 0 {
		return 0
	}
	return n
}

// Conversation is a serializable, provider-neutral transcript. It is a plain
// value: it owns no client and is safe to JSON-marshal, persist, and restore.
// Generation stays stateless (Client.Complete/Stream take a Request); Conversation
// just accumulates the history and usage. For an ergonomic, concurrency-safe
// driver — including cross-provider handoff — wrap it in a Session.
type Conversation struct {
	SchemaVersion int    `json:"schema_version"`
	System        string `json:"system,omitempty"`
	Tools         []Tool `json:"tools,omitempty"`
	// ServerTools are provider-run tools (web search) sent with every turn.
	ServerTools []ServerTool `json:"server_tools,omitempty"`
	Messages    []Message    `json:"messages"`
	Usage       Usage        `json:"usage,omitempty"`
}

// NewConversation creates an empty conversation with an optional system prompt
// and tool set.
// exactNumbers converts the json.Number values of a UseNumber decode to
// float64, except an integer that float64 cannot hold exactly.
func exactNumbers(v any) any {
	switch t := v.(type) {
	case json.Number:
		if f, err := t.Float64(); err == nil {
			if i, err := t.Int64(); err != nil || (i <= 1<<53 && i >= -(1<<53)) {
				return f
			}
		}
		return t
	case map[string]any:
		for k, e := range t {
			t[k] = exactNumbers(e)
		}
		return t
	case []any:
		for i, e := range t {
			t[i] = exactNumbers(e)
		}
		return t
	default:
		return v
	}
}

// clampNegative sets negative token counts and cost to zero.
func (u *Usage) clampNegative() {
	for _, n := range []*int{&u.InputTokens, &u.OutputTokens, &u.ThinkingTokens, &u.CacheCreationTokens, &u.CacheReadTokens, &u.CacheCreation1hTokens} {
		*n = max(*n, 0)
	}
	if u.Cost < 0 || math.IsNaN(u.Cost) {
		u.Cost = 0
	}
}

func NewConversation(system string, tools ...Tool) *Conversation {
	return &Conversation{
		SchemaVersion: ConversationSchemaVersion,
		System:        system,
		Tools:         tools,
	}
}

// Append adds messages to the transcript.
func (c *Conversation) Append(msgs ...Message) {
	c.Messages = append(c.Messages, msgs...)
}

// AddResponse appends the assistant message for resp — tagged with the producing
// provider/model for handoff decisions — and folds its usage into the running
// totals. It returns the appended assistant Message.
func (c *Conversation) AddResponse(provider string, resp *Response) Message {
	am := NewAssistantMessage(resp)
	am.Provider = provider
	c.Append(am)
	c.Usage.AddResponse(resp)
	return am
}

// ToRequest returns a copy of base with System, Tools, and Messages populated
// from the conversation. Fields already set on base (Model, MaxTokens,
// ThinkingLevel, Temperature, …) are preserved, so callers configure generation
// once on the base Request and let the conversation supply the history.
func (c *Conversation) ToRequest(base Request) Request {
	req := base
	if c.System != "" {
		req.System = c.System
	}
	if len(c.Tools) > 0 {
		req.Tools = c.Tools
	}
	if len(c.ServerTools) > 0 {
		req.ServerTools = c.ServerTools
	}
	// Clipped, so a caller appending to req.Messages (or Tools) gets a new
	// array instead of writing into the conversation's spare capacity (B23).
	req.Messages = slices.Clip(c.Messages)
	req.Tools = slices.Clip(req.Tools)
	return req
}

// Clone returns a deep copy of the conversation, made through the versioned
// JSON round-trip (the same machinery persistence uses), so no backing arrays,
// maps, or tool-input values are shared with the original. As with
// persistence, ToolInput values come back in their JSON form (a struct
// becomes a map[string]any).
func (c *Conversation) Clone() *Conversation {
	if data, err := json.Marshal(c); err == nil {
		if cp, err := LoadConversation(data); err == nil {
			return cp
		}
	}
	// Fallback for a non-JSON-serializable ToolInput (e.g. a channel), which
	// makes Marshal fail. A structural deep copy still isolates every backing
	// array, map, and pointer source — only a truly-opaque ToolInput value
	// (which can't be deep-copied without reflection, and isn't persistable
	// anyway) stays shared.
	cp := *c
	cp.Messages = cloneMessages(c.Messages)
	cp.Tools = cloneTools(c.Tools)
	cp.ServerTools = cloneServerTools(c.ServerTools)
	return &cp
}

// cloneMessages deep-copies a message slice (used by Clone's fallback path).
func cloneMessages(src []Message) []Message {
	if src == nil {
		return nil
	}
	out := make([]Message, len(src))
	for i, m := range src {
		m.Content = cloneParts(m.Content)
		out[i] = m
	}
	return out
}

// cloneParts deep-copies a content-part slice, including nested tool-result
// parts and the *ImageSource / *DocumentSource pointers.
func cloneParts(src []ContentPart) []ContentPart {
	if src == nil {
		return nil
	}
	out := make([]ContentPart, len(src))
	for i, p := range src {
		if p.Source != nil {
			s := *p.Source
			p.Source = &s
		}
		if p.Document != nil {
			d := *p.Document
			p.Document = &d
		}
		if p.ToolResultParts != nil {
			p.ToolResultParts = cloneParts(p.ToolResultParts)
		}
		if p.ToolInput != nil {
			p.ToolInput = deepCopyJSONValue(p.ToolInput)
		}
		p.Citations = cloneCitations(p.Citations)
		p.WebSearchResults = append([]WebSearchResult(nil), p.WebSearchResults...)
		if p.ServerToolError != nil {
			e := *p.ServerToolError
			p.ServerToolError = &e
		}
		p.ProviderBlock = append(json.RawMessage(nil), p.ProviderBlock...)
		out[i] = p
	}
	return out
}

// cloneTools deep-copies a tool slice, including each InputSchema map.
func cloneTools(src []Tool) []Tool {
	if src == nil {
		return nil
	}
	out := make([]Tool, len(src))
	for i, t := range src {
		if t.InputSchema != nil {
			m := make(map[string]any, len(t.InputSchema))
			for k, v := range t.InputSchema {
				m[k] = deepCopyJSONValue(v)
			}
			t.InputSchema = m
		}
		out[i] = t
	}
	return out
}

// maxDeepCopyDepth bounds deepCopyJSONValue's recursion. This path is the
// fallback only a non-JSON-serializable ToolInput reaches; a real JSON-sourced
// value is acyclic and far shallower than this. The bound exists so a
// hand-built *cyclic* Go map (which json.Marshal rejects, sending it here)
// can't recurse forever into a stack-overflow crash — it bails and shares the
// remainder instead of crashing the whole process.
const maxDeepCopyDepth = 1000

// deepCopyJSONValue deep-copies a JSON-shaped value (maps and slices of any).
// Scalars and opaque values (chan/func/etc.) are returned as-is — they can't be
// copied without reflection and aren't JSON-serializable in the first place.
func deepCopyJSONValue(v any) any {
	return deepCopyJSONValueDepth(v, 0)
}

func deepCopyJSONValueDepth(v any, depth int) any {
	if depth >= maxDeepCopyDepth {
		// Pathologically deep or cyclic — stop recursing and share the rest
		// rather than overflowing the stack.
		return v
	}
	switch x := v.(type) {
	case map[string]any:
		m := make(map[string]any, len(x))
		for k, val := range x {
			m[k] = deepCopyJSONValueDepth(val, depth+1)
		}
		return m
	case []any:
		s := make([]any, len(x))
		for i, val := range x {
			s[i] = deepCopyJSONValueDepth(val, depth+1)
		}
		return s
	default:
		return v
	}
}

// MarshalJSON stamps the current schema version even on a zero-value Conversation
// constructed without NewConversation, so every persisted blob is versioned.
func (c Conversation) MarshalJSON() ([]byte, error) {
	type alias Conversation // break the method set to avoid recursion
	tmp := alias(c)
	if tmp.SchemaVersion == 0 {
		tmp.SchemaVersion = ConversationSchemaVersion
	}
	return json.Marshal(tmp)
}

// LoadConversation deserializes a Conversation previously produced by json.Marshal,
// validating the schema version. A zero/absent version is tolerated as v1 (data
// written before versioning); a version newer than this build is rejected rather
// than silently mis-read.
func LoadConversation(data []byte) (*Conversation, error) {
	var c Conversation
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	if err := dec.Decode(&c); err != nil {
		return nil, fmt.Errorf("llm: decode conversation: %w", err)
	}
	if dec.More() {
		return nil, fmt.Errorf("llm: decode conversation: trailing data")
	}
	// Numbers decode as float64, as with json.Unmarshal, except integers
	// beyond ±2^53, which float64 would change (a tool argument holding an
	// ID): those stay json.Number, so a save/load round trip is exact (B23).
	for i := range c.Messages {
		for j := range c.Messages[i].Content {
			p := &c.Messages[i].Content[j]
			p.ToolInput = exactNumbers(p.ToolInput)
		}
	}
	for i := range c.Tools {
		if c.Tools[i].InputSchema != nil {
			c.Tools[i].InputSchema = exactNumbers(c.Tools[i].InputSchema).(map[string]any)
		}
	}
	for i := range c.ServerTools {
		if c.ServerTools[i].Params != nil {
			c.ServerTools[i].Params = exactNumbers(c.ServerTools[i].Params).(map[string]any)
		}
	}
	// Token counts are never negative; a stored -1 (TokensNotReported) or a
	// tampered file would corrupt every later total.
	c.Usage.clampNegative()
	if c.SchemaVersion <= 0 {
		c.SchemaVersion = ConversationSchemaVersion // absent/zero/negative → treat as v1
	}
	if c.SchemaVersion > ConversationSchemaVersion {
		return nil, fmt.Errorf("llm: conversation schema_version %d is newer than supported %d", c.SchemaVersion, ConversationSchemaVersion)
	}
	return &c, nil
}

// cloneCitations deep-copies a citation slice, including each Raw.
func cloneCitations(src []Citation) []Citation {
	if src == nil {
		return nil
	}
	out := make([]Citation, len(src))
	for i, c := range src {
		c.Raw = append(json.RawMessage(nil), c.Raw...)
		out[i] = c
	}
	return out
}

// cloneServerTools deep-copies a server tool slice (domain lists, location,
// callers and Params).
func cloneServerTools(src []ServerTool) []ServerTool {
	if src == nil {
		return nil
	}
	out := make([]ServerTool, len(src))
	for i, t := range src {
		t.AllowedDomains = append([]string(nil), t.AllowedDomains...)
		t.BlockedDomains = append([]string(nil), t.BlockedDomains...)
		t.AllowedCallers = append([]string(nil), t.AllowedCallers...)
		if t.UserLocation != nil {
			loc := *t.UserLocation
			t.UserLocation = &loc
		}
		if t.Params != nil {
			m := make(map[string]any, len(t.Params))
			for k, v := range t.Params {
				m[k] = deepCopyJSONValue(v)
			}
			t.Params = m
		}
		out[i] = t
	}
	return out
}
