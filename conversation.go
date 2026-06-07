package llm

import (
	"encoding/json"
	"fmt"
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
type Usage struct {
	InputTokens         int     `json:"input_tokens"`
	OutputTokens        int     `json:"output_tokens"`
	ThinkingTokens      int     `json:"thinking_tokens,omitempty"`
	CacheCreationTokens int     `json:"cache_creation_tokens,omitempty"`
	CacheReadTokens     int     `json:"cache_read_tokens,omitempty"`
	Cost                float64 `json:"cost,omitempty"` // estimated USD, summed per response
}

// AddResponse folds one response's usage into the running totals, estimating its
// cost from the response's own model (registry pricing). Nil-safe.
//
// Token counts are clamped to >= 0 first: a streamed turn whose provider never
// reported usage carries the TokensNotReported (-1) sentinel, which must not
// pollute the accumulated totals or produce a negative cost.
func (u *Usage) AddResponse(resp *Response) {
	if resp == nil {
		return
	}
	in := nonNeg(resp.InputTokens)
	out := nonNeg(resp.OutputTokens)
	think := nonNeg(resp.ThinkingTokens)
	cacheW := nonNeg(resp.CacheCreationTokens)
	cacheR := nonNeg(resp.CacheReadTokens)

	u.InputTokens += in
	u.OutputTokens += out
	u.ThinkingTokens += think
	u.CacheCreationTokens += cacheW
	u.CacheReadTokens += cacheR
	u.Cost += EstimateCost(CostInput{
		Model:             resp.Model,
		InputTokens:       in,
		OutputTokens:      out,
		ThinkingTokens:    think,
		CacheCreateTokens: cacheW,
		CacheReadTokens:   cacheR,
	})
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
	SchemaVersion int       `json:"schema_version"`
	System        string    `json:"system,omitempty"`
	Tools         []Tool    `json:"tools,omitempty"`
	Messages      []Message `json:"messages"`
	Usage         Usage     `json:"usage,omitempty"`
}

// NewConversation creates an empty conversation with an optional system prompt
// and tool set.
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
	req.Messages = c.Messages
	return req
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
	if err := json.Unmarshal(data, &c); err != nil {
		return nil, fmt.Errorf("llm: decode conversation: %w", err)
	}
	if c.SchemaVersion <= 0 {
		c.SchemaVersion = ConversationSchemaVersion // absent/zero/negative → treat as v1
	}
	if c.SchemaVersion > ConversationSchemaVersion {
		return nil, fmt.Errorf("llm: conversation schema_version %d is newer than supported %d", c.SchemaVersion, ConversationSchemaVersion)
	}
	return &c, nil
}
