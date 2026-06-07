package llm

import (
	"context"
	"iter"
	"strings"
	"sync"
)

// Session is an ergonomic, concurrency-safe driver around a Conversation. It
// holds the current Client, appends each turn automatically (with provider/model
// provenance and accumulated usage), and supports cross-provider handoff via
// SwitchProvider. The underlying Conversation stays a plain serializable value
// that you can snapshot and persist between turns.
//
// A Session serializes its own calls with a mutex, so concurrent Send/Stream on
// one Session are safe but run one at a time. For real parallelism, use separate
// Sessions (each over its own Conversation).
//
// Generation itself stays stateless: a Session just wires a Conversation to a
// Client and applies NormalizeForProvider before each request so the accumulated
// history is valid for whichever provider is currently active.
type Session struct {
	mu     sync.Mutex
	conv   *Conversation
	client Client
	base   Request
}

// SessionOption configures a Session at construction.
type SessionOption func(*Session)

// WithSystem sets the conversation's system prompt.
func WithSystem(system string) SessionOption {
	return func(s *Session) { s.conv.System = system }
}

// WithTools sets the tool set available for the conversation.
func WithTools(tools ...Tool) SessionOption {
	return func(s *Session) { s.conv.Tools = tools }
}

// WithBaseRequest sets a template Request whose generation settings (Model,
// MaxTokens, ThinkingLevel, Temperature, StopSequences, …) apply to every turn.
// Its Messages/System/Tools are overwritten by the conversation each turn.
func WithBaseRequest(base Request) SessionOption {
	return func(s *Session) { s.base = base }
}

// WithConversation resumes from an existing (e.g. restored) Conversation instead
// of starting a fresh, empty one.
func WithConversation(conv *Conversation) SessionOption {
	return func(s *Session) {
		if conv != nil {
			s.conv = conv
		}
	}
}

// NewSession creates a Session driving client. By default it starts an empty
// conversation; pass WithConversation to resume one.
func NewSession(client Client, opts ...SessionOption) *Session {
	s := &Session{
		conv:   NewConversation(""),
		client: client,
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// Conversation returns a snapshot copy of the underlying conversation, safe to
// read or json.Marshal at any time — even while other Session calls run
// concurrently. Mutating the returned value does not affect the Session; to
// resume from a modified transcript, build a new Session with WithConversation.
func (s *Session) Conversation() *Conversation {
	s.mu.Lock()
	defer s.mu.Unlock()
	cp := *s.conv
	cp.Messages = append([]Message(nil), s.conv.Messages...)
	cp.Tools = append([]Tool(nil), s.conv.Tools...)
	return &cp
}

// Usage returns a snapshot of the accumulated token/cost usage.
func (s *Session) Usage() Usage {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.conv.Usage
}

// Provider returns the current client's provider name.
func (s *Session) Provider() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.client.Provider()
}

// SwitchProvider performs a provider handoff: subsequent turns run against the
// new client, and the accumulated history is translated into that provider's
// format on the next Send/Stream (see NormalizeForProvider). Thinking blocks not
// native to the new provider degrade to text; tool calls, text, images, and
// documents carry over.
func (s *Session) SwitchProvider(client Client) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.client = client
}

// Append adds messages to the conversation without generating — e.g. to inject a
// tool result before the next Send.
func (s *Session) Append(msgs ...Message) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.conv.Append(msgs...)
}

// Send appends a user prompt, generates a completion, appends the assistant
// reply (with provenance + usage), and returns the response.
func (s *Session) Send(ctx context.Context, prompt string) (*Response, error) {
	return s.SendMessages(ctx, NewTextMessage(RoleUser, prompt))
}

// SendMessages appends the given messages (e.g. a user message, or tool results
// continuing a tool loop), generates, appends the assistant reply, and returns
// the response.
func (s *Session) SendMessages(ctx context.Context, msgs ...Message) (*Response, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := len(s.conv.Messages)
	s.conv.Append(msgs...)
	resp, err := s.client.Complete(ctx, s.requestLocked())
	if err != nil {
		// Roll back the appended input so a failed turn leaves the transcript
		// unchanged (otherwise a retry would stack two user turns in a row,
		// which providers like Anthropic reject).
		s.conv.Messages = s.conv.Messages[:n]
		return nil, err
	}
	s.conv.AddResponse(s.client.Provider(), resp)
	return resp, nil
}

// Stream is the streaming counterpart of Send. It yields events as they arrive;
// once the turn completes, it appends the assembled assistant message (text +
// thinking + tool calls, with provenance) and folds in usage. If the caller
// breaks the range early, no partial turn is appended.
func (s *Session) Stream(ctx context.Context, prompt string) iter.Seq2[StreamEvent, error] {
	return s.StreamMessages(ctx, NewTextMessage(RoleUser, prompt))
}

// StreamMessages is the streaming counterpart of SendMessages.
func (s *Session) StreamMessages(ctx context.Context, msgs ...Message) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		s.mu.Lock()
		defer s.mu.Unlock()

		n := len(s.conv.Messages)
		s.conv.Append(msgs...)
		// Roll back the appended input unless the turn completes and is recorded,
		// so an errored / aborted / abandoned stream leaves the transcript clean.
		committed := false
		defer func() {
			if !committed {
				s.conv.Messages = s.conv.Messages[:n]
			}
		}()

		req := s.requestLocked()
		provider := s.client.Provider()

		var (
			text, thinking   strings.Builder
			thinkingSig      string
			thinkingRedacted bool
			toolCalls        []ToolCall
			final            StreamEvent
			sawDone          bool
		)
		for ev, err := range s.client.Stream(ctx, req) {
			if err != nil {
				yield(StreamEvent{}, err)
				return // do not append a partial/failed turn
			}
			switch ev.Type {
			case EventContent:
				text.WriteString(ev.Text)
			case EventThinking:
				thinking.WriteString(ev.Thinking)
			case EventToolUse:
				if ev.ToolCall != nil {
					toolCalls = append(toolCalls, *ev.ToolCall)
				}
			case EventDone:
				final, sawDone = ev, true
				if ev.ThinkingSignature != "" {
					thinkingSig = ev.ThinkingSignature
				}
				if ev.ThinkingRedacted {
					thinkingRedacted = true
				}
			}
			if !yield(ev, nil) {
				return // caller broke; leave the conversation unchanged
			}
		}
		if !sawDone {
			return // stream ended without a clean completion; don't record it
		}

		resp := &Response{
			Model:               req.Model,
			Content:             text.String(),
			Thinking:            thinking.String(),
			ThinkingSignature:   thinkingSig,
			ThinkingRedacted:    thinkingRedacted,
			ToolCalls:           toolCalls,
			StopReason:          final.StopReason,
			InputTokens:         final.InputTokens,
			OutputTokens:        final.OutputTokens,
			ThinkingTokens:      final.ThinkingTokens,
			CacheCreationTokens: final.CacheCreationTokens,
			CacheReadTokens:     final.CacheReadTokens,
		}
		s.conv.AddResponse(provider, resp)
		committed = true
	}
}

// requestLocked builds the next Request from the conversation, normalized for the
// current provider. The caller must hold s.mu.
func (s *Session) requestLocked() Request {
	req := s.conv.ToRequest(s.base)
	req.Messages = NormalizeForProvider(req.Messages, s.client.Provider())
	if req.Model == "" {
		req.Model = s.client.Model()
	}
	return req
}
