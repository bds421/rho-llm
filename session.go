package llm

import (
	"context"
	"fmt"
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
// Concurrency: turns (Send/Stream) are serialized — concurrent calls are safe
// but run one at a time. Snapshot reads (Conversation, Usage, Provider) and
// SwitchProvider never wait for an in-flight turn: a turn is committed to the
// transcript only after it completes, so snapshots taken mid-turn simply do
// not include it, and a provider switch mid-turn applies from the next turn.
// For real parallelism, use separate Sessions (each over its own Conversation).
//
// Generation itself stays stateless: a Session just wires a Conversation to a
// Client and applies NormalizeForProvider before each request so the accumulated
// history is valid for whichever provider is currently active.
type Session struct {
	turnMu  sync.Mutex // serializes turns (Send/Stream/Append)
	stateMu sync.Mutex // guards conv, client, base — held only briefly
	conv    *Conversation
	client  Client
	base    Request
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

// Conversation returns a deep-copied snapshot of the underlying conversation,
// safe to read, mutate, or json.Marshal at any time — even while a turn is
// streaming (an uncommitted in-flight turn is not included). Mutating the
// returned value never affects the Session; to resume from a modified
// transcript, build a new Session with WithConversation.
func (s *Session) Conversation() *Conversation {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	return s.conv.Clone()
}

// Usage returns a snapshot of the accumulated token/cost usage.
func (s *Session) Usage() Usage {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	return s.conv.Usage
}

// Provider returns the current client's provider name.
func (s *Session) Provider() string {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	return s.client.Provider()
}

// SwitchProvider performs a provider handoff: subsequent turns run against the
// new client, and the accumulated history is translated into that provider's
// format on the next Send/Stream (see NormalizeForProvider). Thinking blocks not
// native to the new provider degrade to text; tool calls, text, images, and
// documents carry over. Switching while a turn is in flight does not affect
// that turn — it completes (and is recorded) against the client it started with.
func (s *Session) SwitchProvider(client Client) {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	s.client = client
}

// Append adds messages to the conversation without generating — e.g. to inject a
// tool result before the next Send. Like a turn, it waits for any in-flight
// turn to finish so transcript order stays deterministic.
func (s *Session) Append(msgs ...Message) {
	s.turnMu.Lock()
	defer s.turnMu.Unlock()
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	s.conv.Append(msgs...)
}

// Send appends a user prompt, generates a completion, appends the assistant
// reply (with provenance + usage), and returns the response.
func (s *Session) Send(ctx context.Context, prompt string) (*Response, error) {
	return s.SendMessages(ctx, NewTextMessage(RoleUser, prompt))
}

// SendMessages appends the given messages (e.g. a user message, or tool results
// continuing a tool loop), generates, appends the assistant reply, and returns
// the response. On failure nothing is recorded — input and reply are committed
// to the transcript together, only after the turn succeeds.
func (s *Session) SendMessages(ctx context.Context, msgs ...Message) (*Response, error) {
	s.turnMu.Lock()
	defer s.turnMu.Unlock()

	req, client, provider := s.prepareTurn(msgs)
	resp, err := client.Complete(ctx, req)
	if err != nil {
		return nil, err // nothing was appended — the transcript is untouched
	}
	if resp == nil {
		// A misbehaving Client returned (nil, nil). Surface it as an error
		// rather than panicking in NewAssistantMessage and taking down the
		// caller; the transcript stays untouched.
		return nil, fmt.Errorf("llm: client %q returned a nil response and nil error", provider)
	}
	if resp.Model == "" {
		// A client that left Model empty still gets correct cost/provenance:
		// attribute the turn to the model it actually ran against (Stream already
		// stamps req.Model). Without this, EstimateCost("") silently returns $0.
		resp.Model = req.Model
	}
	s.commitTurn(msgs, provider, resp)
	return resp, nil
}

// Stream is the streaming counterpart of Send. It yields events as they arrive;
// once the turn completes, it appends the assembled assistant message (text +
// thinking + tool calls, with provenance) and folds in usage. If the stream
// errors, ends without EventDone, or the caller breaks the range early, no
// partial turn is recorded.
func (s *Session) Stream(ctx context.Context, prompt string) iter.Seq2[StreamEvent, error] {
	return s.StreamMessages(ctx, NewTextMessage(RoleUser, prompt))
}

// StreamMessages is the streaming counterpart of SendMessages.
func (s *Session) StreamMessages(ctx context.Context, msgs ...Message) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		s.turnMu.Lock()
		defer s.turnMu.Unlock()

		req, client, provider := s.prepareTurn(msgs)

		var (
			text, thinking   strings.Builder
			thinkingSig      string
			thinkingRedacted bool
			toolCalls        []ToolCall
			final            StreamEvent
			sawDone          bool
		)
		for ev, err := range client.Stream(ctx, req) {
			if err != nil {
				yield(StreamEvent{}, err)
				return // do not record a partial/failed turn
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
		s.commitTurn(msgs, provider, resp)
	}
}

// prepareTurn builds the next Request from a snapshot of the conversation plus
// this turn's input messages, normalized for the current provider, and captures
// the client the turn will run against. The stored transcript is not mutated —
// input is committed together with the reply only after the turn succeeds.
// The caller must hold turnMu (but NOT stateMu).
func (s *Session) prepareTurn(msgs []Message) (Request, Client, string) {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()

	client := s.client
	provider := client.Provider()

	req := s.conv.ToRequest(s.base)
	combined := make([]Message, 0, len(s.conv.Messages)+len(msgs))
	combined = append(combined, s.conv.Messages...)
	combined = append(combined, msgs...)
	req.Messages = NormalizeForProvider(combined, provider)
	if req.Model == "" {
		req.Model = client.Model()
	}
	return req, client, provider
}

// commitTurn atomically records a completed turn: the input messages and the
// assistant reply (with provenance + usage). The caller must hold turnMu.
func (s *Session) commitTurn(msgs []Message, provider string, resp *Response) {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	s.conv.Append(msgs...)
	s.conv.AddResponse(provider, resp)
}
