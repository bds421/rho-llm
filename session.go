package llm

import (
	"context"
	"fmt"
	"iter"
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

// WithServerTools sets the provider-run tools (web search) for the
// conversation. They are sent unchanged with every turn and continuation.
func WithServerTools(tools ...ServerTool) SessionOption {
	return func(s *Session) { s.conv.ServerTools = tools }
}

// WithBaseRequest sets a template Request whose generation settings (Model,
// MaxTokens, ThinkingLevel, Temperature, StopSequences, …) apply to every turn.
// Its Messages/System/Tools are overwritten by the conversation each turn.
// SwitchProvider clears its Model, so the new client's model applies.
func WithBaseRequest(base Request) SessionOption {
	return func(s *Session) { s.base = base }
}

// WithConversation resumes from an existing (e.g. restored) Conversation instead
// of starting a fresh, empty one. The session works on a copy: it kept the
// caller's pointer and wrote to it from every turn, a data race with a caller
// still reading or saving it (B23). Read the live transcript through
// Session.Conversation.
func WithConversation(conv *Conversation) SessionOption {
	return func(s *Session) {
		if conv != nil {
			s.conv = conv.Clone()
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
//
// The new client's model is used from then on: a Model set through
// WithBaseRequest named the previous provider's model, and was sent to the new
// one, which refused it on every turn (B21). Other base settings stay; ones the
// new provider does not support (Effort, ContainerID, …) are refused by it.
func (s *Session) SwitchProvider(client Client) {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	s.client = client
	s.base.Model = ""
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
	return s.send(ctx, msgs, false)
}

// Continue continues a paused turn (StopPauseTurn): it sends the transcript,
// which ends with the paused assistant message, unchanged as the next
// request, with the same system prompt, tools and server tools, and the
// paused turn's container. The reply is appended as a further assistant
// message and can pause again; check StopReason and cap the rounds. It
// returns ErrNotPaused, without sending a request, when the last message is
// not a paused assistant turn.
func (s *Session) Continue(ctx context.Context) (*Response, error) {
	return s.send(ctx, nil, true)
}

func (s *Session) send(ctx context.Context, msgs []Message, continuation bool) (*Response, error) {
	s.turnMu.Lock()
	defer s.turnMu.Unlock()

	req, client, provider, err := s.prepareTurn(msgs, continuation)
	if err != nil {
		return nil, err
	}
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
// errors, ends without EventDone, or the caller breaks the range before
// EventDone, no partial turn is recorded. Breaking on EventDone itself (the
// usual `if ev.Type == EventDone { break }`) keeps the completed turn.
func (s *Session) Stream(ctx context.Context, prompt string) iter.Seq2[StreamEvent, error] {
	return s.StreamMessages(ctx, NewTextMessage(RoleUser, prompt))
}

// StreamMessages is the streaming counterpart of SendMessages.
func (s *Session) StreamMessages(ctx context.Context, msgs ...Message) iter.Seq2[StreamEvent, error] {
	return s.stream(ctx, msgs, false)
}

// StreamContinue is the streaming counterpart of Continue. It yields
// ErrNotPaused when the last message is not a paused assistant turn.
func (s *Session) StreamContinue(ctx context.Context) iter.Seq2[StreamEvent, error] {
	return s.stream(ctx, nil, true)
}

func (s *Session) stream(ctx context.Context, msgs []Message, continuation bool) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		s.turnMu.Lock()
		defer s.turnMu.Unlock()

		req, client, provider, err := s.prepareTurn(msgs, continuation)
		if err != nil {
			yield(StreamEvent{}, err)
			return
		}

		var acc StreamAccumulator
		for ev, err := range client.Stream(ctx, req) {
			if err != nil {
				// Pass the event on: after a provider reported usage it carries
				// what the broken turn cost. The turn itself is not recorded.
				yield(ev, err)
				return
			}
			acc.Add(ev)
			if !yield(ev, nil) {
				// The caller broke. On the terminal event the turn is
				// complete and is recorded; before it, the conversation is
				// left unchanged (B19).
				if acc.Done() {
					s.commitStreamed(msgs, provider, req.Model, &acc)
				}
				return
			}
		}
		if !acc.Done() {
			return // stream ended without a clean completion; don't record it
		}
		s.commitStreamed(msgs, provider, req.Model, &acc)
	}
}

// commitStreamed records a completed streamed turn. The caller must hold turnMu.
func (s *Session) commitStreamed(msgs []Message, provider, model string, acc *StreamAccumulator) {
	resp := acc.Response()
	if resp.Model == "" {
		resp.Model = model
	}
	s.commitTurn(msgs, provider, resp)
}

// prepareTurn builds the next Request from a snapshot of the conversation plus
// this turn's input messages, normalized for the current provider, and captures
// the client the turn will run against. The stored transcript is not mutated —
// input is committed together with the reply only after the turn succeeds.
// A continuation sends no new input: the transcript must end with a paused
// assistant turn, whose container the request reuses unless the base request
// names one. The caller must hold turnMu (but NOT stateMu).
func (s *Session) prepareTurn(msgs []Message, continuation bool) (Request, Client, string, error) {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()

	paused, container := pausedTurn(s.conv.Messages)
	if continuation && !paused {
		return Request{}, nil, "", ErrNotPaused
	}

	client := s.client
	provider := client.Provider()

	req := s.conv.ToRequest(s.base)
	combined := make([]Message, 0, len(s.conv.Messages)+len(msgs))
	combined = append(combined, s.conv.Messages...)
	combined = append(combined, msgs...)
	if req.Model == "" {
		req.Model = client.Model()
	}
	req.Messages = NormalizeForModel(combined, provider, req.Model)
	if continuation {
		keepContainer(&req, container)
	}
	return req, client, provider, nil
}

// commitTurn atomically records a completed turn: the input messages and the
// assistant reply (with provenance + usage). The caller must hold turnMu.
func (s *Session) commitTurn(msgs []Message, provider string, resp *Response) {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	s.conv.Append(msgs...)
	s.conv.AddResponse(provider, resp)
}
