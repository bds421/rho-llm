package llm

import (
	"context"
	"fmt"
	"iter"
	"strings"
	"sync"
)

// MockClient is an in-memory Client for tests and examples — it makes no network
// calls. Script it with PushResponse/PushText/PushStream/PushError (consumed FIFO,
// one per Complete/Stream call) or SetResponseFunc for dynamic behavior, and read
// back the requests it received with Requests()/LastRequest().
//
// Scripting is symmetric: a queued Response is synthesized into events when
// consumed by Stream, and queued events are assembled into a Response when
// consumed by Complete — so one script drives either call style. Pass a
// MockClient anywhere a Client is expected (e.g. llm.NewSession(mock)).
//
// MockClient is safe for concurrent use.
type MockClient struct {
	mu       sync.Mutex
	provider string
	model    string
	queue    []mockTurn
	requests []Request
	respFunc func(Request) (*Response, error)
}

type mockTurn struct {
	resp   *Response
	events []StreamEvent
	err    error
}

// NewMockClient creates a mock with the given provider and model names (returned
// by Provider() and Model()).
func NewMockClient(provider, model string) *MockClient {
	return &MockClient{provider: provider, model: model}
}

// SetResponseFunc installs a dynamic handler that takes precedence over the
// scripted queue: it receives each request and returns the response (or error).
func (m *MockClient) SetResponseFunc(fn func(Request) (*Response, error)) *MockClient {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.respFunc = fn
	return m
}

// PushResponse queues a Response for the next Complete (or, for the next Stream,
// synthesized into events). Returns the receiver for chaining.
func (m *MockClient) PushResponse(r *Response) *MockClient {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.queue = append(m.queue, mockTurn{resp: r})
	return m
}

// PushText queues a plain-text assistant reply.
func (m *MockClient) PushText(text string) *MockClient {
	return m.PushResponse(&Response{Content: text, StopReason: StopEndTurn})
}

// PushStream queues a sequence of stream events for the next Stream (or, for the
// next Complete, assembled into a Response).
func (m *MockClient) PushStream(events ...StreamEvent) *MockClient {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.queue = append(m.queue, mockTurn{events: events})
	return m
}

// PushError queues an error for the next Complete/Stream.
func (m *MockClient) PushError(err error) *MockClient {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.queue = append(m.queue, mockTurn{err: err})
	return m
}

// pop records the request and dequeues the next scripted turn (under lock).
func (m *MockClient) pop(req Request) (fn func(Request) (*Response, error), turn mockTurn, has bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.requests = append(m.requests, req)
	if m.respFunc != nil {
		return m.respFunc, mockTurn{}, false
	}
	if len(m.queue) > 0 {
		turn = m.queue[0]
		m.queue = m.queue[1:]
		has = true
	}
	return nil, turn, has
}

// Complete implements Client.
func (m *MockClient) Complete(_ context.Context, req Request) (*Response, error) {
	fn, turn, has := m.pop(req)
	if fn != nil {
		return fn(req)
	}
	if !has {
		return nil, fmt.Errorf("llm: MockClient has no scripted response for Complete")
	}
	if turn.err != nil {
		return nil, turn.err
	}
	if turn.resp != nil {
		r := *turn.resp
		if r.Model == "" {
			r.Model = m.model
		}
		return &r, nil
	}
	return assembleMockResponse(turn.events, m.model), nil
}

// Stream implements Client.
func (m *MockClient) Stream(_ context.Context, req Request) iter.Seq2[StreamEvent, error] {
	fn, turn, has := m.pop(req)
	return func(yield func(StreamEvent, error) bool) {
		if fn != nil {
			resp, err := fn(req)
			if err != nil {
				yield(StreamEvent{}, err)
				return
			}
			for _, ev := range mockResponseToEvents(resp) {
				if !yield(ev, nil) {
					return
				}
			}
			return
		}
		if !has {
			yield(StreamEvent{}, fmt.Errorf("llm: MockClient has no scripted response for Stream"))
			return
		}
		if turn.err != nil {
			yield(StreamEvent{}, turn.err)
			return
		}
		events := turn.events
		if events == nil {
			events = mockResponseToEvents(turn.resp)
		}
		for _, ev := range events {
			if !yield(ev, nil) {
				return
			}
		}
	}
}

// Requests returns a copy of every request the mock has received, in order.
func (m *MockClient) Requests() []Request {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]Request(nil), m.requests...)
}

// LastRequest returns the most recent request the mock received.
func (m *MockClient) LastRequest() (Request, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.requests) == 0 {
		return Request{}, false
	}
	return m.requests[len(m.requests)-1], true
}

// Provider implements Client.
func (m *MockClient) Provider() string { return m.provider }

// Model implements Client.
func (m *MockClient) Model() string { return m.model }

// Close implements Client.
func (m *MockClient) Close() error { return nil }

// FauxResponse builds a plain-text assistant Response (stop_reason "end_turn").
func FauxResponse(text string) *Response {
	return &Response{Content: text, StopReason: StopEndTurn}
}

// FauxToolCall builds an assistant Response that issues a single tool call
// (stop_reason "tool_use").
func FauxToolCall(id, name string, input any) *Response {
	return &Response{
		ToolCalls:  []ToolCall{{ID: id, Name: name, Input: input}},
		StopReason: StopToolUse,
	}
}

// mockResponseToEvents synthesizes a stream from a Response (thinking → text →
// tool calls → done), mirroring how a real adapter would stream it.
func mockResponseToEvents(r *Response) []StreamEvent {
	if r == nil {
		return nil
	}
	var out []StreamEvent
	if r.Thinking != "" {
		out = append(out, StreamEvent{Type: EventThinking, Thinking: r.Thinking})
	}
	if r.Content != "" {
		out = append(out, StreamEvent{Type: EventContent, Text: r.Content})
	}
	for i := range r.ToolCalls {
		tc := r.ToolCalls[i]
		out = append(out, StreamEvent{Type: EventToolUse, ToolCall: &tc})
	}
	stop := r.StopReason
	if stop == "" {
		stop = StopEndTurn
	}
	out = append(out, StreamEvent{
		Type: EventDone, StopReason: stop,
		InputTokens: r.InputTokens, OutputTokens: r.OutputTokens,
		ThinkingTokens: r.ThinkingTokens, ThinkingSignature: r.ThinkingSignature,
		ThinkingRedacted:    r.ThinkingRedacted,
		CacheCreationTokens: r.CacheCreationTokens, CacheReadTokens: r.CacheReadTokens,
	})
	return out
}

// assembleMockResponse folds a scripted event stream into a Response, mirroring
// how Session.Stream reassembles a turn.
func assembleMockResponse(events []StreamEvent, model string) *Response {
	r := &Response{Model: model}
	var text, thinking strings.Builder
	for _, ev := range events {
		switch ev.Type {
		case EventContent:
			text.WriteString(ev.Text)
		case EventThinking:
			thinking.WriteString(ev.Thinking)
		case EventToolUse:
			if ev.ToolCall != nil {
				r.ToolCalls = append(r.ToolCalls, *ev.ToolCall)
			}
		case EventDone:
			r.StopReason = ev.StopReason
			r.InputTokens = ev.InputTokens
			r.OutputTokens = ev.OutputTokens
			r.ThinkingTokens = ev.ThinkingTokens
			r.ThinkingSignature = ev.ThinkingSignature
			r.ThinkingRedacted = ev.ThinkingRedacted
			r.CacheCreationTokens = ev.CacheCreationTokens
			r.CacheReadTokens = ev.CacheReadTokens
		}
	}
	r.Content = text.String()
	r.Thinking = thinking.String()
	if r.StopReason == "" {
		r.StopReason = StopEndTurn
	}
	return r
}
