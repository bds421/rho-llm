package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"strings"
	"sync"
	"testing"
	"time"
)

// threadServer is a Client that behaves like the Anthropic message-threads
// API (docs: build-with-claude/message-threads): a create request stores the
// conversation under the response ID, a continue request appends to the thread
// that ID names, each ID continues once, and unknown IDs answer 404
// thread_not_found. It keeps every request for inspection.
type threadServer struct {
	mu       sync.Mutex
	provider string
	threads  map[string][]Message // response ID -> conversation held, reply included
	used     map[string]bool
	reqs     []Request
	// inject returns an error for the n-th request (1-based) instead of serving it.
	inject func(n int, req Request) error
	// reply shapes the n-th response; the ID is filled in.
	reply func(n int) *Response
}

func newThreadServer() *threadServer {
	return &threadServer{provider: "anthropic", threads: map[string][]Message{}, used: map[string]bool{}}
}

func (s *threadServer) Provider() string { return s.provider }
func (s *threadServer) Model() string    { return "claude-test" }
func (s *threadServer) Close() error     { return nil }

func threadErr(status int, code, msg string) error {
	body := fmt.Sprintf(`{"type":"error","error":{"type":"invalid_request_error","message":%q,"details":{"error_code":%q}}}`, msg, code)
	if code == "" {
		body = fmt.Sprintf(`{"type":"error","error":{"type":"invalid_request_error","message":%q}}`, msg)
	}
	return &APIError{StatusCode: status, Provider: "anthropic", Message: body}
}

func (s *threadServer) serve(req Request) (*Response, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.reqs = append(s.reqs, req)
	n := len(s.reqs)
	if s.inject != nil {
		if err := s.inject(n, req); err != nil {
			return nil, err
		}
	}
	var held []Message
	if th := req.Thread; th != nil {
		if th.PreviousMessageID != "" {
			prev, ok := s.threads[th.PreviousMessageID]
			if !ok {
				return nil, threadErr(404, ThreadNotFound, "thread not found")
			}
			if s.used[th.PreviousMessageID] {
				return nil, threadErr(400, ThreadAlreadyContinued, "already continued")
			}
			s.used[th.PreviousMessageID] = true
			held = append(held, prev...)
		}
	}
	held = append(held, req.Messages...)
	resp := &Response{Content: fmt.Sprintf("reply %d", n), StopReason: StopEndTurn, Model: "claude-test"}
	if s.reply != nil {
		resp = s.reply(n)
	}
	r := *resp
	r.ID = fmt.Sprintf("msg_%d", n)
	if resp.ID == "-" {
		r.ID = ""
	}
	if req.Thread != nil && r.ID != "" {
		s.threads[r.ID] = append(held, NewAssistantMessage(&r))
	}
	return &r, nil
}

func (s *threadServer) Complete(_ context.Context, req Request) (*Response, error) {
	return s.serve(req)
}

func (s *threadServer) Stream(_ context.Context, req Request) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		resp, err := s.serve(req)
		if err != nil {
			yield(StreamEvent{}, err)
			return
		}
		if !yield(StreamEvent{Type: EventContent, Text: resp.Content}, nil) {
			return
		}
		yield(StreamEvent{Type: EventDone, ID: resp.ID, Model: resp.Model, StopReason: resp.StopReason, Iterations: resp.Iterations}, nil)
	}
}

func (s *threadServer) calls() []Request {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]Request(nil), s.reqs...)
}

// expireAll forgets every thread, as the five-minute expiry does.
func (s *threadServer) expireAll() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.threads = map[string][]Message{}
}

// holds returns the role/text shape of the conversation the thread with this
// tip ID holds.
func (s *threadServer) holds(id string) []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return shapeOf(s.threads[id])
}

func shapeOf(msgs []Message) []string {
	var out []string
	for _, m := range msgs {
		var b strings.Builder
		for _, p := range m.Content {
			b.WriteString(p.Text)
			b.WriteString(p.ToolResultContent)
		}
		out = append(out, string(m.Role)+":"+b.String())
	}
	return out
}

func equalShape(a, b []string) bool {
	return strings.Join(a, "\x00") == strings.Join(b, "\x00")
}

func threadSession(srv *threadServer, opts ...SessionOption) *Session {
	opts = append([]SessionOption{WithMessageThreads(), WithSystem("sys"), WithBaseRequest(Request{MaxTokens: 100})}, opts...)
	return NewSession(srv, opts...)
}

func send(t *testing.T, s *Session, text string) *Response {
	t.Helper()
	resp, err := s.Send(context.Background(), text)
	if err != nil {
		t.Fatalf("Send(%q): %v", text, err)
	}
	return resp
}

// assertServerMirrors checks that what the server holds for the latest reply
// is exactly the session's transcript: sending only deltas must never let the
// two drift apart.
func assertServerMirrors(t *testing.T, srv *threadServer, s *Session, resp *Response) {
	t.Helper()
	want := shapeOf(s.Conversation().Messages)
	if got := srv.holds(resp.ID); !equalShape(got, want) {
		t.Fatalf("server holds %q, transcript is %q", got, want)
	}
}

func TestThreadsAreOffByDefault(t *testing.T) {
	srv := newThreadServer()
	s := NewSession(srv, WithSystem("sys"))
	for _, q := range []string{"a", "b", "c"} {
		send(t, s, q)
	}
	for i, r := range srv.calls() {
		if r.Thread != nil {
			t.Errorf("request %d carries a thread without WithMessageThreads", i+1)
		}
		if want := 2*i + 1; len(r.Messages) != want {
			t.Errorf("request %d sends %d messages, want the full %d", i+1, len(r.Messages), want)
		}
	}
}

func TestThreadsContinueSendsOnlyTheDelta(t *testing.T) {
	srv := newThreadServer()
	s := threadSession(srv, WithTools(Tool{Name: "f", Description: "d", InputSchema: map[string]any{"type": "object"}}))
	var last *Response
	for _, q := range []string{"one", "two", "three"} {
		last = send(t, s, q)
		assertServerMirrors(t, srv, s, last)
	}
	reqs := srv.calls()
	if len(reqs) != 3 {
		t.Fatalf("%d requests, want 3", len(reqs))
	}
	if th := reqs[0].Thread; th == nil || th.PreviousMessageID != "" || len(reqs[0].Messages) != 1 {
		t.Errorf("first request should create: thread %+v, %d messages", th, len(reqs[0].Messages))
	}
	for i, prev := range []string{"msg_1", "msg_2"} {
		r := reqs[i+1]
		if r.Thread == nil || r.Thread.PreviousMessageID != prev {
			t.Errorf("request %d continues %+v, want %s", i+2, r.Thread, prev)
		}
		if len(r.Messages) != 1 || r.Messages[0].Role != RoleUser {
			t.Errorf("request %d sends %d messages, want just the new user turn", i+2, len(r.Messages))
		}
		// The adapter drops them from the wire; the neutral request keeps the
		// fixed fields so the adapter builds the same tool_choice and thinking.
		if r.System != "sys" || len(r.Tools) != 1 || r.MaxTokens != 100 {
			t.Errorf("request %d lost its settings: %+v", i+2, r)
		}
	}
	if n := len(s.Conversation().Messages); n != 6 {
		t.Errorf("transcript has %d messages, want 6", n)
	}
}

func TestThreadsToolLoopSendsOnlyTheResults(t *testing.T) {
	srv := newThreadServer()
	srv.reply = func(n int) *Response {
		if n == 1 {
			return &Response{StopReason: StopToolUse, Model: "claude-test", ToolCalls: []ToolCall{{ID: "toolu_1", Name: "f", Input: map[string]any{}}}}
		}
		return &Response{Content: "done", StopReason: StopEndTurn, Model: "claude-test"}
	}
	s := threadSession(srv)
	send(t, s, "go")
	resp, err := s.SendMessages(context.Background(), NewToolResultMessage("toolu_1", "result", false))
	if err != nil {
		t.Fatal(err)
	}
	r := srv.calls()[1]
	if r.Thread == nil || r.Thread.PreviousMessageID != "msg_1" || len(r.Messages) != 1 || r.Messages[0].Content[0].Type != ContentToolResult {
		t.Fatalf("tool result turn: thread %+v, messages %d", r.Thread, len(r.Messages))
	}
	assertServerMirrors(t, srv, s, resp)
}

func TestThreadsAppendedMessagesRideTheDelta(t *testing.T) {
	srv := newThreadServer()
	s := threadSession(srv)
	send(t, s, "a")
	s.Append(NewTextMessage(RoleUser, "context note"))
	// Two user messages in a row merge into one on some paths; either way the
	// delta carries both and nothing else.
	resp := send(t, s, "b")
	r := srv.calls()[1]
	if r.Thread == nil || r.Thread.PreviousMessageID == "" {
		t.Fatalf("expected a continuation, got %+v", r.Thread)
	}
	if got := strings.Join(shapeOf(r.Messages), "|"); !strings.Contains(got, "context note") || !strings.Contains(got, "user:b") || strings.Contains(got, "reply") {
		t.Errorf("delta = %q", got)
	}
	assertServerMirrors(t, srv, s, resp)
}

// A thread the server cannot continue is replayed once as a full new thread
// and the turn succeeds; the next turn continues the new thread.
func TestThreadsReplayWhenTheServerCannotContinue(t *testing.T) {
	cases := []struct {
		name string
		err  error
	}{
		{"thread_not_found", threadErr(404, ThreadNotFound, "gone")},
		{"already continued", threadErr(400, ThreadAlreadyContinued, "x")},
		{"continue in flight", threadErr(400, ThreadContinueInFlight, "x")},
		{"fingerprint mismatch", threadErr(400, ThreadFingerprint, "x")},
		{"400 without a code", threadErr(400, "", "continuation not understood")},
		{"404 without a code", threadErr(404, "", "x")},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := newThreadServer()
			s := threadSession(srv)
			send(t, s, "one")
			srv.inject = func(n int, req Request) error {
				if n == 2 {
					return tc.err
				}
				return nil
			}
			resp := send(t, s, "two")
			reqs := srv.calls()
			if len(reqs) != 3 {
				t.Fatalf("%d requests, want continue then replay", len(reqs))
			}
			if th := reqs[1].Thread; th == nil || th.PreviousMessageID == "" {
				t.Fatalf("attempt 2 should continue, got %+v", th)
			}
			replay := reqs[2]
			if replay.Thread == nil || replay.Thread.PreviousMessageID != "" || len(replay.Messages) != 3 {
				t.Fatalf("replay should create with the full 3 messages: thread %+v, %d messages", replay.Thread, len(replay.Messages))
			}
			assertServerMirrors(t, srv, s, resp)
			resp = send(t, s, "three")
			r := srv.calls()[3]
			if r.Thread == nil || r.Thread.PreviousMessageID != "msg_3" || len(r.Messages) != 1 {
				t.Fatalf("after the replay the thread should continue: %+v / %d", r.Thread, len(r.Messages))
			}
			assertServerMirrors(t, srv, s, resp)
		})
	}
}

func TestThreadsExpiredOnTheServerIsReplayed(t *testing.T) {
	srv := newThreadServer()
	s := threadSession(srv)
	send(t, s, "one")
	srv.expireAll()
	resp := send(t, s, "two")
	assertServerMirrors(t, srv, s, resp)
	if th := srv.calls()[2].Thread; th == nil || th.PreviousMessageID != "" {
		t.Errorf("replay should be a create, got %+v", th)
	}
}

// Errors that are not the thread's must not trigger a replay, and must leave
// the thread usable.
func TestThreadsOtherErrorsPropagate(t *testing.T) {
	for _, status := range []int{401, 403, 413, 429, 500, 529} {
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			srv := newThreadServer()
			s := threadSession(srv)
			send(t, s, "one")
			fail := true
			srv.inject = func(n int, req Request) error {
				if fail && n == 2 {
					return &APIError{StatusCode: status, Provider: "anthropic", Message: `{"error":{"message":"nope"}}`}
				}
				return nil
			}
			if _, err := s.Send(context.Background(), "two"); err == nil {
				t.Fatal("expected the error")
			}
			if n := len(srv.calls()); n != 2 {
				t.Fatalf("%d requests, an unrelated error must not be replayed", n)
			}
			if n := len(s.Conversation().Messages); n != 2 {
				t.Fatalf("failed turn left %d messages", n)
			}
			fail = false
			resp := send(t, s, "two again")
			if r := srv.calls()[2]; r.Thread == nil || r.Thread.PreviousMessageID != "msg_1" {
				t.Errorf("the thread should still continue from msg_1: %+v", r.Thread)
			}
			assertServerMirrors(t, srv, s, resp)
		})
	}
}

func TestThreadsContextCancelIsNotReplayed(t *testing.T) {
	srv := newThreadServer()
	s := threadSession(srv)
	send(t, s, "one")
	srv.inject = func(n int, req Request) error { return context.Canceled }
	if _, err := s.Send(context.Background(), "two"); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v", err)
	}
	if n := len(srv.calls()); n != 2 {
		t.Fatalf("%d requests", n)
	}
}

func TestThreadsReplayThatFailsReturnsTheError(t *testing.T) {
	srv := newThreadServer()
	s := threadSession(srv)
	send(t, s, "one")
	srv.inject = func(n int, req Request) error {
		switch n {
		case 2:
			return threadErr(404, ThreadNotFound, "gone")
		case 3:
			return &APIError{StatusCode: 500, Provider: "anthropic", Message: "boom", Retryable: true}
		}
		return nil
	}
	if _, err := s.Send(context.Background(), "two"); err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("got %v", err)
	}
	if n := len(s.Conversation().Messages); n != 2 {
		t.Fatalf("%d messages after the failed turn", n)
	}
	// The dead thread is forgotten: the retry creates.
	send(t, s, "two")
	if th := srv.calls()[3].Thread; th == nil || th.PreviousMessageID != "" {
		t.Errorf("after a failed replay the next turn should create, got %+v", th)
	}
}

// A server that refuses threads for this request or organization turns them
// off: plain requests from then on, no retries of the thread.
func TestThreadsRefusedByTheServerFallBackToPlain(t *testing.T) {
	t.Run("on create", func(t *testing.T) {
		srv := newThreadServer()
		srv.inject = func(n int, req Request) error {
			if req.Thread != nil {
				return threadErr(400, ThreadUnsupported, "org has HIPAA readiness")
			}
			return nil
		}
		s := threadSession(srv)
		send(t, s, "one")
		send(t, s, "two")
		send(t, s, "three")
		reqs := srv.calls()
		if len(reqs) != 4 {
			t.Fatalf("%d requests, want create+plain, then plain, plain", len(reqs))
		}
		for i, r := range reqs[1:] {
			if r.Thread != nil {
				t.Errorf("request %d still threaded", i+2)
			}
			if want := 2*i + 1; len(r.Messages) != want {
				t.Errorf("plain request %d sends %d messages, want %d", i+2, len(r.Messages), want)
			}
		}
	})
	t.Run("on continue then create", func(t *testing.T) {
		srv := newThreadServer()
		s := threadSession(srv)
		send(t, s, "one")
		srv.inject = func(n int, req Request) error {
			if req.Thread != nil {
				return threadErr(400, ThreadUnsupported, "x")
			}
			return nil
		}
		send(t, s, "two")
		reqs := srv.calls()
		if len(reqs) != 4 || reqs[3].Thread != nil || len(reqs[3].Messages) != 3 {
			t.Fatalf("want continue, create, plain with the full conversation; got %d requests", len(reqs))
		}
	})
}

func TestThreadsChangedSettingsStartANewThread(t *testing.T) {
	cases := []struct {
		name   string
		change func(s *Session)
	}{
		{"system prompt", func(s *Session) { s.conv.System = "different" }},
		{"tools", func(s *Session) { s.conv.Tools = []Tool{{Name: "new"}} }},
		{"max tokens", func(s *Session) { s.base.MaxTokens = 50 }},
		{"thinking level", func(s *Session) { s.base.ThinkingLevel = ThinkingHigh }},
		{"tool choice", func(s *Session) { s.base.ToolChoice = &ToolChoice{Mode: ToolChoiceNone} }},
		{"model", func(s *Session) { s.base.Model = "other-model" }},
		{"sampling params", func(s *Session) { s.base.SamplingParams = map[string]any{"x": 1} }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := newThreadServer()
			s := threadSession(srv)
			send(t, s, "one")
			tc.change(s)
			send(t, s, "two")
			reqs := srv.calls()
			if len(reqs) != 2 {
				t.Fatalf("%d requests: a changed fixed field must not try the old thread first", len(reqs))
			}
			if th := reqs[1].Thread; th == nil || th.PreviousMessageID != "" || len(reqs[1].Messages) != 3 {
				t.Errorf("want a full create, got thread %+v with %d messages", th, len(reqs[1].Messages))
			}
		})
	}
}

// The container is sent on every request regardless of the thread, so a
// paused turn's container must not read as a changed setting.
func TestThreadsIgnoreTheContainer(t *testing.T) {
	srv := newThreadServer()
	s := threadSession(srv)
	send(t, s, "one")
	s.base.ContainerID = "container_1"
	send(t, s, "two")
	if r := srv.calls()[1]; r.Thread == nil || r.Thread.PreviousMessageID == "" {
		t.Errorf("a container alone should not start a new thread: %+v", r.Thread)
	}
}

// The server may only be trusted to hold what the transcript says. Any edit
// to the transcript the thread already holds sends the whole conversation.
func TestThreadsEditedHistoryStartsANewThread(t *testing.T) {
	edits := map[string]func(s *Session){
		"edit an earlier user message": func(s *Session) { s.conv.Messages[0].Content[0].Text = "edited" },
		"edit the reply":               func(s *Session) { s.conv.Messages[1].Content[0].Text = "edited reply" },
		"drop the reply":               func(s *Session) { s.conv.Messages = s.conv.Messages[:1] },
		"truncate to nothing":          func(s *Session) { s.conv.Messages = nil },
		"insert a message at the front": func(s *Session) {
			s.conv.Messages = append([]Message{NewTextMessage(RoleUser, "first")}, s.conv.Messages...)
		},
		"swap roles":            func(s *Session) { s.conv.Messages[1].Role = RoleUser },
		"edit the latest reply": func(s *Session) { s.conv.Messages[3].Content[0].Text = "edited latest" },
	}
	for name, edit := range edits {
		t.Run(name, func(t *testing.T) {
			srv := newThreadServer()
			s := threadSession(srv)
			send(t, s, "one")
			send(t, s, "two") // a continuation, so a thread state exists
			edit(s)
			if _, err := s.Send(context.Background(), "three"); err != nil {
				t.Fatal(err)
			}
			reqs := srv.calls()
			last := reqs[len(reqs)-1]
			if last.Thread == nil || last.Thread.PreviousMessageID != "" {
				t.Fatalf("edited history must create a new thread, got %+v", last.Thread)
			}
			if len(last.Messages) != len(s.Conversation().Messages)-1 {
				t.Errorf("create sends %d messages, transcript (without the reply) has %d", len(last.Messages), len(s.Conversation().Messages)-1)
			}
		})
	}
}

func TestThreadsExpireByTheClock(t *testing.T) {
	srv := newThreadServer()
	now := time.Unix(1_800_000_000, 0)
	s := threadSession(srv)
	s.now = func() time.Time { return now }
	send(t, s, "one")
	now = now.Add(threadReuseWindow - time.Second)
	send(t, s, "two") // still inside the window: continues
	if th := srv.calls()[1].Thread; th == nil || th.PreviousMessageID == "" {
		t.Fatalf("inside the window should continue: %+v", th)
	}
	now = now.Add(threadReuseWindow + time.Second)
	send(t, s, "three")
	reqs := srv.calls()
	if len(reqs) != 3 {
		t.Fatalf("%d requests: an expired thread should not be tried", len(reqs))
	}
	if th := reqs[2].Thread; th == nil || th.PreviousMessageID != "" || len(reqs[2].Messages) != 5 {
		t.Errorf("want a full create, got %+v with %d messages", th, len(reqs[2].Messages))
	}
}

// After these replies the API holds no usable turn to continue from.
func TestThreadsEndAfterUnstorableReplies(t *testing.T) {
	cases := map[string]*Response{
		"refusal":        {Content: "no", StopReason: StopRefusal, Model: "claude-test"},
		"fallback model": {Content: "x", StopReason: StopEndTurn, Model: "claude-test", Iterations: []UsageIteration{{Type: "fallback_message", Model: "other"}}},
		"no message id":  {ID: "-", Content: "x", StopReason: StopEndTurn, Model: "claude-test"},
		"errored turn":   {Content: "x", StopReason: StopError, Model: "claude-test"},
	}
	for name, reply := range cases {
		t.Run(name, func(t *testing.T) {
			srv := newThreadServer()
			srv.reply = func(n int) *Response {
				if n == 1 {
					r := *reply
					return &r
				}
				return &Response{Content: "ok", StopReason: StopEndTurn, Model: "claude-test"}
			}
			s := threadSession(srv)
			send(t, s, "one")
			send(t, s, "two")
			reqs := srv.calls()
			if len(reqs) != 2 {
				t.Fatalf("%d requests; a continuation from an unstorable turn would only earn a 404", len(reqs))
			}
			// Not a continuation that happens to have an empty ID: that would
			// send only the delta as a new thread.
			got := strings.Join(shapeOf(reqs[1].Messages), "|")
			if th := reqs[1].Thread; th == nil || th.PreviousMessageID != "" || !strings.Contains(got, "user:one") || !strings.Contains(got, "user:two") {
				t.Errorf("the next turn should create with the full conversation, got %+v with %q", th, got)
			}
		})
	}
}

func TestThreadsInertOffAnthropic(t *testing.T) {
	srv := newThreadServer()
	srv.provider = "openai"
	s := threadSession(srv)
	send(t, s, "one")
	send(t, s, "two")
	for i, r := range srv.calls() {
		if r.Thread != nil {
			t.Errorf("request %d to a non-Anthropic client carries a thread", i+1)
		}
	}
}

func TestThreadsSwitchProviderForgetsTheThread(t *testing.T) {
	a, b := newThreadServer(), newThreadServer()
	s := threadSession(a)
	send(t, s, "one")
	s.SwitchProvider(b)
	send(t, s, "two")
	r := b.calls()[0]
	if r.Thread == nil || r.Thread.PreviousMessageID != "" || len(r.Messages) != 3 {
		t.Errorf("the new client should get a full create, got %+v with %d messages", r.Thread, len(r.Messages))
	}
	s.SwitchProvider(a)
	send(t, s, "three")
	if r := a.calls()[1]; r.Thread == nil || r.Thread.PreviousMessageID != "" {
		t.Errorf("switching back must not resurrect the old thread: %+v", r.Thread)
	}
}

// A restored conversation has no thread: the first turn creates one, and the
// serialized form carries no thread state.
func TestThreadsResumedConversationCreates(t *testing.T) {
	srv := newThreadServer()
	conv := NewConversation("sys")
	conv.Append(NewTextMessage(RoleUser, "old"), NewTextMessage(RoleAssistant, "old reply"))
	s := NewSession(srv, WithMessageThreads(), WithConversation(conv))
	send(t, s, "new")
	r := srv.calls()[0]
	if r.Thread == nil || r.Thread.PreviousMessageID != "" || len(r.Messages) != 3 {
		t.Fatalf("got %+v with %d messages", r.Thread, len(r.Messages))
	}
	data, err := json.Marshal(s.Conversation())
	if err != nil {
		t.Fatal(err)
	}
	for _, leak := range []string{"thread", "previous_message_id", "msg_1"} {
		if strings.Contains(string(data), leak) {
			t.Errorf("the serialized conversation contains %q", leak)
		}
	}
	restored, err := LoadConversation(data)
	if err != nil {
		t.Fatal(err)
	}
	s2 := NewSession(srv, WithMessageThreads(), WithConversation(restored))
	send(t, s2, "again")
	if th := srv.calls()[1].Thread; th == nil || th.PreviousMessageID != "" {
		t.Errorf("a session over a restored conversation must create, got %+v", th)
	}
}

func TestThreadsNothingNewToSendIsAFullRequest(t *testing.T) {
	srv := newThreadServer()
	s := threadSession(srv)
	send(t, s, "one")
	if _, err := s.SendMessages(context.Background()); err != nil {
		t.Fatal(err)
	}
	r := srv.calls()[1]
	if r.Thread == nil || r.Thread.PreviousMessageID != "" {
		t.Errorf("an empty delta outside a paused turn cannot continue: %+v", r.Thread)
	}
}

func TestThreadsContinueAPausedTurn(t *testing.T) {
	srv := newThreadServer()
	srv.reply = func(n int) *Response {
		if n == 1 {
			return &Response{Content: "searching", StopReason: StopPauseTurn, Model: "claude-test"}
		}
		return &Response{Content: "done", StopReason: StopEndTurn, Model: "claude-test"}
	}
	s := threadSession(srv)
	send(t, s, "go")
	if _, err := s.Continue(context.Background()); err != nil {
		t.Fatal(err)
	}
	r := srv.calls()[1]
	if r.Thread == nil || r.Thread.PreviousMessageID != "msg_1" || len(r.Messages) != 0 {
		t.Fatalf("a paused turn resumes with an empty message list on the thread: %+v, %d messages", r.Thread, len(r.Messages))
	}
}

func TestThreadsStream(t *testing.T) {
	drain := func(t *testing.T, s *Session, q string) (string, error) {
		t.Helper()
		var text strings.Builder
		for ev, err := range s.Stream(context.Background(), q) {
			if err != nil {
				return text.String(), err
			}
			if ev.Type == EventContent {
				text.WriteString(ev.Text)
			}
		}
		return text.String(), nil
	}
	t.Run("continues and records the id", func(t *testing.T) {
		srv := newThreadServer()
		s := threadSession(srv)
		if _, err := drain(t, s, "one"); err != nil {
			t.Fatal(err)
		}
		if _, err := drain(t, s, "two"); err != nil {
			t.Fatal(err)
		}
		r := srv.calls()[1]
		if r.Thread == nil || r.Thread.PreviousMessageID != "msg_1" || len(r.Messages) != 1 {
			t.Fatalf("streamed turn 2: %+v with %d messages", r.Thread, len(r.Messages))
		}
		want := shapeOf(s.Conversation().Messages)
		if got := srv.holds("msg_2"); !equalShape(got, want) {
			t.Fatalf("server holds %q, transcript %q", got, want)
		}
	})
	t.Run("replays when the continuation is refused before any event", func(t *testing.T) {
		srv := newThreadServer()
		s := threadSession(srv)
		if _, err := drain(t, s, "one"); err != nil {
			t.Fatal(err)
		}
		srv.expireAll()
		text, err := drain(t, s, "two")
		if err != nil || text != "reply 3" {
			t.Fatalf("text %q err %v", text, err)
		}
		reqs := srv.calls()
		if len(reqs) != 3 || reqs[2].Thread == nil || reqs[2].Thread.PreviousMessageID != "" || len(reqs[2].Messages) != 3 {
			t.Fatalf("want continue then a full create, got %d requests", len(reqs))
		}
		if n := len(s.Conversation().Messages); n != 4 {
			t.Errorf("transcript has %d messages, want 4 (no duplicate from the replay)", n)
		}
	})
	t.Run("a failure after events is not replayed", func(t *testing.T) {
		srv := newThreadServer()
		s := threadSession(srv)
		if _, err := drain(t, s, "one"); err != nil {
			t.Fatal(err)
		}
		fc := &midStreamFail{threadServer: srv}
		s.SwitchProvider(fc) // same provider name; also resets the thread
		if _, err := drain(t, s, "two"); err != nil {
			t.Fatal(err)
		}
		fc.fail = true
		if _, err := drain(t, s, "three"); err == nil {
			t.Fatal("expected the mid-stream error")
		}
		if fc.streams != 2 {
			t.Fatalf("%d stream attempts after the switch; a broken stream must not be replayed", fc.streams)
		}
		if n := len(s.Conversation().Messages); n != 4 {
			t.Errorf("a broken turn left %d messages, want 4", n)
		}
	})
	t.Run("breaking on the done event keeps the thread", func(t *testing.T) {
		srv := newThreadServer()
		s := threadSession(srv)
		for ev, err := range s.Stream(context.Background(), "one") {
			if err != nil {
				t.Fatal(err)
			}
			if ev.Type == EventDone {
				break
			}
		}
		send(t, s, "two")
		if r := srv.calls()[1]; r.Thread == nil || r.Thread.PreviousMessageID != "msg_1" {
			t.Errorf("got %+v", r.Thread)
		}
	})
	t.Run("an abandoned stream forgets nothing it must not", func(t *testing.T) {
		srv := newThreadServer()
		s := threadSession(srv)
		send(t, s, "one")
		for ev, err := range s.Stream(context.Background(), "two") {
			_, _ = ev, err
			break // before EventDone: not recorded
		}
		// The server consumed msg_1 for the abandoned turn, so the thread is
		// stale: the next turn gets already_continued and replays.
		resp := send(t, s, "two")
		assertServerMirrors(t, srv, s, resp)
	})
}

// midStreamFail streams one event and then fails once armed.
type midStreamFail struct {
	*threadServer
	fail    bool
	streams int
}

func (m *midStreamFail) Stream(ctx context.Context, req Request) iter.Seq2[StreamEvent, error] {
	m.streams++
	if !m.fail {
		return m.threadServer.Stream(ctx, req)
	}
	return func(yield func(StreamEvent, error) bool) {
		if !yield(StreamEvent{Type: EventContent, Text: "part"}, nil) {
			return
		}
		yield(StreamEvent{}, threadErr(404, ThreadNotFound, "mid-stream"))
	}
}

// Concurrent callers serialize on the session; the thread must stay coherent
// and the race detector quiet.
func TestThreadsConcurrentTurns(t *testing.T) {
	srv := newThreadServer()
	s := threadSession(srv)
	var wg sync.WaitGroup
	errs := make(chan error, 64)
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 4; i++ {
				if _, err := s.Send(context.Background(), fmt.Sprintf("g%d-%d", g, i)); err != nil {
					errs <- err
				}
				_ = s.Conversation()
				_ = s.Usage()
			}
		}(g)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	if n := len(s.Conversation().Messages); n != 64 {
		t.Fatalf("transcript has %d messages, want 64", n)
	}
	for i, r := range srv.calls() {
		if i > 0 && (r.Thread == nil || r.Thread.PreviousMessageID == "") {
			t.Fatalf("request %d replayed; serialized turns should all continue", i+1)
		}
	}
}

func TestThreadErrorCode(t *testing.T) {
	wrapped := fmt.Errorf("attempt 2: %w", threadErr(404, ThreadNotFound, "x"))
	cases := []struct {
		name string
		err  error
		want string
	}{
		{"nil", nil, ""},
		{"plain error", errors.New("thread_not_found"), ""},
		{"not found", threadErr(404, ThreadNotFound, "x"), ThreadNotFound},
		{"fingerprint", threadErr(400, ThreadFingerprint, "x"), ThreadFingerprint},
		{"already continued", threadErr(400, ThreadAlreadyContinued, "x"), ThreadAlreadyContinued},
		{"in flight", threadErr(400, ThreadContinueInFlight, "x"), ThreadContinueInFlight},
		{"unsupported", threadErr(400, ThreadUnsupported, "x"), ThreadUnsupported},
		{"wrapped", wrapped, ThreadNotFound},
		{"no code", threadErr(400, "", "x"), ""},
		{"wrong status", threadErr(500, ThreadNotFound, "x"), ""},
		{"429 with the words in the message", &APIError{StatusCode: 429, Message: "thread_not_found"}, ""},
		{"code only in prose", &APIError{StatusCode: 400, Message: `the "error_code" thread_not_found`}, ""},
		{"truncated body", &APIError{StatusCode: 404, Message: `{"type":"error","error":{"details":{"error_co`}, ""},
		{"empty body", &APIError{StatusCode: 404}, ""},
		{"other error_code", &APIError{StatusCode: 400, Message: `{"error":{"details":{"error_code":"rate_limit"}}}`}, ""},
	}
	for _, tc := range cases {
		if got := ThreadErrorCode(tc.err); got != tc.want {
			t.Errorf("%s: got %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestUnsupportedServerToolsRefusesThreads(t *testing.T) {
	if err := UnsupportedServerTools("openai", Request{Thread: &Thread{}}); err == nil {
		t.Error("an adapter without threads must refuse a Thread, not silently send the whole conversation as a plain request")
	}
	if err := UnsupportedServerTools("openai", Request{}); err != nil {
		t.Errorf("plain request: %v", err)
	}
}

// A system message cannot ride a continuation (the API takes system on create
// only), so a turn that adds one sends the whole conversation.
func TestThreadsSystemMessageInTheDeltaCreates(t *testing.T) {
	srv := newThreadServer()
	s := threadSession(srv)
	send(t, s, "one")
	s.Append(NewSystemMessage("new rules"))
	send(t, s, "two")
	r := srv.calls()[1]
	if r.Thread == nil || r.Thread.PreviousMessageID != "" || len(r.Messages) < 3 {
		t.Fatalf("got %+v with %d messages", r.Thread, len(r.Messages))
	}
}

// A reply normalization drops (empty content) leaves no message to anchor the
// continuation on; skipping ahead would lose the user message after it.
func TestThreadsEmptyReplyDoesNotSkipTheNextMessage(t *testing.T) {
	srv := newThreadServer()
	srv.reply = func(n int) *Response {
		if n == 1 {
			return &Response{StopReason: StopEndTurn, Model: "claude-test"}
		}
		return &Response{Content: "ok", StopReason: StopEndTurn, Model: "claude-test"}
	}
	s := threadSession(srv)
	send(t, s, "one")
	s.Append(NewTextMessage(RoleUser, "note"), NewTextMessage(RoleAssistant, "ack"))
	send(t, s, "two")
	r := srv.calls()[1]
	got := strings.Join(shapeOf(r.Messages), "|")
	if !strings.Contains(got, "user:note") || !strings.Contains(got, "user:two") {
		t.Fatalf("the second request lost a message: %q", got)
	}
}

// What the Session believes the server holds is re-proved every turn: a state
// that no longer matches the transcript is never trusted.
func TestThreadsDoNotTrustAStaleState(t *testing.T) {
	tamper := map[string]func(st *threadState){
		"held hash":      func(st *threadState) { st.heldHash[0] ^= 1 },
		"raw hash":       func(st *threadState) { st.rawHash[0] ^= 1 },
		"raw length":     func(st *threadState) { st.rawLen += 5 },
		"held count":     func(st *threadState) { st.heldN++ },
		"held count low": func(st *threadState) { st.heldN-- },
		"fingerprint":    func(st *threadState) { st.fp[0] ^= 1 },
	}
	for name, f := range tamper {
		t.Run(name, func(t *testing.T) {
			srv := newThreadServer()
			s := threadSession(srv)
			send(t, s, "one")
			f(s.thread)
			send(t, s, "two")
			r := srv.calls()[1]
			if r.Thread == nil || r.Thread.PreviousMessageID != "" || len(r.Messages) != 3 {
				t.Fatalf("got %+v with %d messages", r.Thread, len(r.Messages))
			}
		})
	}
}
