package llm

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"time"
)

// threadReuseWindow is how long after a turn a Session still continues its
// thread: Anthropic expires one five minutes after its last request, and a
// continuation sent later would only earn a 404 and a replay.
const threadReuseWindow = 4*time.Minute + 30*time.Second

// WithMessageThreads makes the Session use Anthropic message threads (beta
// anthropic.MessageThreadsBeta): the API keeps the conversation for five
// minutes, so a turn sends only the messages added since the last one. Off by
// default; it never changes what a request means, only how much of it travels.
//
// The Session still keeps the whole transcript and stays the source of truth.
// A turn continues the thread only when everything the server holds is
// provably what the transcript says: same provider, model, system prompt,
// tools and generation settings, an unchanged transcript prefix, a reply that
// came from the thread, and less than five minutes since the last turn.
// Otherwise it sends the full conversation as a new thread, which is also what
// it does when the server answers that the thread cannot be continued
// (expired, already continued, evicted). If the server refuses threads for
// this request or organization, the Session turns them off and sends
// ordinary requests. Thread state lives in the Session and is never
// serialized: a Conversation restored from disk starts a new thread.
//
// Threads apply to the native "anthropic" provider only; with any other
// client the option is inert. A threaded transcript is held by the API under
// the key that created it: with a multi-key pool a rotation to another key
// costs one replay. Message threads are not eligible for zero data retention.
func WithMessageThreads() SessionOption {
	return func(s *Session) { s.threads = true }
}

// threadState is what the Session knows the server holds after the last
// turn: the tip's message ID, a fingerprint of the fixed request fields, and
// hashes of the normalized messages sent plus of the raw transcript through
// the reply, to prove the next turn extends exactly that.
type threadState struct {
	prevID   string
	fp       [32]byte
	heldHash [32]byte // the normalized messages the server received
	heldN    int
	rawLen   int      // transcript length through the reply
	rawHash  [32]byte // hash of those rawLen messages
	at       time.Time
}

// threadStage is how far a turn has got through its fallbacks.
type threadStage int

const (
	stageContinue threadStage = iota // delta on the existing thread
	stageCreate                      // full conversation, new thread
	stagePlain                       // full conversation, no thread
)

// turnPlan carries one turn's threaded attempts: the continuation (when the
// held prefix is proven) and the full request to fall back on.
type turnPlan struct {
	s     *Session
	full  Request // the whole conversation, Thread unset
	cont  *Request
	norm  []Message // the full normalized conversation of this turn
	fp    [32]byte
	stage threadStage
}

func (p *turnPlan) threaded() bool { return p != nil && p.stage != stagePlain }

// first returns the request to send first and positions the plan on it.
func (p *turnPlan) first() Request {
	if p.cont != nil {
		p.stage = stageContinue
		return *p.cont
	}
	p.stage = stageCreate
	return p.create()
}

func (p *turnPlan) create() Request {
	r := p.full
	r.Thread = &Thread{}
	return r
}

// fallback returns the next request to try after err on the current one, and
// false when err is final. A continuation that fails with a thread error, or
// any 400/404 (the server's answer to a thread it cannot continue carries a
// code, but a continuation it cannot parse does not), is replayed once as a
// full create; a create that fails with a thread error means the server will
// not thread this request, so threads are turned off and the request goes out
// plain. Any other error (auth, rate limit, a real 400) is not the thread's.
func (p *turnPlan) fallback(err error) (Request, bool) {
	if p == nil {
		return Request{}, false
	}
	switch p.stage {
	case stageContinue:
		if !isThreadRejection(err) {
			return Request{}, false
		}
		p.s.dropThread()
		p.stage = stageCreate
		return p.create(), true
	case stageCreate:
		if ThreadErrorCode(err) == "" {
			return Request{}, false
		}
		p.s.disableThreads()
		p.stage = stagePlain
		return p.full, true
	}
	return Request{}, false
}

// isThreadRejection reports whether err is a 400 or 404 from the API, which
// is how a thread that cannot be continued is refused.
func isThreadRejection(err error) bool {
	var apiErr *APIError
	return errors.As(err, &apiErr) && (apiErr.StatusCode == 400 || apiErr.StatusCode == 404)
}

// planThread decides how this turn travels. norm is the normalized full
// conversation; the Session keeps req (full) unchanged and, when the thread
// provably holds a prefix of it, prepares the delta. The caller holds stateMu.
func (s *Session) planThread(req Request, norm []Message, provider string, continuation bool) *turnPlan {
	if !s.threads || s.threadOff || provider != "anthropic" {
		return nil
	}
	fp, ok := requestFingerprint(req, provider)
	if !ok {
		return nil
	}
	p := &turnPlan{s: s, full: req, norm: norm, fp: fp}
	st := s.thread
	if st == nil || st.fp != fp || s.clock().Sub(st.at) >= threadReuseWindow {
		return p
	}
	if st.rawLen > len(s.conv.Messages) || hashMessages(s.conv.Messages[:st.rawLen]) != st.rawHash {
		return p // the transcript changed under the thread
	}
	// The server holds the messages of the previous request plus its reply.
	// norm must start with exactly those, the reply being the next message.
	n := st.heldN
	if len(norm) <= n || hashMessages(norm[:n]) != st.heldHash || norm[n].Role != RoleAssistant {
		return p
	}
	delta := norm[n+1:]
	if len(delta) == 0 && !continuation {
		return p // nothing new to send
	}
	for _, m := range delta {
		if m.Role == RoleSystem {
			return p // a system prompt cannot ride a continuation
		}
	}
	c := req
	c.Messages = delta
	c.Thread = &Thread{PreviousMessageID: st.prevID}
	p.cont = &c
	return p
}

// requestFingerprint hashes everything about a request but its messages that a
// thread fixes at creation (system, tools, model, thinking, output format,
// tool choice, …) — over-approximating the API's set, so a changed setting
// starts a new thread rather than risking thread_fingerprint_mismatch. The
// container ID is excluded: it is sent on every request anyway.
func requestFingerprint(req Request, provider string) ([32]byte, bool) {
	req.Messages = nil
	req.ContainerID = ""
	req.Thread = nil
	b, err := json.Marshal(struct {
		Provider string
		Req      Request
	}{provider, req})
	if err != nil {
		return [32]byte{}, false
	}
	return sha256.Sum256(b), true
}

// hashMessages hashes messages by their JSON form, which is the language-
// agnostic identity of a transcript; unmarshalable input hashes distinctly
// from everything else so it never matches.
func hashMessages(msgs []Message) [32]byte {
	b, err := json.Marshal(msgs)
	if err != nil {
		return sha256.Sum256([]byte("unmarshalable: " + err.Error()))
	}
	return sha256.Sum256(b)
}

// recordThread stores what the server now holds, or forgets the thread when
// the turn cannot be continued from: it ran without a thread, the response
// carried no ID, it was refused (the API may hold the refused turn), or a
// fallback model wrote it (the API stores no such turn). The caller holds
// stateMu and has appended the turn to the transcript.
func (s *Session) recordThread(p *turnPlan, resp *Response) {
	if p == nil {
		return
	}
	if !p.threaded() || resp.ID == "" || resp.StopReason == StopRefusal || fellBack(resp) {
		s.thread = nil
		return
	}
	s.thread = &threadState{
		prevID:   resp.ID,
		fp:       p.fp,
		heldHash: hashMessages(p.norm),
		heldN:    len(p.norm),
		rawLen:   len(s.conv.Messages),
		rawHash:  hashMessages(s.conv.Messages),
		at:       s.clock(),
	}
}

func fellBack(resp *Response) bool {
	for _, it := range resp.Iterations {
		if it.Type == "fallback_message" {
			return true
		}
	}
	return false
}

func (s *Session) clock() time.Time {
	if s.now != nil {
		return s.now()
	}
	return time.Now()
}

func (s *Session) dropThread() {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	s.thread = nil
}

func (s *Session) disableThreads() {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	s.thread = nil
	s.threadOff = true
}
