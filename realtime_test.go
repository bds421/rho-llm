package llm_test

import (
	"context"
	"encoding/json"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
)

// pipeConn is a duplex in-memory RealtimeConn for unit tests.
type pipeConn struct {
	mu        sync.Mutex
	inbox     [][]byte
	lastWrite []byte
	wait      chan struct{}
	closed    bool
}

func newPipeConn() *pipeConn {
	return &pipeConn{wait: make(chan struct{}, 16)}
}

func (p *pipeConn) Read(ctx context.Context) ([]byte, error) {
	for {
		p.mu.Lock()
		if p.closed && len(p.inbox) == 0 {
			p.mu.Unlock()
			return nil, io.EOF
		}
		if len(p.inbox) > 0 {
			b := p.inbox[0]
			p.inbox = p.inbox[1:]
			p.mu.Unlock()
			return b, nil
		}
		ch := p.wait
		p.mu.Unlock()
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-ch:
		case <-time.After(20 * time.Millisecond):
		}
	}
}

func (p *pipeConn) Write(ctx context.Context, data []byte) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return io.ErrClosedPipe
	}
	p.lastWrite = append([]byte(nil), data...)
	return nil
}

func (p *pipeConn) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.closed = true
	select {
	case p.wait <- struct{}{}:
	default:
	}
	return nil
}

func (p *pipeConn) Push(raw []byte) {
	p.mu.Lock()
	p.inbox = append(p.inbox, append([]byte(nil), raw...))
	p.mu.Unlock()
	select {
	case p.wait <- struct{}{}:
	default:
	}
}

func (p *pipeConn) LastWrite() []byte {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]byte(nil), p.lastWrite...)
}

func TestRealtimeEncodeDecodeRoundTrip(t *testing.T) {
	raw, err := llm.EncodeRealtimeEvent(llm.RealtimeEvent{
		Type: llm.RealtimeEventInputAudioAppend, AudioB64: "AAAA", EventID: "e1",
	})
	if err != nil {
		t.Fatal(err)
	}
	var obj map[string]any
	if err := json.Unmarshal(raw, &obj); err != nil {
		t.Fatal(err)
	}
	if obj["type"] != "input_audio_buffer.append" || obj["audio"] != "AAAA" {
		t.Fatalf("wire=%v", obj)
	}

	ev, err := llm.DecodeRealtimeEvent([]byte(`{"type":"response.audio.delta","delta":"QQ==","event_id":"s1"}`))
	if err != nil {
		t.Fatal(err)
	}
	if ev.Type != llm.RealtimeEventResponseAudio || ev.AudioB64 != "QQ==" || ev.EventID != "s1" {
		t.Fatalf("ev=%+v", ev)
	}
	ev, err = llm.DecodeRealtimeEvent([]byte(`{"type":"error","error":{"message":"boom"}}`))
	if err != nil {
		t.Fatal(err)
	}
	if ev.Type != llm.RealtimeEventError || ev.Error != "boom" {
		t.Fatalf("error ev=%+v", ev)
	}
	if _, err := llm.DecodeRealtimeEvent([]byte(`{}`)); err == nil {
		t.Fatal("missing type must error")
	}
}

func TestRealtimeSessionStateMachine(t *testing.T) {
	conn := newPipeConn()
	session, err := llm.OpenRealtimeSession(context.Background(), llm.Config{
		Provider: "openai", Model: "gpt-realtime", APIKey: "k",
	}, func(ctx context.Context, cfg llm.Config) (llm.RealtimeConn, error) {
		if cfg.Model != "gpt-realtime" {
			t.Fatalf("model=%q", cfg.Model)
		}
		return conn, nil
	})
	if err != nil {
		t.Fatalf("OpenRealtimeSession: %v", err)
	}
	if !session.Open() {
		t.Fatal("session should be open")
	}

	if err := session.SendAudio(context.Background(), "AAAA"); err != nil {
		t.Fatal(err)
	}
	var sent map[string]any
	if err := json.Unmarshal(conn.LastWrite(), &sent); err != nil {
		t.Fatal(err)
	}
	if sent["type"] != "input_audio_buffer.append" {
		t.Fatalf("send audio wire=%v", sent)
	}
	if err := session.CommitAudio(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := session.RequestResponse(context.Background()); err != nil {
		t.Fatal(err)
	}

	conn.Push([]byte(`{"type":"session.created","event_id":"1"}`))
	conn.Push([]byte(`{"type":"response.text.delta","delta":"hi"}`))
	conn.Push([]byte(`{"type":"response.done"}`))

	ev, err := session.Recv(context.Background())
	if err != nil || ev.Type != llm.RealtimeEventSessionCreated {
		t.Fatalf("recv1=%+v err=%v", ev, err)
	}
	ev, err = session.Recv(context.Background())
	if err != nil || ev.Type != llm.RealtimeEventResponseText || ev.Text != "hi" {
		t.Fatalf("recv2=%+v err=%v", ev, err)
	}
	ev, err = session.Recv(context.Background())
	if err != nil || ev.Type != llm.RealtimeEventResponseDone {
		t.Fatalf("recv3=%+v err=%v", ev, err)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	if session.Open() {
		t.Fatal("session should be closed")
	}
	if err := session.SendAudio(context.Background(), "AA"); err == nil {
		t.Fatal("send on closed session must fail")
	}
}

func TestRealtimeOpenRequiresDialerAndModel(t *testing.T) {
	if _, err := llm.OpenRealtimeSession(context.Background(), llm.Config{Provider: "openai", APIKey: "k"}, nil); err == nil {
		t.Fatal("nil dialer")
	}
	if _, err := llm.OpenRealtimeSession(context.Background(), llm.Config{Provider: "openai", APIKey: "k"}, func(context.Context, llm.Config) (llm.RealtimeConn, error) {
		return newPipeConn(), nil
	}); err == nil {
		t.Fatal("missing model")
	}
}

// TestRealtimeConcurrentRecvSendClose proves Recv does not hold the session
// mutex across blocking Read: a blocked Recv must not prevent SendAudio or Close.
func TestRealtimeConcurrentRecvSendClose(t *testing.T) {
	conn := newPipeConn()
	session, err := llm.OpenRealtimeSession(context.Background(), llm.Config{
		Provider: "openai", Model: "gpt-realtime", APIKey: "k",
	}, func(context.Context, llm.Config) (llm.RealtimeConn, error) {
		return conn, nil
	})
	if err != nil {
		t.Fatal(err)
	}

	recvStarted := make(chan struct{})
	recvDone := make(chan error, 1)
	go func() {
		close(recvStarted)
		// Blocks until Push or Close (EOF).
		_, err := session.Recv(context.Background())
		recvDone <- err
	}()
	<-recvStarted
	// Give Recv time to enter conn.Read.
	time.Sleep(30 * time.Millisecond)

	sendDone := make(chan error, 1)
	go func() {
		sendDone <- session.SendAudio(context.Background(), "AAAA")
	}()
	select {
	case err := <-sendDone:
		if err != nil {
			t.Fatalf("SendAudio while Recv blocked: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("SendAudio deadlocked behind blocked Recv (mutex held across Read)")
	}

	if got := string(conn.LastWrite()); !strings.Contains(got, "input_audio_buffer.append") {
		t.Fatalf("expected append write, got %s", got)
	}

	// Unblock Recv with a server event, then Close from another goroutine while
	// a second Recv could block — Close must not hang.
	conn.Push([]byte(`{"type":"session.created","event_id":"c1"}`))
	select {
	case err := <-recvDone:
		if err != nil {
			t.Fatalf("Recv after push: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Recv did not complete after Push")
	}

	// Start another blocking Recv, then Close must succeed promptly.
	recv2Done := make(chan error, 1)
	go func() {
		_, err := session.Recv(context.Background())
		recv2Done <- err
	}()
	time.Sleep(30 * time.Millisecond)

	closeDone := make(chan error, 1)
	go func() {
		closeDone <- session.Close()
	}()
	select {
	case err := <-closeDone:
		if err != nil {
			t.Fatalf("Close while Recv blocked: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Close deadlocked behind blocked Recv")
	}

	select {
	case <-recv2Done:
		// EOF / closed is fine
	case <-time.After(2 * time.Second):
		t.Fatal("blocked Recv did not unblock after Close")
	}

	if session.Open() {
		t.Fatal("session still open after Close")
	}
}
