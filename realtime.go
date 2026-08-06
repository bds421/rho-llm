package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
)

// Realtime is a distinct surface from Complete/Stream: a long-lived bidirectional
// session that exchanges framed events (audio/text/control). The reference driver
// is OpenAI Realtime over a dialer-injected connection so unit tests can drive the
// state machine without live network access.

// RealtimeEventType is a provider-neutral realtime event kind.
type RealtimeEventType string

const (
	RealtimeEventSessionCreated   RealtimeEventType = "session.created"
	RealtimeEventSessionUpdated   RealtimeEventType = "session.updated"
	RealtimeEventInputAudioAppend RealtimeEventType = "input_audio_buffer.append"
	RealtimeEventInputAudioCommit RealtimeEventType = "input_audio_buffer.commit"
	RealtimeEventResponseCreate   RealtimeEventType = "response.create"
	RealtimeEventResponseAudio    RealtimeEventType = "response.audio.delta"
	RealtimeEventResponseText     RealtimeEventType = "response.text.delta"
	RealtimeEventResponseDone     RealtimeEventType = "response.done"
	RealtimeEventError            RealtimeEventType = "error"
	RealtimeEventClosed           RealtimeEventType = "session.closed"
)

// RealtimeEvent is one framed realtime message.
type RealtimeEvent struct {
	Type       RealtimeEventType `json:"type"`
	EventID    string            `json:"event_id,omitempty"`
	Text       string            `json:"text,omitempty"`
	AudioB64   string            `json:"audio,omitempty"`
	Error      string            `json:"error,omitempty"`
	Raw        json.RawMessage   `json:"-"`
	ProviderID string            `json:"-"` // vendor event name when distinct
}

// RealtimeDialer opens a bidirectional byte stream for a realtime session.
// Production code may wrap a WebSocket; tests inject a pipe or fake conn.
type RealtimeDialer func(ctx context.Context, cfg Config) (RealtimeConn, error)

// RealtimeConn is the minimal duplex transport realtime needs.
type RealtimeConn interface {
	Read(ctx context.Context) ([]byte, error)
	Write(ctx context.Context, data []byte) error
	Close() error
}

// RealtimeSession is an open realtime conversation.
type RealtimeSession struct {
	mu     sync.Mutex
	cfg    Config
	conn   RealtimeConn
	closed bool
	state  realtimeState
}

type realtimeState int

const (
	realtimeStateNew realtimeState = iota
	realtimeStateOpen
	realtimeStateClosed
)

// OpenRealtimeSession dials a realtime connection using the registered dialer
// for the resolved protocol (OpenAI Realtime is the reference implementation).
func OpenRealtimeSession(ctx context.Context, cfg Config, dialer RealtimeDialer) (*RealtimeSession, error) {
	if dialer == nil {
		return nil, fmt.Errorf("llm: realtime dialer is required")
	}
	cfg.Model = configuredModel(cfg)
	if cfg.Model == "" {
		return nil, fmt.Errorf("llm: model is required for realtime")
	}
	if cfg.APIKey == "" && !IsNoAuthProvider(cfg.Provider) {
		return nil, fmt.Errorf("llm: API key is required for realtime")
	}
	conn, err := dialer(ctx, cfg)
	if err != nil {
		return nil, err
	}
	return &RealtimeSession{cfg: cfg, conn: conn, state: realtimeStateOpen}, nil
}

// SendAudio appends base64-encoded PCM audio to the input buffer (OpenAI shape).
func (s *RealtimeSession) SendAudio(ctx context.Context, audioB64 string) error {
	if strings.TrimSpace(audioB64) == "" {
		return fmt.Errorf("llm: realtime audio payload is empty")
	}
	return s.send(ctx, RealtimeEvent{
		Type:     RealtimeEventInputAudioAppend,
		AudioB64: audioB64,
	})
}

// CommitAudio commits the input audio buffer for model processing.
func (s *RealtimeSession) CommitAudio(ctx context.Context) error {
	return s.send(ctx, RealtimeEvent{Type: RealtimeEventInputAudioCommit})
}

// RequestResponse asks the model to generate a response.
func (s *RealtimeSession) RequestResponse(ctx context.Context) error {
	return s.send(ctx, RealtimeEvent{Type: RealtimeEventResponseCreate})
}

// SendEvent writes an arbitrary framed event (advanced use / tests).
func (s *RealtimeSession) SendEvent(ctx context.Context, ev RealtimeEvent) error {
	return s.send(ctx, ev)
}

// Recv reads and decodes the next server event.
// The session mutex is not held across the blocking transport Read so concurrent
// Send*/Close can proceed (bidirectional realtime).
func (s *RealtimeSession) Recv(ctx context.Context) (RealtimeEvent, error) {
	s.mu.Lock()
	if s.closed || s.state != realtimeStateOpen {
		s.mu.Unlock()
		return RealtimeEvent{}, fmt.Errorf("llm: realtime session is closed")
	}
	conn := s.conn
	s.mu.Unlock()

	raw, err := conn.Read(ctx)
	if err != nil {
		if err == io.EOF {
			s.mu.Lock()
			s.closed = true
			s.state = realtimeStateClosed
			s.mu.Unlock()
			return RealtimeEvent{Type: RealtimeEventClosed}, nil
		}
		return RealtimeEvent{}, err
	}
	return DecodeRealtimeEvent(raw)
}

// Close ends the session.
func (s *RealtimeSession) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	s.state = realtimeStateClosed
	conn := s.conn
	s.mu.Unlock()
	if conn != nil {
		return conn.Close()
	}
	return nil
}

// Open reports whether the session is open.
func (s *RealtimeSession) Open() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state == realtimeStateOpen && !s.closed
}

func (s *RealtimeSession) send(ctx context.Context, ev RealtimeEvent) error {
	payload, err := EncodeRealtimeEvent(ev)
	if err != nil {
		return err
	}
	s.mu.Lock()
	if s.closed || s.state != realtimeStateOpen {
		s.mu.Unlock()
		return fmt.Errorf("llm: realtime session is closed")
	}
	conn := s.conn
	s.mu.Unlock()
	return conn.Write(ctx, payload)
}

// EncodeRealtimeEvent maps a neutral event to the OpenAI Realtime JSON wire shape.
func EncodeRealtimeEvent(ev RealtimeEvent) ([]byte, error) {
	wireType := string(ev.Type)
	switch ev.Type {
	case RealtimeEventInputAudioAppend:
		wireType = "input_audio_buffer.append"
	case RealtimeEventInputAudioCommit:
		wireType = "input_audio_buffer.commit"
	case RealtimeEventResponseCreate:
		wireType = "response.create"
	case RealtimeEventSessionUpdated:
		wireType = "session.update"
	}
	obj := map[string]any{"type": wireType}
	if ev.EventID != "" {
		obj["event_id"] = ev.EventID
	}
	if ev.AudioB64 != "" {
		obj["audio"] = ev.AudioB64
	}
	if ev.Text != "" {
		obj["text"] = ev.Text
	}
	return json.Marshal(obj)
}

// DecodeRealtimeEvent parses an OpenAI Realtime server event into a neutral event.
func DecodeRealtimeEvent(raw []byte) (RealtimeEvent, error) {
	var base struct {
		Type    string `json:"type"`
		EventID string `json:"event_id"`
		Delta   string `json:"delta"`
		Audio   string `json:"audio"`
		Error   *struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &base); err != nil {
		return RealtimeEvent{}, fmt.Errorf("llm: decode realtime event: %w", err)
	}
	if base.Type == "" {
		return RealtimeEvent{}, fmt.Errorf("llm: realtime event missing type")
	}
	ev := RealtimeEvent{
		EventID:    base.EventID,
		Raw:        append(json.RawMessage(nil), raw...),
		ProviderID: base.Type,
	}
	switch base.Type {
	case "session.created":
		ev.Type = RealtimeEventSessionCreated
	case "session.updated":
		ev.Type = RealtimeEventSessionUpdated
	case "response.audio.delta", "response.output_audio.delta":
		// Live OpenAI Realtime uses response.output_audio.delta; older docs use response.audio.delta.
		ev.Type = RealtimeEventResponseAudio
		ev.AudioB64 = base.Delta
		if ev.AudioB64 == "" {
			ev.AudioB64 = base.Audio
		}
	case "response.text.delta", "response.output_text.delta",
		"response.output_audio_transcript.delta":
		ev.Type = RealtimeEventResponseText
		ev.Text = base.Delta
	case "response.done":
		ev.Type = RealtimeEventResponseDone
	case "error":
		ev.Type = RealtimeEventError
		if base.Error != nil {
			ev.Error = base.Error.Message
			if ev.Error == "" {
				ev.Error = base.Error.Type
			}
		}
		if ev.Error == "" {
			ev.Error = "realtime error"
		}
	default:
		// Preserve unknown server events as error-typed text carriers for callers.
		ev.Type = RealtimeEventType(base.Type)
		ev.Text = base.Delta
	}
	return ev, nil
}
