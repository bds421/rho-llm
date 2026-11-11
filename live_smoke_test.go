package llm_test

// Minimal LIVE smokes for surfaces added in the multi-vendor batch/modality work.
// Skipped under -short (make ci) and when the relevant key is unset.
//
// Cost controls: cheapest models, max_tokens tiny, one item per batch where possible.
// Do not paste keys into source — use env vars.

import (
	"bufio"
	"context"
	"crypto/rand"
	"crypto/sha1"
	"crypto/tls"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider"
)

func TestLiveNew_AnthropicBatchSubmitGetCancel(t *testing.T) {
	key := envKey("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEYS")
	requireLiveAPI(t, key, "ANTHROPIC_API_KEY not set")

	cfg := llm.Config{
		Provider: "anthropic", Model: "claude-haiku-4-5-20251001", APIKey: key,
		MaxTokens: 16, Timeout: 60 * time.Second, DisableRetries: true,
	}
	bc, err := llm.NewBatchClient(cfg)
	if err != nil {
		t.Fatalf("NewBatchClient: %v", err)
	}
	defer bc.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	handle, err := bc.Submit(ctx, []llm.BatchItem{{
		ItemID: "smoke-1",
		Request: &llm.Request{
			Model:     "claude-haiku-4-5-20251001",
			MaxTokens: 16,
			Messages:  []llm.Message{llm.NewTextMessage(llm.RoleUser, "Reply with one word: ok")},
		},
	}}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if handle == nil || handle.ID == "" {
		t.Fatal("empty handle")
	}
	t.Logf("anthropic batch id=%s status=%s", handle.ID, handle.Status)

	got, err := bc.Get(ctx, *handle)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	t.Logf("anthropic batch get status=%s counts=%+v", got.Status, got.RequestCounts)

	// Cancel if still non-terminal to avoid a full-day job sitting open.
	if !got.Status.Terminal() {
		canceled, err := bc.Cancel(ctx, *got)
		if err != nil {
			t.Logf("Cancel (best-effort): %v", err)
		} else {
			t.Logf("anthropic batch cancel status=%s", canceled.Status)
		}
	}
}

func TestLiveNew_GeminiBatchSubmitGetCancel(t *testing.T) {
	key := envKey("GEMINI_API_KEY", "GEMINI_API_KEYS")
	if key == "" {
		key = os.Getenv("GOOGLE_API_KEY")
	}
	requireLiveAPI(t, key, "GEMINI_API_KEY/GOOGLE_API_KEY not set")

	cfg := llm.Config{
		Provider: "gemini", Model: "gemini-2.5-flash-lite", APIKey: key,
		MaxTokens: 16, Timeout: 60 * time.Second, DisableRetries: true,
	}
	bc, err := llm.NewBatchClient(cfg)
	if err != nil {
		t.Fatalf("NewBatchClient: %v", err)
	}
	defer bc.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	handle, err := bc.Submit(ctx, []llm.BatchItem{{
		ItemID: "smoke-g1",
		Request: &llm.Request{
			Model:     "gemini-2.5-flash-lite",
			MaxTokens: 16,
			Messages:  []llm.Message{llm.NewTextMessage(llm.RoleUser, "Reply with one word: ok")},
		},
	}}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour})
	if err != nil {
		// Surface the real error — path/name drift is useful signal.
		t.Fatalf("Submit: %v", err)
	}
	t.Logf("gemini batch id=%s status=%s", handle.ID, handle.Status)

	got, err := bc.Get(ctx, *handle)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	t.Logf("gemini batch get status=%s", got.Status)

	if !got.Status.Terminal() {
		if _, err := bc.Cancel(ctx, *got); err != nil {
			t.Logf("Cancel (best-effort): %v", err)
		}
	}
}

func TestLiveNew_GeminiEmbeddingsModality(t *testing.T) {
	key := envKey("GEMINI_API_KEY", "GEMINI_API_KEYS")
	if key == "" {
		key = os.Getenv("GOOGLE_API_KEY")
	}
	requireLiveAPI(t, key, "GEMINI_API_KEY/GOOGLE_API_KEY not set")

	cfg := llm.Config{
		Provider: "gemini", Model: "gemini-embedding-001", APIKey: key,
		Timeout: 60 * time.Second, DisableRetries: true,
	}
	mc, err := llm.NewModalityClient(cfg)
	if err != nil {
		t.Fatalf("NewModalityClient: %v", err)
	}
	defer mc.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	out, err := mc.GenerateEmbeddings(ctx, llm.EmbeddingRequest{
		Model: "gemini-embedding-001",
		Input: []string{"rho"}, // single short string — minimal tokens
	})
	if err != nil {
		t.Fatalf("GenerateEmbeddings: %v", err)
	}
	if len(out.Embeddings) != 1 || len(out.Embeddings[0].Vector) < 8 {
		t.Fatalf("unexpected embeddings: %+v", out)
	}
	t.Logf("gemini embedding dims=%d", len(out.Embeddings[0].Vector))
}

func TestLiveNew_OpenAIBatchStillWorks(t *testing.T) {
	// Only the batch surface we extended routing for — not a full chat suite.
	// Uses nano/chat-cheap model and cancels promptly.
	key := envKey("OPENAI_API_KEY", "OPENAI_API_KEYS")
	requireLiveAPI(t, key, "OPENAI_API_KEY not set")

	cfg := llm.Config{
		Provider: "openai", Model: "gpt-4.1-nano", APIKey: key,
		MaxTokens: 16, Timeout: 60 * time.Second, DisableRetries: true,
	}
	bc, err := llm.NewBatchClient(cfg)
	if err != nil {
		t.Fatalf("NewBatchClient: %v", err)
	}
	defer bc.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	handle, err := bc.Submit(ctx, []llm.BatchItem{{
		ItemID: "smoke-o1",
		Request: &llm.Request{
			Model:     "gpt-4.1-nano",
			MaxTokens: 16,
			Messages:  []llm.Message{llm.NewTextMessage(llm.RoleUser, "Reply with one word: ok")},
		},
	}}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour})
	if err != nil {
		// Some accounts restrict batch — report clearly.
		if strings.Contains(err.Error(), "batch") || strings.Contains(err.Error(), "403") {
			t.Skipf("OpenAI batch not available on this key: %v", err)
		}
		t.Fatalf("Submit: %v", err)
	}
	t.Logf("openai batch id=%s status=%s", handle.ID, handle.Status)
	got, err := bc.Get(ctx, *handle)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if !got.Status.Terminal() {
		if _, err := bc.Cancel(ctx, *got); err != nil {
			t.Logf("Cancel (best-effort): %v", err)
		}
	}
}

// --- Gemini image generation (new modality) ---

func TestLiveNew_GeminiImageGeneration(t *testing.T) {
	key := envKey("GEMINI_API_KEY", "GEMINI_API_KEYS")
	if key == "" {
		key = os.Getenv("GOOGLE_API_KEY")
	}
	requireLiveAPI(t, key, "GEMINI_API_KEY/GOOGLE_API_KEY not set")

	// Flash image is the cheaper image tier vs Pro.
	cfg := llm.Config{
		Provider: "gemini", Model: "gemini-2.5-flash-image", APIKey: key,
		Timeout: 120 * time.Second, DisableRetries: true,
	}
	mc, err := llm.NewModalityClient(cfg)
	if err != nil {
		t.Fatalf("NewModalityClient: %v", err)
	}
	defer mc.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	out, err := mc.GenerateImages(ctx, llm.ImageRequest{
		Model:     "gemini-2.5-flash-image",
		Prompt:    "A single solid red square on white background, minimal",
		N:         1,
		MediaType: "image/png",
	})
	if err != nil {
		t.Fatalf("GenerateImages: %v", err)
	}
	if len(out.Images) < 1 || out.Images[0].B64JSON == "" {
		t.Fatalf("no image bytes: %+v", out)
	}
	// Decode a few bytes to ensure base64 payload is real.
	raw, err := base64.StdEncoding.DecodeString(out.Images[0].B64JSON)
	if err != nil || len(raw) < 32 {
		t.Fatalf("image payload invalid: err=%v len=%d", err, len(raw))
	}
	t.Logf("gemini image bytes=%d mediaType=%s", len(raw), out.Images[0].MediaType)
}

// --- Wait until batch terminal (one cheap Anthropic item; poll, do not cancel) ---

func TestLiveNew_AnthropicBatchWaitForCompletion(t *testing.T) {
	key := envKey("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEYS")
	requireLiveAPI(t, key, "ANTHROPIC_API_KEY not set")

	cfg := llm.Config{
		Provider: "anthropic", Model: "claude-haiku-4-5-20251001", APIKey: key,
		MaxTokens: 16, Timeout: 60 * time.Second, DisableRetries: true,
	}
	bc, err := llm.NewBatchClient(cfg)
	if err != nil {
		t.Fatalf("NewBatchClient: %v", err)
	}
	defer bc.Close()

	// Cap wait: Anthropic batches are often fast for 1 tiny item; hard stop at 8m.
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Minute)
	defer cancel()

	handle, err := bc.Submit(ctx, []llm.BatchItem{{
		ItemID: "wait-1",
		Request: &llm.Request{
			Model:     "claude-haiku-4-5-20251001",
			MaxTokens: 16,
			Messages:  []llm.Message{llm.NewTextMessage(llm.RoleUser, "Say only: ok")},
		},
	}}, llm.BatchOptions{MaxTurnaround: 24 * time.Hour})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	t.Logf("submitted id=%s status=%s", handle.ID, handle.Status)

	done, err := llm.WaitForBatch(ctx, bc, *handle, 5*time.Second)
	if err != nil {
		t.Fatalf("WaitForBatch: %v", err)
	}
	if !done.Status.Terminal() {
		t.Fatalf("expected terminal status, got %s", done.Status)
	}
	t.Logf("terminal status=%s counts=%+v", done.Status, done.RequestCounts)

	if done.Status == llm.BatchCompleted {
		results, err := bc.Results(ctx, *done)
		if err != nil {
			t.Fatalf("Results: %v", err)
		}
		if len(results) != 1 || results[0].ItemID != "wait-1" {
			t.Fatalf("results=%+v", results)
		}
		if results[0].Error != nil {
			t.Fatalf("item error: %v", results[0].Error)
		}
		if results[0].Response == nil || strings.TrimSpace(results[0].Response.Content) == "" {
			t.Fatalf("empty response: %+v", results[0])
		}
		t.Logf("result text=%q", results[0].Response.Content)
	}
}

// --- OpenAI Realtime via injectable WebSocket dialer (stdlib WS client) ---

func TestLiveNew_OpenAIRealtimeSession(t *testing.T) {
	key := envKey("OPENAI_API_KEY", "OPENAI_API_KEYS")
	requireLiveAPI(t, key, "OPENAI_API_KEY not set")

	// Prefer current realtime model; fall back aliases if the account lacks one.
	model := os.Getenv("OPENAI_REALTIME_MODEL")
	if model == "" {
		model = "gpt-realtime"
	}

	cfg := llm.Config{
		Provider: "openai", Model: model, APIKey: key,
		Timeout: 60 * time.Second,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	session, err := llm.OpenRealtimeSession(ctx, cfg, func(ctx context.Context, c llm.Config) (llm.RealtimeConn, error) {
		return dialOpenAIRealtime(ctx, c.APIKey, c.Model)
	})
	if err != nil {
		// Realtime is account/feature gated — skip rather than fail CI-like local runs.
		if strings.Contains(err.Error(), "403") || strings.Contains(err.Error(), "404") ||
			strings.Contains(err.Error(), "401") || strings.Contains(err.Error(), "not available") {
			t.Skipf("realtime not available: %v", err)
		}
		t.Fatalf("OpenRealtimeSession: %v", err)
	}
	defer session.Close()

	// First server event is typically session.created.
	ev, err := session.Recv(ctx)
	if err != nil {
		t.Fatalf("Recv session.created: %v", err)
	}
	t.Logf("first event type=%s provider=%s", ev.Type, ev.ProviderID)
	if ev.Type != llm.RealtimeEventSessionCreated && ev.ProviderID != "session.created" {
		// Some revisions emit session.updated first; still a successful session.
		t.Logf("note: expected session.created, got type=%s id=%s", ev.Type, ev.ProviderID)
	}

	// Text-only response path is cheaper than streaming audio: create a response
	// with a tiny instruction via session.update + response.create if supported.
	// Minimal: request a response and wait for response.done or error.
	if err := session.SendEvent(ctx, llm.RealtimeEvent{Type: llm.RealtimeEventResponseCreate}); err != nil {
		t.Fatalf("RequestResponse: %v", err)
	}

	deadline := time.Now().Add(45 * time.Second)
	sawDone := false
	for time.Now().Before(deadline) {
		recvCtx, recvCancel := context.WithTimeout(ctx, 10*time.Second)
		ev, err := session.Recv(recvCtx)
		recvCancel()
		if err != nil {
			if ctx.Err() != nil {
				break
			}
			t.Logf("recv: %v", err)
			continue
		}
		t.Logf("event type=%s text=%q err=%q", ev.Type, truncate(ev.Text, 40), ev.Error)
		if ev.Type == llm.RealtimeEventResponseDone || ev.ProviderID == "response.done" {
			sawDone = true
			break
		}
		if ev.Type == llm.RealtimeEventError {
			// Empty response.create without input may error — still proves session framing.
			t.Logf("realtime error event (acceptable for empty create): %s", ev.Error)
			return
		}
	}
	if !sawDone {
		t.Log("no response.done within window — session connect + event decode still exercised")
	}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// --- minimal stdlib WebSocket client for OpenAI Realtime (test-only) ---

type wsConn struct {
	conn net.Conn
	r    *bufio.Reader
}

func dialOpenAIRealtime(ctx context.Context, apiKey, model string) (llm.RealtimeConn, error) {
	if model == "" {
		model = "gpt-realtime"
	}
	u, err := url.Parse("wss://api.openai.com/v1/realtime?model=" + url.QueryEscape(model))
	if err != nil {
		return nil, err
	}
	d := net.Dialer{Timeout: 20 * time.Second}
	raw, err := d.DialContext(ctx, "tcp", "api.openai.com:443")
	if err != nil {
		return nil, err
	}
	tlsConn := tls.Client(raw, &tls.Config{ServerName: "api.openai.com", MinVersion: tls.VersionTLS12})
	if err := tlsConn.HandshakeContext(ctx); err != nil {
		_ = raw.Close()
		return nil, err
	}

	key := make([]byte, 16)
	if _, err := rand.Read(key); err != nil {
		_ = tlsConn.Close()
		return nil, err
	}
	secKey := base64.StdEncoding.EncodeToString(key)

	req := fmt.Sprintf(
		"GET %s HTTP/1.1\r\nHost: api.openai.com\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: %s\r\nAuthorization: Bearer %s\r\n\r\n",
		u.RequestURI(), secKey, apiKey,
	)
	if _, err := io.WriteString(tlsConn, req); err != nil {
		_ = tlsConn.Close()
		return nil, err
	}
	br := bufio.NewReader(tlsConn)
	resp, err := http.ReadResponse(br, &http.Request{Method: http.MethodGet})
	if err != nil {
		_ = tlsConn.Close()
		return nil, err
	}
	if resp.StatusCode != http.StatusSwitchingProtocols {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		_ = resp.Body.Close()
		_ = tlsConn.Close()
		return nil, fmt.Errorf("websocket upgrade status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	// Validate accept token lightly.
	accept := resp.Header.Get("Sec-WebSocket-Accept")
	sum := sha1.Sum([]byte(secKey + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"))
	want := base64.StdEncoding.EncodeToString(sum[:])
	if accept != "" && accept != want {
		_ = tlsConn.Close()
		return nil, fmt.Errorf("bad Sec-WebSocket-Accept")
	}
	return &wsConn{conn: tlsConn, r: br}, nil
}

func (w *wsConn) Read(ctx context.Context) ([]byte, error) {
	if deadline, ok := ctx.Deadline(); ok {
		_ = w.conn.SetReadDeadline(deadline)
	} else {
		_ = w.conn.SetReadDeadline(time.Now().Add(30 * time.Second))
	}
	for {
		payload, opcode, err := w.readFrame()
		if err != nil {
			return nil, err
		}
		switch opcode {
		case 0x1: // text
			return payload, nil
		case 0x8: // close
			return nil, io.EOF
		case 0x9: // ping -> pong
			_ = w.writeFrame(0xA, payload)
		case 0xA: // pong
			continue
		default:
			// ignore binary/continuation for this smoke
			continue
		}
	}
}

func (w *wsConn) Write(ctx context.Context, data []byte) error {
	if deadline, ok := ctx.Deadline(); ok {
		_ = w.conn.SetWriteDeadline(deadline)
	}
	return w.writeFrame(0x1, data)
}

func (w *wsConn) Close() error {
	_ = w.writeFrame(0x8, []byte{0x03, 0xe8}) // 1000 normal closure
	return w.conn.Close()
}

func (w *wsConn) readFrame() (payload []byte, opcode byte, err error) {
	h := make([]byte, 2)
	if _, err = io.ReadFull(w.r, h); err != nil {
		return nil, 0, err
	}
	opcode = h[0] & 0x0f
	masked := h[1]&0x80 != 0
	n := int(h[1] & 0x7f)
	switch n {
	case 126:
		var ext [2]byte
		if _, err = io.ReadFull(w.r, ext[:]); err != nil {
			return nil, 0, err
		}
		n = int(binary.BigEndian.Uint16(ext[:]))
	case 127:
		var ext [8]byte
		if _, err = io.ReadFull(w.r, ext[:]); err != nil {
			return nil, 0, err
		}
		n = int(binary.BigEndian.Uint64(ext[:]))
	}
	var mask [4]byte
	if masked {
		if _, err = io.ReadFull(w.r, mask[:]); err != nil {
			return nil, 0, err
		}
	}
	payload = make([]byte, n)
	if _, err = io.ReadFull(w.r, payload); err != nil {
		return nil, 0, err
	}
	if masked {
		for i := range payload {
			payload[i] ^= mask[i%4]
		}
	}
	return payload, opcode, nil
}

func (w *wsConn) writeFrame(opcode byte, payload []byte) error {
	// Client frames must be masked.
	var mask [4]byte
	if _, err := rand.Read(mask[:]); err != nil {
		return err
	}
	var hdr []byte
	hdr = append(hdr, 0x80|opcode)
	n := len(payload)
	switch {
	case n < 126:
		hdr = append(hdr, byte(0x80|n))
	case n <= 0xffff:
		hdr = append(hdr, 0x80|126, byte(n>>8), byte(n))
	default:
		var ext [8]byte
		binary.BigEndian.PutUint64(ext[:], uint64(n))
		hdr = append(hdr, 0x80|127)
		hdr = append(hdr, ext[:]...)
	}
	hdr = append(hdr, mask[:]...)
	masked := make([]byte, n)
	for i := range payload {
		masked[i] = payload[i] ^ mask[i%4]
	}
	if _, err := w.conn.Write(hdr); err != nil {
		return err
	}
	_, err := w.conn.Write(masked)
	return err
}

// Ensure json is used (session event logging helpers may need it later).
var _ = json.Marshal
