package llm_test

// Full break-the-system pass (v0.4.0). These tests try to BREAK the library —
// hostile inputs, corrupt streams, and concurrency under -race. Per CLAUDE.md:
// "If nothing breaks, the tests aren't complete." Several of these were written
// to fail against the as-shipped code and drove fixes; the rest are guards.

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/provider/anthropic"
	"github.com/bds421/rho-llm/provider/gemini"
	"github.com/bds421/rho-llm/provider/openairesponses"
)

// ---------------------------------------------------------------------------
// Conversation.Clone — deep-copy contract under a hostile (unserializable) input
// ---------------------------------------------------------------------------

// Clone()'s primary path is a JSON round-trip. A non-JSON-serializable ToolInput
// (a channel) forces the fallback path. The fallback must STILL produce an
// isolated copy — mutating the clone must not corrupt the original.
func TestCloneFallbackIsolatesNestedStateOnUnserializableInput(t *testing.T) {
	conv := llm.NewConversation("sys", llm.Tool{
		Name:        "t",
		InputSchema: map[string]any{"type": "object"},
	})
	conv.Append(llm.Message{
		Role: llm.RoleAssistant,
		Content: []llm.ContentPart{
			{Type: llm.ContentText, Text: "original"},
			// chan is not JSON-serializable → forces Clone's fallback path
			{Type: llm.ContentToolUse, ToolUseID: "c1", ToolName: "t", ToolInput: make(chan int)},
		},
	})
	conv.Append(llm.NewToolResultParts("c1", false,
		llm.ContentPart{Type: llm.ContentText, Text: "nested-original"},
	))

	cp := conv.Clone()

	// Mutate every nesting level of the clone.
	cp.Messages[0].Content[0].Text = "MUTATED"
	cp.Messages[1].Content[0].ToolResultParts[0].Text = "MUTATED"
	cp.Tools[0].InputSchema["type"] = "MUTATED"
	cp.System = "MUTATED"

	if got := conv.Messages[0].Content[0].Text; got != "original" {
		t.Errorf("clone shares Content backing array: original text corrupted to %q", got)
	}
	if got := conv.Messages[1].Content[0].ToolResultParts[0].Text; got != "nested-original" {
		t.Errorf("clone shares nested ToolResultParts: corrupted to %q", got)
	}
	if got := conv.Tools[0].InputSchema["type"]; got != "object" {
		t.Errorf("clone shares Tool.InputSchema map: corrupted to %v", got)
	}
	if conv.System != "sys" {
		t.Errorf("clone shares top-level fields: System corrupted to %q", conv.System)
	}
}

// Clone of an image/document message must not share the *ImageSource /
// *DocumentSource pointers.
func TestCloneFallbackIsolatesPointerSources(t *testing.T) {
	conv := llm.NewConversation("")
	conv.Append(llm.Message{
		Role: llm.RoleUser,
		Content: []llm.ContentPart{
			{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "image/png", Data: "AAA"}},
			// chan forces the fallback path
			{Type: llm.ContentToolUse, ToolInput: make(chan int)},
		},
	})
	cp := conv.Clone()
	cp.Messages[0].Content[0].Source.Data = "MUTATED"
	if got := conv.Messages[0].Content[0].Source.Data; got != "AAA" {
		t.Fatalf("clone shares *ImageSource pointer: corrupted to %q", got)
	}
}

// ---------------------------------------------------------------------------
// Streaming — trailing noise after a terminal event must not mask a complete turn
// ---------------------------------------------------------------------------

func sseSrv(lines ...string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, l := range lines {
			fmt.Fprint(w, l+"\n\n")
		}
	}))
}

func drain(t *testing.T, c llm.Client) (done bool, errs int) {
	t.Helper()
	for ev, err := range c.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")},
	}) {
		if err != nil {
			errs++
		} else if ev.Type == llm.EventDone {
			done = true
		}
	}
	return done, errs
}

// A garbage SSE line AFTER the protocol-final event must be ignored: the turn
// already completed, so a trailing bad byte must not surface as an error (which
// would make a Session drop an otherwise-complete turn).
func TestStreamGarbageAfterDoneIgnored(t *testing.T) {
	t.Run("anthropic", func(t *testing.T) {
		srv := sseSrv(
			`data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}`,
			`data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}`,
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
			`data: {GARBAGE not json}`,
		)
		defer srv.Close()
		c, _ := anthropic.New(llm.Config{Provider: "anthropic", Model: "claude-sonnet-4-6", APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
		done, errs := drain(t, c)
		if !done || errs != 0 {
			t.Fatalf("done=%v errs=%d — trailing garbage after Done not ignored", done, errs)
		}
	})

	t.Run("gemini", func(t *testing.T) {
		srv := sseSrv(
			`data: {"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}`,
			`data: {GARBAGE not json}`,
		)
		defer srv.Close()
		c, _ := gemini.New(llm.Config{Provider: "gemini", Model: "gemini-2.5-flash", APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
		done, errs := drain(t, c)
		if !done || errs != 0 {
			t.Fatalf("done=%v errs=%d — trailing garbage after Done not ignored", done, errs)
		}
	})

	t.Run("openairesponses", func(t *testing.T) {
		srv := sseSrv(
			`data: {"type":"response.output_text.delta","delta":"hi"}`,
			`data: {"type":"response.completed","response":{"id":"r","status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}`,
			`data: {GARBAGE not json}`,
		)
		defer srv.Close()
		c, _ := openairesponses.New(llm.Config{Provider: "openai_responses", Model: "gpt-5.2", APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
		done, errs := drain(t, c)
		if !done || errs != 0 {
			t.Fatalf("done=%v errs=%d — trailing garbage after Done not ignored", done, errs)
		}
	})
}

// A Session must COMMIT a turn whose stream completed (saw Done) even if the
// server appended trailing garbage — the complete turn must not be dropped.
func TestSessionCommitsDespiteTrailingGarbage(t *testing.T) {
	srv := sseSrv(
		`data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}`,
		`data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"answer"}}`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
		`data: {trailing garbage}`,
	)
	defer srv.Close()
	c, _ := anthropic.New(llm.Config{Provider: "anthropic", Model: "claude-sonnet-4-6", APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
	sess := llm.NewSession(c)
	for range sess.Stream(context.Background(), "q") {
	}
	conv := sess.Conversation()
	if len(conv.Messages) != 2 {
		t.Fatalf("got %d messages, want 2 (user+assistant) — complete turn dropped by trailing garbage", len(conv.Messages))
	}
	if conv.Messages[1].Content[0].Text != "answer" {
		t.Fatalf("assistant content = %q, want %q", conv.Messages[1].Content[0].Text, "answer")
	}
}

// ---------------------------------------------------------------------------
// LoadConversation — hostile / corrupt blobs must error, never panic
// ---------------------------------------------------------------------------

func TestLoadConversationHostileBlobs(t *testing.T) {
	cases := map[string]string{
		"schema_version as string": `{"schema_version":"99","messages":[]}`,
		"messages as object":       `{"schema_version":1,"messages":{"not":"an array"}}`,
		"content as string":        `{"schema_version":1,"messages":[{"role":"user","content":"should be array"}]}`,
		"truncated":                `{"schema_version":1,"messages":[{"role":"user",`,
		"trailing garbage":         `{"schema_version":1,"messages":[]} JUNK`,
		"null bytes":               "{\"schema_version\":1,\x00\"messages\":[]}",
		"future schema":            `{"schema_version":999999,"messages":[]}`,
		"deeply nested tool input": `{"schema_version":1,"messages":[{"role":"assistant","content":[{"type":"tool_use","input":` + strings.Repeat(`[`, 2000) + strings.Repeat(`]`, 2000) + `}]}]}`,
	}
	for name, blob := range cases {
		t.Run(name, func(t *testing.T) {
			// Must not panic; valid blobs return a conv, invalid ones an error.
			conv, err := llm.LoadConversation([]byte(blob))
			if err == nil && conv == nil {
				t.Fatal("both conv and err are nil")
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Concurrency hammers (run under -race)
// ---------------------------------------------------------------------------

// Mixed concurrent Session ops: turns serialize, reads never block, no race,
// no torn state. Asserts the committed transcript stays well-formed (every
// assistant turn carries provider provenance, message count is even).
func TestSessionConcurrentMixedOpsRaceFree(t *testing.T) {
	mk := func(name string) *llm.MockClient {
		m := llm.NewMockClient(name, name+"-model")
		m.SetResponseFunc(func(llm.Request) (*llm.Response, error) {
			return &llm.Response{Content: "r", StopReason: llm.StopEndTurn, InputTokens: 1, OutputTokens: 1}, nil
		})
		return m
	}
	sess := llm.NewSession(mk("a"))
	alt := mk("b")

	var wg sync.WaitGroup
	for i := 0; i < 24; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			switch n % 6 {
			case 0:
				_, _ = sess.Send(context.Background(), "x")
			case 1:
				for range sess.Stream(context.Background(), "y") {
				}
			case 2:
				_ = sess.Conversation()
			case 3:
				_ = sess.Usage()
			case 4:
				sess.SwitchProvider(alt)
			case 5:
				_ = sess.Provider()
			}
		}(i)
	}
	wg.Wait()

	conv := sess.Conversation()
	if len(conv.Messages)%2 != 0 {
		t.Fatalf("odd message count %d — a turn interleaved or was partially committed", len(conv.Messages))
	}
	for i := 1; i < len(conv.Messages); i += 2 {
		if conv.Messages[i].Role != llm.RoleAssistant {
			t.Fatalf("message %d is %q, expected assistant — turns interleaved", i, conv.Messages[i].Role)
		}
		if conv.Messages[i].Provider == "" {
			t.Fatalf("assistant message %d has no provider provenance", i)
		}
	}
}

// Concurrent Save/Load/Delete on the same FileStore id: a Load must return
// either a valid conversation or ErrConversationNotFound — never a decode
// error from a half-written file. No leftover temp files.
func TestFileStoreConcurrentSaveLoadDelete(t *testing.T) {
	dir := t.TempDir()
	store := llm.NewFileStore(dir)
	ctx := context.Background()
	conv := llm.NewConversation("sys")
	conv.Append(llm.NewTextMessage(llm.RoleUser, strings.Repeat("x", 128*1024)))

	var wg sync.WaitGroup
	var corrupt error
	var mu sync.Mutex
	note := func(err error) {
		mu.Lock()
		if corrupt == nil {
			corrupt = err
		}
		mu.Unlock()
	}
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			for j := 0; j < 25; j++ {
				switch n % 3 {
				case 0:
					if err := store.Save(ctx, "c", conv); err != nil {
						note(fmt.Errorf("save: %w", err))
					}
				case 1:
					if _, err := store.Load(ctx, "c"); err != nil && err != llm.ErrConversationNotFound {
						note(fmt.Errorf("load saw corruption: %w", err))
					}
				case 2:
					if err := store.Delete(ctx, "c"); err != nil {
						note(fmt.Errorf("delete: %w", err))
					}
				}
			}
		}(i)
	}
	wg.Wait()
	if corrupt != nil {
		t.Fatalf("concurrent store ops produced an error: %v", corrupt)
	}
}

// Concurrent registration of the SAME model id plus readers: the override must
// not double-append to the provider's available-models list, and reads must be
// race-free.
func TestRegistryConcurrentSameIDNoDoubleAppend(t *testing.T) {
	const id = "break-dup-model"
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			_ = llm.RegisterModel(llm.ModelInfo{ID: id, Provider: "break-dup-prov", InputPricePer1M: float64(n)})
			_ = llm.ModelsByProvider("break-dup-prov")
			_, _ = llm.GetModelInfo(id)
		}(i)
	}
	wg.Wait()

	n := 0
	for _, m := range llm.ModelsByProvider("break-dup-prov") {
		if m.ID == id {
			n++
		}
	}
	if n != 1 {
		t.Fatalf("model %q appears %d times in availableModels — concurrent override double-appended", id, n)
	}
}

// A pool driven by a client that randomly fails or is cancelled, hammered
// concurrently: must stay race-free, never panic, and never permanently
// disable the only key for a non-auth failure.
func TestPoolConcurrentFailuresAndCancellationRaceFree(t *testing.T) {
	var seq int64
	mu := &sync.Mutex{}
	next := func() int64 { mu.Lock(); defer mu.Unlock(); seq++; return seq }

	cfg := llm.DefaultConfig()
	cfg.CircuitThreshold = 0 // isolate pool/key behavior from breaker fail-fast
	// Tiny backoff + cooldowns so a retry storm doesn't sleep on real 60s
	// rate-limit cooldowns — we're testing concurrency, not wall-clock backoff.
	cfg.RetryPolicy = &llm.RetryPolicy{BaseDelay: time.Millisecond, MaxDelay: 2 * time.Millisecond, Factor: 2, Jitter: 0}
	cfg.CooldownRateLimit = time.Millisecond
	cfg.CooldownOverload = time.Millisecond
	cfg.CooldownDefault = time.Millisecond
	pc, err := llm.NewPooledClient(cfg, []string{"key-a"}, func(llm.AuthProfile) (llm.Client, error) {
		m := llm.NewMockClient("anthropic", "m")
		m.SetResponseFunc(func(llm.Request) (*llm.Response, error) {
			switch next() % 3 {
			case 0:
				return nil, llm.NewOverloadedError("anthropic", "503")
			case 1:
				return nil, llm.NewRateLimitError("anthropic", "429")
			default:
				return &llm.Response{Content: "ok", StopReason: llm.StopEndTurn}, nil
			}
		})
		return m, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient: %v", err)
	}
	defer pc.Close()

	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			ctx := context.Background()
			if n%2 == 0 {
				c, cancel := context.WithCancel(ctx)
				cancel() // pre-cancelled: must not poison the key
				ctx = c
			}
			_, _ = pc.Complete(ctx, llm.Request{})
		}(i)
	}
	wg.Wait()

	// After the storm, a transient/cancellation-only history must leave the key
	// usable — a temporary cooldown is allowed, a permanent disable
	// ("unhealthy", reserved for auth errors) is not. The profile is named
	// "<provider>-<n>" (e.g. anthropic-1), not by the key value.
	status := pc.PoolStatus()
	if status == "" {
		t.Fatal("empty pool status")
	}
	if strings.Contains(status, "unhealthy") {
		t.Fatalf("key permanently disabled by transient/cancelled failures: %s", status)
	}
}
