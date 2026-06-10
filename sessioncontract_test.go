package llm_test

// Regression tests for the v0.4.0 session/persistence fixes (architecture
// review findings R-M1 shallow snapshot, R-M2 turn-long lock, R-M7 non-atomic
// FileStore writes). Break-the-system tests: each fails without its fix.

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
)

// R-M1: the snapshot returned by Session.Conversation() must not share any
// mutable state with the live transcript — mutating nested content of the
// snapshot must never corrupt the session.
func TestSessionConversationSnapshotIsDeep(t *testing.T) {
	mock := llm.NewMockClient("mock", "m1").PushText("hello")
	sess := llm.NewSession(mock,
		llm.WithTools(llm.Tool{
			Name:        "search",
			InputSchema: map[string]any{"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}}},
		}),
	)
	if _, err := sess.Send(context.Background(), "hi"); err != nil {
		t.Fatalf("Send: %v", err)
	}

	snap := sess.Conversation()
	snap.Messages[0].Content[0].Text = "TAMPERED"
	snap.Tools[0].InputSchema["type"] = "TAMPERED"

	live := sess.Conversation()
	if got := live.Messages[0].Content[0].Text; got != "hi" {
		t.Fatalf("mutating the snapshot corrupted the live transcript: %q", got)
	}
	if got := live.Tools[0].InputSchema["type"]; got != "object" {
		t.Fatalf("mutating the snapshot corrupted the live tool schema: %v", got)
	}
}

// R-M1: Conversation.Clone must deep-copy every nesting level the serialized
// form can carry — content parts, tool-result sub-parts, and tool inputs.
func TestConversationCloneIsDeep(t *testing.T) {
	conv := llm.NewConversation("sys")
	conv.Append(llm.Message{
		Role: llm.RoleAssistant,
		Content: []llm.ContentPart{
			{Type: llm.ContentToolUse, ToolUseID: "c1", ToolName: "t", ToolInput: map[string]any{"k": "v"}},
		},
	})
	conv.Append(llm.NewToolResultParts("c1", false,
		llm.ContentPart{Type: llm.ContentText, Text: "nested"},
	))

	cp := conv.Clone()
	cp.Messages[0].Content[0].ToolInput.(map[string]any)["k"] = "TAMPERED"
	cp.Messages[1].Content[0].ToolResultParts[0].Text = "TAMPERED"
	cp.System = "TAMPERED"

	if got := conv.Messages[0].Content[0].ToolInput.(map[string]any)["k"]; got != "v" {
		t.Fatalf("Clone shares tool input map: %v", got)
	}
	if got := conv.Messages[1].Content[0].ToolResultParts[0].Text; got != "nested" {
		t.Fatalf("Clone shares nested tool-result parts: %q", got)
	}
	if conv.System != "sys" {
		t.Fatal("Clone shares top-level fields")
	}
}

// R-M2: snapshot reads (Conversation/Usage/Provider) must not block while a
// turn is streaming — a long generation must not freeze every observer.
func TestSessionReadsDoNotBlockDuringStream(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	mock := llm.NewMockClient("mock", "m1").SetResponseFunc(func(llm.Request) (*llm.Response, error) {
		close(started)
		<-release
		return llm.FauxResponse("done"), nil
	})
	sess := llm.NewSession(mock)

	streamDone := make(chan struct{})
	go func() {
		defer close(streamDone)
		for ev, err := range sess.Stream(context.Background(), "hi") {
			_ = ev
			if err != nil {
				t.Errorf("stream error: %v", err)
			}
		}
	}()
	<-started // the turn is now in flight and blocked inside the client

	reads := make(chan int, 1)
	go func() {
		_ = sess.Usage()
		_ = sess.Provider()
		conv := sess.Conversation()
		reads <- len(conv.Messages)
	}()

	select {
	case n := <-reads:
		// The uncommitted in-flight turn must be invisible to snapshots.
		if n != 0 {
			t.Fatalf("in-flight (uncommitted) turn visible in snapshot: %d messages", n)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Session reads blocked while a stream is in flight")
	}

	close(release)
	<-streamDone

	if got := len(sess.Conversation().Messages); got != 2 {
		t.Fatalf("after the turn completes the transcript must hold user+assistant, got %d", got)
	}
}

// R-M2 guard: turns stay strictly serialized — two concurrent Sends must not
// interleave or fork the history.
func TestSessionTurnsStillSerialized(t *testing.T) {
	mock := llm.NewMockClient("mock", "m1")
	mock.SetResponseFunc(func(llm.Request) (*llm.Response, error) {
		time.Sleep(5 * time.Millisecond)
		return llm.FauxResponse("r"), nil
	})
	sess := llm.NewSession(mock)

	const turns = 8
	var wg sync.WaitGroup
	for i := 0; i < turns; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := sess.Send(context.Background(), "x"); err != nil {
				t.Errorf("Send: %v", err)
			}
		}()
	}
	wg.Wait()

	conv := sess.Conversation()
	if got := len(conv.Messages); got != 2*turns {
		t.Fatalf("got %d messages, want %d (user+assistant per turn)", got, 2*turns)
	}
	for i, m := range conv.Messages {
		wantRole := llm.RoleUser
		if i%2 == 1 {
			wantRole = llm.RoleAssistant
		}
		if m.Role != wantRole {
			t.Fatalf("message %d role %q — turns interleaved", i, m.Role)
		}
	}
}

// R-M7: FileStore.Save must be atomic — a reader (or a crash) must never
// observe a partially written conversation. With in-place writes the file is
// truncated first, so concurrent readers see invalid JSON.
func TestFileStoreSaveIsAtomic(t *testing.T) {
	dir := t.TempDir()
	store := llm.NewFileStore(dir)
	ctx := context.Background()

	// A conversation big enough that writing it takes a non-trivial window.
	big := llm.NewConversation("sys")
	filler := strings.Repeat("x", 64*1024)
	for i := 0; i < 64; i++ { // ~4 MB
		big.Append(llm.NewTextMessage(llm.RoleUser, filler))
	}
	if err := store.Save(ctx, "conv", big); err != nil {
		t.Fatalf("initial save: %v", err)
	}

	stop := make(chan struct{})
	var corrupt error
	var mu sync.Mutex
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			if _, err := store.Load(ctx, "conv"); err != nil && !errors.Is(err, llm.ErrConversationNotFound) {
				mu.Lock()
				corrupt = err
				mu.Unlock()
				return
			}
		}
	}()

	for i := 0; i < 40; i++ {
		if err := store.Save(ctx, "conv", big); err != nil {
			t.Fatalf("save %d: %v", i, err)
		}
		mu.Lock()
		failed := corrupt
		mu.Unlock()
		if failed != nil {
			break
		}
	}
	close(stop)
	wg.Wait()

	if corrupt != nil {
		t.Fatalf("a concurrent reader observed a partially written conversation: %v", corrupt)
	}

	// No temp droppings left behind, and the stored file is valid JSON.
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if e.Name() != "conv.json" {
			t.Fatalf("unexpected leftover file in store dir: %s", e.Name())
		}
	}
	data, err := os.ReadFile(filepath.Join(dir, "conv.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !json.Valid(data) {
		t.Fatal("stored conversation is not valid JSON")
	}
}
