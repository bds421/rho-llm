package llm_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	llm "github.com/bds421/rho-llm"
)

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

	// The atomic-save path (temp file + rename) must never leak a temp file,
	// regardless of which op ran last under contention.
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".tmp") {
			t.Fatalf("leftover temp file from atomic save: %s", e.Name())
		}
	}
}

func sampleConv() *llm.Conversation {
	c := llm.NewConversation("sys")
	c.Append(llm.NewTextMessage(llm.RoleUser, "hi"))
	c.AddResponse("anthropic", &llm.Response{Model: "claude-sonnet-4-6", Content: "hello", InputTokens: 3, OutputTokens: 2})
	return c
}

func TestMemoryStoreRoundTrip(t *testing.T) {
	s := llm.NewMemoryStore()
	ctx := context.Background()
	if err := s.Save(ctx, "c1", sampleConv()); err != nil {
		t.Fatal(err)
	}
	got, err := s.Load(ctx, "c1")
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Messages) != 2 || got.System != "sys" {
		t.Errorf("round-trip lost data: %+v", got)
	}
}

func TestMemoryStoreNotFound(t *testing.T) {
	s := llm.NewMemoryStore()
	if _, err := s.Load(context.Background(), "missing"); !errors.Is(err, llm.ErrConversationNotFound) {
		t.Errorf("Load(missing) = %v, want ErrConversationNotFound", err)
	}
}

func TestMemoryStoreDeleteAndReload(t *testing.T) {
	s := llm.NewMemoryStore()
	ctx := context.Background()
	_ = s.Save(ctx, "c1", sampleConv())
	if err := s.Delete(ctx, "c1"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Load(ctx, "c1"); !errors.Is(err, llm.ErrConversationNotFound) {
		t.Errorf("Load after Delete = %v, want not found", err)
	}
	if err := s.Delete(ctx, "c1"); err != nil { // idempotent
		t.Errorf("Delete missing = %v, want nil", err)
	}
}

func TestMemoryStoreCopyIsolation(t *testing.T) {
	s := llm.NewMemoryStore()
	ctx := context.Background()
	conv := sampleConv()
	_ = s.Save(ctx, "c1", conv)
	// Mutate the original after saving — the stored copy must be unaffected.
	conv.Append(llm.NewTextMessage(llm.RoleUser, "mutation"))
	got, _ := s.Load(ctx, "c1")
	if len(got.Messages) != 2 {
		t.Errorf("stored conversation was mutated by caller: %d messages", len(got.Messages))
	}
}

func TestMemoryStoreSaveNil(t *testing.T) {
	if err := llm.NewMemoryStore().Save(context.Background(), "x", nil); err == nil {
		t.Error("Save(nil) should error")
	}
}

func TestMemoryStoreConcurrent(t *testing.T) {
	s := llm.NewMemoryStore()
	ctx := context.Background()
	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			id := "c"
			for i := 0; i < 20; i++ {
				switch i % 3 {
				case 0:
					_ = s.Save(ctx, id, sampleConv())
				case 1:
					_, _ = s.Load(ctx, id)
				case 2:
					_ = s.Delete(ctx, id)
				}
			}
		}(g)
	}
	wg.Wait()
}

func TestFileStoreRoundTrip(t *testing.T) {
	dir := t.TempDir()
	s := llm.NewFileStore(dir)
	ctx := context.Background()
	if err := s.Save(ctx, "chat-1", sampleConv()); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "chat-1.json")); err != nil {
		t.Errorf("expected file on disk: %v", err)
	}
	got, err := s.Load(ctx, "chat-1")
	if err != nil || len(got.Messages) != 2 {
		t.Errorf("round-trip failed: %v %+v", err, got)
	}
}

func TestFileStoreRejectsTraversal(t *testing.T) {
	dir := t.TempDir()
	s := llm.NewFileStore(dir)
	ctx := context.Background()
	for _, bad := range []string{"../evil", "a/b", "..", "", "sub/../../x", "nul\x00byte"} {
		if err := s.Save(ctx, bad, sampleConv()); err == nil {
			t.Errorf("Save(%q) should be rejected (path traversal)", bad)
		}
		if _, err := s.Load(ctx, bad); err == nil {
			t.Errorf("Load(%q) should be rejected", bad)
		}
	}
	// Nothing must have been written outside the store dir.
	if _, err := os.Stat(filepath.Join(dir, "..", "evil.json")); err == nil {
		t.Error("traversal wrote a file outside the store directory!")
	}
}

func TestFileStoreNotFoundAndDeleteIdempotent(t *testing.T) {
	s := llm.NewFileStore(t.TempDir())
	ctx := context.Background()
	if _, err := s.Load(ctx, "nope"); !errors.Is(err, llm.ErrConversationNotFound) {
		t.Errorf("Load(missing) = %v, want not found", err)
	}
	if err := s.Delete(ctx, "nope"); err != nil {
		t.Errorf("Delete(missing) = %v, want nil", err)
	}
}

// BUG 5: "." is a reserved path component; FileStore must reject it as an id
// rather than silently writing a meaningless "..json" file. (".." is already
// rejected; "." slipped through the validation.)
func TestFileStoreRejectsDotID(t *testing.T) {
	s := llm.NewFileStore(t.TempDir())
	ctx := context.Background()
	if err := s.Save(ctx, ".", llm.NewConversation("")); err == nil {
		t.Error(`FileStore.Save(".") must reject the reserved "." id`)
	}
	if _, err := s.Load(ctx, "."); err == nil {
		t.Error(`FileStore.Load(".") must reject the reserved "." id`)
	}
}

// R-M7: FileStore.Save must be atomic — a reader (or a crash) must never
// observe a partially written conversation. With in-place writes the file is
// truncated first, so concurrent readers see invalid JSON.
//
// NOTE: this is a *probabilistic* regression test — it relies on a reader
// landing inside the writer's truncate/write window. Against the buggy
// (non-atomic) implementation it catches the corruption most of the time but
// not every run; the deterministic guarantees (no leftover temp file, valid
// final JSON) are asserted unconditionally below. Run with -count to raise
// confidence in the racing assertion.
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
