package llm_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"

	llm "github.com/bds421/rho-llm"
)

func sampleConv() *llm.Conversation {
	c := llm.NewConversation("sys")
	c.Append(llm.NewTextMessage(llm.RoleUser, "hi"))
	c.AddResponse("anthropic", &llm.Response{Model: "claude-sonnet-4-6", Content: "hello", InputTokens: 3, OutputTokens: 2})
	return c
}

// ---- MemoryStore -----------------------------------------------------------

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

// ---- FileStore -------------------------------------------------------------

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
