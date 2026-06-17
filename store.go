package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// ErrConversationNotFound is returned by Store.Load when no conversation has the
// requested id.
var ErrConversationNotFound = errors.New("llm: conversation not found")

// Store persists conversations by id. Implementations must be safe for concurrent
// use. Conversations are serialized via the versioned JSON form (see
// LoadConversation), so a Store round-trips them losslessly and independently of
// Go types.
type Store interface {
	Save(ctx context.Context, id string, conv *Conversation) error
	Load(ctx context.Context, id string) (*Conversation, error)
	Delete(ctx context.Context, id string) error
}

// MemoryStore is an in-memory Store backed by a mutex-guarded map. It stores the
// serialized bytes, so loaded conversations are independent copies — mutating one
// never affects what's stored or what another Load returns.
type MemoryStore struct {
	mu    sync.RWMutex
	items map[string][]byte
}

// NewMemoryStore creates an empty in-memory Store.
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{items: make(map[string][]byte)}
}

// Save serializes and stores conv under id.
func (s *MemoryStore) Save(_ context.Context, id string, conv *Conversation) error {
	if conv == nil {
		return fmt.Errorf("llm: cannot save nil conversation")
	}
	data, err := json.Marshal(conv)
	if err != nil {
		return fmt.Errorf("llm: marshal conversation: %w", err)
	}
	s.mu.Lock()
	s.items[id] = data
	s.mu.Unlock()
	return nil
}

// Load returns the conversation stored under id, or ErrConversationNotFound.
func (s *MemoryStore) Load(_ context.Context, id string) (*Conversation, error) {
	s.mu.RLock()
	data, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		return nil, ErrConversationNotFound
	}
	return LoadConversation(data)
}

// Delete removes id if present (no error if absent).
func (s *MemoryStore) Delete(_ context.Context, id string) error {
	s.mu.Lock()
	delete(s.items, id)
	s.mu.Unlock()
	return nil
}

// FileStore persists each conversation as a JSON file (`<dir>/<id>.json`). It uses
// only the standard library. IDs are restricted to a single path segment with no
// separators or "..", so a hostile id cannot escape the directory.
type FileStore struct {
	dir string
}

// NewFileStore creates a Store rooted at dir (created lazily on first Save).
func NewFileStore(dir string) *FileStore {
	return &FileStore{dir: dir}
}

// path validates id and returns its on-disk path, rejecting traversal attempts.
func (s *FileStore) path(id string) (string, error) {
	if id == "" || strings.ContainsAny(id, `/\`) || strings.Contains(id, "..") || strings.ContainsRune(id, 0) {
		return "", fmt.Errorf("llm: invalid conversation id %q (must be a single path segment)", id)
	}
	return filepath.Join(s.dir, id+".json"), nil
}

// Save writes conv to `<dir>/<id>.json` (0600), creating dir if needed.
// The write is atomic and durable (temp file + fsync + rename, then a
// best-effort directory fsync): a concurrent reader always sees either the old
// or the new complete file, and a crash mid-write cannot leave a partially
// written or empty conversation behind.
func (s *FileStore) Save(_ context.Context, id string, conv *Conversation) error {
	if conv == nil {
		return fmt.Errorf("llm: cannot save nil conversation")
	}
	p, err := s.path(id)
	if err != nil {
		return err
	}
	data, err := json.Marshal(conv)
	if err != nil {
		return fmt.Errorf("llm: marshal conversation: %w", err)
	}
	if err := os.MkdirAll(s.dir, 0o750); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(s.dir, id+".*.tmp") // 0600 by default
	if err != nil {
		return err
	}
	// On any failure below, the primary error is what matters; the temp-file
	// cleanup is best-effort (errors deliberately ignored).
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmp.Name())
		return err
	}
	// fsync the data before the rename so a crash can't leave the renamed file
	// pointing at unflushed (empty/partial) contents — this is what makes the
	// atomicity guarantee hold across a power loss, not just a concurrent reader.
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmp.Name())
		return err
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmp.Name())
		return err
	}
	if err := os.Rename(tmp.Name(), p); err != nil {
		_ = os.Remove(tmp.Name())
		return err
	}
	// fsync the directory so the rename itself is durable (best-effort: not all
	// platforms/filesystems support directory fsync, and a failure here doesn't
	// invalidate the already-renamed file).
	if dir, derr := os.Open(s.dir); derr == nil {
		_ = dir.Sync()
		_ = dir.Close()
	}
	return nil
}

// Load reads `<dir>/<id>.json`, returning ErrConversationNotFound if absent.
func (s *FileStore) Load(_ context.Context, id string) (*Conversation, error) {
	p, err := s.path(id)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(p) // #nosec G304 -- p comes from s.path(), which rejects separators and ".." (single path segment only)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, ErrConversationNotFound
		}
		return nil, err
	}
	return LoadConversation(data)
}

// Delete removes the file for id (no error if absent).
func (s *FileStore) Delete(_ context.Context, id string) error {
	p, err := s.path(id)
	if err != nil {
		return err
	}
	if err := os.Remove(p); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}
