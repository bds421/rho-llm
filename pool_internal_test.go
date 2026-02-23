package llm

import (
	"context"
	"iter"
	"sync"
	"sync/atomic"
	"testing"
)

// closeSpy is a minimal Client that tracks whether Close was called.
type closeSpy struct {
	closed      atomic.Bool
	provider    string
	model       string
	completeErr error                                                  // if set, Complete returns this error
	blockCh     chan struct{}                                           // if set, Complete blocks until closed
	completeFn  func(context.Context, Request) (*Response, error)      // optional override
}

func (c *closeSpy) Complete(ctx context.Context, req Request) (*Response, error) {
	if c.completeFn != nil {
		return c.completeFn(ctx, req)
	}
	if c.blockCh != nil {
		select {
		case <-c.blockCh:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if c.completeErr != nil {
		return nil, c.completeErr
	}
	return &Response{Content: "ok"}, nil
}

func (c *closeSpy) Stream(_ context.Context, _ Request) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		if c.completeErr != nil {
			yield(StreamEvent{}, c.completeErr)
			return
		}
		if !yield(StreamEvent{Type: EventContent, Text: "streamed"}, nil) {
			return
		}
		yield(StreamEvent{Type: EventDone, StopReason: "end_turn"}, nil)
	}
}

func (c *closeSpy) Provider() string { return c.provider }
func (c *closeSpy) Model() string    { return c.model }
func (c *closeSpy) Close() error     { c.closed.Store(true); return nil }

// --- refCountedClient unit tests ---

func TestRefCountedClientCloseOnLastRelease(t *testing.T) {
	spy := &closeSpy{provider: "test", model: "m"}
	rc := newRefCountedClient(spy)

	// Born with refs=1. Single Release should close.
	rc.Release()

	if !spy.closed.Load() {
		t.Fatal("expected Close to be called after final Release")
	}
}

func TestRefCountedClientMultiAcquireRelease(t *testing.T) {
	spy := &closeSpy{provider: "test", model: "m"}
	rc := newRefCountedClient(spy)

	const n = 10
	for i := 0; i < n; i++ {
		rc.Acquire()
	}

	// Release n times — should NOT close (pool's own ref still held)
	for i := 0; i < n; i++ {
		rc.Release()
		if spy.closed.Load() {
			t.Fatalf("Close called too early after %d releases (of %d acquires + 1 initial)", i+1, n)
		}
	}

	// Final release (the pool's own ref)
	rc.Release()
	if !spy.closed.Load() {
		t.Fatal("expected Close after all references released")
	}
}

// --- PooledClient rotation cleanup tests ---

func TestPooledClientRotationClosesOldClient(t *testing.T) {
	var (
		mu      sync.Mutex
		clients []*closeSpy
	)

	callCount := 0
	pc, err := NewPooledClient(DefaultConfig(), []string{"key-a", "key-b"}, func(profile AuthProfile) (Client, error) {
		mu.Lock()
		defer mu.Unlock()
		callCount++
		spy := &closeSpy{provider: "test", model: "m"}
		if callCount == 1 {
			// First client: will fail with rate limit to trigger rotation
			spy.completeErr = NewRateLimitError("test", "rate limited")
		}
		clients = append(clients, spy)
		return spy, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient: %v", err)
	}
	defer pc.Close()

	// This call triggers rotation: first client fails, second succeeds
	resp, err := pc.Complete(context.Background(), Request{})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Content != "ok" {
		t.Fatalf("Content = %q, want ok", resp.Content)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(clients) < 2 {
		t.Fatalf("expected at least 2 clients, got %d", len(clients))
	}

	// Old client (index 0) should be closed — no in-flight requests held it
	if !clients[0].closed.Load() {
		t.Error("old client was not closed after rotation")
	}

	// New client (index 1) should still be open
	if clients[1].closed.Load() {
		t.Error("new client should not be closed")
	}
}

func TestPooledClientInFlightPreventsClose(t *testing.T) {
	blockCh := make(chan struct{})
	inFlightStarted := make(chan struct{})

	var (
		mu      sync.Mutex
		clients []*closeSpy
	)

	callCount := 0
	pc, err := NewPooledClient(DefaultConfig(), []string{"key-a", "key-b"}, func(profile AuthProfile) (Client, error) {
		mu.Lock()
		defer mu.Unlock()
		callCount++
		spy := &closeSpy{provider: "test", model: "m"}
		if callCount == 1 {
			// First client: blocks on Complete until we unblock it
			spy.completeFn = func(ctx context.Context, req Request) (*Response, error) {
				close(inFlightStarted)
				select {
				case <-blockCh:
					return &Response{Content: "delayed"}, nil
				case <-ctx.Done():
					return nil, ctx.Err()
				}
			}
		}
		clients = append(clients, spy)
		return spy, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient: %v", err)
	}
	defer pc.Close()

	// Start an in-flight request on the first client
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		resp, err := pc.Complete(context.Background(), Request{})
		if err != nil {
			t.Errorf("in-flight Complete: %v", err)
			return
		}
		if resp.Content != "delayed" {
			t.Errorf("Content = %q, want delayed", resp.Content)
		}
	}()

	// Wait for the in-flight request to start
	<-inFlightStarted

	// Force rotation by directly calling rotateClient.
	// First, mark the current profile as failed so rotation proceeds.
	pc.pool.MarkFailedByName(pc.activeName, NewRateLimitError("test", "rate limited"))

	mu.Lock()
	firstName := pc.activeName
	mu.Unlock()

	if rotErr := pc.rotateClient(firstName); rotErr != nil {
		t.Fatalf("rotateClient: %v", rotErr)
	}

	// Old client should NOT be closed yet (in-flight request holds a ref)
	mu.Lock()
	oldClosed := clients[0].closed.Load()
	mu.Unlock()
	if oldClosed {
		t.Fatal("old client closed while in-flight request still active")
	}

	// Unblock the in-flight request
	close(blockCh)
	wg.Wait()

	// Now the old client should be closed (in-flight released its ref)
	mu.Lock()
	oldClosed = clients[0].closed.Load()
	mu.Unlock()
	if !oldClosed {
		t.Error("old client not closed after in-flight request completed")
	}
}

func TestPooledClientCompleteAfterClose(t *testing.T) {
	pc, err := NewPooledClient(DefaultConfig(), []string{"key-a"}, func(profile AuthProfile) (Client, error) {
		return &closeSpy{provider: "test", model: "m"}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient: %v", err)
	}

	pc.Close()

	_, err = pc.Complete(context.Background(), Request{})
	if err != ErrClientClosed {
		t.Fatalf("Complete after Close: got %v, want ErrClientClosed", err)
	}
}

func TestPooledClientStreamAfterClose(t *testing.T) {
	pc, err := NewPooledClient(DefaultConfig(), []string{"key-a"}, func(profile AuthProfile) (Client, error) {
		return &closeSpy{provider: "test", model: "m"}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient: %v", err)
	}

	pc.Close()

	var gotErr error
	for _, err := range pc.Stream(context.Background(), Request{}) {
		if err != nil {
			gotErr = err
			break
		}
	}
	if gotErr != ErrClientClosed {
		t.Fatalf("Stream after Close: got %v, want ErrClientClosed", gotErr)
	}
}
