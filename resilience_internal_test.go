package llm

// Regression tests for the v0.4.0 resilience-correctness fixes (architecture
// review findings R-H1, R-L7, R-L2). These are break-the-system tests: each
// fails without its fix.

import (
	"context"
	"errors"
	"iter"
	"net/url"
	"testing"
	"time"
)

// cancelAwareClient simulates a real adapter: when the caller's context is
// done, it returns the *url.Error that net/http produces (which satisfies
// net.Error and therefore looks "retryable" to a naive classifier).
type cancelAwareClient struct{}

func (c *cancelAwareClient) Complete(ctx context.Context, _ Request) (*Response, error) {
	if ctx.Err() != nil {
		return nil, &url.Error{Op: "Post", URL: "https://api.example.com/v1", Err: ctx.Err()}
	}
	return &Response{Content: "ok"}, nil
}

func (c *cancelAwareClient) Stream(ctx context.Context, _ Request) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		if ctx.Err() != nil {
			yield(StreamEvent{}, &url.Error{Op: "Post", URL: "https://api.example.com/v1", Err: ctx.Err()})
			return
		}
		yield(StreamEvent{Type: EventDone, StopReason: "end_turn"}, nil)
	}
}

func (c *cancelAwareClient) Provider() string { return "test" }
func (c *cancelAwareClient) Model() string    { return "test-model" }
func (c *cancelAwareClient) Close() error     { return nil }

// scriptedStreamClient yields the given events, then the given error.
type scriptedStreamClient struct {
	events []StreamEvent
	err    error
}

func (c *scriptedStreamClient) Complete(_ context.Context, _ Request) (*Response, error) {
	return &Response{Content: "ok"}, nil
}

func (c *scriptedStreamClient) Stream(_ context.Context, _ Request) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		for _, ev := range c.events {
			if !yield(ev, nil) {
				return
			}
		}
		if c.err != nil {
			yield(StreamEvent{}, c.err)
		}
	}
}

func (c *scriptedStreamClient) Provider() string { return "test" }
func (c *scriptedStreamClient) Model() string    { return "test-model" }
func (c *scriptedStreamClient) Close() error     { return nil }

// poolHealth reports whether the single profile of pc is still usable and the
// breaker state, under the pool's lock.
func poolHealth(pc *PooledClient) (profileAvailable bool, state CircuitState) {
	pc.pool.mu.RLock()
	profileAvailable = pc.pool.profiles[0].IsAvailable()
	pc.pool.mu.RUnlock()
	return profileAvailable, pc.breaker.State()
}

// R-H1: a caller-cancelled Complete must not put the key in cooldown, must not
// record a breaker failure, and must not burn retries on a dead context.
func TestCompleteCallerCancellationDoesNotPoisonPool(t *testing.T) {
	cfg := DefaultConfig()
	cfg.CircuitThreshold = 1 // any recorded failure opens the circuit — sharp signal
	pc, err := NewPooledClient(cfg, []string{"key-a"}, func(AuthProfile) (Client, error) {
		return &cancelAwareClient{}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient: %v", err)
	}
	defer pc.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = pc.Complete(ctx, Request{})
	if err == nil {
		t.Fatal("want error from cancelled context")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("want context.Canceled in the chain, got %v", err)
	}

	available, state := poolHealth(pc)
	if state != CircuitClosed {
		t.Fatalf("caller cancellation tripped the breaker: state=%v", state)
	}
	if !available {
		t.Fatal("caller cancellation put the key in cooldown")
	}
}

// R-H1 (stream): same invariant for a pre-data stream failure caused by a
// cancelled caller context.
func TestStreamCallerCancellationDoesNotPoisonPool(t *testing.T) {
	cfg := DefaultConfig()
	cfg.CircuitThreshold = 1
	pc, err := NewPooledClient(cfg, []string{"key-a"}, func(AuthProfile) (Client, error) {
		return &cancelAwareClient{}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient: %v", err)
	}
	defer pc.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	var lastErr error
	for _, err := range pc.Stream(ctx, Request{}) {
		if err != nil {
			lastErr = err
		}
	}
	if lastErr == nil {
		t.Fatal("want error from cancelled context")
	}
	if !errors.Is(lastErr, context.Canceled) {
		t.Fatalf("want context.Canceled in the chain, got %v", lastErr)
	}

	available, state := poolHealth(pc)
	if state != CircuitClosed {
		t.Fatalf("caller cancellation tripped the breaker: state=%v", state)
	}
	if !available {
		t.Fatal("caller cancellation put the key in cooldown")
	}
}

// R-L7: a genuine provider failure mid-stream must count against the breaker
// exactly like a pre-data failure does (the asymmetry let a flapping endpoint
// stay "healthy" as long as it died only after the first event).
func TestMidStreamRetryableErrorRecordsBreakerFailure(t *testing.T) {
	cfg := DefaultConfig()
	cfg.CircuitThreshold = 1
	pc, err := NewPooledClient(cfg, []string{"key-a"}, func(AuthProfile) (Client, error) {
		return &scriptedStreamClient{
			events: []StreamEvent{{Type: EventContent, Text: "partial"}},
			err:    NewOverloadedError("test", "503 mid-stream"),
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient: %v", err)
	}
	defer pc.Close()

	var lastErr error
	for _, err := range pc.Stream(context.Background(), Request{}) {
		if err != nil {
			lastErr = err
		}
	}
	if lastErr == nil {
		t.Fatal("want the mid-stream error passed through")
	}
	if got := pc.breaker.State(); got != CircuitOpen {
		t.Fatalf("mid-stream provider failure did not trip the breaker: state=%v", got)
	}
}

// R-L7 guard: a caller breaking out of the stream early is NOT a failure and
// must keep recording success (regression guard for the fix above).
func TestStreamConsumerBreakStillRecordsSuccess(t *testing.T) {
	cfg := DefaultConfig()
	cfg.CircuitThreshold = 1
	pc, err := NewPooledClient(cfg, []string{"key-a"}, func(AuthProfile) (Client, error) {
		return &scriptedStreamClient{
			events: []StreamEvent{
				{Type: EventContent, Text: "a"},
				{Type: EventContent, Text: "b"},
			},
		}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient: %v", err)
	}
	defer pc.Close()

	for range pc.Stream(context.Background(), Request{}) {
		break // consumer abandons mid-stream
	}
	available, state := poolHealth(pc)
	if state != CircuitClosed || !available {
		t.Fatalf("consumer break counted as failure: state=%v available=%v", state, available)
	}
}

// R-L2: an abandoned half-open probe (consumer never reports success or
// failure) must not wedge the circuit half-open forever — a fresh probe is
// allowed after another cooldown.
func TestCircuitBreakerAbandonedProbeDoesNotWedgeHalfOpen(t *testing.T) {
	cb := NewCircuitBreaker(1, 20*time.Millisecond)
	cb.RecordFailure()
	if cb.Allow() {
		t.Fatal("circuit must be open immediately after the failure")
	}

	time.Sleep(25 * time.Millisecond)
	if !cb.Allow() {
		t.Fatal("cooldown elapsed — the first probe must be allowed")
	}
	// The probe is abandoned: neither RecordSuccess nor RecordFailure follows.
	if cb.Allow() {
		t.Fatal("second request must be rejected while the probe is in flight")
	}

	time.Sleep(25 * time.Millisecond)
	if !cb.Allow() {
		t.Fatal("abandoned probe wedged the circuit half-open: no new probe after another cooldown")
	}
}
