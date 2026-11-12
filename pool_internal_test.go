package llm

import (
	"context"
	"errors"
	"iter"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// closeSpy is a minimal Client that tracks whether Close was called.
type closeSpy struct {
	closed      atomic.Bool
	provider    string
	model       string
	completeErr error                                             // if set, Complete returns this error
	blockCh     chan struct{}                                     // if set, Complete blocks until closed
	completeFn  func(context.Context, Request) (*Response, error) // optional override
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

func (c *closeSpy) Model() string { return c.model }

func (c *closeSpy) Close() error { c.closed.Store(true); return nil }

func TestGetAvailableExpiredCooldownReturnsProfile(t *testing.T) {
	// A profile with IsHealthy=true and Cooldown in the past (or zero)
	// should be available, not reported as "in cooldown".
	pool := &AuthPool{
		profiles: []*AuthProfile{
			{
				Name:      "key-1",
				APIKey:    "k1",
				IsHealthy: true,
				Cooldown:  time.Now().Add(-5 * time.Second), // expired cooldown
			},
		},
	}

	profile, err := pool.GetAvailable()
	if err != nil {
		t.Fatalf("GetAvailable() returned error for expired cooldown: %v", err)
	}
	if profile.Name != "key-1" {
		t.Fatalf("got profile %q, want key-1", profile.Name)
	}
}

func TestGetAvailableMixedHealthyDisabledReturnsCooldownWait(t *testing.T) {
	// One permanently disabled profile, one healthy profile in future cooldown.
	// Should return CooldownError with positive wait time, NOT a garbage value.
	futureTime := time.Now().Add(30 * time.Second)
	pool := &AuthPool{
		profiles: []*AuthProfile{
			{
				Name:      "disabled",
				APIKey:    "k1",
				IsHealthy: false, // permanently disabled
			},
			{
				Name:      "cooling",
				APIKey:    "k2",
				IsHealthy: true,
				Cooldown:  futureTime, // 30s from now
			},
		},
	}

	_, err := pool.GetAvailable()
	if err == nil {
		t.Fatal("expected CooldownError, got nil")
	}

	var cooldownErr *CooldownError
	if !errors.As(err, &cooldownErr) {
		t.Fatalf("expected *CooldownError, got %T: %v", err, err)
	}

	// Wait time should be ~30s, definitely positive and less than 60s
	if cooldownErr.Wait <= 0 {
		t.Errorf("CooldownError.Wait = %v, want positive", cooldownErr.Wait)
	}
	if cooldownErr.Wait > 60*time.Second {
		t.Errorf("CooldownError.Wait = %v, want <= 60s (got garbage value)", cooldownErr.Wait)
	}
}

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

// closeSpyClient counts how many times its Close is invoked, and blocks each
// Complete on a gate so a test can hold N requests in-flight (refs held) while
// Close races them.
type closeSpyClient struct {
	closes atomic.Int32
	ready  *sync.WaitGroup
	gate   chan struct{}
}

func (c *closeSpyClient) Complete(_ context.Context, _ Request) (*Response, error) {
	c.ready.Done()
	<-c.gate
	return &Response{Content: "ok"}, nil
}

func (c *closeSpyClient) Stream(_ context.Context, _ Request) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		c.ready.Done()
		<-c.gate
		yield(StreamEvent{Type: EventDone, StopReason: "end_turn"}, nil)
	}
}

func (c *closeSpyClient) Provider() string { return "test" }

func (c *closeSpyClient) Model() string { return "test-model" }

func (c *closeSpyClient) Close() error { c.closes.Add(1); return nil }

// PIN (Round2): refcount exactness. With N requests in-flight (each holding a
// ref) when Close races them — including concurrent double-Close — the
// underlying client's Close must fire EXACTLY once, never zero (leak) and never
// twice (double-close). Negative assertion under -race.
func TestPoolCloseFiresUnderlyingCloseExactlyOnce(t *testing.T) {
	const n = 8
	var ready sync.WaitGroup
	ready.Add(n)
	spy := &closeSpyClient{ready: &ready, gate: make(chan struct{})}

	pc, err := NewPooledClient(DefaultConfig(), []string{"k"}, func(AuthProfile) (Client, error) {
		return spy, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient: %v", err)
	}

	var inflight sync.WaitGroup
	for i := 0; i < n; i++ {
		inflight.Add(1)
		go func() { defer inflight.Done(); _, _ = pc.Complete(context.Background(), Request{}) }()
	}
	ready.Wait() // all n requests are in-flight, each holding a ref

	var closers sync.WaitGroup
	for i := 0; i < 3; i++ { // concurrent double(triple)-Close
		closers.Add(1)
		go func() { defer closers.Done(); _ = pc.Close() }()
	}
	closers.Wait()
	close(spy.gate) // release the in-flight requests; the last ref drop triggers Close
	inflight.Wait()

	if got := spy.closes.Load(); got != 1 {
		t.Fatalf("underlying client Close fired %d times, want exactly 1 (0=leak, >1=double-close)", got)
	}
}

// alwaysOverloadedClient always returns a retryable 503, ignoring its context —
// it forces the pool to rotate (Complete) or treat the failure as retryable.
type alwaysOverloadedClient struct{}

func (c *alwaysOverloadedClient) Complete(_ context.Context, _ Request) (*Response, error) {
	return nil, NewOverloadedError("test", "503 always")
}

func (c *alwaysOverloadedClient) Stream(_ context.Context, _ Request) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		yield(StreamEvent{}, NewOverloadedError("test", "503 always"))
	}
}

func (c *alwaysOverloadedClient) Provider() string { return "test" }

func (c *alwaysOverloadedClient) Model() string { return "test-model" }

func (c *alwaysOverloadedClient) Close() error { return nil }

// probeTestClient runs a swappable per-call behavior, so a test can open the
// breaker and then control what the half-open probe and the following request
// each return. Safe for concurrent use.
type probeTestClient struct {
	fn atomic.Pointer[func(context.Context) (*Response, error)]
}

func TestPooledClientDisableRetriesMakesOneProviderAttempt(t *testing.T) {
	var calls atomic.Int32
	client := newProbeTestClient(func(context.Context) (*Response, error) {
		calls.Add(1)
		return nil, NewOverloadedError("test", "503")
	})
	cfg := DefaultConfig()
	cfg.DisableRetries = true
	pc, err := NewPooledClient(cfg, []string{"key-a"}, func(AuthProfile) (Client, error) {
		return client, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer pc.Close()
	if _, err = pc.Complete(context.Background(), Request{}); err == nil {
		t.Fatal("Complete unexpectedly succeeded")
	}
	if calls.Load() != 1 {
		t.Fatalf("provider calls=%d, want exactly one", calls.Load())
	}
}

func newProbeTestClient(f func(context.Context) (*Response, error)) *probeTestClient {
	c := &probeTestClient{}
	c.fn.Store(&f)
	return c
}

func (c *probeTestClient) set(f func(context.Context) (*Response, error)) { c.fn.Store(&f) }

func (c *probeTestClient) Complete(ctx context.Context, _ Request) (*Response, error) {
	return (*c.fn.Load())(ctx)
}

func (c *probeTestClient) Stream(ctx context.Context, _ Request) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		resp, err := (*c.fn.Load())(ctx)
		if err != nil {
			yield(StreamEvent{}, err)
			return
		}
		yield(StreamEvent{Type: EventDone, StopReason: "end_turn"}, nil)
		_ = resp
	}
}

func (c *probeTestClient) Provider() string { return "test" }

func (c *probeTestClient) Model() string { return "test-model" }

func (c *probeTestClient) Close() error { return nil }

// halfOpenPool builds a single-key PooledClient whose breaker is OPEN and has
// cooled, so the NEXT Complete/Stream is admitted as a half-open probe. It
// returns the pool and its shared client (swap the client's behavior with .set
// to control the probe and the following request). Key cooldowns are tiny so
// key health doesn't interfere; the breaker cooldown is 40ms.
func halfOpenPool(t *testing.T) (*PooledClient, *probeTestClient) {
	t.Helper()
	cfg := DefaultConfig()
	cfg.CircuitThreshold = 1
	cfg.CircuitCooldown = 40 * time.Millisecond
	cfg.CooldownOverload = time.Millisecond
	cfg.CooldownDefault = time.Millisecond
	cfg.CooldownRateLimit = time.Millisecond
	cfg.MaxRetries = 2

	client := newProbeTestClient(func(context.Context) (*Response, error) {
		return nil, NewOverloadedError("test", "503") // trips the breaker
	})
	pc, err := NewPooledClient(cfg, []string{"key-a"}, func(AuthProfile) (Client, error) {
		return client, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient: %v", err)
	}
	_, _ = pc.Complete(context.Background(), Request{}) // one failure opens it (threshold 1)
	if pc.breaker.State() != CircuitOpen {
		t.Fatalf("setup: breaker not open, got %v", pc.breaker.State())
	}
	time.Sleep(50 * time.Millisecond) // let the breaker cooldown elapse
	return pc, client
}

func isCircuitOpenErr(err error) bool {
	return err != nil && (errors.Is(err, ErrCircuitOpen) || strings.Contains(err.Error(), "circuit breaker open"))
}

// cancelURLError is the *url.Error net/http produces when the caller's context
// is cancelled (looks like a net.Error to a naive classifier).
func cancelURLError(ctx context.Context) error {
	return &url.Error{Op: "Post", URL: "https://api.example.com/v1", Err: ctx.Err()}
}

// R2: the Complete caller-cancel ReleaseProbe wiring — a cancelled half-open
// probe must re-arm so the NEXT request is admitted, not rejected for an extra
// breaker cooldown. Without the pc.breaker.ReleaseProbe call this goes red
// (the follow-up Complete is rejected with a circuit-open error).
func TestPoolCompleteCancelReArmsHalfOpenProbe(t *testing.T) {
	pc, client := halfOpenPool(t)
	defer pc.Close()

	client.set(func(ctx context.Context) (*Response, error) { return nil, cancelURLError(ctx) })
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, _ = pc.Complete(ctx, Request{}) // the half-open probe, caller-cancelled

	client.set(func(context.Context) (*Response, error) { return &Response{Content: "ok"}, nil })
	_, err := pc.Complete(context.Background(), Request{})
	if isCircuitOpenErr(err) {
		t.Fatalf("cancelled half-open probe was not re-armed: next request rejected: %v", err)
	}
	if err != nil {
		t.Fatalf("unexpected error after a re-armed probe: %v", err)
	}
}

// R2: the Complete non-retryable-non-auth (e.g. 400) ReleaseProbe wiring — a
// reachable endpoint returning a client-side error is not a breaker failure, so
// it must re-arm the probe rather than wedge the circuit for an extra cooldown.
func TestPoolCompleteNonRetryableReArmsHalfOpenProbe(t *testing.T) {
	pc, client := halfOpenPool(t)
	defer pc.Close()

	client.set(func(context.Context) (*Response, error) {
		return nil, &APIError{StatusCode: 400, Message: "bad request", Provider: "test", Retryable: false}
	})
	_, _ = pc.Complete(context.Background(), Request{}) // the half-open probe, 400

	client.set(func(context.Context) (*Response, error) { return &Response{Content: "ok"}, nil })
	_, err := pc.Complete(context.Background(), Request{})
	if isCircuitOpenErr(err) {
		t.Fatalf("400 half-open probe was not re-armed: next request rejected: %v", err)
	}
	if err != nil {
		t.Fatalf("unexpected error after a re-armed probe: %v", err)
	}
}

// R2: the Stream caller-cancel ReleaseProbe wiring — same invariant on the
// streaming pre-data path.
func TestPoolStreamCancelReArmsHalfOpenProbe(t *testing.T) {
	pc, client := halfOpenPool(t)
	defer pc.Close()

	client.set(func(ctx context.Context) (*Response, error) { return nil, cancelURLError(ctx) })
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	for _, err := range pc.Stream(ctx, Request{}) { // the half-open probe, cancelled
		_ = err
	}

	client.set(func(context.Context) (*Response, error) { return &Response{Content: "ok"}, nil })
	var gotErr error
	for _, err := range pc.Stream(context.Background(), Request{}) {
		if err != nil {
			gotErr = err
		}
	}
	if isCircuitOpenErr(gotErr) {
		t.Fatalf("cancelled half-open stream probe was not re-armed: %v", gotErr)
	}
}

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

// Close() concurrent with an in-flight Complete that is mid-rotation must not
// panic (Release on a nil refCountedClient) or resurrect/leak a client. The
// factory deterministically signals when a rotation is in flight (its 2nd call,
// which happens between GetAvailable and the rc swap) and holds that window open
// so Close races the swap every iteration. Without the closed-check in
// rotateClient, `old := pc.rc` reads nil and old.Release() panics.
func TestCloseDuringRotationDoesNotPanic(t *testing.T) {
	// Millisecond cooldowns + a bounded context so that an iteration where Close
	// loses the 300us window (rotation completes, both keys then 503 into
	// cooldown) backs off in ms and the context expires fast — instead of the
	// Complete goroutine sleeping a 30s default cooldown on a Background context.
	cfg := DefaultConfig()
	cfg.CooldownOverload = time.Millisecond
	cfg.CooldownDefault = time.Millisecond
	cfg.CooldownRateLimit = time.Millisecond

	for it := 0; it < 100; it++ {
		var calls atomic.Int32
		inRotate := make(chan struct{})
		factory := func(AuthProfile) (Client, error) {
			// Call 1 is NewPooledClient's initial client; call 2 is the first
			// rotation — we're now inside rotateClient, before the rc swap.
			if calls.Add(1) == 2 {
				close(inRotate)
				time.Sleep(300 * time.Microsecond) // hold the rotate window open
			}
			return &alwaysOverloadedClient{}, nil
		}
		pc, err := NewPooledClient(cfg, []string{"key-a", "key-b"}, factory)
		if err != nil {
			t.Fatalf("NewPooledClient: %v", err)
		}

		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			_, _ = pc.Complete(ctx, Request{}) // forces a rotation
		}()
		go func() {
			defer wg.Done()
			<-inRotate     // wait until a rotation is mid-flight
			_ = pc.Close() // race the rc swap
		}()
		wg.Wait()
		cancel()

		// After Close races the rotation, the pool must be truly closed: rc nil,
		// nothing resurrected. Without the closed-check, rotateClient sets
		// pc.rc = newRefCountedClient(...) after Close already niled it — a client
		// that nothing will ever Close. (This catches the bug even when the
		// nil-safe Release suppresses the would-be panic.)
		pc.mu.Lock()
		leaked := pc.rc != nil
		pc.mu.Unlock()
		if leaked {
			t.Fatalf("iter %d: rotateClient resurrected rc after Close — leaked client", it)
		}
	}
}

// #14: isolates the pool.go `if ctx.Err() != nil` early-return from the
// IsRetryable(context.Canceled)==false classification. The client returns a
// *retryable* 503 (not context.Canceled) while the caller's context is already
// cancelled. IsRetryable(503)==true, so only the pool's early-return keeps this
// from recording a breaker failure / cooling the key. Remove that early-return
// and the breaker trips — proving this test pins it.
func TestCompleteCancelledContextWithRetryableErrorDoesNotPoisonPool(t *testing.T) {
	cfg := DefaultConfig()
	cfg.CircuitThreshold = 1
	pc, err := NewPooledClient(cfg, []string{"key-a"}, func(AuthProfile) (Client, error) {
		return &alwaysOverloadedClient{}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient: %v", err)
	}
	defer pc.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err = pc.Complete(ctx, Request{}); err == nil {
		t.Fatal("want error from cancelled context")
	}
	available, state := poolHealth(pc)
	if state != CircuitClosed {
		t.Fatalf("retryable error under a cancelled context tripped the breaker: state=%v", state)
	}
	if !available {
		t.Fatal("retryable error under a cancelled context put the key in cooldown")
	}
}

// A caller whose loop body panics inside the stream: the in-flight reference
// was never released, so the client was never closed (B18).
func TestStreamReleasesTheClientWhenTheCallerPanics(t *testing.T) {
	var spies []*closeSpyClient
	pc, err := NewPooledClient(Config{Provider: "test"}, []string{"a"}, func(AuthProfile) (Client, error) {
		var ready sync.WaitGroup
		ready.Add(1 << 20) // never waited on
		gate := make(chan struct{})
		close(gate)
		spy := &closeSpyClient{ready: &ready, gate: gate}
		spies = append(spies, spy)
		return spy, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	func() {
		defer func() { _ = recover() }()
		for range pc.Stream(context.Background(), Request{}) {
			panic("caller bug")
		}
	}()
	_ = pc.Close()
	if n := spies[0].closes.Load(); n != 1 {
		t.Fatalf("client closed %d times after a panicking caller and Close, want 1", n)
	}
}
