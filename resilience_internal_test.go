package llm

// Regression tests for the v0.4.0 resilience-correctness fixes (architecture
// review findings R-H1, R-L7, R-L2). These are break-the-system tests: each
// fails without its fix.

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

// BUG 15: a negative CircuitCooldown (e.g. a computed duration gone negative)
// must be clamped, not silently disable the breaker — an open circuit with a
// negative cooldown is instantly probe-able (time.Since(...) >= negative is
// always true), so it never actually blocks.
func TestNegativeCircuitCooldownClampedSoBreakerStillBlocks(t *testing.T) {
	cfg := DefaultConfig()
	cfg.CircuitThreshold = 1
	cfg.CircuitCooldown = -30 * time.Second
	pc, err := NewPooledClient(cfg, []string{"k"}, func(AuthProfile) (Client, error) {
		return &alwaysOverloadedClient{}, nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient: %v", err)
	}
	defer pc.Close()

	// Drive the breaker directly — a real Complete backs off for the (now
	// clamped) cooldown, which would let it elapse and make this flaky.
	pc.breaker.RecordFailure() // threshold 1 → open
	if pc.breaker.State() != CircuitOpen {
		t.Fatalf("breaker not open after a failure: %v", pc.breaker.State())
	}
	// Checked immediately: a positive (clamped) cooldown must still BLOCK. An
	// unclamped negative cooldown makes time.Since(openedAt) >= cooldown true at
	// once, so Allow would wrongly admit.
	if pc.breaker.Allow() {
		t.Fatal("negative CircuitCooldown left the breaker non-blocking (cooldown not clamped)")
	}
}

// PIN (Round2/3): the v0.4.1 token-scoped breaker must be race-free under heavy
// contention across every method, and never end in a torn/invalid state. The
// -race detector is the real assertion here; the state check guards against a
// corrupted transition.
func TestCircuitBreakerConcurrentProbeLifecycleRaceFree(t *testing.T) {
	cb := NewCircuitBreaker(2, time.Millisecond)
	var wg sync.WaitGroup
	for g := 0; g < 24; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 400; i++ {
				switch (g + i) % 6 {
				case 0:
					if admitted, tok := cb.allow(); admitted {
						if i%2 == 0 {
							cb.recordSuccess(tok)
						} else {
							cb.recordFailure(tok)
						}
					}
				case 1:
					if admitted, tok := cb.allow(); admitted {
						cb.ReleaseProbe(tok)
					}
				case 2:
					cb.RecordFailure() // public wildcard
				case 3:
					cb.RecordSuccess()
				case 4:
					_ = cb.State()
					_ = cb.Allow()
				case 5:
					if i%50 == 0 {
						cb.Reset()
					}
				}
			}
		}(g)
	}
	wg.Wait()
	switch cb.State() {
	case CircuitClosed, CircuitOpen, CircuitHalfOpen:
	default:
		t.Fatalf("breaker ended in an invalid state: %v", cb.State())
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
func (c *closeSpyClient) Model() string    { return "test-model" }
func (c *closeSpyClient) Close() error     { c.closes.Add(1); return nil }

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

// PIN (Round3): RetryPolicy.Delay must never return a non-positive duration even
// under maximal jitter — a zero/negative backoff would busy-spin retries. Pins
// the post-jitter floor clamp.
func TestRetryPolicyDelayNeverNonPositive(t *testing.T) {
	rp := RetryPolicy{BaseDelay: 100 * time.Millisecond, MaxDelay: 100 * time.Millisecond, Factor: 2, Jitter: 0.99}
	for attempt := 0; attempt < 5; attempt++ {
		for i := 0; i < 1000; i++ {
			if d := rp.Delay(attempt); d <= 0 {
				t.Fatalf("Delay(%d) = %v, must stay > 0 (jitter underflow must be clamped)", attempt, d)
			}
		}
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
func (c *alwaysOverloadedClient) Model() string    { return "test-model" }
func (c *alwaysOverloadedClient) Close() error     { return nil }

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
func (c *probeTestClient) Model() string    { return "test-model" }
func (c *probeTestClient) Close() error     { return nil }

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

// The half-open probe re-arm (ReleaseProbe): a probe admitted but whose outcome
// is non-diagnostic (caller cancelled, or a client-side 400) must not strand
// other traffic for a full extra cooldown. ReleaseProbe re-admits a fresh probe
// immediately. Without the re-arm, the second Allow() below stays rejected.
func TestCircuitBreakerReleaseProbeReArmsHalfOpen(t *testing.T) {
	cb := NewCircuitBreaker(1, 50*time.Millisecond)
	cb.RecordFailure() // open
	if cb.Allow() {
		t.Fatal("circuit must be open immediately after the failure")
	}
	time.Sleep(60 * time.Millisecond)
	admitted, tok := cb.allow()
	if !admitted {
		t.Fatal("cooldown elapsed — the first probe must be admitted")
	}
	if tok == 0 {
		t.Fatal("a half-open probe admission must return a non-zero probe token")
	}
	if cb.State() != CircuitHalfOpen {
		t.Fatalf("want half-open after admitting a probe, got %v", cb.State())
	}
	if cb.Allow() {
		t.Fatal("a second request must be rejected while the probe is in flight")
	}
	// The probe's outcome was non-diagnostic — re-arm instead of recording.
	cb.ReleaseProbe(tok)
	if !cb.Allow() {
		t.Fatal("ReleaseProbe must re-admit a fresh probe immediately, not after another cooldown")
	}
	if cb.State() != CircuitHalfOpen {
		t.Fatal("still half-open until a probe actually resolves")
	}
}

// Probe identity: ReleaseProbe must re-arm ONLY the probe its token identifies.
// A stale token — from a request that never held the probe slot (admitted while
// closed → token 0) or that has been superseded by a newer probe — must be
// ignored, otherwise a second concurrent probe would be admitted while the live
// one is still in flight (violating "one probe at a time").
func TestCircuitBreakerReleaseProbeIgnoresForeignToken(t *testing.T) {
	cb := NewCircuitBreaker(1, 50*time.Millisecond)

	// A request admitted while CLOSED never holds a probe slot → token 0.
	admitted, closedTok := cb.allow()
	if !admitted || closedTok != 0 {
		t.Fatalf("closed admission: admitted=%v token=%d, want true/0", admitted, closedTok)
	}

	// Trip open, cool down, admit the real single probe.
	cb.RecordFailure()
	time.Sleep(60 * time.Millisecond)
	_, probeTok := cb.allow()
	if probeTok == 0 {
		t.Fatal("half-open probe admission must return a non-zero token")
	}
	if cb.Allow() {
		t.Fatal("a second request must be rejected while the probe is in flight")
	}

	// The closed-admitted request returns/cancels and calls ReleaseProbe with its
	// (zero) token — must NOT re-arm the live probe.
	cb.ReleaseProbe(closedTok)
	if cb.Allow() {
		t.Fatal("a foreign/zero token re-armed the live probe — would admit a 2nd concurrent probe")
	}

	// Re-arm with the real token (admits a fresh probe, bumping the generation),
	// then replay the now-superseded token — it must be ignored.
	cb.ReleaseProbe(probeTok)
	_, newTok := cb.allow()
	if newTok == probeTok {
		t.Fatal("a re-armed probe must get a fresh token/generation")
	}
	cb.ReleaseProbe(probeTok)
	if cb.Allow() {
		t.Fatal("a superseded token re-armed the current probe — would admit a 2nd concurrent probe")
	}
}

// BUG 6: a real probe token must never equal the reserved sentinels — 0
// ("admitted while closed / no probe") or anyProbe (the wildcard that bypasses
// the identity check). If probeGen wrapped onto anyProbe, that probe's recorder
// would be treated as the wildcard and could flip an unrelated circuit; a wrap
// onto 0 would make its outcome silently ignored. Force the edge and assert the
// token stays a distinct identity.
func TestCircuitBreakerProbeTokenNeverHitsReservedValues(t *testing.T) {
	cb := NewCircuitBreaker(1, 50*time.Millisecond)
	cb.RecordFailure() // open
	time.Sleep(60 * time.Millisecond)

	// Drive probeGen to the edge so the next admission would wrap onto anyProbe.
	cb.mu.Lock()
	cb.probeGen = anyProbe - 1
	cb.mu.Unlock()

	_, tok := cb.allow() // half-open admission bumps the generation
	if tok == anyProbe {
		t.Fatalf("probe token wrapped onto the anyProbe wildcard (%d) — would bypass the identity check", tok)
	}
	if tok == 0 {
		t.Fatal("probe token wrapped onto 0 — its outcome would be silently ignored")
	}
}

// Probe identity for outcome recording: a half-open probe is single-owner, so a
// stale/foreign success must not CLOSE it and a stale/foreign failure must not
// RE-OPEN it — only the request holding the current probe (or the public
// wildcard) may resolve the circuit. Without the probeGen check, a request
// admitted while the circuit was closed (token 0) that resolves late would flip
// an unrelated goroutine's probe.
func TestCircuitBreakerRecordIgnoresForeignToken(t *testing.T) {
	// recordSuccess: a foreign success must not close the live probe.
	cb := NewCircuitBreaker(1, 50*time.Millisecond)
	_, closedTok := cb.allow() // admitted while closed → token 0
	if closedTok != 0 {
		t.Fatalf("closed admission token = %d, want 0", closedTok)
	}
	cb.RecordFailure() // open (threshold 1)
	time.Sleep(60 * time.Millisecond)
	_, probeTok := cb.allow() // the half-open probe
	if cb.State() != CircuitHalfOpen {
		t.Fatalf("want half-open after admitting the probe, got %v", cb.State())
	}
	cb.recordSuccess(closedTok) // the closed-admitted request resolves late
	if cb.State() != CircuitHalfOpen {
		t.Fatalf("a foreign success closed the live probe: state=%v", cb.State())
	}
	cb.recordSuccess(probeTok) // the probe's own success
	if cb.State() != CircuitClosed {
		t.Fatalf("the probe's own success did not close the circuit: state=%v", cb.State())
	}

	// recordFailure: a foreign failure must not re-open the live probe.
	cb2 := NewCircuitBreaker(1, 50*time.Millisecond)
	_, closedTok2 := cb2.allow() // token 0
	cb2.RecordFailure()          // open
	time.Sleep(60 * time.Millisecond)
	_, probeTok2 := cb2.allow() // the half-open probe
	cb2.recordFailure(closedTok2)
	if cb2.State() != CircuitHalfOpen {
		t.Fatalf("a foreign failure changed the live probe: state=%v", cb2.State())
	}
	cb2.recordFailure(probeTok2) // the probe's own failure
	if cb2.State() != CircuitOpen {
		t.Fatalf("the probe's own failure did not re-open the circuit: state=%v", cb2.State())
	}
}
