package llm

import (
	"context"
	"iter"
	"net/url"
	"sync"
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

func (c *cancelAwareClient) Model() string { return "test-model" }

func (c *cancelAwareClient) Close() error { return nil }

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

func (c *scriptedStreamClient) Model() string { return "test-model" }

func (c *scriptedStreamClient) Close() error { return nil }

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
