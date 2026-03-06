package llm

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ErrNoAvailableProfiles is returned when all profiles are in cooldown.
var ErrNoAvailableProfiles = errors.New("no available auth profiles (all in cooldown)")

// ErrClientClosed is returned when Complete or Stream is called after Close.
var ErrClientClosed = errors.New("llm: client is closed")

// CooldownError indicates all profiles are in cooldown and provides the wait time.
type CooldownError struct {
	Wait time.Duration
}

func (e *CooldownError) Error() string {
	return fmt.Sprintf("%v: next available in %v", ErrNoAvailableProfiles.Error(), e.Wait.Round(time.Second))
}

func (e *CooldownError) Unwrap() error {
	return ErrNoAvailableProfiles
}

// AuthPool manages a pool of auth profiles with rotation on failure.
type AuthPool struct {
	provider string
	profiles []*AuthProfile
	current  int
	mu       sync.RWMutex
}

// NewAuthPool creates a new auth pool from config.
func NewAuthPool(provider string, keys []string) *AuthPool {
	profiles := make([]*AuthProfile, len(keys))
	for i, key := range keys {
		api := key
		var base string
		if idx := strings.Index(key, "|"); idx >= 0 {
			api = key[:idx]
			base = key[idx+1:]
		}

		profiles[i] = &AuthProfile{
			Name:      fmt.Sprintf("%s-%d", provider, i+1),
			APIKey:    api,
			BaseURL:   base,
			IsHealthy: true,
		}
	}
	return &AuthPool{
		provider: provider,
		profiles: profiles,
		current:  0,
	}
}

// GetCurrent returns a snapshot of the current active profile.
// Returns by value to prevent data races — callers cannot mutate pool state.
func (p *AuthPool) GetCurrent() (AuthProfile, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if len(p.profiles) == 0 {
		return AuthProfile{}, false
	}
	return *p.profiles[p.current], true
}

// GetAvailable returns a snapshot of the first available profile, rotating if needed.
// Returns by value to prevent data races — callers cannot mutate pool state.
func (p *AuthPool) GetAvailable() (AuthProfile, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if len(p.profiles) == 0 {
		return AuthProfile{}, ErrNoAvailableProfiles
	}

	// Try current first
	if p.profiles[p.current].IsAvailable() {
		p.profiles[p.current].MarkUsed()
		return *p.profiles[p.current], nil
	}

	// Rotate through all profiles
	start := p.current
	for {
		p.current = (p.current + 1) % len(p.profiles)
		if p.profiles[p.current].IsAvailable() {
			slog.Info("auth pool rotated", "profile", p.profiles[p.current].Name)
			p.profiles[p.current].MarkUsed()
			return *p.profiles[p.current], nil
		}
		if p.current == start {
			break
		}
	}

	// All profiles in cooldown - find soonest available
	var soonestTime time.Time
	for _, profile := range p.profiles {
		if !profile.IsHealthy {
			continue // Skip permanently disabled profiles
		}
		if soonestTime.IsZero() || profile.Cooldown.Before(soonestTime) {
			soonestTime = profile.Cooldown
		}
	}

	if soonestTime.IsZero() {
		return AuthProfile{}, fmt.Errorf("%w: all keys permanently disabled", ErrNoAvailableProfiles)
	}

	waitTime := time.Until(soonestTime)
	return AuthProfile{}, &CooldownError{Wait: waitTime}
}

// MarkFailedByName marks a specific profile as failed.
// Auth errors (401/403) permanently disable the key. Other errors apply a temporary cooldown
// using the default durations (60s rate limit, 30s overload, 10s other).
func (p *AuthPool) MarkFailedByName(name string, err error) {
	p.MarkFailedByNameWithCooldown(name, err, DefaultCooldownRateLimit, DefaultCooldownOverload, DefaultCooldownDefault)
}

// MarkFailedByNameWithCooldown marks a specific profile as failed with configurable cooldowns.
// Auth errors (401/403) permanently disable the key. Other errors apply the appropriate cooldown.
func (p *AuthPool) MarkFailedByNameWithCooldown(name string, err error, rateLimitCD, overloadCD, defaultCD time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()

	var profile *AuthProfile
	for _, prof := range p.profiles {
		if prof.Name == name {
			profile = prof
			break
		}
	}
	if profile == nil {
		return
	}

	// Auth errors permanently disable the key — it's revoked, not temporarily overloaded
	if IsAuthError(err) {
		profile.IsHealthy = false
		profile.LastError = err.Error()
		slog.Warn("profile permanently disabled (auth error)", "profile", profile.Name, "error", err)
		return
	}

	var cooldown time.Duration
	// Cooldown based on error type
	if IsRateLimited(err) {
		cooldown = rateLimitCD
		slog.Warn("profile rate limited", "profile", profile.Name, "cooldown", cooldown)
	} else if IsOverloaded(err) {
		cooldown = overloadCD
		slog.Warn("profile overloaded", "profile", profile.Name, "cooldown", cooldown)
	} else {
		cooldown = defaultCD
		slog.Warn("profile failed", "profile", profile.Name, "error", err, "cooldown", cooldown)
	}

	profile.MarkFailed(err, cooldown)
}

// MarkSuccessByName marks a specific profile as successful.
func (p *AuthPool) MarkSuccessByName(name string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	for _, prof := range p.profiles {
		if prof.Name == name {
			prof.MarkHealthy()
			break
		}
	}
}

// Count returns the number of profiles.
func (p *AuthPool) Count() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return len(p.profiles)
}

// HealthyCount returns the number of profiles that have not been
// permanently disabled (e.g. by auth errors). Profiles in temporary
// cooldown are still counted as healthy.
func (p *AuthPool) HealthyCount() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	n := 0
	for _, prof := range p.profiles {
		if prof.IsHealthy {
			n++
		}
	}
	return n
}

// Status returns a status string for all profiles.
func (p *AuthPool) Status() string {
	p.mu.RLock()
	defer p.mu.RUnlock()

	var parts []string
	for i, profile := range p.profiles {
		status := "ok"
		if !profile.IsHealthy {
			status = "unhealthy"
		} else if !profile.Cooldown.IsZero() && time.Now().Before(profile.Cooldown) {
			status = fmt.Sprintf("cooldown %v", time.Until(profile.Cooldown).Round(time.Second))
		}
		marker := ""
		if i == p.current {
			marker = "*"
		}
		parts = append(parts, fmt.Sprintf("%s%s:%s", marker, profile.Name, status))
	}
	return strings.Join(parts, ", ")
}

// refCountedClient wraps a Client with atomic reference counting.
// Born with refs=1 (the pool's own reference). Each in-flight request
// calls Acquire/Release. When the count drops to zero, Close fires
// exactly once via sync.Once.
type refCountedClient struct {
	client Client
	refs   atomic.Int64
	once   sync.Once
}

func newRefCountedClient(client Client) *refCountedClient {
	rc := &refCountedClient{client: client}
	rc.refs.Store(1)
	return rc
}

// Acquire increments the reference count. Must be called inside pc.mu.RLock().
func (rc *refCountedClient) Acquire() {
	rc.refs.Add(1)
}

// Release decrements the reference count. When it reaches zero, the
// underlying client is closed exactly once. Close errors are logged
// because there is no caller to return them to.
func (rc *refCountedClient) Release() {
	if rc.refs.Add(-1) == 0 {
		rc.once.Do(func() {
			if err := rc.client.Close(); err != nil {
				slog.Warn("ref-counted client close error", "error", err)
			}
		})
	}
}

// PooledClient wraps a Client with auth profile rotation.
type PooledClient struct {
	pool       *AuthPool
	clientFunc func(profile AuthProfile) (Client, error)
	rc         *refCountedClient
	activeName string // Name of the profile currently active
	cfg        Config
	breaker    *CircuitBreaker // nil when CircuitThreshold == 0
	retryHook  RetryHook       // nil when no hook configured
	mu         sync.RWMutex
	rotateMu   sync.Mutex // Serializes rotation to prevent thundering herd
}

// retryPolicy returns the configured RetryPolicy or the default.
func (pc *PooledClient) retryPolicy() RetryPolicy {
	if pc.cfg.RetryPolicy != nil {
		return *pc.cfg.RetryPolicy
	}
	return DefaultRetryPolicy
}

// cooldownForError returns rate-limit, overload, or default cooldown
// based on the error type, using config values with defaults.
func (pc *PooledClient) cooldownForError() (rateLimitCD, overloadCD, defaultCD time.Duration) {
	rateLimitCD = pc.cfg.CooldownRateLimit
	if rateLimitCD == 0 {
		rateLimitCD = DefaultCooldownRateLimit
	}
	overloadCD = pc.cfg.CooldownOverload
	if overloadCD == 0 {
		overloadCD = DefaultCooldownOverload
	}
	defaultCD = pc.cfg.CooldownDefault
	if defaultCD == 0 {
		defaultCD = DefaultCooldownDefault
	}
	return
}

// markFailed marks the named profile as failed with configured cooldowns.
func (pc *PooledClient) markFailed(name string, err error) {
	rl, ol, df := pc.cooldownForError()
	pc.pool.MarkFailedByNameWithCooldown(name, err, rl, ol, df)
}

// emitRetryEvent fires the retry hook if configured.
func (pc *PooledClient) emitRetryEvent(evt RetryEvent) {
	if pc.retryHook != nil {
		evt.Provider = pc.cfg.Provider
		pc.retryHook(evt)
	}
}

// NewPooledClient creates a new pooled client with auth rotation.
// If cfg.CircuitThreshold > 0, a circuit breaker is automatically created.
// If cfg.RetryHook is set, it is wired into the retry loop.
func NewPooledClient(cfg Config, keys []string, clientFunc func(profile AuthProfile) (Client, error)) (*PooledClient, error) {
	if len(keys) == 0 {
		return nil, fmt.Errorf("no API keys provided for provider %s", cfg.Provider)
	}

	pool := NewAuthPool(cfg.Provider, keys)

	// Create initial client with first available profile
	profile, err := pool.GetAvailable()
	if err != nil {
		return nil, err
	}

	client, err := clientFunc(profile)
	if err != nil {
		return nil, err
	}

	slog.Info("pooled client created", "profiles", pool.Count(), "provider", cfg.Provider)

	pc := &PooledClient{
		pool:       pool,
		clientFunc: clientFunc,
		rc:         newRefCountedClient(client),
		activeName: profile.Name,
		cfg:        cfg,
		retryHook:  cfg.RetryHook,
	}

	// Wire circuit breaker from config
	if cfg.CircuitThreshold > 0 {
		cooldown := cfg.CircuitCooldown
		if cooldown == 0 {
			cooldown = DefaultCircuitCooldown
		}
		pc.breaker = NewCircuitBreaker(cfg.CircuitThreshold, cooldown,
			WithSuccessPredicate(func(err error) bool {
				return IsAuthError(err)
			}),
		)
	}

	return pc, nil
}

// Complete implements Client.Complete with retry/rotation and exponential backoff.
func (pc *PooledClient) Complete(ctx context.Context, req Request) (*Response, error) {
	maxRetries := pc.pool.HealthyCount()
	if maxRetries < 3 {
		maxRetries = 3 // Minimum 3 retries for single-key resilience
	}
	if maxRetries > maxRetryAttempts {
		maxRetries = maxRetryAttempts // Prevent pathological retry storms with large key pools
	}

	rp := pc.retryPolicy()

	var lastErr error
	for i := 0; i < maxRetries; i++ {
		// Circuit breaker gate: if open, fail fast. The circuit's own cooldown
		// timer controls when the next probe is allowed — burning retry
		// iterations with backoff delays accomplishes nothing.
		if !pc.breaker.Allow() {
			pc.emitRetryEvent(RetryEvent{Type: RetryCircuitOpen, Attempt: i})
			if lastErr != nil {
				return nil, fmt.Errorf("circuit breaker open: %w", lastErr)
			}
			return nil, fmt.Errorf("circuit breaker open: %w", ErrCircuitOpen)
		}

		pc.mu.RLock()
		rc := pc.rc
		usedName := pc.activeName
		if rc == nil {
			pc.mu.RUnlock()
			return nil, ErrClientClosed
		}
		rc.Acquire()
		pc.mu.RUnlock()

		resp, err := rc.client.Complete(ctx, req)
		rc.Release()
		if err == nil {
			pc.pool.MarkSuccessByName(usedName)
			pc.breaker.RecordSuccess()
			return resp, nil
		}

		lastErr = err
		pc.emitRetryEvent(RetryEvent{Type: RetryAttemptFailed, Attempt: i, Err: err})

		// Auth errors and retryable errors trigger rotation to try another key.
		// Other errors (400 bad request, etc.) are not key-related — return immediately.
		if !IsRetryable(err) && !IsAuthError(err) {
			return nil, err
		}

		// Auth errors do NOT trip the circuit — bad key ≠ broken endpoint
		if !IsAuthError(err) {
			pc.breaker.RecordFailure()
		}

		// Mark failed (auth errors get permanently disabled, others get cooldown)
		pc.markFailed(usedName, err)

		if rotErr := pc.rotateClient(usedName); rotErr != nil {
			// Rotation failed (all keys in cooldown or single-key pool).
			// Auth errors are permanent — no point retrying with the same dead key.
			if IsAuthError(err) {
				slog.Warn("auth error with no healthy keys, giving up", "error", err)
				return nil, err
			}

			// Transient errors (429, 503) — backoff and retry with same client.
			var cooldownErr *CooldownError
			var backoff time.Duration
			if errors.As(rotErr, &cooldownErr) {
				backoff = cooldownErr.Wait
				if backoff <= 0 {
					backoff = time.Second
				}
			} else {
				backoff = rp.Delay(i)
			}
			pc.emitRetryEvent(RetryEvent{Type: RetryBackingOff, Attempt: i, Err: rotErr, Backoff: backoff})
			slog.Info("rotation failed, backing off", "attempt", i+1, "backoff", backoff, "error", rotErr)

			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
				// Continue to next iteration with same client
			}
		} else {
			pc.emitRetryEvent(RetryEvent{Type: RetryRotating, Attempt: i, Err: err})
			slog.Info("retrying with new profile", "attempt", i+2, "max", maxRetries)
		}
	}

	pc.emitRetryEvent(RetryEvent{Type: RetryExhausted, Err: lastErr})
	return nil, fmt.Errorf("all retries exhausted: %w", lastErr)
}

// Stream implements Client.Stream with pre-data retry and exponential backoff.
// If the stream fails before any events are yielded to the caller (e.g.,
// 429/503 on the initial HTTP connection), the error is retryable and the
// pool rotates to the next profile — identical to Complete's retry logic.
// Once any event has been yielded, retrying would duplicate content, so
// mid-stream errors are passed through immediately.
func (pc *PooledClient) Stream(ctx context.Context, req Request) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		maxRetries := pc.pool.HealthyCount()
		if maxRetries < 3 {
			maxRetries = 3 // Minimum 3 retries for single-key resilience
		}
		if maxRetries > maxRetryAttempts {
			maxRetries = maxRetryAttempts // Prevent pathological retry storms with large key pools
		}

		rp := pc.retryPolicy()

		var lastErr error
		for attempt := 0; attempt < maxRetries; attempt++ {
			// Circuit breaker gate: if open, fail fast (see Complete for rationale).
			if !pc.breaker.Allow() {
				pc.emitRetryEvent(RetryEvent{Type: RetryCircuitOpen, Attempt: attempt})
				if lastErr != nil {
					yield(StreamEvent{}, fmt.Errorf("stream: circuit breaker open: %w", lastErr))
				} else {
					yield(StreamEvent{}, fmt.Errorf("stream: circuit breaker open: %w", ErrCircuitOpen))
				}
				return
			}

			pc.mu.RLock()
			rc := pc.rc
			usedName := pc.activeName
			if rc == nil {
				pc.mu.RUnlock()
				yield(StreamEvent{}, ErrClientClosed)
				return
			}
			rc.Acquire()
			pc.mu.RUnlock()

			firstEvent := true
			retryable := false

			for event, err := range rc.client.Stream(ctx, req) {
				if err != nil {
					if firstEvent && (IsRetryable(err) || IsAuthError(err)) {
						// Connection failed before any data — safe to retry
						lastErr = err
						retryable = true
						break
					}
					// Mid-stream error or non-retryable — pass through
					if IsRetryable(err) || IsAuthError(err) {
						pc.markFailed(usedName, err)
					}
					rc.Release()
					yield(StreamEvent{}, err)
					return
				}
				firstEvent = false
				if !yield(event, nil) {
					pc.pool.MarkSuccessByName(usedName)
					pc.breaker.RecordSuccess()
					rc.Release()
					return
				}
			}

			rc.Release()

			if !retryable {
				// Stream completed normally
				pc.pool.MarkSuccessByName(usedName)
				pc.breaker.RecordSuccess()
				return
			}

			pc.emitRetryEvent(RetryEvent{Type: RetryAttemptFailed, Attempt: attempt, Err: lastErr})

			// Auth errors do NOT trip the circuit — bad key ≠ broken endpoint
			if !IsAuthError(lastErr) {
				pc.breaker.RecordFailure()
			}

			// Pre-data retryable error — rotate and retry
			pc.markFailed(usedName, lastErr)
			if rotErr := pc.rotateClient(usedName); rotErr != nil {
				// Rotation failed (all keys in cooldown or single-key pool).
				// Auth errors are permanent — no point retrying with the same dead key.
				if IsAuthError(lastErr) {
					slog.Warn("stream: auth error with no healthy keys, giving up", "error", lastErr)
					yield(StreamEvent{}, lastErr)
					return
				}

				// Transient errors (429, 503) — backoff and retry with same client.
				var cooldownErr *CooldownError
				var backoff time.Duration
				if errors.As(rotErr, &cooldownErr) {
					backoff = cooldownErr.Wait
					if backoff <= 0 {
						backoff = time.Second
					}
				} else {
					backoff = rp.Delay(attempt)
				}
				pc.emitRetryEvent(RetryEvent{Type: RetryBackingOff, Attempt: attempt, Err: rotErr, Backoff: backoff})
				slog.Info("stream: rotation failed, backing off", "attempt", attempt+1, "backoff", backoff, "error", rotErr)

				select {
				case <-ctx.Done():
					yield(StreamEvent{}, ctx.Err())
					return
				case <-time.After(backoff):
					// Continue to next iteration with same client
				}
			} else {
				pc.emitRetryEvent(RetryEvent{Type: RetryRotating, Attempt: attempt, Err: lastErr})
				slog.Info("stream: retrying with new profile", "attempt", attempt+2, "max", maxRetries)
			}
		}

		pc.emitRetryEvent(RetryEvent{Type: RetryExhausted, Err: lastErr})
		yield(StreamEvent{}, fmt.Errorf("stream: all retries exhausted: %w", lastErr))
	}
}

// rotateClient creates a new client with the next available profile.
//
// Uses double-checked locking to prevent thundering herd: rotateMu serializes
// rotation attempts, and the name check inside ensures only one goroutine
// actually rotates while others short-circuit to use the new client.
//
// The old client's reference is released after the swap. If in-flight
// requests still hold references, the actual Close is deferred until
// the last one finishes (via refCountedClient).
func (pc *PooledClient) rotateClient(failedName string) error {
	pc.rotateMu.Lock()
	defer pc.rotateMu.Unlock()

	// Double-check inside the lock: if another goroutine already rotated,
	// the active profile will differ from the failed one.
	pc.mu.RLock()
	currentName := pc.activeName
	pc.mu.RUnlock()

	if currentName != failedName {
		// Another goroutine already rotated — use their client
		return nil
	}

	profile, err := pc.pool.GetAvailable()
	if err != nil {
		return err
	}

	newClient, err := pc.clientFunc(profile)
	if err != nil {
		return err
	}

	pc.mu.Lock()
	old := pc.rc
	pc.rc = newRefCountedClient(newClient)
	pc.activeName = profile.Name
	pc.mu.Unlock()

	old.Release() // outside lock — avoids holding mu during potential I/O

	return nil
}

// Provider implements Client.Provider.
// Acquires a ref to prevent calling Provider() on a concurrently-closed client.
func (pc *PooledClient) Provider() string {
	pc.mu.RLock()
	rc := pc.rc
	if rc == nil {
		pc.mu.RUnlock()
		return pc.cfg.Provider
	}
	rc.Acquire()
	pc.mu.RUnlock()

	s := rc.client.Provider()
	rc.Release()
	return s
}

// Model implements Client.Model.
// Acquires a ref to prevent calling Model() on a concurrently-closed client.
func (pc *PooledClient) Model() string {
	pc.mu.RLock()
	rc := pc.rc
	if rc == nil {
		pc.mu.RUnlock()
		return pc.cfg.Model
	}
	rc.Acquire()
	pc.mu.RUnlock()

	s := rc.client.Model()
	rc.Release()
	return s
}

// Close implements Client.Close.
// Returns nil immediately; actual client Close may fire later when the
// last in-flight request finishes. Calling Complete/Stream after Close
// is a programming error (matches Go conventions: sql.DB, http.Server).
func (pc *PooledClient) Close() error {
	pc.mu.Lock()
	rc := pc.rc
	pc.rc = nil
	pc.mu.Unlock()
	if rc != nil {
		rc.Release()
	}
	return nil
}

// PoolStatus returns the status of the auth pool.
func (pc *PooledClient) PoolStatus() string {
	return pc.pool.Status()
}
