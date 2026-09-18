package llm_test

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider"
)

// Attack: a negative MaxTokens must be REJECTED, never silently floored into a
// valid-looking request. Flooring a negative would hide a caller bug.
func TestBreakNegativeMaxTokensRejectedNotFloored(t *testing.T) {
	_, err := llm.NewClient(llm.Config{
		Provider: "anthropic", Model: "claude-sonnet-4-6", APIKey: "k", MaxTokens: -1,
	})
	if err == nil {
		t.Fatal("MaxTokens=-1 accepted: a caller bug was silently converted into a valid request")
	}
	if !strings.Contains(err.Error(), "MaxTokens") {
		t.Errorf("error = %q, want it to name MaxTokens", err.Error())
	}
}

// TestBreakTimeoutFlooredOnEveryConstructor pins the Timeout floor.
//
// Found by mutation audit: disabling the floor in applyConfigFloors left the
// whole suite green. A zero or negative Timeout produces an http.Client with
// no deadline, so a hung endpoint stalls the caller forever instead of
// returning a timeout error.
//
// Asserted at NewSafeHTTPClient — the exported seam every adapter builds its
// transport through. Note this pins NewSafeHTTPClient's OWN floor: the
// applyConfigFloors Timeout branch is redundant with it (see the comment
// there), so no test can kill a mutation of that branch alone. The property
// that matters — no client is ever unbounded — is what this asserts.
// Driving a real request instead would measure the pool's retry/backoff
// budget, not the floor.
func TestBreakTimeoutFlooredOnEveryConstructor(t *testing.T) {
	for _, tc := range []struct {
		name    string
		timeout time.Duration
	}{
		{"zero (struct-literal default)", 0},
		{"negative", -5 * time.Second},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, err := llm.NewClient(llm.Config{
				Provider: "anthropic", Model: "claude-sonnet-4-6", APIKey: "k",
				Timeout: tc.timeout,
			})
			if err != nil {
				t.Fatalf("NewClient: %v", err)
			}
			defer c.Close()

			hc, err := llm.NewSafeHTTPClient(llm.Config{Timeout: tc.timeout})
			if err != nil {
				t.Fatalf("NewSafeHTTPClient: %v", err)
			}
			if hc.Timeout <= 0 {
				t.Errorf("http.Client.Timeout = %v for input %v: an unbounded client stalls "+
					"the caller indefinitely against a hung endpoint", hc.Timeout, tc.timeout)
			}
			if hc.Timeout != llm.DefaultTimeout {
				t.Errorf("http.Client.Timeout = %v, want DefaultTimeout (%v)", hc.Timeout, llm.DefaultTimeout)
			}
		})
	}
}

// Attack: the retirement error must not leak into unrelated failures, and a
// retired ID must never silently dispatch to the network.
func TestBreakRetiredModelNeverDispatches(t *testing.T) {
	c, err := llm.NewClient(llm.Config{
		Provider: "anthropic",
		Model:    "claude-sonnet-4-20250514",
		APIKey:   "k",
		BaseURL:  "http://127.0.0.1:1", // any dial here means validation was bypassed
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer c.Close()

	_, err = c.Complete(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "x")},
	})
	if err == nil {
		t.Fatal("retired model dispatched successfully")
	}
	if !strings.Contains(err.Error(), "retired") {
		t.Errorf("error = %q; want the retirement error, not a transport error — "+
			"the caller must learn the model is gone, not that the network failed", err.Error())
	}
}

// Attack: RetiredModelReplacement under concurrent load. The map is read by
// every dispatch; a data race here corrupts every request.
func TestBreakRetiredModelReplacementConcurrent(t *testing.T) {
	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 2000; i++ {
				if r, ok := llm.RetiredModelReplacement("claude-sonnet-4-20250514"); !ok || r == "" {
					t.Error("retired lookup returned empty under concurrency")
					return
				}
				if _, ok := llm.RetiredModelReplacement("claude-sonnet-4-6"); ok {
					t.Error("live model reported as retired under concurrency")
					return
				}
				_, _ = llm.RetiredModelReplacement("")
			}
		}()
	}
	wg.Wait()
}

// Attack: RegisterModel must win over a retirement entry — otherwise the
// documented escape hatch ("re-add it with RegisterModel") is a lie.
//
// This mutates the process-global registry and there is no UnregisterModel, so
// it deliberately uses the ID no other test asserts on; picking a shared one
// leaks into TestRetiredModelGivesActionableError.
func TestBreakRegisterModelOverridesRetirement(t *testing.T) {
	const id = "claude-3-haiku-20240307"
	if _, retired := llm.RetiredModelReplacement(id); !retired {
		t.Fatalf("%s is not a retired ID; this test no longer exercises the escape hatch", id)
	}
	if err := llm.RegisterModel(llm.ModelInfo{
		ID: id, Provider: "anthropic", MaxTokens: 4096, ContextWindow: 200000,
		Capabilities: llm.CapabilitySet(llm.CapabilityChat),
	}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}
	c, err := llm.NewClient(llm.Config{Provider: "anthropic", Model: id, APIKey: "k"})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer c.Close()
	_, err = c.Complete(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "x")},
	})
	if err != nil && strings.Contains(err.Error(), "retired") {
		t.Error("RegisterModel could not resurrect a retired ID: the documented escape hatch does not work")
	}
}
