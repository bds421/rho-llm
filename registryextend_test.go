package llm_test

// Regression tests for the v0.4.0 registry-extension fixes (architecture
// review findings R-M5 no runtime registration, R-M6 -short gating, plus the
// registryMu race protection). Break-the-system tests: each fails without its
// fix, and the concurrency test fails under -race without the lock.

import (
	"context"
	"sync"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// R-M5: an unlisted model must be registrable at runtime so it gets a real
// cost estimate instead of the silent 0.
func TestRegisterModelEnablesCostEstimate(t *testing.T) {
	const id = "test-only-custom-model-v1"
	if got := llm.EstimateCost(llm.CostInput{Model: id, InputTokens: 1_000_000}); got != 0 {
		t.Fatalf("unlisted model should cost 0 before registration, got %v", got)
	}

	if err := llm.RegisterModel(llm.ModelInfo{
		ID:               id,
		Provider:         "custom",
		InputPricePer1M:  2.0,
		OutputPricePer1M: 6.0,
	}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}

	got := llm.EstimateCost(llm.CostInput{Model: id, InputTokens: 1_000_000, OutputTokens: 1_000_000})
	if want := 8.0; got != want {
		t.Fatalf("after registration cost = %v, want %v", got, want)
	}

	info, ok := llm.GetModelInfo(id)
	if !ok || info.Provider != "custom" {
		t.Fatalf("GetModelInfo(%q) = %+v, %v", id, info, ok)
	}

	// It must also surface through discovery for its provider.
	found := false
	for _, m := range llm.ModelsByProvider("custom") {
		if m.ID == id {
			found = true
		}
	}
	if !found {
		t.Fatal("registered model missing from ModelsByProvider")
	}
}

// R-M5: registering with missing required fields must error, not corrupt the
// registry.
func TestRegisterModelRejectsInvalid(t *testing.T) {
	if err := llm.RegisterModel(llm.ModelInfo{Provider: "custom"}); err == nil {
		t.Error("RegisterModel without ID must error")
	}
	if err := llm.RegisterModel(llm.ModelInfo{ID: "x-no-provider"}); err == nil {
		t.Error("RegisterModel without Provider must error")
	}
	// An alias to an unknown model must be rejected (else it silently resolves
	// to a non-existent id).
	if err := llm.RegisterModelAlias("test-alias-x", "no-such-model-xyz"); err == nil {
		t.Error("RegisterModelAlias to unknown model must error")
	}
}

// R-M5: a registered alias must resolve and feed cost estimation.
func TestRegisterModelAliasResolves(t *testing.T) {
	const id = "test-only-alias-target-v1"
	if err := llm.RegisterModel(llm.ModelInfo{ID: id, Provider: "custom", InputPricePer1M: 1.0}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}
	if err := llm.RegisterModelAlias("test-only-alias-v1", id); err != nil {
		t.Fatalf("RegisterModelAlias: %v", err)
	}
	if got := llm.ResolveModelAlias("test-only-alias-v1"); got != id {
		t.Fatalf("ResolveModelAlias = %q, want %q", got, id)
	}
}

// registryMu race guard: concurrent RegisterModel and readers must be
// data-race-free. Run under -race; without the lock the map access trips the
// detector (or panics on concurrent map read/write).
func TestRegistryConcurrentAccessRaceFree(t *testing.T) {
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			id := "race-model-" + string(rune('a'+n))
			_ = llm.RegisterModel(llm.ModelInfo{ID: id, Provider: "race", InputPricePer1M: float64(n)})
			_ = llm.EstimateCost(llm.CostInput{Model: id, InputTokens: 100})
			_, _ = llm.GetModelInfo(id)
			_ = llm.Models()
			_ = llm.ModelsByProvider("race")
			_ = llm.ResolveModelAlias(id)
		}(i)
	}
	wg.Wait()
}

// R-M5 end-to-end: a client built for an unlisted model/provider (via BaseURL)
// must work, and registering metadata for it must light up cost in the
// session usage. Uses a mock client so no network is needed.
func TestRegisteredModelFeedsSessionUsage(t *testing.T) {
	const id = "test-only-session-model-v1"
	if err := llm.RegisterModel(llm.ModelInfo{ID: id, Provider: "custom", InputPricePer1M: 10.0, OutputPricePer1M: 30.0}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}
	mock := llm.NewMockClient("custom", id)
	mock.PushResponse(&llm.Response{Model: id, Content: "hi", StopReason: llm.StopEndTurn, InputTokens: 1_000_000, OutputTokens: 1_000_000})
	sess := llm.NewSession(mock, llm.WithBaseRequest(llm.Request{Model: id}))
	if _, err := sess.Send(context.Background(), "x"); err != nil {
		t.Fatalf("Send: %v", err)
	}
	if got := sess.Usage().Cost; got != 40.0 {
		t.Fatalf("session cost = %v, want 40.0 (registered pricing not applied)", got)
	}
}
