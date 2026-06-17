package llm_test

// Regression tests for the v0.4.0 registry-extension fixes (architecture
// review findings R-M5 no runtime registration, R-M6 -short gating, plus the
// registryMu race protection). Break-the-system tests: each fails without its
// fix, and the concurrency test fails under -race without the lock.

import (
	"context"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// regTestSeq hands out process-unique suffixes so a test whose precondition is
// "this model is not registered yet" stays valid across repeated runs (e.g.
// `go test -count=2`), since RegisterModel mutates the process-global registry.
var regTestSeq atomic.Uint64

func uniqueModelID(prefix string) string {
	return fmt.Sprintf("%s-%d", prefix, regTestSeq.Add(1))
}

// countOccurrences reports how many times id appears in s.
func countOccurrences(s []string, id string) int {
	n := 0
	for _, v := range s {
		if v == id {
			n++
		}
	}
	return n
}

// R-M5: an unlisted model must be registrable at runtime so it gets a real
// cost estimate instead of the silent 0.
func TestRegisterModelEnablesCostEstimate(t *testing.T) {
	id := uniqueModelID("test-only-custom-model")
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

// #6: GetAvailableModels (the curated discovery list) must stay in sync with the
// registry across an override. The old code gated the append on `!existed`, so
// re-registering a model already in modelRegistry but absent from
// availableModels never reached discovery. Re-register such a built-in (with its
// own metadata, unchanged) and it must surface.
func TestRegisterModelOverrideSurfacesInAvailableModels(t *testing.T) {
	var gapID, gapProvider string
	for _, m := range llm.Models() {
		if !slices.Contains(llm.GetAvailableModels(m.Provider), m.ID) {
			gapID, gapProvider = m.ID, m.Provider
			break
		}
	}
	if gapID == "" {
		t.Skip("no built-in is currently absent from GetAvailableModels; nothing to pin")
	}
	info, ok := llm.GetModelInfo(gapID)
	if !ok {
		t.Fatalf("GetModelInfo(%q) missing", gapID)
	}
	// Re-register identical metadata — non-destructive, but exercises the
	// override path (existed == true).
	if err := llm.RegisterModel(info); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}
	if !slices.Contains(llm.GetAvailableModels(gapProvider), gapID) {
		t.Fatalf("after RegisterModel override, %q still absent from GetAvailableModels(%q) — override never reached discovery", gapID, gapProvider)
	}
	if c := countOccurrences(llm.GetAvailableModels(gapProvider), gapID); c != 1 {
		t.Fatalf("model %q appears %d times in discovery, want exactly 1 (no duplicates)", gapID, c)
	}
}

// #6: an override that moves a model to a different provider must update both
// providers' discovery lists — add to the new, remove from the old — and never
// duplicate. The old `!existed` append-gate left the new provider missing it.
func TestRegisterModelProviderChangeUpdatesDiscovery(t *testing.T) {
	const id = "test-only-move-provider-v1"
	const p1, p2 = "test-only-p1", "test-only-p2"

	if err := llm.RegisterModel(llm.ModelInfo{ID: id, Provider: p1, InputPricePer1M: 1}); err != nil {
		t.Fatalf("RegisterModel p1: %v", err)
	}
	if !slices.Contains(llm.GetAvailableModels(p1), id) {
		t.Fatal("new model missing from its provider's discovery list")
	}

	// Move it to a different provider.
	if err := llm.RegisterModel(llm.ModelInfo{ID: id, Provider: p2, InputPricePer1M: 1}); err != nil {
		t.Fatalf("RegisterModel p2: %v", err)
	}
	if !slices.Contains(llm.GetAvailableModels(p2), id) {
		t.Fatal("override to a new provider didn't add the model to the new provider's discovery (append gated on !existed)")
	}
	if slices.Contains(llm.GetAvailableModels(p1), id) {
		t.Fatal("override to a new provider left a stale entry in the old provider's discovery")
	}

	// Re-register under the same provider: still exactly one entry, no dup.
	if err := llm.RegisterModel(llm.ModelInfo{ID: id, Provider: p2, InputPricePer1M: 2}); err != nil {
		t.Fatalf("RegisterModel p2 again: %v", err)
	}
	if c := countOccurrences(llm.GetAvailableModels(p2), id); c != 1 {
		t.Fatalf("model appears %d times after same-provider re-register, want 1", c)
	}
}

// #10: an alias key equal to a real model ID would shadow that model
// (ResolveModelAlias checks aliases first) — RegisterModelAlias must reject it,
// and the real model must keep resolving to itself.
func TestRegisterModelAliasRejectsShadowingRealModel(t *testing.T) {
	const a = "test-only-shadow-a-v1"
	const b = "test-only-shadow-b-v1"
	if err := llm.RegisterModel(llm.ModelInfo{ID: a, Provider: "custom", InputPricePer1M: 1}); err != nil {
		t.Fatalf("RegisterModel a: %v", err)
	}
	if err := llm.RegisterModel(llm.ModelInfo{ID: b, Provider: "custom", InputPricePer1M: 2}); err != nil {
		t.Fatalf("RegisterModel b: %v", err)
	}
	if err := llm.RegisterModelAlias(a, b); err == nil {
		t.Fatal("RegisterModelAlias with an alias key equal to an existing model ID must be rejected (would shadow it)")
	}
	if got := llm.ResolveModelAlias(a); got != a {
		t.Fatalf("real model %q was shadowed by an alias: ResolveModelAlias(%q) = %q", a, a, got)
	}
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
