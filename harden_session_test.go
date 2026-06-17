package llm_test

// Hardening pass 1 — session cost/model provenance.

import (
	"context"
	"iter"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// emptyModelClient is a (minimal/misbehaving) custom Client that returns a
// Response with an empty Model verbatim — unlike MockClient, it does NOT
// auto-fill it. Model() reports the model the turn actually runs against.
type emptyModelClient struct{ id string }

func (c *emptyModelClient) Complete(context.Context, llm.Request) (*llm.Response, error) {
	return &llm.Response{Model: "", Content: "ok", StopReason: llm.StopEndTurn, InputTokens: 1_000_000, OutputTokens: 1_000_000}, nil
}

func (c *emptyModelClient) Stream(context.Context, llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		yield(llm.StreamEvent{Type: llm.EventDone, StopReason: llm.StopEndTurn}, nil)
	}
}

func (c *emptyModelClient) Provider() string { return "custom" }
func (c *emptyModelClient) Model() string    { return c.id }
func (c *emptyModelClient) Close() error     { return nil }

// A Client that returns a Response with an empty Model must not silently zero
// out the cost: the turn ran against a known model (Model() / the request's), so
// usage must be priced against that model, not $0. EstimateCost("") returns 0 —
// a wrong, under-reported cost.
func TestSessionCostFallsBackToRequestModelWhenResponseModelEmpty(t *testing.T) {
	id := uniqueModelID("test-only-session-cost")
	if err := llm.RegisterModel(llm.ModelInfo{ID: id, Provider: "custom", InputPricePer1M: 10, OutputPricePer1M: 30}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}
	sess := llm.NewSession(&emptyModelClient{id: id})
	if _, err := sess.Send(context.Background(), "x"); err != nil {
		t.Fatalf("Send: %v", err)
	}
	if got := sess.Usage().Cost; got != 40.0 {
		t.Fatalf("session cost = %v, want 40.0 — empty Response.Model silently zeroed the cost", got)
	}
}
