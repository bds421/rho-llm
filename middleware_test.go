package llm_test

import (
	"context"
	"iter"
	"testing"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider"
)

// A misbehaving inner Client that returns (nil, nil) must not crash the
// LoggingClient decorator: it checks err but then dereferences resp.InputTokens.
func TestLoggingClientDoesNotPanicOnNilResponse(t *testing.T) {
	lc := llm.WithLogging(nilClient{})
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("LoggingClient.Complete panicked on a (nil,nil) inner response: %v", r)
		}
	}()
	resp, err := lc.Complete(context.Background(), llm.Request{Model: "m"})
	// Transparent pass-through of the (nil,nil) is fine; the contract is just
	// "do not panic" — the caller (e.g. Session) handles the nil response.
	if resp != nil {
		t.Fatalf("expected nil response passed through, got %+v", resp)
	}
	_ = err
}

func (m *mockClient) Complete(ctx context.Context, req llm.Request) (*llm.Response, error) {
	if m.completeFunc != nil {
		return m.completeFunc(ctx, req)
	}
	return m.resp, m.err
}

func (m *mockClient) Stream(_ context.Context, _ llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		if m.err != nil {
			yield(llm.StreamEvent{}, m.err)
			return
		}
		if !yield(llm.StreamEvent{Type: llm.EventContent, Text: "hello"}, nil) {
			return
		}
		yield(llm.StreamEvent{Type: llm.EventDone, StopReason: "end_turn", InputTokens: 10, OutputTokens: 5}, nil)
	}
}

func (m *mockClient) Provider() string { return m.provider }

func (m *mockClient) Model() string { return m.model }

func (m *mockClient) Close() error { return nil }

// TestLoggingClientDelegates verifies the logging wrapper delegates correctly.
func TestLoggingClientDelegates(t *testing.T) {
	inner := &mockClient{
		provider: "test",
		model:    "test-model",
		resp: &llm.Response{
			Content:      "hello",
			InputTokens:  10,
			OutputTokens: 5,
			StopReason:   "end_turn",
		},
	}

	client := llm.WithLogging(inner)

	if client.Provider() != "test" {
		t.Errorf("Provider() = %q, want test", client.Provider())
	}
	if client.Model() != "test-model" {
		t.Errorf("Model() = %q, want test-model", client.Model())
	}

	// Test Complete
	resp, err := client.Complete(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}
	if resp.Content != "hello" {
		t.Errorf("Content = %q, want hello", resp.Content)
	}

	// Test Stream
	var chunks int
	for event, err := range client.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage("user", "hi")},
	}) {
		if err != nil {
			t.Fatalf("Stream failed: %v", err)
		}
		if event.Type == llm.EventContent {
			chunks++
		}
	}
	if chunks != 1 {
		t.Errorf("chunks = %d, want 1", chunks)
	}
}

// TestLoggingClientError verifies error propagation.
func TestLoggingClientError(t *testing.T) {
	apiErr := llm.NewRateLimitError("test", "too fast")
	inner := &mockClient{
		provider: "test",
		model:    "test-model",
		err:      apiErr,
	}

	client := llm.WithLogging(inner)

	_, err := client.Complete(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage("user", "hi")},
	})
	if err == nil {
		t.Fatal("Expected error")
	}
	if !llm.IsRateLimited(err) {
		t.Errorf("Expected rate limit error, got: %v", err)
	}
}

// TestWithLoggingPrefix verifies the prefix logging wrapper.
func TestWithLoggingPrefix(t *testing.T) {
	inner := &mockClient{
		provider: "test",
		model:    "test-model",
		resp:     &llm.Response{Content: "hello"},
	}

	client := llm.WithLoggingPrefix(inner, "[MyApp]")

	if client.Provider() != "test" {
		t.Errorf("Provider() = %q, want test", client.Provider())
	}

	resp, err := client.Complete(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}
	if resp.Content != "hello" {
		t.Errorf("Content = %q, want hello", resp.Content)
	}
}

// TestLoggingClientClose verifies Close delegation.
func TestLoggingClientClose(t *testing.T) {
	inner := &mockClient{provider: "test", model: "test-model"}
	client := llm.WithLogging(inner)

	err := client.Close()
	if err != nil {
		t.Errorf("Close() error: %v", err)
	}
}

// TestLoggingClientStreamEarlyBreak verifies stream cleanup on early break.
func TestLoggingClientStreamEarlyBreak(t *testing.T) {
	inner := &mockClient{
		provider: "test",
		model:    "test-model",
	}

	client := llm.WithLogging(inner)

	// Break after first event
	for event, err := range client.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage("user", "hi")},
	}) {
		if err != nil {
			t.Fatalf("Stream error: %v", err)
		}
		if event.Type == llm.EventContent {
			break // Early break
		}
	}
	// Should not panic or leak
}

// TestLoggingClientStreamError verifies stream error logging.
func TestLoggingClientStreamError(t *testing.T) {
	apiErr := llm.NewRateLimitError("test", "too fast")
	inner := &mockClient{
		provider: "test",
		model:    "test-model",
		err:      apiErr,
	}

	client := llm.WithLogging(inner)

	for _, err := range client.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage("user", "hi")},
	}) {
		if err != nil {
			if !llm.IsRateLimited(err) {
				t.Errorf("expected rate limit error, got: %v", err)
			}
			break
		}
	}
}
