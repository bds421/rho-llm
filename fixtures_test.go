package llm_test

import (
	"context"
	"encoding/base64"
	"iter"
	"os"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider"
)

// TestPrompt is a simple prompt that all models should handle.
const TestPrompt = "Reply with exactly one word: Hello"

// floatPtr is a helper to create *float64 for Temperature fields.
func floatPtr(v float64) *float64 { return &v }

// envKey reads an API key from env, supporting both singular and plural forms.
// Checks KEY first, then KEYS (returns first comma-separated value).
func envKey(singular, plural string) string {
	if v := os.Getenv(singular); v != "" {
		return v
	}
	if v := os.Getenv(plural); v != "" {
		if idx := strings.Index(v, ","); idx > 0 {
			return v[:idx]
		}
		return v
	}
	return ""
}

// envKeys reads all API keys from env (plural form, comma-separated).
// Falls back to singular form as a single-element list.
func envKeys(singular, plural string) []string {
	if v := os.Getenv(plural); v != "" {
		var keys []string
		for _, k := range strings.Split(v, ",") {
			k = strings.TrimSpace(k)
			if k != "" {
				keys = append(keys, k)
			}
		}
		if len(keys) > 0 {
			return keys
		}
	}
	if v := os.Getenv(singular); v != "" {
		return []string{v}
	}
	return nil
}

// mockClient is a minimal Client for testing middleware.
type mockClient struct {
	provider     string
	model        string
	resp         *llm.Response
	err          error
	completeFunc func(context.Context, llm.Request) (*llm.Response, error)
}

// requireLiveAPI gates a test that calls a real provider API: it is skipped in
// -short mode (what `make test` runs — keys in the developer's env must not
// trigger live calls and spend) and when no key is configured.
func requireLiveAPI(t *testing.T, key, skipMsg string) {
	t.Helper()
	if testing.Short() {
		t.Skip("integration test: skipped in -short mode")
	}
	if key == "" {
		t.Skip(skipMsg)
	}
}

// failingMockClient is a configurable mock for pool tests.
// It returns responses from the calls slice in order. When exhausted, it reuses the last entry.
type failingMockClient struct {
	calls    []mockCall
	callIdx  int
	provider string
	model    string
	closed   bool
}

type mockCall struct {
	resp         *llm.Response
	err          error
	midStreamErr error // if set, yield one content event then this error
}

// threadSafeMockClient is a simple mock that always returns the same result.
// Safe for concurrent use because it has no mutable state.
type threadSafeMockClient struct {
	provider string
	model    string
	resp     *llm.Response
	err      error
}

// loadTestImage reads a test image from testdata/ and returns its base64 encoding.
func loadTestImage(t *testing.T, filename string) string {
	t.Helper()
	data, err := os.ReadFile("testdata/" + filename)
	if err != nil {
		t.Fatalf("failed to read testdata/%s: %v", filename, err)
	}
	return base64.StdEncoding.EncodeToString(data)
}

// nilClient is a hostile Client: Complete returns (nil, nil) — no response and
// no error, which a buggy custom adapter could do.
type nilClient struct{}

func (nilClient) Complete(context.Context, llm.Request) (*llm.Response, error) {
	return nil, nil
}

func (nilClient) Stream(_ context.Context, _ llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {}
}

func (nilClient) Provider() string { return "nilprov" }

func (nilClient) Model() string { return "nilmodel" }

func (nilClient) Close() error { return nil }

// keyClient answers per API key: fn decides each call from the key, so a pool
// test can make one key fail and another succeed. Stream yields fn's result as
// one content event plus Done, or its error.
type keyClient struct {
	key string
	fn  func(key string, req llm.Request) (*llm.Response, error)
}

func (c *keyClient) Complete(_ context.Context, req llm.Request) (*llm.Response, error) {
	return c.fn(c.key, req)
}

func (c *keyClient) Stream(_ context.Context, req llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		resp, err := c.fn(c.key, req)
		if err != nil {
			yield(llm.StreamEvent{}, err)
			return
		}
		if yield(llm.StreamEvent{Type: llm.EventContent, Text: resp.Content}, nil) {
			yield(llm.StreamEvent{Type: llm.EventDone, StopReason: llm.StopEndTurn}, nil)
		}
	}
}

func (c *keyClient) Provider() string { return "test" }
func (c *keyClient) Model() string    { return "m" }
func (c *keyClient) Close() error     { return nil }

// keyPool builds a PooledClient whose keys are served by keyClient.
func keyPool(t *testing.T, cfg llm.Config, keys []string, fn func(string, llm.Request) (*llm.Response, error)) *llm.PooledClient {
	t.Helper()
	pc, err := llm.NewPooledClient(cfg, keys, func(p llm.AuthProfile) (llm.Client, error) {
		return &keyClient{key: p.APIKey, fn: fn}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = pc.Close() })
	return pc
}
