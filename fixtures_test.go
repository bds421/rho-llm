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
