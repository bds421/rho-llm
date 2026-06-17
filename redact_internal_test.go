package llm

// Break-the-system: secret leakage. A provider error body can echo the request
// API key (e.g. "401: key sk-... rejected"); the library must never leak that
// key into its own logs or serialized state. Negative assertions — the secret
// must be ABSENT.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"testing"
	"time"
)

const leakKey = "sk-secret-key-abcdef0123456789XYZ"

// BUG 8: AuthProfile.MarshalJSON redacts APIKey but a server-echoed key stored in
// LastError leaked through — defeating the very purpose of the redacting marshaler.
func TestAuthProfileNeverSerializesAPIKeyEvenViaLastError(t *testing.T) {
	p := AuthProfile{Name: "p1", APIKey: leakKey}
	p.MarkFailed(fmt.Errorf("401 Unauthorized: key %s is invalid", leakKey), time.Second)

	if strings.Contains(p.LastError, leakKey) {
		t.Fatalf("MarkFailed stored the raw API key in LastError: %q", p.LastError)
	}
	b, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(b), leakKey) {
		t.Fatalf("AuthProfile JSON leaked the API key: %s", b)
	}
}

// BUG 9: the pool must not log the API key when a profile fails with an error
// whose message echoes the key.
func TestPoolDoesNotLogAPIKeyOnFailure(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))
	defer slog.SetDefault(prev)

	pc, err := NewPooledClient(DefaultConfig(), []string{leakKey}, func(AuthProfile) (Client, error) {
		return newProbeTestClient(func(context.Context) (*Response, error) {
			return nil, NewAuthError("test", "invalid key "+leakKey+" rejected", 401)
		}), nil
	})
	if err != nil {
		t.Fatalf("NewPooledClient: %v", err)
	}
	defer pc.Close()
	_, _ = pc.Complete(context.Background(), Request{})

	if strings.Contains(buf.String(), leakKey) {
		t.Fatalf("pool logged the API key in its output:\n%s", buf.String())
	}
}
