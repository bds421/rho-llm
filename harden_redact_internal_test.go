package llm

// Hardening pass 16 (internal) — PooledClient.redactErr must scrub credentials
// embedded in a per-key BaseURL override, not just the APIKey.

import (
	"errors"
	"strings"
	"testing"
)

func TestRedactErrStripsBaseURLCredentials(t *testing.T) {
	const apiKey = "sk-secret-longkey-1234"
	const baseURL = "https://user:badPassword99@internal-api:8443/v1?token=topSecretToken1"
	pc, err := NewPooledClient(DefaultConfig(), []string{apiKey + "|" + baseURL},
		func(AuthProfile) (Client, error) { return &closeSpy{provider: "zai", model: "glm-5.2"}, nil })
	if err != nil {
		t.Fatalf("NewPooledClient: %v", err)
	}
	defer pc.Close()

	// An error body that echoes both the key and the per-key BaseURL verbatim.
	redacted := pc.redactErr(errors.New("connection to " + apiKey + "|" + baseURL + " failed: 407"))

	for _, secret := range []string{apiKey, "badPassword99", "topSecretToken1"} {
		if strings.Contains(redacted, secret) {
			t.Errorf("redactErr leaked %q: %q", secret, redacted)
		}
	}
	// Host stays for debugging.
	if !strings.Contains(redacted, "internal-api") {
		t.Errorf("redactErr dropped the host (needed for debugging): %q", redacted)
	}
}
