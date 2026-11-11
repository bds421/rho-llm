package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"testing"
	"time"
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

// BUG 16/17: Config.MarshalJSON redacts APIKey but a credential embedded in
// BaseURL (userinfo or a `key=`/`token=` query param) leaked through.
func TestConfigMarshalRedactsBaseURLCredentials(t *testing.T) {
	const pw = "badpass-secret-9876"
	const q = "urlsecret-abc123"
	cfg := Config{Provider: "anthropic", APIKey: "sk-x", BaseURL: "https://user:" + pw + "@proxy.internal:8080/v1?key=" + q}
	b, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	s := string(b)
	if strings.Contains(s, pw) {
		t.Fatalf("Config JSON leaked a password embedded in BaseURL: %s", s)
	}
	if strings.Contains(s, q) {
		t.Fatalf("Config JSON leaked a credential query param in BaseURL: %s", s)
	}
}

// BUG 18/19: AuthProfile.MarshalJSON must likewise scrub a credential embedded
// in BaseURL (e.g. from the "apikey|baseurl" key form).
func TestAuthProfileMarshalRedactsBaseURLCredentials(t *testing.T) {
	const pw = "badpass-secret-9876"
	p := AuthProfile{Name: "p1", APIKey: "sk-x", BaseURL: "https://user:" + pw + "@proxy.internal/v1"}
	b, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(b), pw) {
		t.Fatalf("AuthProfile JSON leaked a password embedded in BaseURL: %s", b)
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

// L: keys shorter than 8 bytes used to be skipped entirely, so a short proxy
// or test key echoed in an error body leaked. Keys of 4–7 bytes are now
// scrubbed where they stand as a whole token, without mangling words that
// merely contain them.
func TestRedactSecretShortKeys(t *testing.T) {
	cases := []struct{ msg, key, want string }{
		{"Bearer abc1 rejected", "abc1", "Bearer REDACTED rejected"},
		{`{"key":"abc1"}`, "abc1", `{"key":"REDACTED"}`},
		{"abc1", "abc1", "REDACTED"},
		{"abc12 xabc1 abc1_x abc1-y", "abc1", "abc12 xabc1 abc1_x abc1-y"}, // embedded: untouched
		{"a real 5-char key ab3x9 leaked", "ab3x9", "a real 5-char key REDACTED leaked"},
		{"a-a-a-a a-a-a", "a-a-a", "a-a-a-a REDACTED"}, // overlapping/embedded matches never panic
		{"a.a.a.a.a", "a.a.a", "REDACTED.a.a"},         // overlap: first whole-token match wins, no panic
		{"x", "", "x"},                                 // empty key: no-op
		{"long key sk-0123456789 here", "sk-0123456789", "long key REDACTED here"},
		{"embedded-sk-0123456789x", "sk-0123456789", "embedded-REDACTEDx"}, // long keys: substring scrub as before
	}
	for _, tc := range cases {
		if got := redactSecret(tc.msg, tc.key); got != tc.want {
			t.Errorf("redactSecret(%q, %q) = %q, want %q", tc.msg, tc.key, got, tc.want)
		}
	}
}

// N4: keys under 4 bytes and well-known placeholder keys are not secrets;
// redacting them as whole words corrupted ordinary provider text.
func TestRedactSecretLeavesTinyAndPlaceholderKeys(t *testing.T) {
	cases := []struct{ msg, key string }{
		{"temperature must be none or a number", "none"},
		{"field must be None", "None"},
		{"value is EMPTY", "EMPTY"},
		{"value is empty", "EMPTY"},
		{"model not found, try ollama pull", "ollama"},
		{"dummy request rejected", "dummy"},
		{"k=k&k", "k"},
		{"an id is required", "id"},
		{"key key key", "key"},
	}
	for _, tc := range cases {
		if got := redactSecret(tc.msg, tc.key); got != tc.msg {
			t.Errorf("redactSecret(%q, %q) = %q, want unchanged", tc.msg, tc.key, got)
		}
		if got := RedactSecrets(tc.msg, Config{APIKey: tc.key}); got != tc.msg {
			t.Errorf("RedactSecrets(%q, key %q) = %q, want unchanged", tc.msg, tc.key, got)
		}
	}
	// A placeholder of 8+ bytes keeps the v0.9.4 substring scrub.
	if got := redactSecret("key sk-no-key-required", "sk-no-key-required"); got != "key REDACTED" {
		t.Errorf("long placeholder: %q", got)
	}
}

// Fuzz-ish: no input combination may panic or leave a whole-token key of the
// redacted length range (4–7 bytes, non-placeholder).
func TestRedactSecretShortKeyNeverPanics(t *testing.T) {
	alphabet := []string{"a", "b", "-", ".", " ", "ab", "a-"}
	keys := []string{"aaaa", "abab", "a-a-", "-a-a", "a.a.a", "aa-aa", "....", "a", "ab"}
	var gen func(prefix string, depth int)
	gen = func(prefix string, depth int) {
		for _, key := range keys {
			out := redactSecret(prefix, key)
			if len(key) < minWholeTokenSecretLen {
				if out != prefix {
					t.Fatalf("redactSecret(%q,%q) = %q altered text for a tiny key", prefix, key, out)
				}
				continue
			}
			for _, tok := range strings.FieldsFunc(out, func(r rune) bool { return !isTokenByte(byte(r)) }) {
				if tok == key && isTokenByte(key[0]) {
					t.Fatalf("redactSecret(%q,%q) = %q left the key as a token", prefix, key, out)
				}
			}
		}
		if depth == 0 {
			return
		}
		for _, a := range alphabet {
			gen(prefix+a, depth-1)
		}
	}
	gen("", 5)
}
