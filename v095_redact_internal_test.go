package llm

import (
	"strings"
	"testing"
)

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
