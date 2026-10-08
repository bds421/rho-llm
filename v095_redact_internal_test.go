package llm

import (
	"strings"
	"testing"
)

// L: keys shorter than 8 bytes used to be skipped entirely, so a short proxy
// or test key echoed in an error body leaked. They are now scrubbed where they
// stand as a whole token, without mangling words that merely contain them.
func TestRedactSecretShortKeys(t *testing.T) {
	cases := []struct{ msg, key, want string }{
		{"Bearer abc1 rejected", "abc1", "Bearer REDACTED rejected"},
		{`{"key":"abc1"}`, "abc1", `{"key":"REDACTED"}`},
		{"abc1", "abc1", "REDACTED"},
		{"abc12 xabc1 abc1_x abc1-y", "abc1", "abc12 xabc1 abc1_x abc1-y"}, // embedded: untouched
		{"k=k&k", "k", "REDACTED=REDACTED&REDACTED"},
		{"keep the kitchen", "k", "keep the kitchen"},
		{"a-a-a a-a", "a-a", "a-a-a REDACTED"}, // overlapping/embedded matches never panic
		{"a.a.a", "a.a", "REDACTED.a"},         // overlap: first whole-token match wins, no panic
		{"x", "", "x"},                         // empty key: no-op
		{"long key sk-0123456789 here", "sk-0123456789", "long key REDACTED here"},
		{"embedded-sk-0123456789x", "sk-0123456789", "embedded-REDACTEDx"}, // long keys: substring scrub as before
	}
	for _, tc := range cases {
		if got := redactSecret(tc.msg, tc.key); got != tc.want {
			t.Errorf("redactSecret(%q, %q) = %q, want %q", tc.msg, tc.key, got, tc.want)
		}
	}
}

// Fuzz-ish: no input combination may panic or leave a whole-token short key.
func TestRedactSecretShortKeyNeverPanics(t *testing.T) {
	alphabet := []string{"a", "b", "-", ".", " ", "ab", "a-"}
	keys := []string{"a", "ab", "a-", "-a", "a.a", "aa", "."}
	var gen func(prefix string, depth int)
	gen = func(prefix string, depth int) {
		for _, key := range keys {
			out := redactSecret(prefix, key)
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
